
import logging
from typing import Optional, Sequence, Tuple, List
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt.layers.mamba2_block import BidirectionalMamba2Block

logger = logging.getLogger(__name__)
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class Aggregator(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0,
                 num_register_tokens=4, block_fn=Block, qkv_bias=True, proj_bias=True, ffn_bias=True,
                 patch_embed='dinov2_vitl14_reg', aa_order=['frame', 'global'], aa_block_size=1, qk_norm=True,
                 rope_freq=100, init_values=0.01, halo_ratio=0, mamba2_d_state=8, mamba2_d_conv=4,
                 mamba2_expand=2, keep_global_attn_indices: Optional[Sequence[int]] = None,
                 replace_global_with_mamba2_indices: Optional[Sequence[int]] = None, mamba_mlp_ratio: Optional[float] = None):
        super().__init__()
        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        attn_kwargs = dict(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias,
                           ffn_bias=ffn_bias, init_values=init_values, qk_norm=qk_norm, rope=self.rope)
        self.frame_blocks = nn.ModuleList([block_fn(**attn_kwargs) for _ in range(depth)])
        layout = self._resolve_global_layout(depth, halo_ratio, keep_global_attn_indices, replace_global_with_mamba2_indices)
        self.keep_global_attn_indices = layout['attention_indices']
        self.replace_global_with_mamba2_indices = layout['mamba_indices']
        mamba_mlp_ratio = mlp_ratio if mamba_mlp_ratio is None else mamba_mlp_ratio
        self.global_blocks = nn.ModuleList([
            block_fn(**attn_kwargs) if i in self.keep_global_attn_indices else
            BidirectionalMamba2Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mamba_mlp_ratio,
                                     d_state=mamba2_d_state, d_conv=mamba2_d_conv, expand=mamba2_expand,
                                     init_values=init_values, ffn_bias=ffn_bias, rope=self.rope)
            for i in range(depth)
        ])
        logger.info('paper hybrid global layout attention=%s mamba=%s', self.keep_global_attn_indices, self.replace_global_with_mamba2_indices)
        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f'depth ({depth}) must be divisible by aa_block_size ({aa_block_size})')
        self.aa_block_num = self.depth // self.aa_block_size
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        for name, value in (('_resnet_mean', _RESNET_MEAN), ('_resnet_std', _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)
        self.use_reentrant = False

    @staticmethod
    def _normalize_indices(indices: Sequence[int], depth: int, name: str) -> List[int]:
        vals = sorted({int(i) for i in indices})
        for i in vals:
            if i < 0 or i >= depth:
                raise ValueError(f'{name} contains invalid layer index {i} for depth={depth}')
        return vals

    @classmethod
    def _resolve_global_layout(cls, depth: int, halo_ratio: int, keep_global_attn_indices, replace_global_with_mamba2_indices):
        if keep_global_attn_indices is not None and replace_global_with_mamba2_indices is not None:
            raise ValueError('Specify either keep_global_attn_indices or replace_global_with_mamba2_indices, not both')
        if keep_global_attn_indices is not None:
            attn = cls._normalize_indices(keep_global_attn_indices, depth, 'keep_global_attn_indices')
            mamba = [i for i in range(depth) if i not in attn]
        elif replace_global_with_mamba2_indices is not None:
            mamba = cls._normalize_indices(replace_global_with_mamba2_indices, depth, 'replace_global_with_mamba2_indices')
            attn = [i for i in range(depth) if i not in mamba]
        elif halo_ratio and halo_ratio > 0:
            attn = [i for i in range(depth) if (i + 1) % halo_ratio == 0]
            mamba = [i for i in range(depth) if i not in attn]
        else:
            attn = list(range(depth))
            mamba = []
        return {'attention_indices': attn, 'mamba_indices': mamba}

    def __build_patch_embed__(self, patch_embed, img_size, patch_size, num_register_tokens, interpolate_antialias=True, interpolate_offset=0.0, block_chunks=0, init_values=1.0, embed_dim=1024):
        if 'conv' in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {'dinov2_vitl14_reg': vit_large, 'dinov2_vitb14_reg': vit_base, 'dinov2_vits14_reg': vit_small, 'dinov2_vitg2_reg': vit_giant2}
            self.patch_embed = vit_models[patch_embed](img_size=img_size, patch_size=patch_size, num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias, interpolate_offset=interpolate_offset, block_chunks=block_chunks, init_values=init_values)
            if hasattr(self.patch_embed, 'mask_token'):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f'Expected 3 input channels, got {C_in}')
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens['x_norm_patchtokens']
        _, P, C = patch_tokens.shape
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        P = tokens.shape[1]
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
        if pos is not None and self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        frame_idx = 0
        global_idx = 0
        output_list = []
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == 'frame':
                    tokens, frame_idx, frame_inter = self._process_frame_attention(tokens, B, S, P, C, frame_idx, pos=pos)
                elif attn_type == 'global':
                    tokens, global_idx, global_inter = self._process_global_attention(tokens, B, S, P, C, global_idx, pos=pos)
                else:
                    raise ValueError(f'Unknown attention type: {attn_type}')
            for i in range(len(frame_inter)):
                output_list.append(torch.cat([frame_inter[i], global_inter[i]], dim=-1))
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        inter = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            inter.append(tokens.view(B, S, P, C))
        return tokens, frame_idx, inter

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)
        inter = []
        for _ in range(self.aa_block_size):
            blk = self.global_blocks[global_idx]
            if self.training:
                tokens = checkpoint(blk, tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = blk(tokens, pos=pos)
            global_idx += 1
            inter.append(tokens.view(B, S, P, C))
        return tokens, global_idx, inter


def slice_expand_and_flatten(token_tensor, B, S):
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    return combined.view(B * S, *combined.shape[2:])
