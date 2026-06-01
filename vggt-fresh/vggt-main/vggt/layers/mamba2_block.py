
import logging
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layer_scale import LayerScale
from .mlp import Mlp

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba2 as _CUDAMamba2
    MAMBA2_AVAILABLE = True
except Exception:
    _CUDAMamba2 = None
    MAMBA2_AVAILABLE = False

class PureTorchMamba2(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.headdim = d_state
        assert self.d_inner % self.headdim == 0
        self.n_heads = self.d_inner // self.headdim
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.n_heads, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.norm = nn.RMSNorm(self.d_inner)

    def forward(self, x: Tensor) -> Tensor:
        b, l, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z, B_param, C_param, dt = xz.split([self.d_inner, self.d_inner, self.d_state, self.d_state, self.n_heads], dim=-1)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :l].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)
        y = self._scan(x_conv, dt, A, B_param, C_param)
        y = self.norm(y) * F.silu(z)
        return self.out_proj(y)

    def _scan(self, x, dt, A, B, C):
        batch, length, _ = x.shape
        n_heads = self.n_heads
        headdim = self.headdim
        x = x.view(batch, length, n_heads, headdim)
        dt = dt.unsqueeze(-1)                               # (B, L, H, 1)
        dA = torch.exp(A.view(1, 1, n_heads, 1) * dt)      # (B, L, H, 1)
        h = torch.zeros(batch, n_heads, headdim, self.d_state, device=x.device, dtype=x.dtype)
        out = torch.empty(batch, length, n_heads, headdim, device=x.device, dtype=x.dtype)
        for t in range(length):
            x_scaled = x[:, t] * dt[:, t]                  # (B, H, D)
            h = h * dA[:, t].unsqueeze(-1) + x_scaled.unsqueeze(-1) * B[:, t, None, None, :]
            out[:, t] = (h * C[:, t, None, None, :]).sum(dim=-1)
        return out.reshape(batch, length, n_heads * headdim)

class _PositionalInjector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU(), nn.Linear(dim, dim, bias=False))
        self.gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _sincos(pos: Tensor, dim: int) -> Tensor:
        pos = pos.float()
        maxv = pos.amax(dim=1, keepdim=True).clamp_min(1.0)
        p = 2.0 * (pos / maxv) - 1.0
        quarter = max(dim // 4, 1)
        freq = torch.arange(quarter, device=pos.device, dtype=torch.float32)
        freq = torch.exp(-math.log(10000.0) * freq / max(quarter - 1, 1))
        row = p[..., 0:1] * freq
        col = p[..., 1:2] * freq
        emb = torch.cat([torch.sin(row), torch.cos(row), torch.sin(col), torch.cos(col)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb[..., :dim]

    def forward(self, x: Tensor, pos: Optional[Tensor]) -> Tensor:
        if pos is None:
            return x
        pe = self._sincos(pos, x.shape[-1]).to(dtype=x.dtype)
        return x + torch.tanh(self.gate) * self.proj(pe)

class BidirectionalMamba2Block(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0, d_state: int = 8, d_conv: int = 4,
                 expand: int = 2, init_values: Optional[float] = None, drop: float = 0.0, drop_path: float = 0.0,
                 act_layer: Callable[..., nn.Module] = nn.GELU, norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
                 qkv_bias: bool = True, proj_bias: bool = True, ffn_bias: bool = True, qk_norm: bool = False,
                 fused_attn: bool = True, rope=None, local_conv_kernel: int = 3):
        super().__init__()
        del num_heads, qkv_bias, proj_bias, qk_norm, fused_attn, rope, drop_path
        self.norm1 = norm_layer(dim)
        self.pos_inject = _PositionalInjector(dim)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=local_conv_kernel, padding=local_conv_kernel // 2, groups=dim, bias=True)
        Mamba2Cls = _CUDAMamba2 if MAMBA2_AVAILABLE else PureTorchMamba2
        self.mamba_fwd = Mamba2Cls(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba2Cls(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.merge = nn.Linear(2 * dim, dim, bias=False)
        self.out_gate = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        z = self.norm1(x)
        z = self.pos_inject(z, pos)
        z = z + self.local_conv(z.transpose(1, 2)).transpose(1, 2)
        y_fwd = self.mamba_fwd(z)
        y_bwd = self.mamba_bwd(z.flip(dims=[1]).contiguous()).flip(dims=[1])
        y = self.merge(torch.cat([y_fwd, y_bwd], dim=-1))
        y = y * self.out_gate(z)
        x = x + self.ls1(y)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
