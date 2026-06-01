"""
6L Stage 1: Layer-wise Hidden State Alignment
- 대상 레이어: [2, 5, 7, 9, 18, 20] (6개)
- 각 레이어별로 독립 학습 (teacher hidden → student hidden)
- Loss: MSE + 0.5 * (1 - cos)
- STEPS_PER_LAYER=1500 (layerwise_align.log 재현)
- 학습 완료된 블록을 teacher에도 주입하여 cascade 반영
"""
import torch, sys, os, time, logging
import numpy as np
import torch.nn as nn
from pathlib import Path

VGGT_ROOT = str((Path(__file__).resolve().parent.parent / 'vggt-fresh' / 'vggt-main'))
sys.path.insert(0, VGGT_ROOT)
sys.path.insert(0, os.path.join(VGGT_ROOT, 'training'))

from vggt.models.vggt import VGGT
from fla.layers import GatedLinearAttention

PRETRAIN_PATH = '/root/.cache/torch/hub/checkpoints/model.pt'
SAVE_DIR = Path('/root/halo_checkpoints_6L_layerwise')
LOG_PATH = Path('/root/halo_logs/6L_layerwise.log')

SIX_IDX = [2, 5, 7, 9, 18, 20]
# 학습 순서: 안쪽(얕은 층)부터 바깥쪽으로 진행 — cascade 오차 누적을 줄이기 위함
NEW_ORDER = [2, 5, 7, 9, 18, 20]
STEPS_PER_LAYER = 1500

SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()])
log = logging.getLogger(__name__)


class FLABlock(nn.Module):
    def __init__(self, ob, m):
        super().__init__()
        self.mixer = m
        self.norm1 = ob.norm1; self.ls1 = ob.ls1
        self.norm2 = ob.norm2; self.mlp = ob.mlp; self.ls2 = ob.ls2
    def forward(self, x, pos=None):
        o = self.mixer(self.norm1(x))
        if isinstance(o, tuple): o = o[0]
        x = x + self.ls1(o)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class IOHook:
    def __init__(self):
        self.inputs = {}; self.outputs = {}
    def register(self, blocks, indices):
        for idx in indices:
            def mk(i):
                def fn(m, inp, out):
                    self.inputs[i] = inp[0].detach()
                    self.outputs[i] = (out.detach() if isinstance(out, torch.Tensor) else out[0].detach())
                return fn
            blocks[idx].register_forward_hook(mk(idx))


def setup_data(S=8, B=8):
    from omegaconf import OmegaConf
    from data.datasets.co3d import Co3dDataset
    conf = OmegaConf.create({
        'img_size': 518, 'patch_size': 14, 'rescale': True, 'rescale_aug': True,
        'landscape_check': False, 'load_depth': False, 'training': True, 'debug': False,
        'fix_img_num': S, 'img_nums': [S, S], 'max_img_per_gpu': B * S,
        'allow_duplicate_img': False, 'repeat_batch': False, 'inside_random': True,
        'get_nearby': True, 'track_num': 0, 'load_track': False, 'fix_aspect_ratio': 1.0,
        'augs': {'scales': [0.8, 1.2], 'cojitter': True, 'cojitter_ratio': 0.3,
                 'aspects': [1.0, 1.0], 'color_jitter': {'brightness': 0.5, 'contrast': 0.5,
                 'saturation': 0.5, 'hue': 0.1, 'p': 0.9},
                 'gray_scale': True, 'gau_blur': False}
    })
    ds = Co3dDataset(conf, split='train', len_train=200000,
        CO3D_DIR='/data/vggt_kimi_files/VGGT-Dataset',
        CO3D_ANNOTATION_DIR='/root/co3d_annotations', min_num_images=S)
    IMG_H, IMG_W = 518, 518
    def sample_batch(device):
        imgs = []
        for _ in range(B):
            for _ in range(3):
                try:
                    data = ds.get_data(img_per_seq=S)
                    if data['frame_num'] >= S:
                        im = np.stack(data['images'][:S])
                        if im.shape[1] != IMG_H or im.shape[2] != IMG_W:
                            import cv2; im = np.stack([cv2.resize(i, (IMG_W, IMG_H)) for i in im])
                        imgs.append(torch.from_numpy(im).permute(0, 3, 1, 2).float() / 255.)
                        break
                except: pass
            else:
                imgs.append(torch.zeros(S, 3, IMG_H, IMG_W))
        return torch.stack(imgs).to(device)
    return sample_batch


def main():
    device = torch.device('cuda')
    log.info('=' * 60)
    log.info('6L Stage 1: Layer-wise Hidden State Alignment')
    log.info(f'대상 레이어: {SIX_IDX}')
    log.info(f'학습 순서: {NEW_ORDER}')
    log.info(f'Steps/layer: {STEPS_PER_LAYER}')
    log.info('=' * 60)

    pretrained = torch.load(PRETRAIN_PATH, map_location='cpu', weights_only=False)

    # Teacher: pretrained VGGT (원본 어텐션)
    teacher = VGGT(img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=True, enable_depth=False, enable_point=False, enable_track=False).to(device)
    teacher.load_state_dict(pretrained, strict=False)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # Teacher hooks (모든 레이어에 input/output 후킹)
    t_hook = IOHook()
    t_hook.register(teacher.aggregator.global_blocks, list(range(24)))

    sample_batch = setup_data(S=8, B=8)
    log.info('데이터 준비 완료')

    total_start = time.time()

    for layer_idx in NEW_ORDER:
        log.info('')
        log.info('=' * 60)
        log.info(f'  Layer {layer_idx} 독립 정렬 시작 ({STEPS_PER_LAYER} steps)')
        log.info('=' * 60)

        # 새 Lightning block 생성
        gla = GatedLinearAttention(hidden_size=1024, num_heads=16,
            use_output_gate=True, use_short_conv=False, mode='chunk').to(device)
        orig_block = teacher.aggregator.global_blocks[layer_idx]
        wrapped = FLABlock(orig_block, gla).to(device)

        # mixer만 학습
        for p in wrapped.parameters(): p.requires_grad_(False)
        for p in wrapped.mixer.parameters(): p.requires_grad_(True)
        wrapped.mixer.train()

        trainable = sum(p.numel() for p in wrapped.mixer.parameters() if p.requires_grad)
        log.info(f'  파라미터: {trainable/1e6:.2f}M')

        optimizer = torch.optim.AdamW(
            [p for p in wrapped.mixer.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=0.05)

        t0 = time.time()
        mavg = 0.0
        for step in range(1, STEPS_PER_LAYER + 1):
            optimizer.zero_grad(set_to_none=True)
            imgs = sample_batch(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                t_hook.inputs.clear(); t_hook.outputs.clear()
                with torch.no_grad(): teacher(imgs)

                teacher_input = t_hook.inputs[layer_idx].float()
                teacher_output = t_hook.outputs[layer_idx].float()

                student_output = wrapped(teacher_input)
                if not isinstance(student_output, torch.Tensor):
                    student_output = student_output[0]
                student_output = student_output.float()

                l_mse = (teacher_output - student_output).pow(2).mean()
                cos = nn.functional.cosine_similarity(
                    teacher_output.reshape(-1, teacher_output.shape[-1]),
                    student_output.reshape(-1, student_output.shape[-1]), dim=-1)
                loss = l_mse + 0.5 * (1 - cos).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapped.mixer.parameters(), 1.0)
            optimizer.step()

            mavg = 0.99 * mavg + 0.01 * loss.item() if step > 1 else loss.item()

            if step == 1 or step % 50 == 0:
                elapsed = time.time() - t0
                log.info(f'  L[{layer_idx:2d}] step={step:>5}  loss={loss.item():.4f}  cos={cos.mean().item():.4f}  mavg={mavg:.4f}  {elapsed:.0f}s')

        final_cos = cos.mean().item()
        log.info(f'  Layer {layer_idx} 완료: cos={final_cos:.4f}  ({time.time()-t0:.0f}s)')

        # 저장
        wrapped.eval()
        torch.save({
            'layer_idx': layer_idx,
            'block_state': wrapped.state_dict(),
            'final_cos': final_cos,
        }, SAVE_DIR / f'layer{layer_idx}.pt')

        # Teacher에 학습된 블록 주입 (다음 레이어는 이 결과 위에서 학습)
        teacher.aggregator.global_blocks[layer_idx] = wrapped
        teacher.aggregator.global_blocks[layer_idx].eval()
        for p in teacher.aggregator.global_blocks[layer_idx].parameters():
            p.requires_grad_(False)

        del optimizer
        torch.cuda.empty_cache()

    log.info('')
    log.info(f'전체 Stage 1 완료 ({(time.time()-total_start)/60:.1f}분)')
    log.info(f'체크포인트: {SAVE_DIR}')


if __name__ == '__main__':
    main()
