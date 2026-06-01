"""
6L Feature MSE + Task Loss (Pose)
- Feature MSE로 feature 정렬 유지
- Pose L1 loss로 camera token 직접 최적화
- step 1050 체크포인트에서 시작
"""
import torch, sys, os, time, logging, random
import numpy as np
import torch.nn as nn
from pathlib import Path
from collections import deque

VGGT_ROOT = str((Path(__file__).resolve().parent.parent / 'vggt-fresh' / 'vggt-main'))
sys.path.insert(0, VGGT_ROOT)
sys.path.insert(0, os.path.join(VGGT_ROOT, 'training'))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from fla.layers import GatedLinearAttention

PRETRAIN_PATH = '/root/.cache/torch/hub/checkpoints/model.pt'
SAVE_DIR = Path('/root/halo_checkpoints_6L_task')
LOG_PATH = Path('/root/halo_logs/6L_fmse_task.log')

SIX_IDX = [2, 5, 7, 9, 18, 20]
DISTILL_IDX = [2, 4, 5, 7, 9, 11, 12, 14, 17, 18, 20, 23]
DPT_IDX = [4, 11, 17, 23]

S = 8; B = 4; LR = 2e-5  # 낮은 LR (fine-tune)
MAX_STEPS = 2000
CKPT_FREQ = 50
LOG_FREQ = 10
FEAT_STAT_FREQ = 200

# Loss weights
W_FEAT = 1.0      # Feature MSE
W_POSE = 0.5      # Pose L1 (teacher의 pose_enc과 비교)

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


class FeatureHook:
    def __init__(self):
        self.features = {}
    def register(self, blocks, indices):
        for idx in indices:
            def hook_fn(m, inp, out, i=idx):
                self.features[i] = out if isinstance(out, torch.Tensor) else out[0]
            blocks[idx].register_forward_hook(hook_fn)


def setup_data():
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
    datasets = [
        Co3dDataset(conf, split='train', len_train=200000,
            CO3D_DIR='/data/vggt_kimi_files/VGGT-Dataset',
            CO3D_ANNOTATION_DIR='/root/co3d_annotations', min_num_images=S),
    ]
    try:
        from data.datasets.dl3dv import DL3DVDataset
        datasets.append(DL3DVDataset(conf, split='train', len_train=100000,
            DL3DV_DIR='/data/vggt_kimi_files/VGGT-Dataset/DL3DV-10K',
            resolutions=['1K', '2K', '3K'], min_num_images=S))
    except: pass
    ds_weights = [1.0] * len(datasets)

    def sample_one():
        for _ in range(5):
            ds = random.choices(datasets, weights=ds_weights, k=1)[0]
            try:
                data = ds.get_data(img_per_seq=S)
                if data['frame_num'] < S: continue
                imgs_np = np.stack(data['images'][:S])
                if imgs_np.shape[1] != 518 or imgs_np.shape[2] != 518:
                    import cv2
                    imgs_np = np.stack([cv2.resize(im, (518, 518)) for im in imgs_np])
                return torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
            except: continue
        return torch.zeros(S, 3, 518, 518)

    def sample_batch(device):
        return torch.stack([sample_one() for _ in range(B)]).to(device)
    return sample_batch


def main():
    device = torch.device('cuda')
    log.info('=' * 60)
    log.info('6L Feature MSE + Task Loss (Pose)')
    log.info(f'W_FEAT={W_FEAT}, W_POSE={W_POSE}')
    log.info(f'B={B}, S={S}, LR={LR}, Steps={MAX_STEPS}')
    log.info('=' * 60)

    pretrained = torch.load(PRETRAIN_PATH, map_location='cpu', weights_only=False)

    # Teacher (camera head 활성화)
    teacher = VGGT(img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=True, enable_depth=False, enable_point=False, enable_track=False).to(device)
    teacher.load_state_dict(pretrained, strict=False); teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # Student (camera head 활성화)
    student = VGGT(img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=True, enable_depth=False, enable_point=False, enable_track=False).to(device)
    student.load_state_dict(pretrained, strict=False)

    # Lightning 교체
    for idx in SIX_IDX:
        gla = GatedLinearAttention(hidden_size=1024, num_heads=16,
            use_output_gate=True, use_short_conv=False, mode='chunk').to(device)
        wrapped = FLABlock(student.aggregator.global_blocks[idx], gla).to(device)
        student.aggregator.global_blocks[idx] = wrapped

    # Step 1050 체크포인트에서 warm start
    ckpt = torch.load('/root/halo_checkpoints_lightning_fmse/checkpoint.pt', map_location='cpu', weights_only=False)
    model_sd = student.state_dict()
    loaded = 0
    for k, v in ckpt['model'].items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k] = v; loaded += 1
    student.load_state_dict(model_sd, strict=False)
    log.info(f'Step {ckpt["step"]} checkpoint loaded ({loaded} params)')

    # Freeze all, unfreeze Lightning blocks only
    for p in student.parameters():
        p.requires_grad_(False)
    for idx in SIX_IDX:
        for p in student.aggregator.global_blocks[idx].parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info(f'학습 파라미터: {trainable/1e6:.1f}M')

    # Feature Hooks (aggregator에만 — head 전)
    teacher_hook = FeatureHook()
    student_hook = FeatureHook()
    teacher_hook.register(teacher.aggregator.global_blocks, DISTILL_IDX)
    student_hook.register(student.aggregator.global_blocks, DISTILL_IDX)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=LR, weight_decay=0.05)

    sample_batch = setup_data()
    log.info('데이터 준비 완료\n')

    # Train mode
    student.train()
    for i, b in enumerate(student.aggregator.global_blocks):
        if i not in SIX_IDX: b.eval()
    student.aggregator.frame_blocks.eval()
    student.aggregator.patch_embed.eval()
    # Camera head도 eval (frozen)
    if hasattr(student, 'camera_head'):
        student.camera_head.eval()

    loss_hist = deque(maxlen=500)
    t0 = time.time()

    log.info(f'{"Step":>6}  {"Total":>8}  {"Feat":>8}  {"Pose":>8}  {"VRAM":>7}  {"s/step":>7}')
    log.info('-' * 55)

    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        imgs = sample_batch(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Teacher forward
            teacher_hook.features.clear()
            with torch.no_grad():
                t_pred = teacher(imgs)

            # Student forward
            student_hook.features.clear()
            s_pred = student(imgs)

            # 1. Feature MSE
            feat_losses = []
            for idx in DISTILL_IDX:
                if idx in teacher_hook.features and idx in student_hook.features:
                    t_f = teacher_hook.features[idx].float()
                    s_f = student_hook.features[idx].float()
                    w = 3.0 if (idx in DPT_IDX or idx in SIX_IDX) else 1.0
                    feat_losses.append(w * (t_f - s_f).pow(2).mean())
            feat_loss = sum(feat_losses) / len(feat_losses) if feat_losses else torch.tensor(0.0, device=device)

            # 2. Pose Loss (teacher의 pose_enc과 비교)
            t_pose = t_pred['pose_enc'].float()  # [B, S, 9]
            s_pose = s_pred['pose_enc'].float()
            pose_loss = (t_pose - s_pose).abs().mean()  # L1

            # Total
            loss = W_FEAT * feat_loss + W_POSE * pose_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        loss_hist.append(loss.item())

        if step % LOG_FREQ == 0 or step <= 3:
            vram = torch.cuda.max_memory_allocated() / 1024**3
            log.info(f'{step:>6}  {loss.item():>8.5f}  {feat_loss.item():>8.5f}  {pose_loss.item():>8.5f}  {vram:>5.1f}GB  {(time.time()-t0)/step:>6.2f}s')

        if step % FEAT_STAT_FREQ == 0 or step == MAX_STEPS:
            lines = [f'  [FeatStat] step={step}']
            for idx in DISTILL_IDX:
                if idx in teacher_hook.features and idx in student_hook.features:
                    t = teacher_hook.features[idx].float().detach()
                    s = student_hook.features[idx].float().detach()
                    cos = nn.functional.cosine_similarity(
                        t.reshape(-1, t.shape[-1]), s.reshape(-1, s.shape[-1]), dim=-1).mean().item()
                    lines.append(f'    L{idx:2d} cos={cos:.4f}')
            # pose cos
            pc = nn.functional.cosine_similarity(
                t_pose.reshape(-1, t_pose.shape[-1]),
                s_pose.reshape(-1, s_pose.shape[-1]), dim=-1).mean().item()
            lines.append(f'    pose_enc cos={pc:.6f}')
            log.info('\n'.join(lines))

        if step % CKPT_FREQ == 0:
            torch.save({
                'step': step, 'model': student.state_dict(), 'loss': loss.item(),
            }, SAVE_DIR / f'step{step}.pt')
            torch.save({
                'step': step, 'model': student.state_dict(), 'loss': loss.item(),
            }, SAVE_DIR / 'checkpoint.pt')
            log.info(f'  [CKPT] step={step}')

    log.info(f'\n완료 ({(time.time()-t0)/60:.1f}분)')


if __name__ == '__main__':
    main()
