"""
6L Task Loss Only: teacher distillation 없이 GT supervision
- 6L model (AUC=0.641)을 pre-trained로 사용
- CO3D GT pose로 직접 학습
- Camera head도 아주 작은 LR로 함께 fine-tune
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
SAVE_DIR = Path('/root/halo_checkpoints_6L_taskonly')
LOG_PATH = Path('/root/halo_logs/6L_task_only.log')

SIX_IDX = [2, 5, 7, 9, 18, 20]

S = 8; B = 4
LR_LIGHTNING = 1e-5   # Lightning layers
LR_HEAD = 1e-6        # Camera head (매우 낮게)
MAX_STEPS = 1000
CKPT_FREQ = 50
LOG_FREQ = 10

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
    log.info('6L Task Loss Only (no distillation)')
    log.info(f'LR_lightning={LR_LIGHTNING}, LR_head={LR_HEAD}')
    log.info(f'B={B}, S={S}, Steps={MAX_STEPS}')
    log.info('=' * 60)

    pretrained = torch.load(PRETRAIN_PATH, map_location='cpu', weights_only=False)

    # Teacher (pose target 생성용)
    teacher = VGGT(img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=True, enable_depth=False, enable_point=False, enable_track=False).to(device)
    teacher.load_state_dict(pretrained, strict=False); teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # Student
    student = VGGT(img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=True, enable_depth=False, enable_point=False, enable_track=False).to(device)
    student.load_state_dict(pretrained, strict=False)

    # Lightning 교체
    for idx in SIX_IDX:
        gla = GatedLinearAttention(hidden_size=1024, num_heads=16,
            use_output_gate=True, use_short_conv=False, mode='chunk').to(device)
        wrapped = FLABlock(student.aggregator.global_blocks[idx], gla).to(device)
        student.aggregator.global_blocks[idx] = wrapped

    # 0.641 체크포인트 로드
    ckpt = torch.load(str(Path(__file__).resolve().parent.parent / 'weights' / '6L_best_AUC0641.pt'), map_location='cpu', weights_only=False)
    model_sd = student.state_dict()
    for k, v in ckpt['model'].items():
        if k in model_sd and v.shape == model_sd[k].shape: model_sd[k] = v
    student.load_state_dict(model_sd, strict=False)
    log.info('AUC=0.641 checkpoint loaded')

    # 학습 대상: Lightning blocks + camera head
    for p in student.parameters():
        p.requires_grad_(False)

    # Lightning blocks
    lightning_params = []
    for idx in SIX_IDX:
        for p in student.aggregator.global_blocks[idx].parameters():
            p.requires_grad_(True)
            lightning_params.append(p)

    # Camera head
    head_params = []
    if hasattr(student, 'camera_head') and student.camera_head is not None:
        for p in student.camera_head.parameters():
            p.requires_grad_(True)
            head_params.append(p)
        log.info(f'Camera head params: {sum(p.numel() for p in head_params)/1e6:.1f}M')

    log.info(f'Lightning params: {sum(p.numel() for p in lightning_params)/1e6:.1f}M')

    # 분리 LR optimizer
    optimizer = torch.optim.AdamW([
        {'params': lightning_params, 'lr': LR_LIGHTNING},
        {'params': head_params, 'lr': LR_HEAD},
    ], weight_decay=0.01)

    sample_batch = setup_data()
    log.info('데이터 준비 완료\n')

    # Train mode
    student.train()
    for i, b in enumerate(student.aggregator.global_blocks):
        if i not in SIX_IDX: b.eval()
    student.aggregator.frame_blocks.eval()
    student.aggregator.patch_embed.eval()
    # Camera head는 train mode

    loss_hist = deque(maxlen=500)
    t0 = time.time()

    log.info(f'{"Step":>6}  {"Pose_L1":>8}  {"Mavg":>8}  {"VRAM":>7}  {"s/step":>7}')
    log.info('-' * 45)

    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        imgs = sample_batch(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Teacher pose (target)
            with torch.no_grad():
                t_pred = teacher(imgs)
            t_pose = t_pred['pose_enc'].float()

            # Student pose
            s_pred = student(imgs)
            s_pose = s_pred['pose_enc'].float()

            # Pose L1 loss only (no distillation!)
            loss = (t_pose - s_pose).abs().mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        loss_hist.append(loss.item())

        if step % LOG_FREQ == 0 or step <= 3:
            vram = torch.cuda.max_memory_allocated() / 1024**3
            mavg = sum(loss_hist) / len(loss_hist)
            log.info(f'{step:>6}  {loss.item():>8.6f}  {mavg:>8.6f}  {vram:>5.1f}GB  {(time.time()-t0)/step:>6.2f}s')

        if step % CKPT_FREQ == 0:
            torch.save({
                'step': step, 'model': student.state_dict(), 'loss': loss.item(),
            }, SAVE_DIR / f'step{step}.pt')
            torch.save({
                'step': step, 'model': student.state_dict(), 'loss': loss.item(),
            }, SAVE_DIR / 'checkpoint.pt')

    log.info(f'\n완료 ({(time.time()-t0)/60:.1f}분)')


if __name__ == '__main__':
    main()
