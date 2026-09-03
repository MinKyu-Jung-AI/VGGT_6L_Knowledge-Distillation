"""Project paths configurable through environment variables."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


DATA_ROOT = _env_path("VGGT_DATA_ROOT", REPO_ROOT / "data")
OUTPUT_ROOT = _env_path("VGGT_OUTPUT_ROOT", REPO_ROOT / "outputs")
WEIGHTS_DIR = _env_path("VGGT_WEIGHTS_DIR", REPO_ROOT / "weights")

PRETRAIN_PATH = _env_path("VGGT_PRETRAIN_PATH", WEIGHTS_DIR / "model.pt")
FINAL_CHECKPOINT_PATH = _env_path(
    "VGGT_FINAL_CHECKPOINT", WEIGHTS_DIR / "6L_best_AUC0641.pt"
)
VISER_IMAGE_DIR = _env_path(
    "VGGT_VISER_IMAGE_DIR",
    REPO_ROOT / "vggt-fresh" / "vggt-main" / "examples" / "kitchen" / "images",
)

CO3D_DIR = _env_path("VGGT_CO3D_DIR", DATA_ROOT)
CO3D_ANNOTATION_DIR = _env_path(
    "VGGT_CO3D_ANNOTATION_DIR", DATA_ROOT / "co3d_annotations"
)
DL3DV_DIR = _env_path("VGGT_DL3DV_DIR", DATA_ROOT / "DL3DV-10K")
BLENDED_MVS_DIR = _env_path(
    "VGGT_BLENDED_MVS_DIR", DATA_ROOT / "blendedmvs" / "BlendedMVS_plusplus"
)
MEGADEPTH_DIR = _env_path("VGGT_MEGADEPTH_DIR", DATA_ROOT / "megadepth")
SCANNET_DIR = _env_path(
    "VGGT_SCANNET_DIR", DATA_ROOT / "scannet" / "extracted"
)

CHECKPOINT_ROOT = _env_path("VGGT_CHECKPOINT_ROOT", OUTPUT_ROOT / "checkpoints")
LOG_ROOT = _env_path("VGGT_LOG_ROOT", OUTPUT_ROOT / "logs")
STAGE1_CHECKPOINT_DIR = _env_path(
    "VGGT_STAGE1_CHECKPOINT_DIR", CHECKPOINT_ROOT / "stage1_layerwise"
)
STAGE2_CHECKPOINT_DIR = _env_path(
    "VGGT_STAGE2_CHECKPOINT_DIR", CHECKPOINT_ROOT / "stage2_fmse"
)
STAGE3_CHECKPOINT_DIR = _env_path(
    "VGGT_STAGE3_CHECKPOINT_DIR", CHECKPOINT_ROOT / "stage3_task"
)
TASK_ONLY_CHECKPOINT_DIR = _env_path(
    "VGGT_TASK_ONLY_CHECKPOINT_DIR", CHECKPOINT_ROOT / "stage3_task_only"
)
PATCH_CHECKPOINT_DIR = _env_path(
    "VGGT_PATCH_CHECKPOINT_DIR", CHECKPOINT_ROOT / "stage3_patch_focus"
)
STAGE3_WARM_START = _env_path(
    "VGGT_STAGE3_WARM_START", STAGE2_CHECKPOINT_DIR / "checkpoint.pt"
)
PATCH_WARM_START = _env_path(
    "VGGT_PATCH_WARM_START", STAGE3_CHECKPOINT_DIR / "step1250.pt"
)
