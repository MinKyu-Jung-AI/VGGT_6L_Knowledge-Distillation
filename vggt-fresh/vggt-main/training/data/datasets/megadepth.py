"""
MegaDepth Dataset Loader
포맷: COLMAP sparse (cameras.txt / images.txt) + images/
"""

import os
import random
import logging
import numpy as np

from data.dataset_util import read_image_cv2
from data.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def parse_colmap_cameras(cameras_txt):
    """cameras.txt → {cam_id: (model, W, H, K)}"""
    cameras = {}
    with open(cameras_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            if model in ("PINHOLE", "SIMPLE_PINHOLE"):
                if model == "PINHOLE":
                    fl_x, fl_y, cx, cy = params[:4]
                else:
                    fl = params[0]; cx, cy = params[1], params[2]
                    fl_x = fl_y = fl
                K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float64)
            elif model in ("OPENCV", "RADIAL"):
                fl_x, fl_y, cx, cy = params[:4]
                K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float64)
            else:
                fl_x = params[0]; cx = W / 2; cy = H / 2
                K = np.array([[fl_x, 0, cx], [0, fl_x, cy], [0, 0, 1]], dtype=np.float64)
            cameras[cam_id] = (W, H, K)
    return cameras


def parse_colmap_images(images_txt):
    """images.txt → list of (img_name, cam_id, R, t)"""
    frames = []
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        img_name = parts[9]
        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        extr = np.hstack([R, t.reshape(3, 1)])   # (3, 4) w2c
        frames.append((img_name, cam_id, extr))
        i += 2   # skip point2D line
    return frames


def _quat_to_rot(qw, qx, qy, qz):
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    R = np.array([
        [1 - s*(qy*qy+qz*qz), s*(qx*qy-qz*qw),   s*(qx*qz+qy*qw)],
        [s*(qx*qy+qz*qw),     1 - s*(qx*qx+qz*qz), s*(qy*qz-qx*qw)],
        [s*(qx*qz-qy*qw),     s*(qy*qz+qx*qw),   1 - s*(qx*qx+qy*qy)],
    ], dtype=np.float64)
    return R


class MegaDepthDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        MEGADEPTH_DIR: str = None,
        min_num_images: int = 8,
        len_train: int = 100000,
        len_test: int = 5000,
    ):
        super().__init__(common_conf)
        if MEGADEPTH_DIR is None:
            raise ValueError("MEGADEPTH_DIR must be specified")

        self.MEGADEPTH_DIR = MEGADEPTH_DIR
        self.training = getattr(common_conf, 'training', True)
        self.load_depth = getattr(common_conf, 'load_depth', False)
        self.min_num_images = min_num_images
        self.len_train = len_train
        self.len_test = len_test
        self.split = split

        # 씬별 (scene_id, reconstruction_id) 목록
        self.sequences = []
        sfm_root = os.path.join(MEGADEPTH_DIR, "MegaDepth_v1_SfM")
        img_root = os.path.join(MEGADEPTH_DIR, "MegaDepth_v1")

        for scene in sorted(os.listdir(sfm_root)):
            scene_sfm = os.path.join(sfm_root, scene, "sparse", "manhattan")
            if not os.path.isdir(scene_sfm):
                continue
            for recon in sorted(os.listdir(scene_sfm)):
                recon_dir = os.path.join(scene_sfm, recon)
                cameras_txt = os.path.join(recon_dir, "cameras.txt")
                images_txt = os.path.join(recon_dir, "images.txt")
                img_dir = os.path.join(sfm_root, scene, "images")
                if not os.path.exists(cameras_txt) or not os.path.exists(images_txt):
                    continue
                # 이미지 수 빠르게 확인
                try:
                    with open(images_txt) as f:
                        n = sum(1 for l in f if not l.startswith("#") and l.strip()) // 2
                except Exception:
                    continue
                if n >= min_num_images:
                    self.sequences.append((scene, recon_dir, img_dir))

        logger.info(f"MegaDepth: {len(self.sequences)} reconstructions at {MEGADEPTH_DIR}")

    def __len__(self):
        return self.len_train if self.split == "train" else self.len_test

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0, img_per_seq=8):
        if seq_index is None:
            seq_index = random.randint(0, len(self.sequences) - 1)

        scene, recon_dir, img_dir = self.sequences[seq_index % len(self.sequences)]
        cameras_txt = os.path.join(recon_dir, "cameras.txt")
        images_txt = os.path.join(recon_dir, "images.txt")

        try:
            cameras = parse_colmap_cameras(cameras_txt)
            frames = parse_colmap_images(images_txt)
        except Exception as e:
            return self.get_data(seq_index=random.randint(0, len(self.sequences) - 1),
                                 img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        if len(frames) < 2:
            return self.get_data(seq_index=random.randint(0, len(self.sequences) - 1),
                                 img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        chosen = random.sample(frames, min(img_per_seq, len(frames)))
        target_image_shape = self.get_target_shape(aspect_ratio)
        images, depths, extrinsics, intrinsics = [], [], [], []

        for img_name, cam_id, extr in chosen:
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue
            if cam_id not in cameras:
                continue

            _, _, K = cameras[cam_id]
            img = read_image_cv2(img_path)
            original_size = np.array(img.shape[:2])
            dummy_depth = np.zeros(img.shape[:2], dtype=np.float32)

            img, _, extr, intr, _, _, _, _ = self.process_one_image(
                img, dummy_depth, extr, K.copy(), original_size, target_image_shape,
            )

            images.append(img)
            depths.append(None)
            extrinsics.append(extr)
            intrinsics.append(intr)

        if len(images) < 2:
            return self.get_data(seq_index=random.randint(0, len(self.sequences) - 1),
                                 img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        return {
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "seq_name": f"{scene}_{os.path.basename(recon_dir)}",
            "dataset": "megadepth",
        }
