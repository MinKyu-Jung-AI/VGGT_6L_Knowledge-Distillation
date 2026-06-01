"""
DL3DV-10K Dataset Loader
포맷: transforms.json (NeRF / nerfstudio 스타일) + images_8/
"""

import os
import json
import random
import logging
import numpy as np

from data.dataset_util import read_image_cv2, threshold_depth_map
from data.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def c2w_to_extrinsic(c2w):
    """NeRF c2w (4x4) → OpenCV extrinsic (w2c)"""
    c2w = np.array(c2w, dtype=np.float64)
    # NeRF convention (OpenGL) → OpenCV: flip Y and Z
    flip = np.diag([1, -1, -1, 1]).astype(np.float64)
    c2w = c2w @ flip
    w2c = np.linalg.inv(c2w)
    return w2c[:3, :]   # (3, 4)


class DL3DVDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DL3DV_DIR: str = None,
        resolutions: list = None,
        min_num_images: int = 8,
        len_train: int = 100000,
        len_test: int = 5000,
    ):
        super().__init__(common_conf)
        if DL3DV_DIR is None:
            raise ValueError("DL3DV_DIR must be specified")

        self.DL3DV_DIR = DL3DV_DIR
        self.training = getattr(common_conf, 'training', True)
        self.load_depth = getattr(common_conf, 'load_depth', False)
        self.min_num_images = min_num_images
        self.len_train = len_train
        self.len_test = len_test
        self.split = split

        if resolutions is None:
            resolutions = ["1K", "2K", "3K"]

        # 씬 목록 수집
        self.sequences = []
        for res in resolutions:
            res_dir = os.path.join(DL3DV_DIR, res)
            if not os.path.isdir(res_dir):
                continue
            for scene in sorted(os.listdir(res_dir)):
                scene_dir = os.path.join(res_dir, scene)
                transforms_path = os.path.join(scene_dir, "transforms.json")
                img_dir = os.path.join(scene_dir, "images_8")
                if not os.path.exists(transforms_path) or not os.path.isdir(img_dir):
                    continue
                self.sequences.append((scene_dir, transforms_path))

        logger.info(f"DL3DV-10K: {len(self.sequences)} scenes (resolutions={resolutions})")

    def __len__(self):
        return self.len_train if self.split == "train" else self.len_test

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0, img_per_seq=8):
        if seq_index is None:
            seq_index = random.randint(0, len(self.sequences) - 1)

        scene_dir, transforms_path = self.sequences[seq_index % len(self.sequences)]

        with open(transforms_path) as f:
            meta = json.load(f)

        frames = meta["frames"]
        if len(frames) < self.min_num_images:
            # fallback to random other scene
            return self.get_data(seq_index=random.randint(0, len(self.sequences) - 1),
                                 img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        # 랜덤 샘플링
        chosen = random.sample(frames, min(img_per_seq, len(frames)))

        # 전역 intrinsic — images_8 실제 해상도에 맞게 스케일 조정
        W, H = meta["w"], meta["h"]
        fl_x, fl_y = meta["fl_x"], meta["fl_y"]
        cx, cy = meta["cx"], meta["cy"]
        # images_8은 원본의 1/8 해상도
        scale = 1.0 / 8.0
        fl_x, fl_y, cx, cy = fl_x * scale, fl_y * scale, cx * scale, cy * scale
        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float64)

        target_image_shape = self.get_target_shape(aspect_ratio)
        images, depths, extrinsics, intrinsics = [], [], [], []

        for frame in chosen:
            img_path = os.path.join(scene_dir, "images_8", os.path.basename(frame["file_path"]))
            if not os.path.exists(img_path):
                # 확장자 시도
                for ext in [".png", ".jpg", ".JPG"]:
                    candidate = os.path.splitext(img_path)[0] + ext
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

            if not os.path.exists(img_path):
                continue

            img = read_image_cv2(img_path)
            original_size = np.array(img.shape[:2])
            extr = c2w_to_extrinsic(frame["transform_matrix"])
            intr = K.copy()

            dummy_depth = np.zeros(img.shape[:2], dtype=np.float32)
            img, _, extr, intr, _, _, _, _ = self.process_one_image(
                img, dummy_depth, extr, intr, original_size, target_image_shape,
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
            "seq_name": os.path.basename(scene_dir),
            "dataset": "dl3dv",
        }
