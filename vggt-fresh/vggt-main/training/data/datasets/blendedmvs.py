"""
BlendedMVS++ Dataset Loader
포맷: cams/XXXXX_cam.txt + blended_images/ + rendered_depth_maps/
"""

import os
import random
import logging
import numpy as np
import cv2

from data.dataset_util import read_image_cv2, threshold_depth_map
from data.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def parse_cam_file(cam_path):
    """BlendedMVS cam 파일 파싱 → extrinsic (3x4), intrinsic (3x3), depth_range"""
    with open(cam_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # extrinsic
    assert lines[0] == "extrinsic"
    extr = np.array([list(map(float, lines[i+1].split())) for i in range(4)], dtype=np.float64)

    # intrinsic
    assert lines[5] == "intrinsic"
    intr = np.array([list(map(float, lines[i+6].split())) for i in range(3)], dtype=np.float64)

    # depth range (min max interval nviews)
    depth_info = list(map(float, lines[9].split()))
    depth_range = (depth_info[0], depth_info[1]) if len(depth_info) >= 2 else (0.1, 100.0)

    return extr[:3, :], intr, depth_range


class BlendedMVSDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        BLENDEDMVS_DIR: str = None,
        min_num_images: int = 8,
        len_train: int = 100000,
        len_test: int = 5000,
    ):
        super().__init__(common_conf)
        if BLENDEDMVS_DIR is None:
            raise ValueError("BLENDEDMVS_DIR must be specified")

        self.BLENDEDMVS_DIR = BLENDEDMVS_DIR
        self.training = getattr(common_conf, 'training', True)
        self.load_depth = getattr(common_conf, 'load_depth', True)
        self.min_num_images = min_num_images
        self.len_train = len_train
        self.len_test = len_test
        self.split = split

        # 씬 목록
        self.sequences = []
        for scene in sorted(os.listdir(BLENDEDMVS_DIR)):
            scene_dir = os.path.join(BLENDEDMVS_DIR, scene)
            cam_dir = os.path.join(scene_dir, "cams")
            img_dir = os.path.join(scene_dir, "blended_images")
            if not os.path.isdir(cam_dir) or not os.path.isdir(img_dir):
                continue
            cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_cam.txt")])
            if len(cam_files) >= min_num_images:
                self.sequences.append(scene_dir)

        logger.info(f"BlendedMVS++: {len(self.sequences)} scenes at {BLENDEDMVS_DIR}")

    def __len__(self):
        return self.len_train if self.split == "train" else self.len_test

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0, img_per_seq=8):
        if seq_index is None:
            seq_index = random.randint(0, len(self.sequences) - 1)

        scene_dir = self.sequences[seq_index % len(self.sequences)]
        cam_dir = os.path.join(scene_dir, "cams")
        img_dir = os.path.join(scene_dir, "blended_images")
        dep_dir = os.path.join(scene_dir, "rendered_depth_maps")

        cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_cam.txt")])
        if len(cam_files) < 2:
            return self.get_data(seq_index=random.randint(0, len(self.sequences) - 1),
                                 img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        chosen = random.sample(cam_files, min(img_per_seq, len(cam_files)))
        target_image_shape = self.get_target_shape(aspect_ratio)
        images, depths, extrinsics, intrinsics = [], [], [], []

        for cam_file in chosen:
            frame_id = cam_file.replace("_cam.txt", "")
            cam_path = os.path.join(cam_dir, cam_file)

            # 이미지 찾기 (.jpg / .png)
            img_path = None
            for ext in [".jpg", ".png", ".JPG"]:
                candidate = os.path.join(img_dir, frame_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            try:
                extr, intr, depth_range = parse_cam_file(cam_path)
            except Exception:
                continue

            img = read_image_cv2(img_path)
            original_size = np.array(img.shape[:2])

            # depth
            depth_map = None
            if self.load_depth:
                for ext in [".pfm", ".png"]:
                    dep_path = os.path.join(dep_dir, frame_id + ext)
                    if os.path.exists(dep_path):
                        if ext == ".pfm":
                            depth_map = _read_pfm(dep_path)
                        else:
                            depth_map = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
                        depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)
                        break

            img, depth_map, extr, intr, _, _, _, _ = self.process_one_image(
                img, depth_map, extr, intr, original_size, target_image_shape,
            )

            images.append(img)
            depths.append(depth_map)
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
            "dataset": "blendedmvs",
        }


def _read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode().rstrip()
        color = header == "PF"
        dims = f.readline().decode().rstrip().split()
        W, H = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode().rstrip())
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)
        data = np.frombuffer(f.read(), endian + "f")
        data = data.reshape(H, W, 3) if color else data.reshape(H, W)
        return np.flipud(data).astype(np.float32)
