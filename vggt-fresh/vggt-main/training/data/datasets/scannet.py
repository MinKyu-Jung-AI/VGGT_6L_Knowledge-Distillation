import os
import os.path as osp
import logging
import random
import glob

import cv2
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset


class ScanNetDataset(BaseDataset):
    """
    ScanNet RGB-D dataset loader.
    Expects extracted structure:
      SCANNET_EXTRACTED_DIR/
        scene0000_00/
          color/      0.jpg, 10.jpg, ...
          depth/      0.png, 10.png, ...  (16-bit PNG, mm units)
          pose/       0.txt, 10.txt, ...  (4x4 camera_to_world)
          intrinsic/  intrinsic_color.txt (4x4 matrix)
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SCANNET_DIR: str = "/data/vggt_kimi_files/VGGT-Dataset/scannet/extracted",
        min_num_images: int = 8,
        len_train: int = 100000,
        len_test: int = 5000,
        expand_ratio: int = 4,
        depth_max: float = 10.0,   # ScanNet indoor: max 10m
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.SCANNET_DIR = SCANNET_DIR
        self.min_num_images = min_num_images
        self.depth_max = depth_max

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"ScanNet extracted dir: {self.SCANNET_DIR}")

        # Build sequence list: (scene_id, sorted_frame_ids)
        cache_path = osp.join(self.SCANNET_DIR, "sequence_list.txt")
        if osp.exists(cache_path):
            with open(cache_path, 'r') as f:
                scene_ids = [l.strip() for l in f if l.strip()]
        else:
            scene_ids = sorted([
                d for d in os.listdir(self.SCANNET_DIR)
                if osp.isdir(osp.join(self.SCANNET_DIR, d)) and d.startswith('scene')
            ])
            with open(cache_path, 'w') as f:
                f.write('\n'.join(scene_ids))

        # Filter scenes with enough frames
        self.sequences = []
        for scene_id in scene_ids:
            color_dir = osp.join(self.SCANNET_DIR, scene_id, 'color')
            if not osp.exists(color_dir):
                continue
            frames = sorted(
                [int(osp.splitext(f)[0]) for f in os.listdir(color_dir) if f.endswith('.jpg')],
                key=lambda x: x
            )
            if len(frames) >= self.min_num_images:
                self.sequences.append((scene_id, frames))

        self.sequence_list_len = len(self.sequences)
        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: ScanNet scenes: {self.sequence_list_len}, dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        scene_id, frame_ids = self.sequences[seq_index]

        # Load intrinsics (4x4 → 3x3)
        intr_path = osp.join(self.SCANNET_DIR, scene_id, 'intrinsic', 'intrinsic_color.txt')
        intr4x4 = np.loadtxt(intr_path)
        K = intr4x4[:3, :3].copy()  # fx,fy,cx,cy

        num_images = len(frame_ids)
        if ids is None:
            chosen = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)
        else:
            chosen = ids

        if self.get_nearby:
            chosen = self.get_nearby_ids(chosen, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, cam_points, world_points = [], [], [], []
        point_masks, extrinsics, intrinsics, original_sizes = [], [], [], []

        for idx in chosen:
            frame_id = frame_ids[idx]
            color_path = osp.join(self.SCANNET_DIR, scene_id, 'color', f'{frame_id}.jpg')
            depth_path = osp.join(self.SCANNET_DIR, scene_id, 'depth', f'{frame_id}.png')
            pose_path  = osp.join(self.SCANNET_DIR, scene_id, 'pose',  f'{frame_id}.txt')

            image = read_image_cv2(color_path)

            # Depth: 16-bit PNG in mm → meters
            depth_mm = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_m  = depth_mm.astype(np.float32) / 1000.0
            depth_m  = threshold_depth_map(depth_m, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

            # Pose: camera_to_world → world_to_camera (extrinsic)
            c2w = np.loadtxt(pose_path)          # 4x4
            w2c = np.linalg.inv(c2w)             # 4x4
            extri_opencv = w2c[:3]               # 3x4

            intri_opencv = K.copy()
            original_size = np.array(image.shape[:2])

            (
                image,
                depth_m,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_m,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=color_path,
            )

            if (image.shape[:2] != target_image_shape).any():
                logging.error(f"Shape mismatch {scene_id}/{frame_id}: {image.shape[:2]} vs {target_image_shape}")
                continue

            images.append(image)
            depths.append(depth_m)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        batch = {
            "seq_name": f"scannet_{scene_id}",
            "ids": chosen,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch
