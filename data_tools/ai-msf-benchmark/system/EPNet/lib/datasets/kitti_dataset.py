# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

from PIL import Image
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
import numpy as np
import torch.utils.data as torch_data
from lib.datasets.generation_utils import image_shape, lidar_points


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split="train"):
        self.split = split
        is_test = self.split == "test"
        self.imageset_dir = os.path.join(
            root_dir, "KITTI", "object", "testing" if is_test else "training"
        )

        split_dir = os.path.join(root_dir, "KITTI", "ImageSets", split + ".txt")
        with open(split_dir, encoding="utf-8") as split_file:
            self.image_idx_list = [line.strip() for line in split_file]
        self.num_sample = self.image_idx_list.__len__()

        self.image_dir = os.path.join(self.imageset_dir, "image_2")
        self.lidar_dir = os.path.join(self.imageset_dir, "velodyne")
        self.calib_dir = os.path.join(self.imageset_dir, "calib")
        self.label_dir = os.path.join(self.imageset_dir, "label_2")
        self.plane_dir = os.path.join(self.imageset_dir, "planes")
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Don't need to permute while using grid_sample
        self.image_hw_with_padding_np = np.array([1280.0, 384.0])

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def get_image(self, idx):
        if not (False):
            raise AssertionError("DO NOT USE cv2 NOW, AVOID DEADLOCK")
        import cv2

        img_file = os.path.join(self.image_dir, "%06d.png" % idx)
        if not (os.path.exists(img_file)):
            raise AssertionError
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_rgb_with_normal(self, idx):
        """
        return img with normalization in rgb mode
        :param idx:
        :return: imback(H,W,3)
        """
        img_file = os.path.join(self.image_dir, "%06d.png" % idx)
        if not (os.path.exists(img_file)):
            raise AssertionError
        with Image.open(img_file) as image:
            im = np.array(image.convert("RGB")).astype(np.float)
        im = im / 255.0
        im -= self.mean
        im /= self.std
        # make same size padding with 0
        imback = np.zeros([384, 1280, 3], dtype=np.float)
        imback[: im.shape[0], : im.shape[1], :] = im

        return imback  # (H,W,3) RGB mode

    def get_image_shape_with_padding(self, idx=0):
        return 384, 1280, 3

    def get_image_shape(self, idx):
        return image_shape(self.image_dir, idx)

    def get_lidar(self, idx):
        return lidar_points(self.lidar_dir, idx)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, "%06d.txt" % idx)
        if not (os.path.exists(calib_file)):
            raise AssertionError
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, "%06d.txt" % idx)
        if not (os.path.exists(label_file)):
            raise AssertionError
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, "%06d.txt" % idx)
        with open(plane_file, "r") as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane
