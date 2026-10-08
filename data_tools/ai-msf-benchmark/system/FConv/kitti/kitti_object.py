# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""
import logging
import os
import sys

import kitti_util as utils
from shared.kitti_detection import KittiVideoDataset, load_indexed_calibration


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "kitti"))

LOGGER = logging.getLogger(__name__)


class KittiObject(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training"):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == "training":
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 7518
        else:
            raise ValueError(f"Unknown split: {split}")

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")
        self.lidar_dir = os.path.join(self.split_dir, "velodyne")
        self.label_dir = os.path.join(self.split_dir, "label_2")

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        if not (idx < self.num_samples):
            raise AssertionError
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        if not (idx < self.num_samples):
            raise AssertionError
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        return load_indexed_calibration(utils, self.calib_dir, idx, self.num_samples)

    def get_label_objects(self, idx):
        if not (idx < self.num_samples and self.split == "training"):
            raise AssertionError
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return utils.read_label(label_filename)

    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass


class KittiObjectVideo(KittiVideoDataset):
    """Load data for KITTI videos"""

    def __init__(self, img_dir, lidar_dir, calib_dir):
        super().__init__(img_dir, lidar_dir, calib_dir, utils, LOGGER)
