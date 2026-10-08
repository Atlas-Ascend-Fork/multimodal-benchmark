# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import os
import sys

import sunrgbd_utils as utils



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


LOGGER = logging.getLogger(__name__)


class SunrgbdObject(object):
    """Load and parse object data"""

    def __init__(self, root_dir, split="training"):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == "training":
            self.num_samples = 10335
        elif split == "testing":
            self.num_samples = 2860
        else:
            raise ValueError(f"Unknown split: {split}")

        self.image_dir = os.path.join(self.split_dir, "image")
        self.calib_dir = os.path.join(self.split_dir, "calib")
        self.pc_dir = os.path.join(self.split_dir, "pc")
        self.depth_dir = os.path.join(self.split_dir, "depth")
        self.label_dir = os.path.join(self.split_dir, "label")

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, "%06d.jpg" % (idx))
        return utils.load_image(img_filename)

    def get_pointcloud(self, idx):
        depth_filename = os.path.join(self.pc_dir, "%06d.mat" % (idx))
        return utils.load_depth_points(depth_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return utils.SunrgbdCalibration(calib_filename)

    def get_label_objects(self, idx):
        if not (self.split == "training"):
            raise AssertionError
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return utils.read_sunrgbd_label(label_filename)
