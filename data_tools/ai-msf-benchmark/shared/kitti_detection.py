# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""KITTI detection projection helpers."""

import os

import numpy as np


def load_indexed_calibration(loader, calibration_dir, index, sample_count):
    """Load a numbered KITTI calibration after validating the index."""
    if index >= sample_count:
        raise AssertionError
    return loader.Calibration(os.path.join(calibration_dir, "%06d.txt" % index))


class KittiVideoDataset:
    """Common image/LiDAR access for KITTI raw video folders."""

    def __init__(self, image_dir, lidar_dir, calibration_dir, loader, logger):
        self.calib = loader.Calibration(calibration_dir, from_video=True)
        self.loader = loader
        self.img_filenames = sorted(
            os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
        )
        self.lidar_filenames = sorted(
            os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)
        )
        logger.info("%s", len(self.img_filenames))
        logger.info("%s", len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, index):
        if index >= self.num_samples:
            raise AssertionError
        return self.loader.load_image(self.img_filenames[index])

    def get_lidar(self, index):
        if index >= self.num_samples:
            raise AssertionError
        return self.loader.load_velo_scan(self.lidar_filenames[index])

    def get_calibration(self, unused):
        return self.calib


def project_detection_boxes(boxes3d, calibration, image_shape, box_utils):
    corners = box_utils.boxes3d_to_corners3d(boxes3d)
    image_boxes, _ = calibration.corners3d_to_img_boxes(corners)
    image_boxes[:, (0, 2)] = np.clip(
        image_boxes[:, (0, 2)], 0, image_shape[1] - 1
    )
    image_boxes[:, (1, 3)] = np.clip(
        image_boxes[:, (1, 3)], 0, image_shape[0] - 1
    )
    widths = image_boxes[:, 2] - image_boxes[:, 0]
    heights = image_boxes[:, 3] - image_boxes[:, 1]
    valid = (widths < image_shape[1] * 0.8) & (heights < image_shape[0] * 0.8)
    return image_boxes, valid


def format_detection_line(class_name, alpha, image_box, box3d, metadata=None):
    """Format one projected 3D detection in KITTI label order."""
    metadata = metadata or {}
    score = metadata.get("score")
    truncation = metadata.get("truncation", -1)
    occlusion = metadata.get("occlusion", -1)
    values = (
        class_name, truncation, occlusion, alpha, *image_box,
        box3d[3], box3d[4], box3d[5], box3d[0], box3d[1], box3d[2], box3d[6],
    )
    line = "%s %.2f %d " % values[:3]
    line += " ".join("%.4f" % value for value in values[3:])
    if score is not None:
        line += " %.4f" % score
    return line


def observation_angle(box3d):
    """Convert a camera-space box yaw to KITTI observation angle alpha."""
    beta = np.arctan2(box3d[2], box3d[0])
    return -np.sign(beta) * np.pi / 2 + beta + box3d[6]
