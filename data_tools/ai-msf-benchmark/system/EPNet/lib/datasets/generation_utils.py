# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Helpers for EPNet datasets and generation scripts."""

import os
from typing import NamedTuple

import numpy as np
from PIL import Image
from shared.parameters import bind_named as _bind_named


class FilteredLidarSample(NamedTuple):
    points: object
    intensities: object
    image_points: object
    depths: object
    valid: object


CLASS_GROUPS = {
    "Car": ("Background", "Car"),
    "People": ("Background", "Pedestrian", "Cyclist"),
    "Pedestrian": ("Background", "Pedestrian"),
    "Cyclist": ("Background", "Cyclist"),
}


class DatasetGeneratorMixin:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def configure_classes(self, classes):
        self.gt_database = None
        if classes not in CLASS_GROUPS:
            raise ValueError("Invalid classes: %s" % classes)
        self.classes = CLASS_GROUPS[classes]

    def configure_point_filter(self, reduce_by_range, area_scope):
        self._reduce_points_by_range = reduce_by_range
        self._point_area_scope = area_scope

    def get_valid_flag(self, points_rect, points_image, depths, image_dimensions):
        return valid_point_flag(
            points_rect,
            points_image,
            depths,
            image_dimensions,
            self._reduce_points_by_range,
            self._point_area_scope,
        )

    def filtrate_objects(self, objects):
        filtered = []
        for obj in objects:
            if obj.cls_type in self.classes and obj.level_str in (
                "Easy", "Moderate", "Hard"
            ):
                filtered.append(obj)
        return filtered


def valid_point_flag(*call_args, **call_kwargs):
    (
        points_rect,
        points_image,
        depths,
        image_dimensions,
        reduce_by_range,
        area_scope,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('points_rect', 'points_image', 'depths', 'image_dimensions', 'reduce_by_range', 'area_scope'),
    )
    within_width = np.logical_and(
        points_image[:, 0] >= 0, points_image[:, 0] < image_dimensions[1]
    )
    within_height = np.logical_and(
        points_image[:, 1] >= 0, points_image[:, 1] < image_dimensions[0]
    )
    valid = np.logical_and(np.logical_and(within_width, within_height), depths >= 0)
    if reduce_by_range:
        x_range, y_range, z_range = area_scope
        x, y, z = points_rect[:, 0], points_rect[:, 1], points_rect[:, 2]
        valid &= (
            (x >= x_range[0])
            & (x <= x_range[1])
            & (y >= y_range[0])
            & (y <= y_range[1])
            & (z >= z_range[0])
            & (z <= z_range[1])
        )
    return valid


def load_lidar_sample(dataset, sample_id):
    """Load and project one KITTI LiDAR sample."""
    calibration = dataset.get_calib(sample_id)
    image_dimensions = dataset.get_image_shape(sample_id)
    lidar = dataset.get_lidar(sample_id)
    points = calibration.lidar_to_rect(lidar[:, :3])
    return calibration, image_dimensions, points, lidar[:, 3]


def filter_lidar_sample(dataset, calibration, image_dimensions, points, intensities):
    """Restrict projected LiDAR points to the valid image/range region."""
    image_points, depths = calibration.rect_to_img(points)
    valid = dataset.get_valid_flag(points, image_points, depths, image_dimensions)
    return FilteredLidarSample(
        points[valid, :3], intensities[valid], image_points, depths, valid
    )


def sample_point_indices(points, point_count):
    """Choose a shuffled fixed-size point subset, retaining far points."""
    if point_count < len(points):
        near_mask = points[:, 2] < 40.0
        far_indices = np.where(~near_mask)[0]
        near_indices = np.where(near_mask)[0]
        selected_near = np.random.choice(
            near_indices, point_count - len(far_indices), replace=False
        )
        choice = np.concatenate((selected_near, far_indices)) if len(far_indices) else selected_near
    else:
        choice = np.arange(len(points), dtype=np.int32)
        if point_count > len(points):
            extra = np.random.choice(choice, point_count - len(points), replace=False)
            choice = np.concatenate((choice, extra))
    np.random.shuffle(choice)
    return choice


def populate_point_sample(sample_info, points, features, include_features=True):
    """Store point arrays and their concatenated network input."""
    point_input = (
        np.concatenate((points, features), axis=1)
        if include_features and features is not None else points
    )
    sample_info.update(
        {"pts_input": point_input, "pts_rect": points, "pts_features": features}
    )
    return sample_info


def image_shape(image_dir, index):
    image_file = os.path.join(image_dir, "%06d.png" % index)
    if not os.path.exists(image_file):
        raise AssertionError
    with Image.open(image_file) as image:
        width, height = image.size
    return height, width, 3


def lidar_points(lidar_dir, index):
    lidar_file = os.path.join(lidar_dir, "%06d.bin" % index)
    if not os.path.exists(lidar_file):
        raise AssertionError
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
