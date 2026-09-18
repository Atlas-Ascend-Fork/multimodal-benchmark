# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Geometry operations for compatible KITTI datasets."""

import numpy as np


SUNRGBD_CLASSES_10 = (
    "bed", "table", "sofa", "chair", "toilet", "desk", "dresser",
    "night_stand", "bookshelf", "bathtub",
)


def point_in_configured_range(xyz, config):
    return all(
        bounds[0] <= coordinate <= bounds[1]
        for coordinate, bounds in zip(xyz, config.PC_AREA_SCOPE)
    )


def valid_point_flags(pts_rect, pts_img, pts_rect_depth, img_shape, config):
    image_flags = (
        (pts_img[:, 0] >= 0)
        & (pts_img[:, 0] < img_shape[1])
        & (pts_img[:, 1] >= 0)
        & (pts_img[:, 1] < img_shape[0])
        & (pts_rect_depth >= 0)
    )
    if not config.PC_REDUCE_BY_RANGE:
        return image_flags
    range_flags = np.ones(pts_rect.shape[0], dtype=bool)
    for axis, bounds in enumerate(config.PC_AREA_SCOPE):
        range_flags &= (pts_rect[:, axis] >= bounds[0])
        range_flags &= (pts_rect[:, axis] <= bounds[1])
    return image_flags & range_flags


def rpn_training_labels(pts_rect, gt_boxes3d, box_utils):
    cls_label = np.zeros(pts_rect.shape[0], dtype=np.int32)
    reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)
    gt_corners = box_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
    enlarged_boxes = box_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
    enlarged_corners = box_utils.boxes3d_to_corners3d(enlarged_boxes, rotate=True)
    for box, corners, outer_corners in zip(
        gt_boxes3d, gt_corners, enlarged_corners
    ):
        foreground = box_utils.in_hull(pts_rect, corners)
        cls_label[foreground] = 1
        enlarged = box_utils.in_hull(pts_rect, outer_corners)
        cls_label[np.logical_xor(foreground, enlarged)] = -1
        center = box[:3].copy()
        center[1] -= box[3] / 2
        reg_label[foreground, :3] = center - pts_rect[foreground]
        reg_label[foreground, 3:] = box[3:]
    return cls_label, reg_label


def filter_objects(objects, classes, mode, config, check_range):
    allowed = classes
    if mode == "TRAIN" and config.INCLUDE_SIMILAR_TYPE:
        allowed = list(classes)
        if "Car" in classes:
            allowed.append("Van")
        if "Pedestrian" in classes:
            allowed.append("Person_sitting")
    filtered = []
    for obj in objects:
        outside_range = (
            mode == "TRAIN"
            and config.PC_REDUCE_BY_RANGE
            and not check_range(obj.pos)
        )
        if obj.cls_type in allowed and not outside_range:
            filtered.append(obj)
    return filtered


class ConfiguredGeometryMixin:
    geometry_config = None
    geometry_box_utils = None

    @classmethod
    def check_pc_range(cls, xyz):
        return point_in_configured_range(xyz, cls.geometry_config)

    @classmethod
    def get_valid_flag(cls, pts_rect, pts_img, pts_rect_depth, img_shape):
        return valid_point_flags(
            pts_rect, pts_img, pts_rect_depth, img_shape, cls.geometry_config
        )

    @classmethod
    def generate_rpn_training_labels(cls, pts_rect, gt_boxes3d):
        return rpn_training_labels(pts_rect, gt_boxes3d, cls.geometry_box_utils)

    @classmethod
    def rotate_box3d_along_y(cls, box3d, rot_angle):
        old_x, old_z, rotation = box3d[0], box3d[2], box3d[6]
        old_beta = np.arctan2(old_z, old_x)
        alpha = -np.sign(old_beta) * np.pi / 2 + old_beta + rotation
        box3d = cls.geometry_box_utils.rotate_pc_along_y(
            box3d.reshape(1, 7), rot_angle=rot_angle
        )[0]
        new_beta = np.arctan2(box3d[2], box3d[0])
        box3d[6] = np.sign(new_beta) * np.pi / 2 + alpha - new_beta
        return box3d
