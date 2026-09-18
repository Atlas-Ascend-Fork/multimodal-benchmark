# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
from shared.box_geometry import camera_box_corners
from shared.kitti_labels import assign_object_label, format_object_label


def cls_type_to_id(cls_type):
    type_to_id = {"Car": 1, "Pedestrian": 2, "Cyclist": 3, "Van": 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        self.src = line
        assign_object_label(self, line)
        self.cls_id = cls_type_to_id(self.cls_type)
        self.trucation = self.truncation
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = "Easy"
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = "Moderate"
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = "Hard"
            return 3  # Hard
        else:
            self.level_str = "UnKnown"
            return 4

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        return camera_box_corners(
            self.pos, (self.length_value, self.w, self.h), self.ry
        )

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(
                np.int32
            )
            box2d[:, 1] = (
                Object3d.BEV_SHAPE[0]
                - 1
                - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(
                    np.int32
                )
            )
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(
                np.int32
            )
            cv = (
                Object3d.BEV_SHAPE[0]
                - 1
                - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            )
            half_l, half_w = int(self.length_value / voxel_size / 2), int(
                self.w / voxel_size / 2
            )
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = (
            "%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f"
            % (
                self.cls_type,
                self.trucation,
                self.occlusion,
                self.alpha,
                self.box2d,
                self.h,
                self.w,
                self.length_value,
                self.pos,
                self.ry,
            )
        )
        return print_str

    def to_kitti_format(self):
        return format_object_label(self)
