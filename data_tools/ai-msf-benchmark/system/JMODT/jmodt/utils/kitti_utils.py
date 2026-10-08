# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""JMODT compatibility exports for shared KITTI geometry utilities."""

__all__ = [
    "get_objects_from_label", "dist_to_plane", "rotate_pc_along_y",
    "rotate_pc_along_y_torch", "boxes3d_to_corners3d",
    "boxes3d_to_corners3d_torch", "boxes3d_to_bev_torch", "enlarge_box3d",
    "in_hull", "get_iou3d",
]

from pathlib import Path

from shared.compatibility import import_from_root


_MODULE = import_from_root("lib.utils.kitti_utils", Path(__file__).parents[3] / "EPNet")

get_objects_from_label = _MODULE.get_objects_from_label
dist_to_plane = _MODULE.dist_to_plane
rotate_pc_along_y = _MODULE.rotate_pc_along_y
rotate_pc_along_y_torch = _MODULE.rotate_pc_along_y_torch
boxes3d_to_corners3d = _MODULE.boxes3d_to_corners3d
boxes3d_to_corners3d_torch = _MODULE.boxes3d_to_corners3d_torch
boxes3d_to_bev_torch = _MODULE.boxes3d_to_bev_torch
enlarge_box3d = _MODULE.enlarge_box3d
in_hull = _MODULE.in_hull
get_iou3d = _MODULE.get_iou3d
