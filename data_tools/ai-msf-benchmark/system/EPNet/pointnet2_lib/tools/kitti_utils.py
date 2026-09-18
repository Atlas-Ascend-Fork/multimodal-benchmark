# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compatibility exports for EPNet's canonical KITTI utility modules."""

__all__ = [
    "cls_type_to_id", "Object3d", "get_calib_from_file", "Calibration",
    "get_objects_from_label", "objs_to_boxes3d", "boxes3d_to_corners3d",
    "enlarge_box3d", "in_hull",
]

from pathlib import Path

from shared.compatibility import import_from_root


_EPNET_ROOT = Path(__file__).parents[2]
_OBJECTS = import_from_root("lib.utils.object3d", _EPNET_ROOT)
_CALIBRATION = import_from_root("lib.utils.calibration", _EPNET_ROOT)
_GEOMETRY = import_from_root("lib.utils.kitti_utils", _EPNET_ROOT)

cls_type_to_id = _OBJECTS.cls_type_to_id
Object3d = _OBJECTS.Object3d
get_calib_from_file = _CALIBRATION.get_calib_from_file
Calibration = _CALIBRATION.Calibration
get_objects_from_label = _GEOMETRY.get_objects_from_label
objs_to_boxes3d = _GEOMETRY.objs_to_boxes3d
boxes3d_to_corners3d = _GEOMETRY.boxes3d_to_corners3d
enlarge_box3d = _GEOMETRY.enlarge_box3d
in_hull = _GEOMETRY.in_hull
