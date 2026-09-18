# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""EPNet compatibility exports for the shared KITTI annotation helpers."""

__all__ = [
    "get_image_index_str", "get_kitti_info_path", "get_image_path",
    "get_label_path", "get_velodyne_path", "get_calib_path",
    "get_kitti_image_info", "filter_kitti_anno", "filter_annos_low_score",
    "kitti_result_line", "add_difficulty_to_annos", "get_label_anno",
    "get_label_annos", "area", "intersection", "iou",
]

from pathlib import Path

from shared.compatibility import load_from_path


_MODULE = load_from_path(
    "_ai_msf_kitti_common",
    Path(__file__).parents[3] / "clocs/second/data/kitti_common.py",
)

get_image_index_str = _MODULE.get_image_index_str
get_kitti_info_path = _MODULE.get_kitti_info_path
get_image_path = _MODULE.get_image_path
get_label_path = _MODULE.get_label_path
get_velodyne_path = _MODULE.get_velodyne_path
get_calib_path = _MODULE.get_calib_path
get_kitti_image_info = _MODULE.get_kitti_image_info
filter_kitti_anno = _MODULE.filter_kitti_anno
filter_annos_low_score = _MODULE.filter_annos_low_score
kitti_result_line = _MODULE.kitti_result_line
add_difficulty_to_annos = _MODULE.add_difficulty_to_annos
get_label_anno = _MODULE.get_label_anno
get_label_annos = _MODULE.get_label_annos
area = _MODULE.area
intersection = _MODULE.intersection
iou = _MODULE.iou
