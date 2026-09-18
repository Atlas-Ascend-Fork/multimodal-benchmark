# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""EPNet adapters for the shared KITTI evaluation implementation."""

__all__ = [
    "get_thresholds", "clean_data", "image_box_overlap", "bev_box_overlap",
    "d3_box_overlap", "compute_statistics_jit", "get_split_parts",
    "fused_compute_statistics", "calculate_iou_partly", "eval_class",
    "get_m_ap", "print_str", "do_eval", "do_coco_style_eval",
    "get_official_eval_result", "get_coco_eval_result",
]

from pathlib import Path

from shared.compatibility import import_from_root


_MODULE = import_from_root(
    "second.utils.eval",
    Path(__file__).parents[3] / "clocs",
)

get_thresholds = _MODULE.get_thresholds
clean_data = _MODULE.clean_data
image_box_overlap = _MODULE.image_box_overlap
bev_box_overlap = _MODULE.bev_box_overlap
d3_box_overlap = _MODULE.d3_box_overlap
compute_statistics_jit = _MODULE.compute_statistics_jit
get_split_parts = _MODULE.get_split_parts
fused_compute_statistics = _MODULE.fused_compute_statistics
calculate_iou_partly = _MODULE.calculate_iou_partly
eval_class = _MODULE.eval_class_v3
get_m_ap = _MODULE.get_m_ap
print_str = _MODULE.print_str
do_eval = _MODULE.do_eval
do_coco_style_eval = _MODULE.do_coco_style_eval
get_coco_eval_result = _MODULE.get_coco_eval_result


def get_official_eval_result(gt_annos, dt_annos, current_classes):
    """Return EPNet's historical text-and-metrics result pair."""
    result = _MODULE.get_official_eval_result(gt_annos, dt_annos, current_classes)
    evaluation = _MODULE.official_eval_metrics(gt_annos, dt_annos, current_classes)
    metrics = {}
    for metric, values in (
        ("3d", evaluation["3d"]),
        ("bev", evaluation["bev"]),
        ("image", evaluation["bbox"]),
    ):
        for index, difficulty in enumerate(("easy", "moderate", "hard")):
            metrics[f"Car_{metric}_{difficulty}"] = values[0, index, 0]
    return result, metrics
