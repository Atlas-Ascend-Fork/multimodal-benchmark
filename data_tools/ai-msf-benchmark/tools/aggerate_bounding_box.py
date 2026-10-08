# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

from ops.second.my_eval import get_label_anno
from ops.second.utils.eval import calculate_iou_partly
import numpy as np
from shared.detection_merge import (
    MERGE_DP_PARAMETERS_SCHEMA,
    detection_paths,
    merge_all_detections,
    merge_without_overlap,
    read_detection_lines,
    write_detection_lines,
)


from shared.parameters import bind_parameters as _bind_parameters


LOGGER = logging.getLogger(__name__)


def _merge_car_detections(paths, output_dir, merge_config):
    paths1, paths2 = paths
    iou_threshold, score_threshold, metric = merge_config
    added_count = 0
    for path1, path2 in zip(paths1, paths2):
        lines1 = [
            line for line in read_detection_lines(path1) if line.startswith("Car")
        ]
        lines2 = [
            line for line in read_detection_lines(path2) if line.startswith("Car")
        ]
        merged = merge_without_overlap(lines1, lines2, score_threshold)
        if merged is None:
            fusion_results = get_label_anno(path1, "Car")
            signal_results = get_label_anno(path2, "Car")
            overlaps = calculate_iou_partly(
                signal_results, fusion_results, metric=metric, num_parts=1
            )[0][0]
            try:
                maximum_overlaps = np.max(overlaps, axis=1)
            except (TypeError, ValueError) as exc:
                LOGGER.info("%s", overlaps)
                LOGGER.info("%s %s", lines1, lines2)
                raise ValueError("unable to calculate detection overlaps") from exc
            selected = np.where(maximum_overlaps < iou_threshold)[0]
            additions = [
                lines2[index]
                for index in selected
                if float(lines2[index].strip().split(" ")[-1]) >= score_threshold
            ]
            merged = lines1 + additions
        added_count += len(merged) - len(lines1)
        write_detection_lines(path1, output_dir, merged)
    LOGGER.info("add %s objects", added_count)


def merge_dp(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        MERGE_DP_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    dir1, dir2, output_dir, iou_th, score, metric = bound_parameters
    dts1, dts2 = detection_paths(dir1, dir2, metric, require_nonempty=True)
    if iou_th is None:
        merge_all_detections(dts1, dts2, output_dir)
    else:
        _merge_car_detections((dts1, dts2), output_dir, (iou_th, score, metric))
