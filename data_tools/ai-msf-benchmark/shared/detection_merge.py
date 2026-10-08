# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""File handling for detection-result merging tools."""

import glob
import os
from typing import NamedTuple

from natsort import natsorted

from shared.parameters import ParameterSchema


class MergeDpParameters(NamedTuple):
    dir1: object
    dir2: object
    output_dir: object
    iou_th: object
    score: object
    metric: object


MERGE_DP_PARAMETERS_SCHEMA = ParameterSchema(
    MergeDpParameters,
    ("dir1", "dir2", "output_dir", "iou_th", "score", "metric"),
    {"iou_th": None, "score": 0.0, "metric": None},
    frozenset(()),
    6,
    None,
    None,
)


def read_detection_lines(path):
    with open(path, "r") as detection_file:
        lines = detection_file.readlines()
    if lines:
        lines[-1] = lines[-1].strip() + "\n"
    return lines


def write_detection_lines(source_path, output_dir, lines):
    if lines:
        lines[-1] = lines[-1].strip()
    output_path = os.path.join(output_dir, os.path.basename(source_path))
    with open(output_path, "w") as output_file:
        output_file.writelines(lines)


def merge_all_detections(paths1, paths2, output_dir):
    for path1, path2 in zip(paths1, paths2):
        lines = read_detection_lines(path1) + read_detection_lines(path2)
        lines.sort()
        write_detection_lines(path1, output_dir, lines)


def merge_without_overlap(primary, secondary, score_threshold):
    """Resolve a merge when either detector produced no boxes."""
    if not primary:
        return [
            line for line in secondary
            if float(line.strip().split(" ")[-1]) > score_threshold
        ]
    if not secondary:
        return primary
    return None


def detection_paths(dir1, dir2, metric, require_nonempty=False):
    if metric is None:
        raise ValueError()
    paths1 = glob.glob(dir1 + "/*")
    paths2 = glob.glob(dir2 + "/*")
    if len(paths1) != len(paths2):
        raise ValueError(dir1, len(paths1), dir2, len(paths2))
    if require_nonempty and not paths1:
        raise ValueError("input directories contain no detections")
    return natsorted(paths1), natsorted(paths2)
