# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GPU NMS entry points with a shared rotated-IoU implementation."""

__all__ = [
    "nms_gpu", "nms_gpu_cc", "rotate_iou_gpu", "rotate_iou_gpu_eval",
    "rotate_nms_gpu",
]

from pathlib import Path

import numpy as np
from spconv.utils import non_max_suppression

from shared.compatibility import load_from_path


_ROTATED_IOU = load_from_path(
    "_ai_msf_rotate_iou",
    Path(__file__).parents[4] / "EPNet/tools/kitti_object_eval_python/rotate_iou.py",
)
rotate_iou_gpu_eval = _ROTATED_IOU.rotate_iou_gpu_eval


def nms_gpu(dets, nms_overlap_thresh, device_id=0):
    """Apply axis-aligned GPU NMS and return indices in the input ordering."""
    if dets.shape[0] == 0:
        return []
    keep = np.zeros(dets.shape[0], dtype=np.int32)
    order = dets[:, 4].argsort()[::-1].astype(np.int32)
    num_out = non_max_suppression(
        dets[order, :], keep, nms_overlap_thresh, device_id
    )
    return list(order[keep[:num_out]])


def nms_gpu_cc(dets, nms_overlap_thresh, device_id=0):
    """Backward-compatible name for :func:`nms_gpu`."""
    return nms_gpu(dets, nms_overlap_thresh, device_id)


def rotate_iou_gpu(boxes, query_boxes, device_id=0):
    """Compute pairwise rotated IoU using the shared CUDA implementation."""
    return rotate_iou_gpu_eval(boxes, query_boxes, -1, device_id)


def rotate_nms_gpu(dets, nms_overlap_thresh, device_id=0):
    """Apply rotated NMS using one shared pairwise-IoU kernel."""
    if dets.shape[0] == 0:
        return []
    order = dets[:, 5].argsort()[::-1].astype(np.int32)
    boxes = dets[order, :5].astype(np.float32)
    overlaps = rotate_iou_gpu_eval(boxes, boxes, -1, device_id)
    suppressed = np.zeros(len(order), dtype=bool)
    keep = []
    for current, ordered_index in enumerate(order):
        if suppressed[current]:
            continue
        keep.append(ordered_index)
        suppressed[current + 1:] |= overlaps[current, current + 1:] > nms_overlap_thresh
    return keep
