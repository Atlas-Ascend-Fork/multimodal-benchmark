# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from spconv.utils import non_max_suppression_cpu, rotate_non_max_suppression_cpu
import numba
import numpy as np

from second.core import box_np_ops


def nms_cc(dets, thresh):
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    return non_max_suppression_cpu(dets, order, thresh, 1.0)


def rotate_nms_cc(dets, thresh):

    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    dets_corners = box_np_ops.center_to_corner_box2d(
        dets[:, :2], dets[:, 2:4], dets[:, 4]
    )

    dets_standup = box_np_ops.corner_to_standup_nd(dets_corners)

    standup_iou = box_np_ops.iou_jit(dets_standup, dets_standup, eps=0.0)
    return rotate_non_max_suppression_cpu(dets_corners, order, standup_iou, thresh)


@numba.jit(nopython=True)
def nms_jit(dets, thresh, eps=0.0):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + eps) * (y2 - y1 + eps)
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate iou between i and j box
            w = max(min(x2[i], x2[j]) - max(x1[i], x1[j]) + eps, 0.0)
            h = max(min(y2[i], y2[j]) - max(y1[i], y1[j]) + eps, 0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep


@numba.jit(nopython=True)
def _swap_box_rows(boxes, first, second):
    for column in range(5):
        value = boxes[first, column]
        boxes[first, column] = boxes[second, column]
        boxes[second, column] = value


@numba.jit(nopython=True)
def _copy_box_row(boxes, target, source):
    for column in range(5):
        boxes[target, column] = boxes[source, column]


@numba.jit(nopython=True)
def _maximum_score_position(boxes, start, end):
    maximum_position = start
    for position in range(start + 1, end):
        if boxes[maximum_position, 4] < boxes[position, 4]:
            maximum_position = position
    return maximum_position


@numba.jit(nopython=True)
def _soft_nms_weight(overlap, method, threshold, sigma):
    if method == 1:
        return 1 - overlap if overlap > threshold else 1
    if method == 2:
        return np.exp(-(overlap * overlap) / sigma)
    return 0 if overlap > threshold else 1


@numba.jit("float32[:, :], float32, float32, float32, uint32", nopython=True)
def soft_nms_jit(boxes, sigma=0.5, nt=0.3, threshold=0.001, method=0):
    n = boxes.shape[0]
    for i in range(n):
        maximum_position = _maximum_score_position(boxes, i, n)
        _swap_box_rows(boxes, i, maximum_position)
        current_box = boxes[i].copy()
        pos = i + 1
        while pos < n:
            candidate = boxes[pos]
            area = (candidate[2] - candidate[0] + 1) * (candidate[3] - candidate[1] + 1)
            intersection_width = (
                min(current_box[2], candidate[2])
                - max(current_box[0], candidate[0])
                + 1
            )
            intersection_height = (
                min(current_box[3], candidate[3])
                - max(current_box[1], candidate[1])
                + 1
            )
            if intersection_width <= 0 or intersection_height <= 0:
                pos += 1
                continue
            current_area = (current_box[2] - current_box[0] + 1) * (
                current_box[3] - current_box[1] + 1
            )
            union_area = float(
                current_area + area - intersection_width * intersection_height
            )
            overlap = intersection_width * intersection_height / union_area
            boxes[pos, 4] *= _soft_nms_weight(overlap, method, nt, sigma)
            if boxes[pos, 4] < threshold:
                _copy_box_row(boxes, pos, n - 1)
                n -= 1
                pos -= 1
            pos += 1

    keep = [i for i in range(n)]
    return keep
