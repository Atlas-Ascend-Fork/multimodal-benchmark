# -*-coding:utf-8-*
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# author: wangxy
from tracking.cost_function import iou3d, convert_3dbox_to_8corner, iou_batch
import numpy as np


def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def _corner_boxes(items, attribute):
    boxes = [convert_3dbox_to_8corner(getattr(item, attribute)) for item in items]
    return np.stack(boxes, axis=0) if boxes else []


def _iou_assignment(iou_matrix, threshold):
    if min(iou_matrix.shape) == 0:
        return np.empty((0, 2), dtype=int)
    valid = (iou_matrix > threshold).astype(np.int32)
    if valid.sum(1).max() == 1 and valid.sum(0).max() == 1:
        return np.stack(np.where(valid), axis=1)
    return linear_assignment(-iou_matrix)


def _filter_iou_matches(indices, iou_matrix, threshold):
    matches = []
    unmatched_detections = set(range(iou_matrix.shape[0]))
    unmatched_trackers = set(range(iou_matrix.shape[1]))
    for detection, tracker in indices:
        if iou_matrix[detection, tracker] < threshold:
            continue
        matches.append((detection, tracker))
        unmatched_detections.discard(detection)
        unmatched_trackers.discard(tracker)
    return (
        np.asarray(matches, dtype=int).reshape(-1, 2),
        np.asarray(sorted(unmatched_detections), dtype=int),
        np.asarray(sorted(unmatched_trackers), dtype=int),
    )


def associate_detections_to_trackers_fusion(detections, trackers, iou_threshold):
    """
        Assigns detections to tracked object (both represented as bounding boxes)
    detections:  N x 8 x 3
        trackers:    M x 8 x 3
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    dets_8corner = _corner_boxes(detections, "bbox")
    trks_8corner = _corner_boxes(trackers, "pose")
    if len(trks_8corner) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(dets_8corner)),
            np.empty((0, 8, 3), dtype=int),
        )

    iou_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)
    for d, det in enumerate(dets_8corner):
        for t, trk in enumerate(trks_8corner):
            iou_matrix[d, t] = iou3d(det, trk)[0]  # det: 8 x 3, trk: 8 x 3

    matched_indices = _iou_assignment(iou_matrix, iou_threshold)
    return _filter_iou_matches(matched_indices, iou_matrix, iou_threshold)


def trackfusion2_dand3_d(tracker_2_d, trks_3_dto2_d_image, iou_threshold):
    track_indices = list(range(len(tracker_2_d)))  # 跟踪对象索引
    detection_indices = list(range(len(trks_3_dto2_d_image)))  # 检测对象索引
    matches = []
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    iou_matrix = np.zeros(
        (len(tracker_2_d), len(trks_3_dto2_d_image)), dtype=np.float32
    )
    for t, trk in enumerate(tracker_2_d):
        for d, det in enumerate(trks_3_dto2_d_image):
            iou_matrix[t, d] = iou_batch(trk.x1y1x2y2(), det)  # det: 8 x 3, trk: 8 x 3
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, _ in enumerate(trks_3_dto2_d_image):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    unmatched_tracker_2_d = []
    for t, _ in enumerate(tracker_2_d):
        if t not in matched_indices[:, 0]:
            unmatched_tracker_2_d.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[1])
            unmatched_tracker_2_d.append(m[0])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_tracker_2_d), np.array(unmatched_detections)


def associate_2_d_to_3_d_tracking(tracker_2_d, tracks_3_d, calib_file, iou_threshold):
    trks_3_dto2_d_image = [list(i.additional_info[2:6]) for i in tracks_3_d]
    matched_track_2_d, unmatch_tracker_2_d, _ = trackfusion2_dand3_d(
        tracker_2_d, trks_3_dto2_d_image, iou_threshold
    )
    return matched_track_2_d, unmatch_tracker_2_d
