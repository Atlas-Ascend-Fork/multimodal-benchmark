# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Author: wangxy

from typing import NamedTuple
from tracking import kalman_filter_2d
from tracking.cost_function import iou_batch
from tracking.kalman_fileter_3d import KalmanBoxTracker
from tracking.matching import (
    associate_detections_to_trackers_fusion,
    associate_2_d_to_3_d_tracking,
    linear_assignment,
)
from tracking.track_2d import Track2D
from tracking.track_3d import Track3D
import numpy as np


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class TrackerUpdateParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    detection_3_d_fusion: object
    detection_3_d_only: object
    detection_3_dto2_d_only: object
    detection_2_d_only: object
    calib_file: object
    iou_threshold: object


def associate_detections_to_tracks(tracks, detections, threshold):
    track_indices = [index for index, _ in enumerate(tracks)]
    detection_indices = [index for index, _ in enumerate(detections)]
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t, trk in enumerate(tracks):
        for d, det in enumerate(detections):
            iou_matrix[t, d] = iou_batch(
                trk.x1y1x2y2(), det.to_x1y1x2y2()
            )  # det: 8 x 3, trk: 8 x 3

    matches = []
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, _ in enumerate(detections):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, _ in enumerate(tracks):
        if t not in matched_indices[:, 0]:
            unmatched_trackers.append(t)

    # Filter out those pairs with small IoU
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < threshold:
            unmatched_detections.append(m[1])
            unmatched_trackers.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_trackers), np.array(unmatched_detections)


_TRACKER_UPDATE_PARAMETERS_SCHEMA = _ParameterSchema(
    TrackerUpdateParameters,
    (
        "detection_3_d_fusion",
        "detection_3_d_only",
        "detection_3_dto2_d_only",
        "detection_2_d_only",
        "calib_file",
        "iou_threshold",
    ),
    {},
    frozenset(()),
    6,
    None,
    None,
)


class Tracker:
    def __init__(self, max_age, n_init):
        self.max_age = max_age
        self.n_init = n_init
        self.tracks_3d = []
        self.tracks_2d = []
        self.track_id_3d = 0  # The id of 3D track is represented by an even number.
        self.track_id_2d = 1  # The id of 3D track is represented by an odd number.
        self.unmatch_tracks_3d = []
        self.unmatch_tracks_3d1 = []
        self.unmatch_tracks_3d2 = []
        self.kf_2d = kalman_filter_2d.KalmanFilter()
        self.kf_3d = None
        self.additional_info = None

    def predict_3d(self):
        for track in self.tracks_3d:
            track.predict_3d(track.kf_3d)

    def predict_2d(self):
        for track in self.tracks_2d:
            track.predict_2d(self.kf_2d)

    def update(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRACKER_UPDATE_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            detection_3_d_fusion,
            detection_3_d_only,
            detection_3_dto2_d_only,
            detection_2_d_only,
            calib_file,
            iou_threshold,
        ) = bound_parameters
        self._update_fusion_tracks(detection_3_d_fusion)
        self._update_3d_only_tracks(detection_3_d_only)
        self._update_2d_tracks(detection_2_d_only, iou_threshold)
        self._merge_2d_tracks(calib_file, iou_threshold)

    def _update_fusion_tracks(self, detections):
        matched, unmatched_detections, unmatched_tracks = (
            associate_detections_to_trackers_fusion(
                detections, self.tracks_3d, iou_threshold=0.01
            )
        )
        for detection_idx, track_idx in matched:
            track = self.tracks_3d[track_idx]
            track.update_3d(detections[detection_idx])
            track.state = 2
            track.fusion_time_update = 0
        for track_idx in unmatched_tracks:
            self.tracks_3d[track_idx].fusion_time_update += 1
            self.tracks_3d[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track_3d(detections[detection_idx])

    def _update_3d_only_tracks(self, detections):
        self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        matched, unmatched_detections, _ = associate_detections_to_trackers_fusion(
            detections, self.unmatch_tracks_3d1, iou_threshold=0.01
        )
        matched_track_indices = []
        for detection_idx, track_idx in matched:
            unmatched_track = self.unmatch_tracks_3d1[track_idx]
            for track in self.tracks_3d:
                if track.track_id_3d == unmatched_track.track_id_3d:
                    track.update_3d(detections[detection_idx])
                    matched_track_indices.append(track_idx)
                    break
        self.unmatch_tracks_3d1 = [
            track
            for index, track in enumerate(self.unmatch_tracks_3d1)
            if index not in matched_track_indices
        ]
        for detection_idx in unmatched_detections:
            self._initiate_track_3d(detections[detection_idx])
        self.unmatch_tracks_3d2 = [
            track
            for track in self.tracks_3d
            if track.time_since_update == 0 and track.hits == 1
        ]
        self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2

    def _update_2d_tracks(self, detections, iou_threshold):
        matched, unmatched_tracks, unmatched_detections = (
            associate_detections_to_tracks(self.tracks_2d, detections, iou_threshold)
        )
        for track_idx, detection_idx in matched:
            self.tracks_2d[track_idx].update_2d(
                self.kf_2d, detections[detection_idx]
            )
        for track_idx in unmatched_tracks:
            self.tracks_2d[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track_2d(detections[detection_idx])
        self.tracks_2d = [track for track in self.tracks_2d if not track.is_deleted()]

    def _merge_2d_tracks(self, calib_file, iou_threshold):
        matched, _ = associate_2_d_to_3_d_tracking(
            self.tracks_2d, self.unmatch_tracks_3d, calib_file, iou_threshold
        )
        matched_2d_indices = []
        for track_idx_2d, track_idx_3d in matched:
            target_id = self.unmatch_tracks_3d[track_idx_3d].track_id_3d
            for track_3d in self.tracks_3d:
                if track_3d.track_id_3d == target_id:
                    track_2d = self.tracks_2d[track_idx_2d]
                    track_3d.age = track_2d.age + 1
                    track_3d.time_since_update = 0
                    track_3d.hits = (
                        track_2d.hits + 1
                        if track_2d.hits >= 2
                        else track_3d.hits + 1
                    )
                    track_3d.state_update()
            matched_2d_indices.append(track_idx_2d)
        self.tracks_2d = [
            track
            for index, track in enumerate(self.tracks_2d)
            if index not in matched_2d_indices
        ]
        self.tracks_3d = [track for track in self.tracks_3d if not track.is_deleted()]

    def _initiate_track_3d(self, detection):
        self.kf_3d = KalmanBoxTracker(detection.bbox)
        self.additional_info = detection.additional_info
        pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.tracks_3d.append(
            Track3D(
                pose,
                self.kf_3d,
                self.track_id_3d,
                self.n_init,
                self.max_age,
                self.additional_info,
            )
        )
        self.track_id_3d += 2

    def _initiate_track_2d(self, detection):
        mean, covariance = self.kf_2d.initiate(detection.to_xyah())
        self.tracks_2d.append(
            Track2D(mean, covariance, self.track_id_2d, self.n_init, self.max_age)
        )
        self.track_id_2d += 2
