# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import numpy as np


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class Track3dInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    pose: object
    kf_3d: object
    track_id_3d: object
    n_init: object
    max_age: object
    additional_info: object
    feature: object


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4


class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2


_TRACK3D_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    Track3dInitParameters,
    ("pose", "kf_3d", "track_id_3d", "n_init", "max_age", "additional_info", "feature"),
    {"feature": None},
    frozenset(()),
    7,
    None,
    None,
)


class Track3D:
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRACK3D_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        pose, kf_3d, track_id_3d, n_init, max_age, additional_info, feature = (
            bound_parameters
        )
        self.pose = pose
        self.kf_3d = kf_3d
        self.track_id_3d = track_id_3d
        self.hits = 1
        self.age = 1
        self.state = TrackState.Tentative
        self.n_init = n_init
        self._max_age = max_age
        self.is3_d_or_2_d_track = TrackState3Dor2D.Tracking_3D
        self.additional_info = additional_info
        self.time_since_update = 0
        self.fusion_time_update = 0

    def predict_3d(self, trk_3d):
        self.pose = trk_3d.predict()

    def update_3d(self, detection_3d):
        self.kf_3d.update(detection_3d.bbox)
        self.additional_info = detection_3d.additional_info
        self.pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative
        if self.fusion_time_update >= 3:
            self.state = TrackState.Reactivate

    def state_update(self):
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate
        elif self.time_since_update >= 1 and self.state != TrackState.Reactivate:
            self.state = TrackState.Deleted
        elif (
            self.state == TrackState.Reactivate
            and self.time_since_update > self._max_age
        ):
            self.state = TrackState.Deleted

    def fusion_state(self):
        if self.fusion_time_update >= 2:
            self.state = TrackState.Deleted

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_track_id_3d(self):
        track_id_3d = self.track_id_3d
        return track_id_3d
