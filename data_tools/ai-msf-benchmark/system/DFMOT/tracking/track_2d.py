# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class Track2dInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    mean: object
    covariance: object
    track_id: object
    n_init: object
    max_age: object
    feature: object


# Note：The code code is referenced from https://github.com/nwojke/deep_sort

"""
  2D track management
  Reactivate: When a confirmed trajectory is occluded and in turn cannot be associated
  with any detections for several frames, it is then regarded as a reappeared trajectory.
"""


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4


class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2


_TRACK2D_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    Track2dInitParameters,
    ("mean", "covariance", "track_id", "n_init", "max_age", "feature"),
    {"feature": None},
    frozenset(()),
    6,
    None,
    None,
)


class Track2D:
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRACK2D_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        mean, covariance, track_id, n_init, max_age, feature = bound_parameters
        self.mean = mean
        self.covariance = covariance
        self.track_id_2d = track_id  #
        self.hits = 1
        self.age = 1
        self.state = TrackState.Tentative
        self.is3_d_or_2_d_track = TrackState3Dor2D.Tracking_2D  # 2D tracking
        self.time_since_update = 0
        self.n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """
        Get current position in bounding box format `(top left x, top left y, width, height)`.
        Returns
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def x1y1x2y2(self):
        """
        Get current position in bounding box format `(min x, miny, max x, max y)`.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def increment(self):
        self.age += 1
        self.time_since_update += 1

    def predict_2d(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment()

    def update_2d(self, kf, detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Reactivate:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative or self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
