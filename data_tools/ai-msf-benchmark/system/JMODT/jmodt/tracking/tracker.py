# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import numpy as np
import torch

from jmodt.tracking.data_association import ortools_solve, hungarian_match
from jmodt.tracking.track import Track
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters

_HELPER_CONTINUE = object()


class TrackerInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    link_model: object
    se_model: object
    t_miss: object
    t_hit: object
    w_cls: object
    w_app: object
    w_iou: object
    w_dis: object
    w_se: object
    hungarian: object


class TrackerUpdateParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    frame_id: object
    boxes_3d: object
    det_scores: object
    det_features: object
    frame_detections: object


_TRACKER_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    TrackerInitParameters,
    (
        "link_model",
        "se_model",
        "t_miss",
        "t_hit",
        "w_cls",
        "w_app",
        "w_iou",
        "w_dis",
        "w_se",
        "hungarian",
    ),
    {"hungarian": False},
    frozenset(()),
    10,
    None,
    None,
)
_TRACKER_UPDATE_PARAMETERS_SCHEMA = _ParameterSchema(
    TrackerUpdateParameters,
    ("frame_id", "boxes_3d", "det_scores", "det_features", "frame_detections"),
    {},
    frozenset(()),
    5,
    None,
    None,
)


def _tracker_update_142_part_1_chunk_1(_state):
    _state["bound_parameters"] = _bind_parameters(
        _TRACKER_UPDATE_PARAMETERS_SCHEMA, _state["call_args"], _state["call_kwargs"]
    )
    (
        _state["frame_id"],
        _state["boxes_3d"],
        _state["det_scores"],
        _state["det_features"],
        _state["frame_detections"],
    ) = _state["bound_parameters"]
    _state["num_det"] = len(_state["det_scores"])
    _state["num_pred"] = len(_state["self"].tracks)
    if _state["num_det"] == 0:
        return []
    _state["passed_frames"] = _state["frame_id"] - _state["self"].last_frame_idx
    _state["self"].frame_count += _state["passed_frames"]
    _state["self"].last_frame_idx = _state["frame_id"]
    if _state["num_pred"] == 0:
        for _state["d"] in range(_state["num_det"]):
            _state["self"].tracks.append(
                Track(
                    bbox=_state["boxes_3d"][_state["d"]],
                    score=_state["det_scores"][_state["d"]],
                    feature=_state["det_features"][_state["d"]],
                    info=_state["frame_detections"][_state["d"]],
                )
            )
        return _state["self"].track_management()
    _state["pred_boxes"] = []
    _state["pred_scores"] = []
    _state["pred_features"] = []
    for _state["trk"] in _state["self"].tracks:
        _state["box"], _state["score"], _state["feature"] = _state["trk"].predict(
            _state["passed_frames"]
        )
        _state["pred_boxes"].append(_state["box"])
        _state["pred_scores"].append(_state["score"])
        _state["pred_features"].append(_state["feature"].view(1, -1))
    _state["pred_boxes"] = np.vstack(_state["pred_boxes"])
    _state["pred_scores"] = np.array(_state["pred_scores"])
    _state["pred_features"] = torch.cat(_state["pred_features"])
    _state["cor_feat"] = torch.abs(
        _state["pred_features"].unsqueeze(1).repeat(1, _state["num_det"], 1)
        - _state["det_features"].unsqueeze(0).repeat(_state["num_pred"], 1, 1)
    )


def _tracker_update_142_part_1_chunk_2(_state):
    _state["link_scores"] = (
        _state["self"]
        .link_model(
            _state["cor_feat"].view(_state["num_pred"] * _state["num_det"], -1, 1)
        )
        .view(_state["num_pred"], _state["num_det"])
    )
    return _HELPER_CONTINUE


def _tracker_update_142_part_1(_state):
    early_result = _tracker_update_142_part_1_chunk_1(_state)
    if early_result is not None:
        return early_result
    return _tracker_update_142_part_1_chunk_2(_state)


def _tracker_update_142_part_2(_state):
    _state["link_score_pred"] = torch.softmax(_state["link_scores"], dim=1)
    _state["link_score_det"] = torch.softmax(_state["link_scores"], dim=0)
    _state["link_scores"] = (_state["link_score_pred"] + _state["link_score_det"]) / 2


def _tracker_ortools_match(_state):
    _state["cls_scores"] = _state["self"].w_cls * (
        np.concatenate([_state["pred_scores"], _state["det_scores"]]) - 1
    )
    _state["start_scores"] = (
        _state["self"].w_se
        * torch.sigmoid(
            _state["self"].se_model(_state["cor_feat"].mean(dim=0).unsqueeze(-1))
        )
        .cpu()
        .numpy()
        .flatten()
    )
    _state["end_scores"] = (
        _state["self"].w_se
        * torch.sigmoid(
            _state["self"].se_model(_state["cor_feat"].mean(dim=1).unsqueeze(-1))
        )
        .cpu()
        .numpy()
        .flatten()
    )
    _state["start_scores"] = np.concatenate(
        [np.zeros(_state["num_pred"]), _state["start_scores"]]
    )
    _state["end_scores"] = np.concatenate(
        [_state["end_scores"], np.zeros(_state["num_det"])]
    )
    _state["matched"], _state["unmatched_dets"], _state["tentative_dets"] = (
        ortools_solve(
            torch.from_numpy(_state["boxes_3d"]).cuda(non_blocking=True),
            torch.from_numpy(_state["pred_boxes"]).cuda(non_blocking=True),
            _state["cls_scores"],
            _state["link_scores"],
            _state["start_scores"],
            _state["end_scores"],
            w_app=_state["self"].w_app,
            w_iou=_state["self"].w_iou,
            w_dis=_state["self"].w_dis,
        )
    )


def _tracker_update_142_part_3(_state):
    if _state["self"].hungarian:
        _state["matched"], _state["unmatched_dets"], _state["tentative_dets"] = (
            hungarian_match(
                torch.from_numpy(_state["boxes_3d"]).cuda(non_blocking=True),
                torch.from_numpy(_state["pred_boxes"]).cuda(non_blocking=True),
                _state["det_scores"],
                _state["link_scores"],
                w_app=_state["self"].w_app,
                w_iou=_state["self"].w_iou,
                score_threshold=0,
            )
        )
    else:
        _tracker_ortools_match(_state)


def _tracker_update_142_part_4(_state):
    for _state["t"], _state["d"] in _state["matched"]:
        _state["self"].tracks[_state["t"]].update_with_feature(
            _state["boxes_3d"][_state["d"]],
            _state["det_features"][_state["d"]],
            _state["det_scores"][_state["d"]],
            info=_state["frame_detections"][_state["d"]],
        )
    for _state["i"] in _state["unmatched_dets"]:
        _state["trk"] = Track(
            bbox=_state["boxes_3d"][_state["i"]],
            feature=_state["det_features"][_state["i"]],
            score=_state["det_scores"][_state["i"]],
            info=_state["frame_detections"][_state["i"]],
        )
        _state["self"].tracks.append(_state["trk"])
    for _state["i"] in _state["tentative_dets"]:
        _state["trk"] = Track(
            bbox=_state["boxes_3d"][_state["i"]],
            feature=_state["det_features"][_state["i"]],
            score=_state["det_scores"][_state["i"]],
            info=_state["frame_detections"][_state["i"]],
        )
        _state["trk"].misses += 1
        _state["self"].tracks.append(_state["trk"])


class Tracker:
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRACKER_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            link_model,
            se_model,
            t_miss,
            t_hit,
            w_cls,
            w_app,
            w_iou,
            w_dis,
            w_se,
            hungarian,
        ) = bound_parameters
        self.link_model = link_model
        self.se_model = se_model
        self.t_miss = t_miss
        self.t_hit = t_hit
        self.w_cls = w_cls
        self.w_app = w_app
        self.w_iou = w_iou
        self.w_dis = w_dis
        self.w_se = w_se
        self.hungarian = hungarian

        self.tracks = []
        self.frame_count = 0
        self.last_frame_idx = 0

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        self.last_frame_idx = 0
        Track.new_id = 1

    def track_management(self):
        idx = len(self.tracks)
        results = []
        for trk in reversed(self.tracks):
            if trk.hits >= self.t_hit or self.frame_count <= self.t_hit:
                if trk.misses == 0:
                    results.append(trk.get_data())
            idx -= 1
            # remove dead tracks
            if trk.misses >= self.t_miss:
                self.tracks.pop(idx)
        return results

    def update(self, *call_args, **call_kwargs):
        _state = {"self": self, "call_args": call_args, "call_kwargs": call_kwargs}
        _helper_result = _tracker_update_142_part_1(_state)
        if _helper_result is not _HELPER_CONTINUE:
            return _helper_result
        _tracker_update_142_part_2(_state)
        _tracker_update_142_part_3(_state)
        _tracker_update_142_part_4(_state)
        return _state["self"].track_management()
