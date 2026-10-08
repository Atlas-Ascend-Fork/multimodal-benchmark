# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Author: wangxy
from typing import NamedTuple

from tracking.cost_function import iou_batch
from tracking.matching import linear_assignment
import numpy as np


class DataFusionResult(NamedTuple):
    """Named result returned without changing tuple unpacking behavior."""

    value_0: object
    value_1: object
    value_2: object
    value_3: object
    value_4: object
    value_5: object


def _module_datafusion2_dand3_d_28_part_1(_state):
    _state["iou_threshold"] = 0.3
    _state["iou_matrix"] = np.zeros(
        (len(_state["detection_2_d"]), len(_state["detection_3_dto2_d"])),
        dtype=np.float32,
    )
    for _state["d1"], _state["det_2_d"] in enumerate(_state["detection_2_d"]):
        for _state["d2"], _state["det_3_dto2_d"] in enumerate(
            _state["additional_info"]
        ):
            _state["iou_matrix"][_state["d1"], _state["d2"]] = iou_batch(
                _state["det_2_d"], _state["det_3_dto2_d"][2:6]
            )
    if min(_state["iou_matrix"].shape) > 0:
        _state["a"] = (_state["iou_matrix"] > _state["iou_threshold"]).astype(np.int32)
        if _state["a"].sum(1).max() == 1 and _state["a"].sum(0).max() == 1:
            _state["matched_indices"] = np.stack(np.where(_state["a"]), axis=1)
        else:
            _state["matched_indices"] = linear_assignment(-_state["iou_matrix"])
    else:
        _state["matched_indices"] = np.empty(shape=(0, 2))
    (
        _state["matched"],
        _state["unmatched_detections_2_d"],
        _state["unmatched_detection_3_dto2_d"],
    ) = ([], [], [])
    _state["detection_3_d_fusion_info"], _state["detection_3_d_only_info"] = ([], [])
    for _state["d"], _state["_"] in enumerate(_state["detection_2_d"]):
        if _state["d"] not in _state["matched_indices"][:, 0]:
            _state["unmatched_detections_2_d"].append(_state["d"])
    for _state["t"], _state["_"] in enumerate(_state["detection_3_dto2_d"]):
        if _state["t"] not in _state["matched_indices"][:, 1]:
            _state["unmatched_detection_3_dto2_d"].append(_state["t"])
    for _state["m"] in _state["matched_indices"]:
        if (
            _state["iou_matrix"][_state["m"][0], _state["m"][1]]
            < _state["iou_threshold"]
        ):
            _state["unmatched_detections_2_d"].append(_state["m"][0])
            _state["unmatched_detection_3_dto2_d"].append(_state["m"][1])
        else:
            _state["matched"].append(_state["m"].reshape(1, 2))


def _module_datafusion2_dand3_d_28_part_2_chunk_1(_state):
    if len(_state["matched"]) == 0:
        _state["matched"] = np.empty((0, 2), dtype=int)
    else:
        _state["matched"] = np.concatenate(_state["matched"], axis=0)
    (
        _state["detection_2_d_fusion"],
        _state["detection_3_dto2_d_fusion"],
        _state["detection_3_d_fusion"],
        _state["detection_2_d_only"],
        _state["detection_2_dto3_d_only"],
        _state["detection_3_d_only"],
    ) = ([], [], [], [], [], [])
    for _state["detection_2_d_idx"], _state["detection_3_dto2_d_idx"] in _state[
        "matched"
    ]:
        _state["detection_2_d_fusion"].append(
            _state["detection_2_d"][_state["detection_2_d_idx"]].tolist()
        )
        _state["detection_3_d_fusion_info"].append(
            _state["additional_info"][_state["detection_3_dto2_d_idx"]]
        )
        _state["detection_3_dto2_d_fusion"].append(
            _state["detection_3_dto2_d"][_state["detection_3_dto2_d_idx"]].tolist()
        )
        _state["detection_3_d_fusion"].append(
            _state["detections_3_d_camera"][_state["detection_3_dto2_d_idx"]].tolist()
        )
    for _state["unmatched_detections_2_d_idx"] in _state["unmatched_detections_2_d"]:
        _state["detection_2_d_only"].append(
            _state["detection_2_d"][_state["unmatched_detections_2_d_idx"]].tolist()
        )


def _module_datafusion2_dand3_d_28_part_2_chunk_2(_state):
    for _state["unmatched_detections_2_dto3_d_idx"] in _state[
        "unmatched_detection_3_dto2_d"
    ]:
        _state["detection_2_dto3_d_only"].append(
            _state["detection_3_dto2_d"][
                _state["unmatched_detections_2_dto3_d_idx"]
            ].tolist()
        )
        _state["detection_3_d_only"].append(
            _state["detections_3_d_camera"][
                _state["unmatched_detections_2_dto3_d_idx"]
            ].tolist()
        )
        _state["detection_3_d_only_info"].append(
            _state["additional_info"][_state["unmatched_detections_2_dto3_d_idx"]]
        )
    _state["detection_3_d_fusion"] = {
        "dets_3d_fusion": _state["detection_3_d_fusion"],
        "dets_3d_fusion_info": _state["detection_3_d_fusion_info"],
    }
    _state["detection_3_d_only"] = {
        "dets_3d_only": _state["detection_3_d_only"],
        "dets_3d_only_info": _state["detection_3_d_only_info"],
    }


def _module_datafusion2_dand3_d_28_part_2(_state):
    _module_datafusion2_dand3_d_28_part_2_chunk_1(_state)
    _module_datafusion2_dand3_d_28_part_2_chunk_2(_state)


def datafusion2_dand3_d(
    detections_3_d_camera, detection_2_d, detection_3_dto2_d, additional_info
):
    """
    :param detections_3D_camera:  Detections by LiDAR
    :param detection_2D:          Detections by Camera
    :param detection_3Dto2D:      Detections by LiDAR
    :return:
        detection_2D_fusion: Objects detected by LiDAR and camera, represented
            by a 2D bounding box in the pixel domain.
        detection_2D_only: Objects detected by camera only.
        detection_3D_fusion: Objects detected by LiDAR and camera, represented
            by a 3D bounding box in the camera domain.
        detection_3D_only: Objects detected by LiDAR only.
    """
    _state = locals()
    _module_datafusion2_dand3_d_28_part_1(_state)
    _module_datafusion2_dand3_d_28_part_2(_state)

    return DataFusionResult(
        np.array(_state["detection_2_d_fusion"]),
        np.array(_state["detection_3_dto2_d_fusion"]),
        _state["detection_3_d_fusion"],
        np.array(_state["detection_2_d_only"]),
        np.array(_state["detection_2_dto3_d_only"]),
        _state["detection_3_d_only"],
    )
