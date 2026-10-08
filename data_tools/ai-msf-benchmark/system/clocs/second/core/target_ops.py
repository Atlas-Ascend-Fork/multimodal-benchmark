# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import logging

import numpy as np
import numpy.random as npr


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class CreateTargetNpParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    all_anchors: object
    gt_boxes: object
    similarity_fn: object
    box_encoding_fn: object
    prune_anchor_fn: object
    gt_classes: object
    matched_threshold: object
    unmatched_threshold: object
    bbox_inside_weight: object
    positive_fraction: object
    rpn_batch_size: object
    norm_by_num_examples: object
    box_code_size: object


logger = logging.getLogger(__name__)


def unmap(data, count, inds, fill=0):
    """Unmap a data subset back to the original set of ``count`` items."""
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


_CREATE_TARGET_NP_PARAMETERS_SCHEMA = _ParameterSchema(
    CreateTargetNpParameters,
    (
        "all_anchors",
        "gt_boxes",
        "similarity_fn",
        "box_encoding_fn",
        "prune_anchor_fn",
        "gt_classes",
        "matched_threshold",
        "unmatched_threshold",
        "bbox_inside_weight",
        "positive_fraction",
        "rpn_batch_size",
        "norm_by_num_examples",
        "box_code_size",
    ),
    {
        "prune_anchor_fn": None,
        "gt_classes": None,
        "matched_threshold": 0.6,
        "unmatched_threshold": 0.45,
        "bbox_inside_weight": None,
        "positive_fraction": None,
        "rpn_batch_size": 300,
        "norm_by_num_examples": False,
        "box_code_size": 7,
    },
    frozenset(()),
    13,
    None,
    None,
)


def _module_create_target_np_134_part_1(_state):
    _state["bound_parameters"] = _bind_parameters(
        _CREATE_TARGET_NP_PARAMETERS_SCHEMA, _state["call_args"], _state["call_kwargs"]
    )
    (
        _state["all_anchors"],
        _state["gt_boxes"],
        _state["similarity_fn"],
        _state["box_encoding_fn"],
        _state["prune_anchor_fn"],
        _state["gt_classes"],
        _state["matched_threshold"],
        _state["unmatched_threshold"],
        _state["bbox_inside_weight"],
        _state["positive_fraction"],
        _state["rpn_batch_size"],
        _state["norm_by_num_examples"],
        _state["box_code_size"],
    ) = _state["bound_parameters"]
    _state["total_anchors"] = _state["all_anchors"].shape[0]
    if _state["prune_anchor_fn"] is not None:
        _state["inds_inside"] = _state["prune_anchor_fn"](_state["all_anchors"])
        _state["anchors"] = _state["all_anchors"][_state["inds_inside"], :]
        if not isinstance(_state["matched_threshold"], float):
            _state["matched_threshold"] = _state["matched_threshold"][
                _state["inds_inside"]
            ]
        if not isinstance(_state["unmatched_threshold"], float):
            _state["unmatched_threshold"] = _state["unmatched_threshold"][
                _state["inds_inside"]
            ]
    else:
        _state["anchors"] = _state["all_anchors"]
        _state["inds_inside"] = None
    _state["num_inside"] = (
        len(_state["inds_inside"])
        if _state["inds_inside"] is not None
        else _state["total_anchors"]
    )
    _state["box_ndim"] = _state["all_anchors"].shape[1]
    logger.debug("total_anchors: %s", _state["total_anchors"])


def _module_create_target_np_134_part_2(_state):
    logger.debug("inds_inside: %s", _state["num_inside"])
    logger.debug("anchors.shape: %s", _state["anchors"].shape)
    if _state["gt_classes"] is None:
        _state["gt_classes"] = np.ones([_state["gt_boxes"].shape[0]], dtype=np.int32)
    _state["labels"] = np.empty((_state["num_inside"],), dtype=np.int32)
    _state["gt_ids"] = np.empty((_state["num_inside"],), dtype=np.int32)
    _state["labels"].fill(-1)
    _state["gt_ids"].fill(-1)
    if len(_state["gt_boxes"]) > 0 and _state["anchors"].shape[0] > 0:
        _state["anchor_by_gt_overlap"] = _state["similarity_fn"](
            _state["anchors"], _state["gt_boxes"]
        )
        _state["anchor_to_gt_argmax"] = _state["anchor_by_gt_overlap"].argmax(axis=1)
        _state["anchor_to_gt_max"] = _state["anchor_by_gt_overlap"][
            np.arange(_state["num_inside"]), _state["anchor_to_gt_argmax"]
        ]
        _state["gt_to_anchor_argmax"] = _state["anchor_by_gt_overlap"].argmax(axis=0)
        _state["gt_to_anchor_max"] = _state["anchor_by_gt_overlap"][
            _state["gt_to_anchor_argmax"],
            np.arange(_state["anchor_by_gt_overlap"].shape[1]),
        ]
        _state["empty_gt_mask"] = _state["gt_to_anchor_max"] == 0
        _state["gt_to_anchor_max"][_state["empty_gt_mask"]] = -1
        _state["anchors_with_max_overlap"] = np.where(
            _state["anchor_by_gt_overlap"] == _state["gt_to_anchor_max"]
        )[0]
        _state["gt_inds_force"] = _state["anchor_to_gt_argmax"][
            _state["anchors_with_max_overlap"]
        ]
        _state["labels"][_state["anchors_with_max_overlap"]] = _state["gt_classes"][
            _state["gt_inds_force"]
        ]
        _state["gt_ids"][_state["anchors_with_max_overlap"]] = _state["gt_inds_force"]
        _state["pos_inds"] = _state["anchor_to_gt_max"] >= _state["matched_threshold"]
        _state["gt_inds"] = _state["anchor_to_gt_argmax"][_state["pos_inds"]]
        _state["labels"][_state["pos_inds"]] = _state["gt_classes"][_state["gt_inds"]]
        _state["gt_ids"][_state["pos_inds"]] = _state["gt_inds"]
        _state["bg_inds"] = np.where(
            _state["anchor_to_gt_max"] < _state["unmatched_threshold"]
        )[0]
    else:
        _state["bg_inds"] = np.arange(_state["num_inside"])
    _state["fg_inds"] = np.where(_state["labels"] > 0)[0]
    _state["fg_max_overlap"] = None


def _module_create_target_np_134_part_3(_state):
    if len(_state["gt_boxes"]) > 0 and _state["anchors"].shape[0] > 0:
        _state["fg_max_overlap"] = _state["anchor_to_gt_max"][_state["fg_inds"]]
    _state["gt_pos_ids"] = _state["gt_ids"][_state["fg_inds"]]
    if _state["positive_fraction"] is not None:
        _state["num_fg"] = int(_state["positive_fraction"] * _state["rpn_batch_size"])
        if len(_state["fg_inds"]) > _state["num_fg"]:
            _state["disable_inds"] = npr.choice(
                _state["fg_inds"],
                size=len(_state["fg_inds"]) - _state["num_fg"],
                replace=False,
            )
            _state["labels"][_state["disable_inds"]] = -1
            _state["fg_inds"] = np.where(_state["labels"] > 0)[0]
        _state["num_bg"] = _state["rpn_batch_size"] - np.sum(_state["labels"] > 0)
        if len(_state["bg_inds"]) > _state["num_bg"]:
            _state["enable_inds"] = _state["bg_inds"][
                npr.randint(len(_state["bg_inds"]), size=_state["num_bg"])
            ]
            _state["labels"][_state["enable_inds"]] = 0
        _state["bg_inds"] = np.where(_state["labels"] == 0)[0]
    elif len(_state["gt_boxes"]) == 0 or _state["anchors"].shape[0] == 0:
        _state["labels"][:] = 0
    else:
        _state["labels"][_state["bg_inds"]] = 0
        _state["labels"][_state["anchors_with_max_overlap"]] = _state["gt_classes"][
            _state["gt_inds_force"]
        ]
    _state["bbox_targets"] = np.zeros(
        (_state["num_inside"], _state["box_code_size"]),
        dtype=_state["all_anchors"].dtype,
    )
    if len(_state["gt_boxes"]) > 0 and _state["anchors"].shape[0] > 0:
        _state["bbox_targets"][_state["fg_inds"], :] = _state["box_encoding_fn"](
            _state["gt_boxes"][_state["anchor_to_gt_argmax"][_state["fg_inds"]], :],
            _state["anchors"][_state["fg_inds"], :],
        )
    _state["bbox_outside_weights"] = np.zeros(
        (_state["num_inside"],), dtype=_state["all_anchors"].dtype
    )
    if _state["norm_by_num_examples"]:
        _state["num_examples"] = np.sum(_state["labels"] >= 0)
        _state["num_examples"] = np.maximum(1.0, _state["num_examples"])
        _state["bbox_outside_weights"][_state["labels"] > 0] = (
            1.0 / _state["num_examples"]
        )
    else:
        _state["bbox_outside_weights"][_state["labels"] > 0] = 1.0


def _module_create_target_np_134_part_4(_state):
    if _state["inds_inside"] is not None:
        _state["labels"] = unmap(
            _state["labels"], _state["total_anchors"], _state["inds_inside"], fill=-1
        )
        _state["bbox_targets"] = unmap(
            _state["bbox_targets"],
            _state["total_anchors"],
            _state["inds_inside"],
            fill=0,
        )
        _state["bbox_outside_weights"] = unmap(
            _state["bbox_outside_weights"],
            _state["total_anchors"],
            _state["inds_inside"],
            fill=0,
        )
    _state["ret"] = {
        "labels": _state["labels"],
        "bbox_targets": _state["bbox_targets"],
        "bbox_outside_weights": _state["bbox_outside_weights"],
        "assigned_anchors_overlap": _state["fg_max_overlap"],
        "positive_gt_id": _state["gt_pos_ids"],
    }
    if _state["inds_inside"] is not None:
        _state["ret"]["assigned_anchors_inds"] = _state["inds_inside"][
            _state["fg_inds"]
        ]
    else:
        _state["ret"]["assigned_anchors_inds"] = _state["fg_inds"]


def create_target_np(*call_args, **call_kwargs):
    """Modified from FAIR detectron.
    Args:
        all_anchors: [num_of_anchors, box_ndim] float tensor.
        gt_boxes: [num_gt_boxes, box_ndim] float tensor.
        similarity_fn: a function, accept anchors and gt_boxes, return
            similarity matrix(such as IoU).
        box_encoding_fn: a function, accept gt_boxes and anchors, return
            box encodings(offsets).
        prune_anchor_fn: a function, accept anchors, return indices that
            indicate valid anchors.
        gt_classes: [num_gt_boxes] int tensor. indicate gt classes, must
            start with 1.
        matched_threshold: float, iou greater than matched_threshold will
            be treated as positives.
        unmatched_threshold: float, iou smaller than unmatched_threshold will
            be treated as negatives.
        bbox_inside_weight: unused
        positive_fraction: [0-1] float or None. if not None, we will try to
            keep ratio of pos/neg equal to positive_fraction when sample.
            if there is not enough positives, it fills the rest with negatives
        rpn_batch_size: int. sample size
        norm_by_num_examples: bool. norm box_weight by number of examples, but
            I recommend to do this outside.
    Returns:
        labels, bbox_targets, bbox_outside_weights
    """
    _state = locals()
    _module_create_target_np_134_part_1(_state)
    _module_create_target_np_134_part_2(_state)
    _module_create_target_np_134_part_3(_state)
    _module_create_target_np_134_part_4(_state)
    return _state["ret"]
