# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""JMODT adapters for the shared EPNet loss implementation."""

__all__ = ["DiceLoss", "SigmoidFocalClassificationLoss", "get_reg_loss"]

from collections import namedtuple
from pathlib import Path

from shared.compatibility import import_from_root
from jmodt.config import cfg
from shared.parameters import ParameterSchema, bind_parameters


_FIELDS = (
    "cls_score", "pred_reg", "reg_label", "loc_scope", "loc_bin_size",
    "num_head_bin", "anchor_size", "get_xz_fine", "get_y_by_bin",
    "loc_y_scope", "loc_y_bin_size", "get_ry_fine", "use_cls_score",
    "use_mask_score", "use_iou_branch", "iou_branch_pred",
)
GetRegLossParameters = namedtuple("GetRegLossParameters", _FIELDS)


_SCHEMA = ParameterSchema(
    GetRegLossParameters,
    _FIELDS,
    {
        "get_xz_fine": True,
        "get_y_by_bin": False,
        "loc_y_scope": 0.5,
        "loc_y_bin_size": 0.25,
        "get_ry_fine": False,
        "use_cls_score": False,
        "use_mask_score": False,
        "use_iou_branch": False,
        "iou_branch_pred": None,
    },
    frozenset(),
    16,
    None,
    None,
)
_MODULE = import_from_root("lib.utils.loss_utils", Path(__file__).parents[3] / "EPNet")
_MODULE.cfg = cfg

DiceLoss = _MODULE.DiceLoss
SigmoidFocalClassificationLoss = _MODULE.SigmoidFocalClassificationLoss


def get_reg_loss(*call_args, **call_kwargs):
    """Adapt JMODT's historical call signature to the shared implementation."""
    parameters = bind_parameters(_SCHEMA, call_args, call_kwargs)
    return _MODULE.get_reg_loss(
        parameters.cls_score,
        parameters.cls_score,
        parameters.pred_reg,
        parameters.reg_label,
        loc_scope=parameters.loc_scope,
        loc_bin_size=parameters.loc_bin_size,
        num_head_bin=parameters.num_head_bin,
        anchor_size=parameters.anchor_size,
        get_xz_fine=parameters.get_xz_fine,
        get_y_by_bin=parameters.get_y_by_bin,
        loc_y_scope=parameters.loc_y_scope,
        loc_y_bin_size=parameters.loc_y_bin_size,
        get_ry_fine=parameters.get_ry_fine,
        use_cls_score=parameters.use_cls_score,
        use_mask_score=parameters.use_mask_score,
        gt_iou_weight=None,
        use_iou_branch=parameters.use_iou_branch,
        iou_branch_pred=parameters.iou_branch_pred,
    )
