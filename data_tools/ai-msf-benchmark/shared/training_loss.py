# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Classification-loss helpers for compatible trainers."""

from collections import namedtuple
from typing import NamedTuple

import torch
import torch.nn.functional as F

from shared.parameters import ParameterSchema
from shared.parameters import bind_parameters
from shared.parameters import bind_named as _bind_named


class GetRpnLossParameters(NamedTuple):
    model: object
    rpn_cls: object
    rpn_reg: object
    rpn_cls_label: object
    rpn_reg_label: object
    tb_dict: object


class RpnLossResult(NamedTuple):
    value_0: object
    value_1: object
    value_2: object
    value_3: object
    value_4: object
    value_5: object


GET_RPN_LOSS_SCHEMA = ParameterSchema(
    GetRpnLossParameters,
    ("model", "rpn_cls", "rpn_reg", "rpn_cls_label", "rpn_reg_label", "tb_dict"),
    {},
    frozenset(),
    6,
    None,
    None,
)
ModelReturn = namedtuple("ModelReturn", ["loss", "tb_dict", "disp_dict"])


def initialize_rpn_loss_state(state):
    """Bind RPN loss inputs and select the wrapped model's loss function."""
    parameters = bind_parameters(
        GET_RPN_LOSS_SCHEMA, state["call_args"], state["call_kwargs"]
    )
    state.update(parameters._asdict())
    model = state["model"]
    rpn = model.module.rpn if isinstance(model, torch.nn.DataParallel) else model.rpn
    state["rpn_cls_loss_func"] = rpn.rpn_cls_loss_func


def rpn_classification_loss(*call_args, **call_kwargs):
    """Compute the configured RPN classification loss."""
    (
        config,
        loss_function,
        flat_predictions,
        flat_labels,
        foreground_mask,
        metrics,
        dice_predictions,
    ) = _bind_named(
        call_args, call_kwargs,
        (
            'config',
            'loss_function',
            'flat_predictions',
            'flat_labels',
            'foreground_mask',
            'metrics',
            'dice_predictions',
        ),
        {'dice_predictions': None},
    )
    if config.RPN.LOSS_CLS == "DiceLoss":
        predictions = flat_predictions if dice_predictions is None else dice_predictions
        return loss_function(predictions, flat_labels)
    if config.RPN.LOSS_CLS == "SigmoidFocalLoss":
        target = (flat_labels > 0).float()
        positive = target
        negative = (flat_labels == 0).float()
        weights = (positive + negative) / torch.clamp(positive.sum(), min=1.0)
        losses = loss_function(flat_predictions, target, weights)
        metrics["rpn_loss_cls_pos"] = (losses * positive).sum().item()
        metrics["rpn_loss_cls_neg"] = (losses * negative).sum().item()
        return losses.sum()
    if config.RPN.LOSS_CLS == "BinaryCrossEntropy":
        weights = flat_predictions.new(flat_predictions.shape[0]).fill_(1.0)
        weights[foreground_mask] = config.RPN.FG_WEIGHT
        target = (flat_labels > 0).float()
        losses = F.binary_cross_entropy(
            torch.sigmoid(flat_predictions),
            target,
            weight=weights,
            reduction="none",
        )
        valid = (flat_labels >= 0).float()
        return (losses * valid).sum() / torch.clamp(valid.sum(), min=1.0)
    raise NotImplementedError


def rpn_regression_options(config, mean_size):
    return {
        "loc_scope": config.RPN.LOC_SCOPE,
        "loc_bin_size": config.RPN.LOC_BIN_SIZE,
        "num_head_bin": config.RPN.NUM_HEAD_BIN,
        "anchor_size": mean_size,
        "get_xz_fine": config.RPN.LOC_XZ_FINE,
        "use_cls_score": True,
        "use_mask_score": False,
    }


def combine_rpn_regression(state, config, has_foreground):
    """Apply common regression weights and combine classification/regression."""
    if has_foreground:
        state["loss_size"] *= 3
        state["loss_iou"] *= config.TRAIN.CE_WEIGHT
        state["rpn_loss_reg"] = sum(
            state[key]
            for key in ("loss_loc", "loss_angle", "loss_size", "loss_iou")
        )
    else:
        zero = state["rpn_loss_cls"] * 0
        for key in (
            "loss_loc",
            "loss_angle",
            "loss_size",
            "loss_iou",
            "rpn_loss_reg",
        ):
            state[key] = zero
    state["rpn_loss"] = (
        state["rpn_loss_cls"] * config.RPN.LOSS_WEIGHT[0]
        + state["rpn_loss_reg"] * config.RPN.LOSS_WEIGHT[1]
    )
