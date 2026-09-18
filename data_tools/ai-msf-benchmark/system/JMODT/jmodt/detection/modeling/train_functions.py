# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from jmodt.config import cfg
from jmodt.utils import loss_utils


from shared.training_loss import GET_RPN_LOSS_SCHEMA as _GET_RPN_LOSS_PARAMETERS_SCHEMA
from shared.training_loss import ModelReturn as _ModelReturn
from shared.training_loss import RpnLossResult, rpn_classification_loss
from shared.training_loss import (
    combine_rpn_regression,
    initialize_rpn_loss_state,
    rpn_regression_options,
)


LOGGER = logging.getLogger(__name__)


def _jmodt_model_input(data):
    rpn_labels = (None, None)
    if cfg.RPN.ENABLED:
        input_data = {
            "pts_input": torch.from_numpy(data["pts_input"]).cuda(non_blocking=True).float(),
            "gt_boxes3d": torch.from_numpy(data["gt_boxes3d"]).cuda(non_blocking=True).float(),
        }
        if not cfg.RPN.FIXED:
            rpn_labels = (
                torch.from_numpy(data["rpn_cls_label"]).cuda(non_blocking=True).long(),
                torch.from_numpy(data["rpn_reg_label"]).cuda(non_blocking=True).float(),
            )
    else:
        input_data = {
            key: torch.from_numpy(value).contiguous().cuda(non_blocking=True).float()
            for key, value in data.items() if key != "sample_id"
        }
    if cfg.LI_FUSION.ENABLED:
        input_data["img"] = (
            torch.from_numpy(data["img"]).cuda(non_blocking=True).float().permute((0, 3, 1, 2))
        )
        input_data["pts_xy"] = torch.from_numpy(data["pts_xy"]).cuda(non_blocking=True).float()
    if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
        input_data["pts_rgb"] = torch.from_numpy(data["rgb"]).cuda(non_blocking=True).float()
    if cfg.REID.ENABLED:
        input_data["gt_tids"] = (
            torch.from_numpy(data["gt_tids"]).cuda(non_blocking=True).float()
        )
    return input_data, rpn_labels


def _jmodt_training_losses(model, predictions, rpn_labels, mean_size):
    metrics, display, loss = {}, {}, 0
    if cfg.RPN.ENABLED and not cfg.RPN.FIXED:
        rpn_result = get_rpn_loss(
            mean_size, model, predictions["rpn_cls"], predictions["rpn_reg"],
            rpn_labels[0], rpn_labels[1], metrics,
        )
        rpn_loss = rpn_result.value_0 * cfg.TRAIN.RPN_TRAIN_WEIGHT
        loss += rpn_loss
        display["rpn_loss"] = rpn_loss.item()
    if cfg.RCNN.ENABLED:
        rcnn_result = get_rcnn_loss(model, predictions, metrics, mean_size)
        if cfg.USE_IOU_BRANCH:
            rcnn_loss, iou_loss, branch_loss = rcnn_result
            display["rcnn_iou_loss"] = iou_loss.item()
            display["iou_branch_loss"] = branch_loss.item()
        else:
            rcnn_loss = rcnn_result
        maximum_iou = predictions["max_iou"].tolist()
        display["max_iou"] = (
            [round(value, 2) for value in maximum_iou]
            if isinstance(maximum_iou, list) else maximum_iou
        )
        display["link_pos"] = metrics.get("rcnn_link_pos")
        display["link_neg"] = metrics.get("rcnn_link_neg")
        rcnn_loss *= cfg.TRAIN.RCNN_TRAIN_WEIGHT
        display["rcnn_loss"] = (
            rcnn_loss.item() if isinstance(rcnn_loss, torch.Tensor) else rcnn_loss
        )
        loss += rcnn_loss
    return _ModelReturn(loss, metrics, display)


def _model_fn_train_impl(model, data, mean_size):
    input_data, rpn_labels = _jmodt_model_input(data)
    return _jmodt_training_losses(model, model(input_data), rpn_labels, mean_size)


def _module_get_rpn_loss_159_part_1(_state):
    initialize_rpn_loss_state(_state)
    _state["rpn_cls"] = _state["rpn_cls"].squeeze(-1)
    _state["invalid_mask"] = torch.logical_or(
        torch.isnan(_state["rpn_cls"]), torch.isinf(_state["rpn_cls"])
    )
    if _state["invalid_mask"].sum() > 0:
        _state["valid_mask"] = torch.logical_not(_state["invalid_mask"])
        _state["rpn_cls"] = _state["rpn_cls"][_state["valid_mask"]]
        _state["rpn_cls_label"] = _state["rpn_cls_label"][_state["valid_mask"]]
    _state["invalid_mask"] = torch.logical_or(
        torch.isnan(_state["rpn_reg"]), torch.isinf(_state["rpn_reg"])
    )
    if _state["invalid_mask"].sum() > 0:
        _state["invalid_mask"] = _state["invalid_mask"].sum(-1) > 0
        _state["valid_mask"] = torch.logical_not(_state["invalid_mask"])
        _state["rpn_reg"] = _state["rpn_reg"][_state["valid_mask"]]
        _state["rpn_reg_label"] = _state["rpn_reg_label"][_state["valid_mask"]]
    else:
        _state["point_num"] = _state["rpn_reg"].size(0) * _state["rpn_reg"].size(1)
        _state["rpn_reg"] = _state["rpn_reg"].view(_state["point_num"], -1)
        _state["rpn_reg_label"] = _state["rpn_reg_label"].view(
            _state["point_num"], 7
        )
    _state["rpn_cls_flat"] = _state["rpn_cls"].view(-1)
    _state["rpn_cls_label_flat"] = _state["rpn_cls_label"].view(-1)
    _state["fg_mask"] = _state["rpn_cls_label_flat"] > 0


def _module_get_rpn_loss_159_part_2(_state):
    _state["rpn_loss_cls"] = rpn_classification_loss(
        cfg,
        _state["rpn_cls_loss_func"],
        _state["rpn_cls_flat"],
        _state["rpn_cls_label_flat"],
        _state["fg_mask"],
        _state["tb_dict"],
    )


def _module_get_rpn_loss_159_part_3(_state):
    if _state["fg_mask"].sum() > 0:
        (
            _state["loss_loc"],
            _state["loss_angle"],
            _state["loss_size"],
            _state["loss_iou"],
            _state["reg_loss_dict"],
        ) = loss_utils.get_reg_loss(
            cls_score=torch.sigmoid(_state["rpn_cls_flat"])[_state["fg_mask"]],
            pred_reg=_state["rpn_reg"][_state["fg_mask"]],
            reg_label=_state["rpn_reg_label"][_state["fg_mask"]],
            **rpn_regression_options(cfg, _state["mean_size"]),
        )
    combine_rpn_regression(_state, cfg, _state["fg_mask"].sum() > 0)
    _state["tb_dict"].update(
        {
            "rpn_loss_cls": _state["rpn_loss_cls"].item(),
            "rpn_loss_reg": _state["rpn_loss_reg"].item(),
            "rpn_loss": _state["rpn_loss"].item(),
            "rpn_fg_sum": _state["fg_mask"].sum().item(),
        }
    )


def get_rpn_loss(mean_size, *call_args, **call_kwargs):
    _state = locals()
    _module_get_rpn_loss_159_part_1(_state)
    _module_get_rpn_loss_159_part_2(_state)
    _module_get_rpn_loss_159_part_3(_state)

    return RpnLossResult(
        _state["rpn_loss"],
        _state["rpn_loss_cls"],
        _state["loss_loc"],
        _state["loss_angle"],
        _state["loss_size"],
        _state["loss_iou"],
    )


def _prepare_rcnn_loss_data(ret_dict):
    data = {
        "classes": ret_dict["rcnn_cls"],
        "regression": ret_dict["rcnn_reg"],
        "labels": ret_dict["cls_label"].float().view(-1),
        "valid_regression": ret_dict["reg_valid_mask"],
        "roi_size": ret_dict["roi_boxes3d"][:, 3:6],
        "ground_truth": ret_dict["gt_of_rois"],
    }
    data["flat_classes"] = data["classes"].view(-1)
    invalid = torch.isnan(data["flat_classes"]) | torch.isinf(data["flat_classes"])
    if invalid.any():
        valid = ~invalid
        for name in (
            "flat_classes",
            "regression",
            "labels",
            "valid_regression",
            "roi_size",
            "ground_truth",
        ):
            data[name] = data[name][valid]
    return data


def _rcnn_classification_loss(model, data):
    network = (
        model.module.rcnn_net
        if isinstance(model, nn.DataParallel)
        else model.rcnn_net
    )
    loss_function = network.cls_loss_func
    classes, labels = data["flat_classes"], data["labels"]
    if cfg.RCNN.LOSS_CLS == "SigmoidFocalLoss":
        positive = (labels > 0).float()
        weights = positive + (labels == 0).float()
        weights /= torch.clamp(positive.sum(), min=1.0)
        return loss_function(classes, positive, weights).sum()
    if cfg.RCNN.LOSS_CLS == "BinaryCrossEntropy":
        batch_loss = F.binary_cross_entropy_with_logits(
            classes, labels, reduction="none"
        )
        valid = labels >= 0
        loss = batch_loss[valid].sum() / torch.clamp(valid.sum(), min=1.0)
        if torch.isnan(loss).any():
            LOGGER.info("cls loss nan %s", batch_loss[valid].shape)
        return loss
    if cfg.TRAIN.LOSS_CLS == "CrossEntropy":
        reshaped = data["classes"].view(data["classes"].shape[0], -1)
        valid = (labels >= 0).float()
        batch_loss = loss_function(reshaped, labels.long())
        return (batch_loss.mean(dim=1) * valid).sum() / torch.clamp(
            valid.sum(), min=1.0
        )
    raise NotImplementedError


def _rcnn_regression_loss(ret_dict, data, tb_dict, mean_size):
    foreground = data["valid_regression"] > 0
    if foreground.sum() == 0:
        return 0
    iou_prediction = None
    if cfg.USE_IOU_BRANCH:
        iou_prediction = ret_dict["rcnn_iou_branch"][foreground]
    anchor_size = (
        data["roi_size"][foreground] if cfg.RCNN.SIZE_RES_ON_ROI else mean_size
    )
    location, angle, size, iou, details = loss_utils.get_reg_loss(
        cls_score=torch.sigmoid(data["flat_classes"])[foreground],
        pred_reg=data["regression"][foreground],
        reg_label=data["ground_truth"][foreground],
        loc_scope=cfg.RCNN.LOC_SCOPE,
        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
        anchor_size=anchor_size,
        get_xz_fine=True,
        get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
        loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
        get_ry_fine=True,
        use_cls_score=True,
        use_mask_score=True,
        use_iou_branch=cfg.USE_IOU_BRANCH,
        iou_branch_pred=iou_prediction,
    )
    loss = location + angle + 3 * size + cfg.TRAIN.CE_WEIGHT * iou
    if cfg.USE_IOU_BRANCH:
        loss += details["iou_branch_loss"]
    tb_dict.update(details)
    return loss


def _record_reid_counts(labels, prefix, tb_dict):
    tb_dict[f"rcnn_{prefix}_pos"] = torch.sum(labels > 0).item()
    tb_dict[f"rcnn_{prefix}_neg"] = torch.sum(labels == 0).item()


def _rcnn_reid_loss(ret_dict, tb_dict):
    link_labels = ret_dict["gt_links"]
    start_labels = ret_dict["gt_starts"]
    end_labels = ret_dict["gt_ends"]
    _record_reid_counts(link_labels, "link", tb_dict)
    _record_reid_counts(start_labels, "start", tb_dict)
    _record_reid_counts(end_labels, "end", tb_dict)
    link_loss = F.l1_loss(
        ret_dict["rcnn_link"].view(-1), link_labels, reduction="mean"
    )
    if cfg.REID.LOSS_LINK != "L1" or cfg.REID.LOSS_SE != "L1":
        raise NotImplementedError
    start_loss = F.l1_loss(
        torch.sigmoid(ret_dict["rcnn_start"].view(-1)),
        start_labels,
        reduction="mean",
    )
    end_loss = F.l1_loss(
        torch.sigmoid(ret_dict["rcnn_end"].view(-1)), end_labels, reduction="mean"
    )
    loss = 0
    for name, value, weight in (
        ("link", link_loss, cfg.TRAIN.LINK_TRAIN_WEIGHT),
        ("start", start_loss, cfg.TRAIN.SE_TRAIN_WEIGHT),
        ("end", end_loss, cfg.TRAIN.SE_TRAIN_WEIGHT),
    ):
        if not torch.isnan(value):
            loss += value * weight
        if value > 0:
            tb_dict[f"rcnn_loss_{name}_mean"] = value.item()
    return loss


def get_rcnn_loss(model, ret_dict, tb_dict, mean_size):
    data = _prepare_rcnn_loss_data(ret_dict)
    if cfg.TRAIN.FINETUNE:
        rcnn_loss = 0
    else:
        rcnn_loss_cls = _rcnn_classification_loss(model, data)
        rcnn_loss_reg = _rcnn_regression_loss(ret_dict, data, tb_dict, mean_size)
        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict.update(
            {
                "rcnn_loss_cls": rcnn_loss_cls.item(),
                "rcnn_cls_fg": (data["labels"] > 0).sum().item(),
                "rcnn_cls_bg": (data["labels"] == 0).sum().item(),
                "rcnn_loss_reg": (
                    rcnn_loss_reg.item()
                    if isinstance(rcnn_loss_reg, torch.Tensor)
                    else rcnn_loss_reg
                ),
                "rcnn_reg_fg": data["valid_regression"].sum().item(),
            }
        )
    if cfg.REID.ENABLED:
        rcnn_loss += _rcnn_reid_loss(ret_dict, tb_dict)
    if rcnn_loss > 0:
        tb_dict["rcnn_loss"] = rcnn_loss.item()
    return rcnn_loss


def model_joint_fn_decorator():
    mean_size = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    def model_fn_train(model, data):
        return _model_fn_train_impl(model, data, mean_size)

    return model_fn_train
