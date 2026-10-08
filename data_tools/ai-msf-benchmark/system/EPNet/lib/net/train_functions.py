# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from lib.config import cfg
import lib.utils.loss_utils as loss_utils
import torch
import torch.nn as nn
import torch.nn.functional as F


from shared.training_loss import GET_RPN_LOSS_SCHEMA as _GET_RPN_LOSS_PARAMETERS_SCHEMA
from shared.training_loss import ModelReturn as _ModelReturn
from shared.training_loss import RpnLossResult, rpn_classification_loss
from shared.training_loss import (
    combine_rpn_regression,
    initialize_rpn_loss_state,
    rpn_regression_options,
)


def _prepare_epnet_base_input(state):
    data = state["data"]
    if cfg.RPN.ENABLED:
        state["pts_rect"], state["pts_features"] = data["pts_rect"], data["pts_features"]
        state["pts_input"], state["gt_boxes3d"] = data["pts_input"], data["gt_boxes3d"]
        if not cfg.RPN.FIXED:
            state["rpn_cls_label"] = (
                torch.from_numpy(data["rpn_cls_label"]).cuda(non_blocking=True).long()
            )
            state["rpn_reg_label"] = (
                torch.from_numpy(data["rpn_reg_label"]).cuda(non_blocking=True).float()
            )
        state["inputs"] = torch.from_numpy(data["pts_input"]).cuda(non_blocking=True).float()
        state["gt_boxes3d"] = torch.from_numpy(data["gt_boxes3d"]).cuda(non_blocking=True).float()
        state["input_data"] = {
            "pts_input": state["inputs"], "gt_boxes3d": state["gt_boxes3d"],
        }
    else:
        state["input_data"] = {
            key: torch.from_numpy(value).contiguous().cuda(non_blocking=True).float()
            for key, value in data.items() if key != "sample_id"
        }
        if not cfg.RCNN.ROI_SAMPLE_JIT:
            state["pts_input"] = torch.cat(
                (state["input_data"].get("pts_input"), state["input_data"].get("pts_features")), dim=-1
            )
            state["input_data"]["pts_input"] = state["pts_input"]


def _add_epnet_modal_inputs(state):
    data, input_data = state["data"], state["input_data"]
    if cfg.LI_FUSION.ENABLED:
        state["img"] = (
            torch.from_numpy(data["img"]).cuda(non_blocking=True).float().permute((0, 3, 1, 2))
        )
        state["pts_origin_xy"] = torch.from_numpy(data["pts_origin_xy"]).cuda(non_blocking=True).float()
        input_data["img"], input_data["pts_origin_xy"] = state["img"], state["pts_origin_xy"]
    if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
        state["pts_rgb"] = torch.from_numpy(data["rgb"]).cuda(non_blocking=True).float()
        input_data["pts_rgb"] = state["pts_rgb"]


def _module_model_fn_98_part_1(_state):
    _prepare_epnet_base_input(_state)
    _add_epnet_modal_inputs(_state)
    _state["ret_dict"] = _state["model"](_state["input_data"])
    _state["tb_dict"], _state["disp_dict"] = {}, {}


def _module_model_fn_98_part_2(_state):
    _state["loss"] = 0
    if cfg.RPN.ENABLED and (not cfg.RPN.FIXED):
        _state["rpn_cls"], _state["rpn_reg"] = (
            _state["ret_dict"]["rpn_cls"],
            _state["ret_dict"]["rpn_reg"],
        )
        (
            _state["rpn_loss"],
            _state["rpn_loss_cls"],
            _state["rpn_loss_loc"],
            _state["rpn_loss_angle"],
            _state["rpn_loss_size"],
            _state["rpn_loss_iou"],
        ) = get_rpn_loss(
            _state["mean_size"], _state["model"],
            _state["rpn_cls"],
            _state["rpn_reg"],
            _state["rpn_cls_label"],
            _state["rpn_reg_label"],
            _state["tb_dict"],
        )
        _state["rpn_loss"] = _state["rpn_loss"] * cfg.TRAIN.RPN_TRAIN_WEIGHT
        _state["loss"] += _state["rpn_loss"]
        _state["disp_dict"]["rpn_loss"] = _state["rpn_loss"].item()
        _state["disp_dict"]["rpn_loss_cls"] = _state["rpn_loss_cls"].item()
        _state["disp_dict"]["rpn_loss_loc"] = _state["rpn_loss_loc"].item()
        _state["disp_dict"]["rpn_loss_angle"] = _state["rpn_loss_angle"].item()
        _state["disp_dict"]["rpn_loss_size"] = _state["rpn_loss_size"].item()
        _state["disp_dict"]["rpn_loss_iou"] = _state["rpn_loss_iou"].item()


def _module_model_fn_98_part_3(_state):
    if cfg.RCNN.ENABLED:
        if cfg.USE_IOU_BRANCH:
            _state["rcnn_loss"], _state["iou_loss"], _state["iou_branch_loss"] = (
                get_rcnn_loss(
                    _state["model"], _state["ret_dict"], _state["tb_dict"], _state["mean_size"]
                )
            )
            _state["disp_dict"]["reg_fg_sum"] = _state["tb_dict"].get("rcnn_reg_fg")
            _state["rcnn_loss"] = _state["rcnn_loss"] * cfg.TRAIN.RCNN_TRAIN_WEIGHT
            _state["disp_dict"]["rcnn_loss"] = _state["rcnn_loss"].item()
            _state["loss"] += _state["rcnn_loss"]
            _state["disp_dict"]["loss"] = _state["loss"].item()
            _state["disp_dict"]["rcnn_iou_loss"] = _state["iou_loss"].item()
            _state["disp_dict"]["iou_branch_loss"] = _state[
                "iou_branch_loss"
            ].item()
        else:
            _state["rcnn_loss"] = get_rcnn_loss(
                _state["model"], _state["ret_dict"], _state["tb_dict"], _state["mean_size"]
            )
            _state["disp_dict"]["reg_fg_sum"] = _state["tb_dict"].get("rcnn_reg_fg")
            _state["rcnn_loss"] = _state["rcnn_loss"] * cfg.TRAIN.RCNN_TRAIN_WEIGHT
            _state["disp_dict"]["rcnn_loss"] = _state["rcnn_loss"].item()
            _state["loss"] += _state["rcnn_loss"]
            _state["disp_dict"]["loss"] = _state["loss"].item()


def _model_fn_impl(model, data, mean_size):
    _state = locals()
    _module_model_fn_98_part_1(_state)
    _module_model_fn_98_part_2(_state)
    _module_model_fn_98_part_3(_state)

    return _ModelReturn(_state["loss"], _state["tb_dict"], _state["disp_dict"])


def _module_get_rpn_loss_170_part_1(_state):
    initialize_rpn_loss_state(_state)
    _state["rpn_cls_label_flat"] = _state["rpn_cls_label"].view(-1)
    _state["rpn_cls_flat"] = _state["rpn_cls"].view(-1)
    _state["fg_mask"] = _state["rpn_cls_label_flat"] > 0


def _module_get_rpn_loss_170_part_2(_state):
    _state["rpn_loss_cls"] = rpn_classification_loss(
        cfg,
        _state["rpn_cls_loss_func"],
        _state["rpn_cls_flat"],
        _state["rpn_cls_label_flat"],
        _state["fg_mask"],
        _state["tb_dict"],
        dice_predictions=_state["rpn_cls"],
    )
    _state["point_num"] = _state["rpn_reg"].size(0) * _state["rpn_reg"].size(1)
    _state["fg_sum"] = _state["fg_mask"].long().sum().item()


def _module_get_rpn_loss_170_part_3(_state):
    if _state["fg_sum"] != 0:
        (
            _state["loss_loc"],
            _state["loss_angle"],
            _state["loss_size"],
            _state["loss_iou"],
            _state["reg_loss_dict"],
        ) = loss_utils.get_reg_loss(
            torch.sigmoid(_state["rpn_cls_flat"])[_state["fg_mask"]],
            torch.sigmoid(_state["rpn_cls_flat"])[_state["fg_mask"]],
            _state["rpn_reg"].view(_state["point_num"], -1)[_state["fg_mask"]],
            _state["rpn_reg_label"].view(_state["point_num"], 7)[_state["fg_mask"]],
            **rpn_regression_options(cfg, _state["mean_size"]),
        )
    combine_rpn_regression(_state, cfg, _state["fg_sum"] != 0)


def _module_get_rpn_loss_170_part_4(_state):
    _state["tb_dict"].update(
        {
            "rpn_loss_cls": _state["rpn_loss_cls"].item(),
            "rpn_loss_reg": _state["rpn_loss_reg"].item(),
            "rpn_loss": _state["rpn_loss"].item(),
            "rpn_fg_sum": _state["fg_sum"],
            "rpn_loss_loc": _state["loss_loc"].item(),
            "rpn_loss_angle": _state["loss_angle"].item(),
            "rpn_loss_size": _state["loss_size"].item(),
            "rpn_loss_iou": _state["loss_iou"].item(),
        }
    )


def get_rpn_loss(mean_size, *call_args, **call_kwargs):
    _state = locals()
    _module_get_rpn_loss_170_part_1(_state)
    _module_get_rpn_loss_170_part_2(_state)
    _module_get_rpn_loss_170_part_3(_state)
    _module_get_rpn_loss_170_part_4(_state)

    return RpnLossResult(
        _state["rpn_loss"],
        _state["rpn_loss_cls"],
        _state["loss_loc"],
        _state["loss_angle"],
        _state["loss_size"],
        _state["loss_iou"],
    )


def _epnet_rcnn_classification_loss(model, ret_dict, tb_dict):
    predictions = ret_dict["rcnn_cls"]
    labels = ret_dict["cls_label"].float()
    flat_labels = labels.view(-1)
    loss_function = (
        model.module.rcnn_net.cls_loss_func
        if isinstance(model, nn.DataParallel) else model.rcnn_net.cls_loss_func
    )
    flat_predictions = predictions.view(-1)
    if cfg.RCNN.LOSS_CLS == "SigmoidFocalLoss":
        positive, negative = (flat_labels > 0).float(), (flat_labels == 0).float()
        weights = (positive + negative) / torch.clamp(positive.sum(), min=1.0)
        losses = loss_function(flat_predictions, positive, weights)
        tb_dict["rpn_loss_cls_pos"] = (losses * positive).sum().item()
        tb_dict["rpn_loss_cls_neg"] = (losses * negative).sum().item()
        return losses.sum(), flat_predictions, labels
    if cfg.RCNN.LOSS_CLS == "BinaryCrossEntropy":
        losses = F.binary_cross_entropy(
            torch.sigmoid(flat_predictions), labels, reduction="none"
        )
        valid = (flat_labels >= 0).float()
        return (losses * valid).sum() / torch.clamp(valid.sum(), min=1.0), flat_predictions, labels
    if cfg.TRAIN.LOSS_CLS == "CrossEntropy":
        reshaped = predictions.view(predictions.shape[0], -1)
        valid = (flat_labels >= 0).float()
        losses = loss_function(reshaped, flat_labels.long())
        loss = (losses.mean(dim=1) * valid).sum() / torch.clamp(valid.sum(), min=1.0)
        return loss, flat_predictions, labels
    raise NotImplementedError


def _epnet_rcnn_regression_loss(ret_dict, cls_predictions, mean_size, class_loss, tb_dict):
    valid_mask = ret_dict["reg_valid_mask"] > 0
    foreground_count = valid_mask.long().sum().item()
    if foreground_count == 0:
        zero = class_loss * 0
        return zero, {"loc": zero, "angle": zero, "size": zero,
                      "iou": zero, "branch": zero}
    batch_size = ret_dict["pts_input"].shape[0]
    branch_prediction = (
        ret_dict["rcnn_iou_branch"][valid_mask] if cfg.USE_IOU_BRANCH else None
    )
    roi_size = ret_dict["roi_boxes3d"][:, 3:6]
    anchor_size = roi_size[valid_mask] if cfg.RCNN.SIZE_RES_ON_ROI else mean_size
    loc, angle, size, iou, details = loss_utils.get_reg_loss(
        torch.sigmoid(cls_predictions)[valid_mask], ret_dict["mask_score"][valid_mask],
        ret_dict["rcnn_reg"].view(batch_size, -1)[valid_mask],
        ret_dict["gt_of_rois"].view(batch_size, 7)[valid_mask],
        loc_scope=cfg.RCNN.LOC_SCOPE, loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
        num_head_bin=cfg.RCNN.NUM_HEAD_BIN, anchor_size=anchor_size,
        get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
        get_ry_fine=True, use_cls_score=True, use_mask_score=True,
        gt_iou_weight=ret_dict["gt_iou"][valid_mask],
        use_iou_branch=cfg.USE_IOU_BRANCH, iou_branch_pred=branch_prediction,
    )
    size, iou = 3 * size, cfg.TRAIN.CE_WEIGHT * iou
    branch = details["iou_branch_loss"] if cfg.USE_IOU_BRANCH else class_loss * 0
    regression = loc + angle + size + iou + branch
    tb_dict.update(details)
    return regression, {"loc": loc, "angle": angle, "size": size,
                        "iou": iou, "branch": branch}


def _record_epnet_rcnn_loss(tb_dict, losses, labels, valid_mask):
    total, classification, regression, parts = losses
    tb_dict.update({
        "rcnn_loss_cls": classification.item(), "rcnn_loss_reg": regression.item(),
        "rcnn_loss": total.item(), "rcnn_loss_loc": parts["loc"].item(),
        "rcnn_loss_angle": parts["angle"].item(),
        "rcnn_loss_size": parts["size"].item(), "rcnn_loss_iou": parts["iou"].item(),
        "rcnn_cls_fg": (labels > 0).sum().item(),
        "rcnn_cls_bg": (labels == 0).sum().item(),
        "rcnn_reg_fg": valid_mask.sum().item(),
    })


def get_rcnn_loss(model, ret_dict, tb_dict, mean_size):
    classification, cls_predictions, labels = _epnet_rcnn_classification_loss(
        model, ret_dict, tb_dict
    )
    regression, parts = _epnet_rcnn_regression_loss(
        ret_dict, cls_predictions, mean_size, classification, tb_dict
    )
    total = classification + regression
    _record_epnet_rcnn_loss(
        tb_dict, (total, classification, regression, parts),
        labels, ret_dict["reg_valid_mask"],
    )
    if cfg.USE_IOU_BRANCH:
        tb_dict["iou_branch_loss"] = parts["branch"].item()
        return total, parts["iou"], parts["branch"]
    return total


def model_joint_fn_decorator():
    mean_size = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    def model_fn(model, data):
        return _model_fn_impl(model, data, mean_size)

    return model_fn
