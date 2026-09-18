# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from lib.config import cfg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class GetRegLossParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    cls_score: object
    mask_score: object
    pred_reg: object
    reg_label: object
    loc_scope: object
    loc_bin_size: object
    num_head_bin: object
    anchor_size: object
    get_xz_fine: object
    get_y_by_bin: object
    loc_y_scope: object
    loc_y_bin_size: object
    get_ry_fine: object
    use_cls_score: object
    use_mask_score: object
    gt_iou_weight: object
    use_iou_branch: object
    iou_branch_pred: object


class DiceLoss(nn.Module):
    def __init__(self, ignore_target=-1):
        super().__init__()
        self.ignore_target = ignore_target

    def forward(self, input_tensor, target):
        """
        :param input_tensor: (N), logit
        :param target: (N), {0, 1}
        :return:
        """
        input_tensor = torch.sigmoid(input_tensor.view(-1))
        target = target.float().view(-1)
        mask = (target != self.ignore_target).float()
        return 1.0 - (torch.min(input_tensor, target) * mask).sum() / torch.clamp(
            (torch.max(input_tensor, target) * mask).sum(), min=1.0
        )


class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid focal cross entropy loss.
    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor
        )
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = (target_tensor * prediction_probabilities) + (
            (1 - target_tensor) * (1 - prediction_probabilities)
        )
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target_tensor * self._alpha + (1 - target_tensor) * (
                1 - self._alpha
            )

        focal_cross_entropy_loss = (
            modulating_factor * alpha_weight_factor * per_entry_cross_ent
        )
        return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss


_GET_REG_LOSS_PARAMETERS_SCHEMA = _ParameterSchema(
    GetRegLossParameters,
    (
        "cls_score",
        "mask_score",
        "pred_reg",
        "reg_label",
        "loc_scope",
        "loc_bin_size",
        "num_head_bin",
        "anchor_size",
        "get_xz_fine",
        "get_y_by_bin",
        "loc_y_scope",
        "loc_y_bin_size",
        "get_ry_fine",
        "use_cls_score",
        "use_mask_score",
        "gt_iou_weight",
        "use_iou_branch",
        "iou_branch_pred",
    ),
    {
        "get_xz_fine": True,
        "get_y_by_bin": False,
        "loc_y_scope": 0.5,
        "loc_y_bin_size": 0.25,
        "get_ry_fine": False,
        "use_cls_score": False,
        "use_mask_score": False,
        "gt_iou_weight": None,
        "use_iou_branch": False,
        "iou_branch_pred": None,
    },
    frozenset(()),
    18,
    None,
    None,
)


def _regression_one_hot(labels, count):
    values = torch.cuda.FloatTensor(labels.size(0), count).zero_()
    values.scatter_(1, labels.view(-1, 1).long(), 1)
    return values


def _xz_localization_loss(parameters, losses):
    per_bin = int(parameters.loc_scope / parameters.loc_bin_size) * 2
    x_label, y_label, z_label = (
        parameters.reg_label[:, 0], parameters.reg_label[:, 1], parameters.reg_label[:, 2]
    )
    x_shift = torch.clamp(x_label + parameters.loc_scope, 0, parameters.loc_scope * 2 - 1e-3)
    z_shift = torch.clamp(z_label + parameters.loc_scope, 0, parameters.loc_scope * 2 - 1e-3)
    x_bin, z_bin = (x_shift / parameters.loc_bin_size).floor().long(), (
        z_shift / parameters.loc_bin_size
    ).floor().long()
    x_range, z_range = (0, per_bin), (per_bin, per_bin * 2)
    x_bin_loss = F.cross_entropy(parameters.pred_reg[:, slice(*x_range)], x_bin)
    z_bin_loss = F.cross_entropy(parameters.pred_reg[:, slice(*z_range)], z_bin)
    losses.update({"loss_x_bin": x_bin_loss.item(), "loss_z_bin": z_bin_loss.item()})
    result = {"per_bin": per_bin, "x_bin": x_bin, "z_bin": z_bin,
              "y_label": y_label, "offset": z_range[1]}
    location_loss = x_bin_loss + z_bin_loss
    if parameters.get_xz_fine:
        x_res_range, z_res_range = (per_bin * 2, per_bin * 3), (per_bin * 3, per_bin * 4)
        x_res = x_shift - (x_bin.float() * parameters.loc_bin_size + parameters.loc_bin_size / 2)
        z_res = z_shift - (z_bin.float() * parameters.loc_bin_size + parameters.loc_bin_size / 2)
        x_one_hot, z_one_hot = _regression_one_hot(x_bin, per_bin), _regression_one_hot(z_bin, per_bin)
        x_res_loss = F.smooth_l1_loss(
            (parameters.pred_reg[:, slice(*x_res_range)] * x_one_hot).sum(1),
            x_res / parameters.loc_bin_size,
        )
        z_res_loss = F.smooth_l1_loss(
            (parameters.pred_reg[:, slice(*z_res_range)] * z_one_hot).sum(1),
            z_res / parameters.loc_bin_size,
        )
        losses.update({"loss_x_res": x_res_loss.item(), "loss_z_res": z_res_loss.item()})
        location_loss += x_res_loss + z_res_loss
        result.update({"x_res_range": x_res_range, "z_res_range": z_res_range,
                       "x_res": x_res, "z_res": z_res,
                       "x_one_hot": x_one_hot, "z_one_hot": z_one_hot,
                       "offset": z_res_range[1]})
    return location_loss, result


def _y_localization_loss(parameters, offset, y_label, losses):
    if parameters.get_y_by_bin:
        bin_count = int(parameters.loc_y_scope / parameters.loc_y_bin_size) * 2
        bin_range, res_range = (offset, offset + bin_count), (offset + bin_count, offset + bin_count * 2)
        shifted = torch.clamp(y_label + parameters.loc_y_scope, 0, parameters.loc_y_scope * 2 - 1e-3)
        bin_label = (shifted / parameters.loc_y_bin_size).floor().long()
        residual = shifted - (bin_label.float() * parameters.loc_y_bin_size + parameters.loc_y_bin_size / 2)
        one_hot = _regression_one_hot(bin_label, bin_count)
        bin_loss = F.cross_entropy(parameters.pred_reg[:, slice(*bin_range)], bin_label)
        res_loss = F.smooth_l1_loss(
            (parameters.pred_reg[:, slice(*res_range)] * one_hot).sum(1),
            residual / parameters.loc_y_bin_size,
        )
        losses.update({"loss_y_bin": bin_loss.item(), "loss_y_res": res_loss.item()})
        return bin_loss + res_loss, res_range[1], {}
    y_range = (offset, offset + 1)
    loss = F.smooth_l1_loss(parameters.pred_reg[:, slice(*y_range)].sum(1), y_label)
    losses["loss_y_offset"] = loss.item()
    return loss, y_range[1], {"y_range": y_range, "y_label": y_label}


def _rotation_regression_loss(parameters, offset, losses):
    bin_range = (offset, offset + parameters.num_head_bin)
    res_range = (bin_range[1], bin_range[1] + parameters.num_head_bin)
    angle = parameters.reg_label[:, 6]
    if parameters.get_ry_fine:
        angle_per_class = (np.pi / 2) / parameters.num_head_bin
        angle = angle % (2 * np.pi)
        opposite = (angle > np.pi * 0.5) & (angle < np.pi * 1.5)
        angle[opposite] = (angle[opposite] + np.pi) % (2 * np.pi)
        shifted = torch.clamp(
            (angle + np.pi * 0.5) % (2 * np.pi) - np.pi * 0.25,
            min=1e-3, max=np.pi * 0.5 - 1e-3,
        )
    else:
        angle_per_class = (2 * np.pi) / parameters.num_head_bin
        shifted = (angle % (2 * np.pi) + angle_per_class / 2) % (2 * np.pi)
    bin_label = (shifted / angle_per_class).floor().long()
    residual = shifted - (bin_label.float() * angle_per_class + angle_per_class / 2)
    one_hot = _regression_one_hot(bin_label, parameters.num_head_bin)
    bin_loss = F.cross_entropy(parameters.pred_reg[:, slice(*bin_range)], bin_label)
    res_loss = F.smooth_l1_loss(
        (parameters.pred_reg[:, slice(*res_range)] * one_hot).sum(1),
        residual / (angle_per_class / 2),
    )
    losses.update({"loss_ry_bin": bin_loss.item(), "loss_ry_res": res_loss.item()})
    return bin_loss + res_loss, res_range[1]


def _axis_aligned_iou(predicted, target):
    pred_x, pred_y, pred_z, pred_size = predicted
    target_x, target_y, target_z, target_size = target
    epsilon = pred_x.new().resize_(pred_x.shape).fill_(1e-3)
    intersections = []
    for pred_axis, target_axis, pred_width, target_width in (
        (pred_x, target_x, pred_size[:, 2], target_size[:, 2]),
        (pred_y, target_y, pred_size[:, 0], target_size[:, 0]),
        (pred_z, target_z, pred_size[:, 1], target_size[:, 1]),
    ):
        overlap = torch.min(pred_axis + pred_width / 2, target_axis + target_width / 2)
        overlap -= torch.max(pred_axis - pred_width / 2, target_axis - target_width / 2)
        intersections.append(torch.max(overlap, epsilon))
    intersection = intersections[0] * intersections[1] * intersections[2]
    pred_area = torch.max(pred_size.prod(dim=1), epsilon)
    target_area = target_size.prod(dim=1)
    return intersection / (pred_area + target_area - intersection)


def _finish_iou_loss(parameters, iou, losses):
    if parameters.use_iou_branch:
        prediction = torch.clamp(parameters.iou_branch_pred.view(-1), 0.0001, 0.9999)
        target = torch.clamp(iou, 0.0001, 0.9999).detach()
        branch_loss = -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))
        losses["iou_branch_loss"] = branch_loss.mean()
    if parameters.use_cls_score:
        iou = parameters.cls_score * iou
    iou = torch.max(iou, iou.new().resize_(iou.shape).fill_(1e-4))
    return -torch.log(iou).mean()


def _box_values(parameters, xz_values, y_values, size_residual):
    bin_size = parameters.loc_bin_size
    if cfg.TRAIN.IOU_LOSS_TYPE == "raw":
        pred_x = (parameters.pred_reg[:, slice(*xz_values["x_res_range"])] * xz_values["x_one_hot"]).sum(1) * bin_size
        pred_z = (parameters.pred_reg[:, slice(*xz_values["z_res_range"])] * xz_values["z_one_hot"]).sum(1) * bin_size
        target_x, target_z = xz_values["x_res"], xz_values["z_res"]
    else:
        centers = torch.arange(xz_values["per_bin"]).float() * bin_size + bin_size / 2 - parameters.loc_scope
        centers = centers.to(parameters.pred_reg.device)
        x_prob = F.softmax(parameters.pred_reg[:, :xz_values["per_bin"]], 1)
        z_prob = F.softmax(parameters.pred_reg[:, xz_values["per_bin"]:xz_values["per_bin"] * 2], 1)
        x_res = parameters.pred_reg[:, slice(*xz_values["x_res_range"])] * bin_size
        z_res = parameters.pred_reg[:, slice(*xz_values["z_res_range"])] * bin_size
        pred_x, pred_z = ((centers + x_res) * x_prob).sum(1), ((centers + z_res) * z_prob).sum(1)
        target_x = centers[xz_values["x_bin"]] + xz_values["x_res"]
        target_z = centers[xz_values["z_bin"]] + xz_values["z_res"]
    pred_y = parameters.pred_reg[:, slice(*y_values["y_range"])].sum(1)
    pred_size = size_residual * parameters.anchor_size + parameters.anchor_size
    target = (target_x, y_values["y_label"], target_z, parameters.reg_label[:, 3:6])
    return (pred_x, pred_y, pred_z, pred_size), target


def get_reg_loss(*call_args, **call_kwargs):
    """Compute bin-based 3D bounding-box regression losses."""
    parameters = _bind_parameters(
        _GET_REG_LOSS_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    losses = {}
    location_loss, xz_values = _xz_localization_loss(parameters, losses)
    y_loss, offset, y_values = _y_localization_loss(
        parameters, xz_values["offset"], xz_values["y_label"], losses
    )
    location_loss += y_loss
    angle_loss, offset = _rotation_regression_loss(parameters, offset, losses)
    size_range = (offset, offset + 3)
    if parameters.pred_reg.shape[1] != size_range[1]:
        raise AssertionError(
            "%d vs %d" % (parameters.pred_reg.shape[1], size_range[1])
        )
    size_target = (
        parameters.reg_label[:, 3:6] - parameters.anchor_size
    ) / parameters.anchor_size
    size_residual = parameters.pred_reg[:, slice(*size_range)]
    size_loss = F.smooth_l1_loss(size_residual, size_target)
    if cfg.TRAIN.IOU_LOSS_TYPE not in ("raw", "cls_mask_with_bin"):
        raise NotImplementedError(cfg.TRAIN.IOU_LOSS_TYPE)
    predicted, target = _box_values(
        parameters, xz_values, y_values, size_residual
    )
    iou_loss = _finish_iou_loss(
        parameters, _axis_aligned_iou(predicted, target), losses
    )
    losses.update({
        "loss_loc": location_loss, "loss_angle": angle_loss,
        "loss_size": size_loss, "loss_iou": iou_loss,
    })
    return location_loss, angle_loss, size_loss, iou_loss, losses
