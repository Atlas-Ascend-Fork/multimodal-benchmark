# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import NamedTuple


from configs.config import cfg
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair
from ops.query_depth_point.query_depth_point import QueryDepthPoint
import numpy as np
import torch
from utils.model_helpers import (
    forward_conv_features,
    initialize_conv_feature_weights,
    initialize_detector_forward,
    initialize_pointnet_detector,
    pooled_point_features,
)
import torch.nn as nn
import torch.nn.functional as F

from datasets.dataset_info import DATASET_INFO
from models.box_transform import (
    size_decode,
    size_encode,
    center_decode,
    center_encode,
    angle_decode,
    angle_encode,
)
from models.common import conv1d, conv2d, de_conv1d, init_params
from models.common import softmax_focal_loss_ignore, get_accuracy
from models.model_util import get_box3d_corners_helper
from models.model_util import huber_loss
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters

_HELPER_CONTINUE = object()
HELPER_CONTINUE = _HELPER_CONTINUE


class PointnetmoduleInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    infea: object
    mlp: object
    dist: object
    nsample: object
    use_xyz: object
    use_feature: object


# single scale PointNet module

_POINTNETMODULE_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    PointnetmoduleInitParameters,
    ("infea", "mlp", "dist", "nsample", "use_xyz", "use_feature"),
    {"use_xyz": True, "use_feature": True},
    frozenset(()),
    6,
    None,
    None,
)


class PointNetModule(nn.Module):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _POINTNETMODULE_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        infea, mlp, dist, nsample, use_xyz, use_feature = bound_parameters
        super(PointNetModule, self).__init__()
        self.dist = dist
        self.nsample = nsample
        self.use_xyz = use_xyz

        if infea > 0:
            use_feature = True
        else:
            use_feature = False

        self.use_feature = use_feature

        self.query_depth_point = QueryDepthPoint(dist, nsample)

        if self.use_xyz:
            self.conv1 = conv2d(infea + 3, mlp[0], 1)
        else:
            self.conv1 = conv2d(infea, mlp[0], 1)

        self.conv2 = conv2d(mlp[0], mlp[1], 1)
        self.conv3 = conv2d(mlp[1], mlp[2], 1)

        init_params([self.conv1[0], self.conv2[0], self.conv3[0]], "kaiming_normal")
        init_params([self.conv1[1], self.conv2[1], self.conv3[1]], 1)

    def forward(self, pc, feat, new_pc=None):
        batch_size = pc.size(0)

        npoint = new_pc.shape[2]
        k = self.nsample

        indices, num = self.query_depth_point(pc, new_pc)  # b*npoint*nsample

        if not (indices.data.max() < pc.shape[2] and indices.data.min() >= 0):
            raise AssertionError
        grouped_pc = None
        grouped_feature = None

        if self.use_xyz:
            grouped_pc = torch.gather(
                pc, 2, indices.view(batch_size, 1, npoint * k).expand(-1, 3, -1)
            ).view(batch_size, 3, npoint, k)

            grouped_pc = grouped_pc - new_pc.unsqueeze(3)

        if self.use_feature:
            grouped_feature = torch.gather(
                feat,
                2,
                indices.view(batch_size, 1, npoint * k).expand(-1, feat.size(1), -1),
            ).view(batch_size, feat.size(1), npoint, k)

        if self.use_feature and self.use_xyz:
            grouped_feature = torch.cat([grouped_pc, grouped_feature], 1)
        elif self.use_xyz:
            grouped_feature = grouped_pc.contiguous()

        grouped_feature = self.conv1(grouped_feature)
        grouped_feature = self.conv2(grouped_feature)
        grouped_feature = self.conv3(grouped_feature)

        valid = (num > 0).view(batch_size, 1, -1, 1)
        grouped_feature = grouped_feature * valid.float()

        return grouped_feature


# multi-scale PointNet module
class PointNetFeat(nn.Module):
    def __init__(self, input_channel=3, num_vec=0):
        super(PointNetFeat, self).__init__()

        self.num_vec = num_vec
        u = cfg.DATA.HEIGHT_HALF
        if not (len(u) == 4):
            raise AssertionError
        self.pointnet1 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[0], 32, use_xyz=True, use_feature=True
        )

        self.pointnet2 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[1], 64, use_xyz=True, use_feature=True
        )

        self.pointnet3 = PointNetModule(
            input_channel - 3, [128, 128, 256], u[2], 64, use_xyz=True, use_feature=True
        )

        self.pointnet4 = PointNetModule(
            input_channel - 3,
            [256, 256, 512],
            u[3],
            128,
            use_xyz=True,
            use_feature=True,
        )

    def forward(self, point_cloud, sample_pc, feat=None, one_hot_vec=None):
        return pooled_point_features(self, point_cloud, sample_pc, feat, one_hot_vec)


# FCN
class ConvFeatNet(nn.Module):
    def __init__(self, i_c=128, num_vec=3):
        super(ConvFeatNet, self).__init__()

        self.block1_conv1 = conv1d(i_c + num_vec, 128, 3, 1, 1)

        self.block2_conv1 = conv1d(128, 128, 3, 2, 1)
        self.block2_conv2 = conv1d(128, 128, 3, 1, 1)
        self.block2_merge = conv1d(128 + 128 + num_vec, 128, 1, 1)

        self.block3_conv1 = conv1d(128, 256, 3, 2, 1)
        self.block3_conv2 = conv1d(256, 256, 3, 1, 1)
        self.block3_merge = conv1d(256 + 256 + num_vec, 256, 1, 1)

        self.block4_conv1 = conv1d(256, 512, 3, 2, 1)
        self.block4_conv2 = conv1d(512, 512, 3, 1, 1)
        self.block4_merge = conv1d(512 + 512 + num_vec, 512, 1, 1)

        self.block2_deconv = de_conv1d(128, 256, 1, 1, 0)
        self.block3_deconv = de_conv1d(256, 256, 2, 2, 0)
        self.block4_deconv = de_conv1d(512, 256, 4, 4, 0)

        initialize_conv_feature_weights(self, nn)

    def forward(self, x1, x2, x3, x4):
        return forward_conv_features(self, (x1, x2, x3, x4))


# the whole pipeline
def _pointnetdet_forward_374_part_1(_state):
    initialize_detector_forward(_state, 4, flatten_class_scores=True)


def _pointnetdet_forward_374_part_2_chunk_1(_state):
    _state["outputs"] = (
        _state["outputs"].permute(0, 2, 1).contiguous().view(-1, _state["output_size"])
    )
    _state["center_ref2"] = (
        _state["center_ref2"].permute(0, 2, 1).contiguous().view(-1, 3)
    )
    _state["cls_probs"] = F.softmax(_state["cls_scores"], -1)


def _decode_heading_and_size(state):
    state["heading_pred_label"] = torch.argmax(state["heading_probs"], -1)
    state["size_pred_label"] = torch.argmax(state["size_probs"], -1)
    state["heading_preds"] = angle_decode(
        state["heading_res_norm"], state["heading_pred_label"],
        num_bins=state["self"].num_bins,
    )
    state["size_preds"] = size_decode(
        state["size_res_norm"], state["mean_size_array"], state["size_pred_label"]
    )


def _pointnetdet_forward_374_part_2_chunk_2(_state):
    if _state["center_label"] is None:
        if not not _state["self"].training:
            raise AssertionError("Please provide labels for training.")
        _state["det_outputs"] = _state["self"].slice_output(_state["outputs"])
        (
            _state["center_boxnet"],
            _state["heading_scores"],
            _state["heading_res_norm"],
            _state["size_scores"],
            _state["size_res_norm"],
        ) = _state["det_outputs"]
        _state["heading_probs"] = F.softmax(_state["heading_scores"], -1)
        _state["size_probs"] = F.softmax(_state["size_scores"], -1)
        _state["center_preds"] = _state["center_boxnet"] + _state["center_ref2"]
        _decode_heading_and_size(_state)
        _state["cls_probs"] = _state["cls_probs"].view(_state["batch_size"], -1, 2)
        _state["center_preds"] = _state["center_preds"].view(
            _state["batch_size"], -1, 3
        )
        _state["size_preds"] = _state["size_preds"].view(_state["batch_size"], -1, 3)
        _state["size_probs"] = _state["size_probs"].view(
            _state["batch_size"], -1, _state["self"].num_size_cluster
        )
        _state["heading_preds"] = _state["heading_preds"].view(_state["batch_size"], -1)
        _state["heading_probs"] = _state["heading_probs"].view(
            _state["batch_size"], -1, _state["self"].num_bins
        )
        _state["outputs"] = (
            _state["cls_probs"],
            _state["center_preds"],
            _state["heading_preds"],
            _state["size_preds"],
            _state["heading_probs"],
            _state["size_probs"],
        )
        return _state["outputs"]
    return None


def _pointnetdet_forward_374_part_2_chunk_3(_state):
    _state["fg_idx"] = (_state["cls_label"].view(-1) == 1).nonzero().view(-1)
    if not _state["fg_idx"].numel() != 0:
        raise AssertionError
    _state["outputs"] = _state["outputs"][_state["fg_idx"], :]
    _state["center_ref2"] = _state["center_ref2"][_state["fg_idx"]]
    _state["det_outputs"] = _state["self"].slice_output(_state["outputs"])
    (
        _state["center_boxnet"],
        _state["heading_scores"],
        _state["heading_res_norm"],
        _state["size_scores"],
        _state["size_res_norm"],
    ) = _state["det_outputs"]
    return _HELPER_CONTINUE


def _pointnetdet_forward_374_part_2(_state):
    _pointnetdet_forward_374_part_2_chunk_1(_state)
    _pointnetdet_forward_374_part_2_chunk_2(_state)
    _pointnetdet_forward_374_part_2_chunk_3(_state)


def _pointnetdet_forward_374_part_3_chunk_1(_state):
    _state["heading_probs"] = F.softmax(_state["heading_scores"], -1)
    _state["size_probs"] = F.softmax(_state["size_scores"], -1)
    _state["cls_loss"] = softmax_focal_loss_ignore(
        _state["cls_probs"], _state["cls_label"].view(-1), ignore_idx=-1
    )
    _state["center_label"] = (
        _state["center_label"]
        .unsqueeze(1)
        .expand(-1, _state["num_out"], -1)
        .contiguous()
        .view(-1, 3)[_state["fg_idx"]]
    )
    _state["heading_label"] = (
        _state["heading_label"]
        .expand(-1, _state["num_out"])
        .contiguous()
        .view(-1)[_state["fg_idx"]]
    )
    _state["size_label"] = (
        _state["size_label"]
        .unsqueeze(1)
        .expand(-1, _state["num_out"], -1)
        .contiguous()
        .view(-1, 3)[_state["fg_idx"]]
    )
    _state["size_class_label"] = (
        _state["size_class_label"]
        .expand(-1, _state["num_out"])
        .contiguous()
        .view(-1)[_state["fg_idx"]]
    )
    _state["center_gt_offsets"] = center_encode(
        _state["center_label"], _state["center_ref2"]
    )
    _state["heading_class_label"], _state["heading_res_norm_label"] = angle_encode(
        _state["heading_label"], num_bins=_state["self"].num_bins
    )
    _state["size_res_label_norm"] = size_encode(
        _state["size_label"], _state["mean_size_array"], _state["size_class_label"]
    )
    _state["center_loss"] = _state["self"].get_center_loss(
        _state["center_boxnet"], _state["center_gt_offsets"]
    )


def _pointnetdet_forward_374_part_3_chunk_2(_state):
    _state["heading_class_loss"], _state["heading_res_norm_loss"] = _state[
        "self"
    ].get_heading_loss(
        _state["heading_scores"],
        _state["heading_res_norm"],
        _state["heading_class_label"],
        _state["heading_res_norm_label"],
    )
    _state["size_class_loss"], _state["size_res_norm_loss"] = _state[
        "self"
    ].get_size_loss(
        _state["size_scores"],
        _state["size_res_norm"],
        _state["size_class_label"],
        _state["size_res_label_norm"],
    )
    _state["center_preds"] = center_decode(
        _state["center_ref2"], _state["center_boxnet"]
    )
    _state["heading"] = angle_decode(
        _state["heading_res_norm"],
        _state["heading_class_label"],
        num_bins=_state["self"].num_bins,
    )
    _state["size"] = size_decode(
        _state["size_res_norm"], _state["mean_size_array"], _state["size_class_label"]
    )
    _state["corners_loss"], _state["corner_gts"] = _state["self"].get_corner_loss(
        (_state["center_preds"], _state["heading"], _state["size"]),
        (_state["center_label"], _state["heading_label"], _state["size_label"]),
    )
    _state["box_loss_weight"] = cfg.LOSS.BOX_LOSS_WEIGHT
    _state["corner_loss_weight"] = cfg.LOSS.CORNER_LOSS_WEIGHT
    _state["head_reg_weight"] = cfg.LOSS.HEAD_REG_WEIGHT
    _state["size_reg_weight"] = cfg.LOSS.SIZE_REG_WEIGHT


def _pointnetdet_forward_374_part_3(_state):
    _pointnetdet_forward_374_part_3_chunk_1(_state)
    _pointnetdet_forward_374_part_3_chunk_2(_state)


def _pointnetdet_forward_374_part_4_chunk_1(_state):
    _state["loss"] = _state["cls_loss"] + _state["box_loss_weight"] * (
        _state["center_loss"]
        + _state["heading_class_loss"]
        + _state["size_class_loss"]
        + _state["head_reg_weight"] * _state["heading_res_norm_loss"]
        + _state["size_reg_weight"] * _state["size_res_norm_loss"]
        + _state["corner_loss_weight"] * _state["corners_loss"]
    )


def _pointnetdet_forward_374_part_4_chunk_2(_state):
    with torch.no_grad():
        _state["cls_prec"] = get_accuracy(
            _state["cls_probs"], _state["cls_label"].view(-1), ignore=-1
        )
        _state["heading_prec"] = get_accuracy(
            _state["heading_probs"], _state["heading_class_label"].view(-1)
        )
        _state["size_prec"] = get_accuracy(
            _state["size_probs"], _state["size_class_label"].view(-1)
        )
        _decode_heading_and_size(_state)
        _state["corner_preds"] = get_box3d_corners_helper(
            _state["center_preds"], _state["heading_preds"], _state["size_preds"]
        )
        _state["overlap"] = rbbox_iou_3d_pair(
            _state["corner_preds"].detach().cpu().numpy(),
            _state["corner_gts"].detach().cpu().numpy(),
        )
        _state["iou2ds"], _state["iou3ds"] = (
            _state["overlap"][:, 0],
            _state["overlap"][:, 1],
        )
        _state["iou2d_mean"] = _state["iou2ds"].mean()
        _state["iou3d_mean"] = _state["iou3ds"].mean()
        _state["iou3d_gt_mean"] = (_state["iou3ds"] >= cfg.IOU_THRESH).mean()
        _state["iou2d_mean"] = torch.tensor(_state["iou2d_mean"]).type_as(
            _state["cls_prec"]
        )
        _state["iou3d_mean"] = torch.tensor(_state["iou3d_mean"]).type_as(
            _state["cls_prec"]
        )
        _state["iou3d_gt_mean"] = torch.tensor(_state["iou3d_gt_mean"]).type_as(
            _state["cls_prec"]
        )


def _pointnetdet_forward_374_part_4(_state):
    _pointnetdet_forward_374_part_4_chunk_1(_state)
    _pointnetdet_forward_374_part_4_chunk_2(_state)


def _pointnetdet_forward_374_part_5(_state):
    _state["losses"] = {
        "total_loss": _state["loss"],
        "cls_loss": _state["cls_loss"],
        "center_loss": _state["center_loss"],
        "head_cls_loss": _state["heading_class_loss"],
        "head_res_loss": _state["heading_res_norm_loss"],
        "size_cls_loss": _state["size_class_loss"],
        "size_res_loss": _state["size_res_norm_loss"],
        "corners_loss": _state["corners_loss"],
    }
    _state["metrics"] = {
        "cls_acc": _state["cls_prec"],
        "head_acc": _state["heading_prec"],
        "size_acc": _state["size_prec"],
        "IoU_2D": _state["iou2d_mean"],
        "IoU_3D": _state["iou3d_mean"],
        "IoU_" + str(cfg.IOU_THRESH): _state["iou3d_gt_mean"],
    }


def prepare_detector_outputs(state):
    """Normalize detector tensors and return predictions in inference mode."""
    _pointnetdet_forward_374_part_2_chunk_1(state)
    inference_output = _pointnetdet_forward_374_part_2_chunk_2(state)
    if inference_output is not None:
        return inference_output
    return _pointnetdet_forward_374_part_2_chunk_3(state)


def compute_detector_losses(state):
    _pointnetdet_forward_374_part_3(state)


def compute_detector_metrics(state):
    _pointnetdet_forward_374_part_4(state)


def collect_detector_results(state):
    _pointnetdet_forward_374_part_5(state)


class PointNetDet(nn.Module):
    def __init__(self, input_channel=3, num_vec=0, num_classes=2):
        super(PointNetDet, self).__init__()

        initialize_pointnet_detector(
            self, cfg, DATASET_INFO, PointNetFeat, ConvFeatNet,
            input_channel, num_vec, num_classes, 768, nn,
        )

        nn.init.kaiming_uniform_(self.cls_out.weight, mode="fan_in")
        nn.init.kaiming_uniform_(self.reg_out.weight, mode="fan_in")

        self.cls_out.bias.data.zero_()
        self.reg_out.bias.data.zero_()

    def get_center_loss(self, pred_offsets, gt_offsets):

        center_dist = torch.norm(gt_offsets - pred_offsets, 2, dim=-1)
        center_loss = huber_loss(center_dist, delta=3.0)

        return center_loss

    def get_heading_loss(
        self,
        heading_scores,
        heading_res_norm,
        heading_class_label,
        heading_res_norm_label,
    ):

        heading_class_loss = F.cross_entropy(heading_scores, heading_class_label)

        # b, NUM_HEADING_BIN -> b, 1
        heading_res_norm_select = torch.gather(
            heading_res_norm, 1, heading_class_label.view(-1, 1)
        )

        heading_res_norm_loss = huber_loss(
            heading_res_norm_select.squeeze(1) - heading_res_norm_label, delta=1.0
        )

        return heading_class_loss, heading_res_norm_loss

    def get_size_loss(
        self, size_scores, size_res_norm, size_class_label, size_res_label_norm
    ):
        batch_size = size_scores.shape[0]
        size_class_loss = F.cross_entropy(size_scores, size_class_label)

        # b, NUM_SIZE_CLUSTER, 3 -> b, 1, 3
        size_res_norm_select = torch.gather(
            size_res_norm,
            1,
            size_class_label.view(batch_size, 1, 1).expand(batch_size, 1, 3),
        )

        size_norm_dist = torch.norm(
            size_res_label_norm - size_res_norm_select.squeeze(1), 2, dim=-1
        )

        size_res_norm_loss = huber_loss(size_norm_dist, delta=1.0)

        return size_class_loss, size_res_norm_loss

    def get_corner_loss(self, preds, gts):

        center_label, heading_label, size_label = gts
        center_preds, heading_preds, size_preds = preds

        corners_3d_gt = get_box3d_corners_helper(
            center_label, heading_label, size_label
        )
        corners_3d_gt_flip = get_box3d_corners_helper(
            center_label, heading_label + np.pi, size_label
        )

        corners_3d_pred = get_box3d_corners_helper(
            center_preds, heading_preds, size_preds
        )

        # N, 8, 3
        corners_dist = torch.min(
            torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1).mean(-1),
            torch.norm(corners_3d_pred - corners_3d_gt_flip, 2, dim=-1).mean(-1),
        )
        corners_loss = huber_loss(corners_dist, delta=1.0)

        return corners_loss, corners_3d_gt

    def forward(self, data_dicts):
        _state = locals()
        _pointnetdet_forward_374_part_1(_state)
        _helper_result = prepare_detector_outputs(_state)
        if _helper_result is not _HELPER_CONTINUE:
            return _helper_result
        compute_detector_losses(_state)
        compute_detector_metrics(_state)
        collect_detector_results(_state)

        return (_state["losses"], _state["metrics"])

    def slice_output(self, output):

        batch_size = output.shape[0]

        num_bins = self.num_bins
        num_sizes = self.num_size_cluster

        center = output[:, 0:3].contiguous()

        heading_scores = output[:, 3: 3 + num_bins].contiguous()

        heading_res_norm = output[:, 3 + num_bins: 3 + num_bins * 2].contiguous()

        size_scores = output[
            :, 3 + num_bins * 2: 3 + num_bins * 2 + num_sizes
        ].contiguous()

        size_res_norm = output[:, 3 + num_bins * 2 + num_sizes:].contiguous()
        size_res_norm = size_res_norm.view(batch_size, num_sizes, 3)

        return center, heading_scores, heading_res_norm, size_scores, size_res_norm
