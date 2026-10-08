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
import torch.nn.functional as F


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class DecodeBboxTargetParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    roi_box3d: object
    pred_reg: object
    loc_scope: object
    loc_bin_size: object
    num_head_bin: object
    anchor_size: object
    get_xz_fine: object
    get_y_by_bin: object
    loc_y_scope: object
    loc_y_bin_size: object
    get_ry_fine: object


def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    r = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, [0, 2]].unsqueeze(dim=1)  # (N, 1, 2)

    pc[:, [0, 2]] = torch.matmul(pc_temp, r.permute(0, 2, 1)).squeeze(dim=1)
    return pc


_DECODE_BBOX_TARGET_PARAMETERS_SCHEMA = _ParameterSchema(
    DecodeBboxTargetParameters,
    (
        "roi_box3d",
        "pred_reg",
        "loc_scope",
        "loc_bin_size",
        "num_head_bin",
        "anchor_size",
        "get_xz_fine",
        "get_y_by_bin",
        "loc_y_scope",
        "loc_y_bin_size",
        "get_ry_fine",
    ),
    {
        "get_xz_fine": True,
        "get_y_by_bin": False,
        "loc_y_scope": 0.5,
        "loc_y_bin_size": 0.25,
        "get_ry_fine": False,
    },
    frozenset(()),
    11,
    None,
    None,
)


def _decode_xz_location(params):
    bin_count = int(params.loc_scope / params.loc_bin_size) * 2
    x_slice, z_slice = slice(0, bin_count), slice(bin_count, bin_count * 2)
    if not cfg.TRAIN.BBOX_AVG_BY_BIN:
        x_bin = torch.argmax(params.pred_reg[:, x_slice], dim=1)
        z_bin = torch.argmax(params.pred_reg[:, z_slice], dim=1)
        pos_x = x_bin.float() * params.loc_bin_size + params.loc_bin_size / 2
        pos_z = z_bin.float() * params.loc_bin_size + params.loc_bin_size / 2
        pos_x, pos_z = pos_x - params.loc_scope, pos_z - params.loc_scope
        offset = bin_count * 2
        if params.get_xz_fine:
            x_res = torch.gather(
                params.pred_reg[:, offset: offset + bin_count],
                1,
                x_bin.unsqueeze(1),
            ).squeeze(1)
            z_res = torch.gather(
                params.pred_reg[:, offset + bin_count: offset + bin_count * 2],
                1,
                z_bin.unsqueeze(1),
            ).squeeze(1)
            pos_x += x_res * params.loc_bin_size
            pos_z += z_res * params.loc_bin_size
            offset += bin_count * 2
        return pos_x, pos_z, offset
    if not params.get_xz_fine:
        raise AssertionError("now only support bin format!")
    x_probability = F.softmax(params.pred_reg[:, x_slice], 1)
    z_probability = F.softmax(params.pred_reg[:, z_slice], 1)
    centers = torch.arange(bin_count).float().to(params.pred_reg.device)
    centers = centers * params.loc_bin_size + params.loc_bin_size / 2 - params.loc_scope
    offset = bin_count * 2
    x_values = centers + params.pred_reg[:, offset: offset + bin_count] * params.loc_bin_size
    z_values = centers + params.pred_reg[
        :, offset + bin_count: offset + bin_count * 2
    ] * params.loc_bin_size
    return (
        (x_values * x_probability).sum(1),
        (z_values * z_probability).sum(1),
        offset + bin_count * 2,
    )


def _decode_y_location(params, offset):
    if not params.get_y_by_bin:
        return params.roi_box3d[:, 1] + params.pred_reg[:, offset], offset + 1
    bin_count = int(params.loc_y_scope / params.loc_y_bin_size) * 2
    y_bin = torch.argmax(params.pred_reg[:, offset: offset + bin_count], dim=1)
    residual_offset = offset + bin_count
    residual = torch.gather(
        params.pred_reg[:, residual_offset: residual_offset + bin_count],
        1,
        y_bin.unsqueeze(1),
    ).squeeze(1)
    position = (
        y_bin.float() * params.loc_y_bin_size
        + params.loc_y_bin_size / 2
        - params.loc_y_scope
        + residual * params.loc_y_bin_size
        + params.roi_box3d[:, 1]
    )
    return position, residual_offset + bin_count


def _weighted_rotation(probabilities, angles, right_mask):
    right_probability = probabilities * right_mask.float()
    left_probability = probabilities * (~right_mask).float()
    right_mass = right_probability.sum(1, keepdim=True) + 1e-7
    left_mass = left_probability.sum(1, keepdim=True) + 1e-7
    right_angle = (angles * right_probability / right_mass).sum(1)
    left_angle = (angles * left_probability / left_mass).sum(1)
    return torch.where(right_mass.squeeze() >= left_mass.squeeze(), right_angle, left_angle)


def _decode_rotation(params, offset):
    bin_slice = slice(offset, offset + params.num_head_bin)
    residual_slice = slice(offset + params.num_head_bin, offset + params.num_head_bin * 2)
    if not cfg.TEST.RY_WITH_BIN:
        rotation_bin = torch.argmax(params.pred_reg[:, bin_slice], dim=1)
        residual = torch.gather(
            params.pred_reg[:, residual_slice], 1, rotation_bin.unsqueeze(1)
        ).squeeze(1)
        angle_size = (
            np.pi / 2 / params.num_head_bin
            if params.get_ry_fine
            else 2 * np.pi / params.num_head_bin
        )
        residual *= angle_size / 2
        if params.get_ry_fine:
            rotation = rotation_bin.float() * angle_size + angle_size / 2
            return rotation + residual - np.pi / 4, residual_slice.stop
        rotation = (rotation_bin.float() * angle_size + residual) % (2 * np.pi)
        rotation[rotation > np.pi] -= 2 * np.pi
        return rotation, residual_slice.stop
    probability = F.softmax(params.pred_reg[:, bin_slice], 1)
    residual = params.pred_reg[:, residual_slice]
    angle_size = (
        np.pi / 2 / params.num_head_bin
        if params.get_ry_fine
        else 2 * np.pi / params.num_head_bin
    )
    angles = torch.arange(params.num_head_bin).float().to(residual.device) * angle_size
    angles = angles + residual * (angle_size / 2)
    if params.get_ry_fine:
        angles = angles + angle_size / 2 - np.pi / 4
        return _weighted_rotation(probability, angles, angles >= 0), residual_slice.stop
    angles %= 2 * np.pi
    rotation = _weighted_rotation(probability, angles, angles <= np.pi)
    rotation[rotation > np.pi] -= 2 * np.pi
    return rotation, residual_slice.stop


def decode_bbox_target(*call_args, **call_kwargs):
    """Decode network regression output into camera-coordinate 3D boxes."""
    params = _bind_parameters(
        _DECODE_BBOX_TARGET_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    if cfg.TRAIN.BBOX_AVG_BY_BIN != cfg.TEST.BBOX_AVG_BY_BIN:
        raise AssertionError
    if cfg.TRAIN.RY_WITH_BIN != cfg.TEST.RY_WITH_BIN:
        raise AssertionError
    anchor_size = params.anchor_size.to(params.roi_box3d.get_device())
    pos_x, pos_z, offset = _decode_xz_location(params)
    pos_y, offset = _decode_y_location(params, offset)
    rotation, offset = _decode_rotation(params, offset)
    if offset + 3 != params.pred_reg.shape[1]:
        raise AssertionError
    size = params.pred_reg[:, offset: offset + 3] * anchor_size + anchor_size
    shifted = torch.cat(
        (
            pos_x.view(-1, 1),
            pos_y.view(-1, 1),
            pos_z.view(-1, 1),
            size,
            rotation.view(-1, 1),
        ),
        dim=1,
    )
    result = shifted
    if params.roi_box3d.shape[1] == 7:
        roi_rotation = params.roi_box3d[:, 6]
        result = rotate_pc_along_y_torch(shifted, -roi_rotation)
        result[:, 6] += roi_rotation
    result[:, [0, 2]] += params.roi_box3d[:, [0, 2]]
    return result
