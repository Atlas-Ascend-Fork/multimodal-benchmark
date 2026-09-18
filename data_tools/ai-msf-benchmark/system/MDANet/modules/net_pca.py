# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.net import (
    DepthDecoder,
    DepthEncoder,
    RGBEncoder,
    channel_num,
    refinement_outputs,
    refinement_stage,
)


class DepthInputResult(NamedTuple):
    """Depth inputs and weights at the three pyramid resolutions."""

    input_d: object
    input_d12: object
    input_d14: object
    weight_11: object
    weight_12: object
    weight_14: object


def _prepare_depth_inputs(input_d):
    c11 = (input_d > 0).float()
    input_d18 = F.avg_pool2d(input_d, 8, 8) / (F.avg_pool2d(c11, 8, 8) + 0.0001)
    input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(c11, 4, 4) + 0.0001)
    input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(c11, 2, 2) + 0.0001)
    c12, c14 = (input_d12 > 0).float(), (input_d14 > 0).float()

    temp_d14 = F.interpolate(input_d18, scale_factor=2)
    mask14 = torch.logical_and(input_d14 == 0, temp_d14 != 0).float()
    weight_14 = (input_d14 != 0).float() + F.interpolate(
        F.avg_pool2d(c14, 2, 2), scale_factor=2
    ) * mask14
    input_d14 += temp_d14 * mask14

    temp_d12 = F.interpolate(input_d14, scale_factor=2)
    mask12 = torch.logical_and(input_d12 == 0, temp_d12 != 0).float()
    weight_12 = (
        (input_d12 != 0).float()
        + F.interpolate(F.avg_pool2d(c12, 2, 2), scale_factor=2) * mask12
        + F.interpolate(weight_14, scale_factor=2)
        * (1 - F.interpolate(c14, scale_factor=2)) * 0.25
    )
    input_d12 += temp_d12 * mask12

    temp_d11 = F.interpolate(input_d12, scale_factor=2)
    mask11 = torch.logical_and(input_d == 0, temp_d11 != 0).float()
    weight_11 = (
        (input_d != 0).float()
        + F.interpolate(F.avg_pool2d(c11, 2, 2), scale_factor=2) * mask11
        + F.interpolate(weight_12, scale_factor=2)
        * (1 - F.interpolate(c12, scale_factor=2)) * 0.25
    )
    input_d += temp_d11 * mask11
    return DepthInputResult(
        input_d, input_d12, input_d14, weight_11, weight_12, weight_14
    )


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.rgb_encoder = RGBEncoder(3)
        self.depth_encoder1 = DepthEncoder(2, channel_num[2:6])
        self.depth_decoder1 = DepthDecoder(
            channel_num[2:6][::-1], dump_deformation=True
        )
        self.depth_encoder2 = DepthEncoder(3, channel_num[1:6])
        self.depth_decoder2 = DepthDecoder(
            channel_num[1:5][::-1], dump_deformation=True
        )
        self.depth_encoder3 = DepthEncoder(3, channel_num[0:5])
        self.depth_decoder3 = DepthDecoder(
            channel_num[0:4][::-1], dump_deformation=True
        )

    def forward(self, input_d, input_rgb):
        enc_rgb = self.rgb_encoder(input_rgb)
        depth_inputs = _prepare_depth_inputs(input_d)
        input_d, input_d12, input_d14, weight_11, weight_12, weight_14 = depth_inputs

        enc_d14, indices2, indices1 = self.depth_encoder1(
            torch.cat((input_d14, weight_14), dim=1)
        )
        dcd_d14 = self.depth_decoder1(enc_d14, enc_rgb[2:6], indices2, indices1)

        predict_d12 = F.interpolate(
            dcd_d14[4], scale_factor=2, mode="bilinear", align_corners=True
        )
        input_12 = torch.cat((input_d12, weight_12, predict_d12), 1)
        dcd_d12 = refinement_stage(
            encoder=self.depth_encoder2,
            decoder=self.depth_decoder2,
            inputs=input_12,
            previous=dcd_d14,
            rgb_features=enc_rgb[1:5],
        )

        predict_d11 = F.interpolate(
            dcd_d12[4] + predict_d12,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        input_11 = torch.cat((input_d, weight_11, predict_d11), 1)
        dcd_d11 = refinement_stage(
            encoder=self.depth_encoder3,
            decoder=self.depth_decoder3,
            inputs=input_11,
            previous=dcd_d12,
            rgb_features=enc_rgb[0:4],
        )

        return refinement_outputs(dcd_d11, predict_d11, dcd_d14)


if __name__ == "__main__":
    rgb = torch.randn([1, 3, 352, 1216]).cuda()
    sparse_depth = torch.randn([1, 1, 352, 1216]).cuda()
    model = Network().cuda()
    model(sparse_depth, rgb)
