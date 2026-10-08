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
from dataloaders.loader_common import ind2sub, sub2ind


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class ConvbnParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    in_planes: object
    out_planes: object
    kernel_size: object
    stride: object
    padding: object
    stats_flag: object


class ConvtBnReluParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    in_channels: object
    out_channels: object
    kernel_size: object
    stride: object
    padding: object
    output_padding: object
    bn: object
    relu: object
    stats_flag: object


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(
        tensor.narrow(int(dim), int(start), int(length))
        for start, length in zip(splits, split_sizes)
    )


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def init_weights_constant(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight, 0.3)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.constant_(m.weight, 0.3)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


_CONVBN_PARAMETERS_SCHEMA = _ParameterSchema(
    ConvbnParameters,
    ("in_planes", "out_planes", "kernel_size", "stride", "padding", "stats_flag"),
    {"padding": 1, "stats_flag": True},
    frozenset(()),
    6,
    None,
    None,
)
_CONVT_BN_RELU_PARAMETERS_SCHEMA = _ParameterSchema(
    ConvtBnReluParameters,
    (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "output_padding",
        "bn",
        "relu",
        "stats_flag",
    ),
    {
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "bn": True,
        "relu": True,
        "stats_flag": True,
    },
    frozenset(()),
    9,
    None,
    None,
)


def convbn(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CONVBN_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    in_planes, out_planes, kernel_size, stride, padding, stats_flag = bound_parameters
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes, track_running_stats=stats_flag),
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True
        ),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def convt_bn_relu(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CONVT_BN_RELU_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bn,
        relu,
        stats_flag,
    ) = bound_parameters
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=bias,
        )
    )
    if bn:
        layers.append(nn.BatchNorm2d(out_channels, track_running_stats=stats_flag))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


def _init_encoder_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(module.weight)
        nn.init.constant_(module.bias, 0.01)


class _FeatureEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size, depth):
        super(_FeatureEncoder, self).__init__()
        padding = int((filter_size - 1) / 2)
        self.init = nn.Sequential(
            nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
        )
        for index in range(1, depth + 1):
            stage = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                nn.ReLU(),
                nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
            )
            setattr(self, "enc{}".format(index), stage)
        self.depth = depth
        self.apply(_init_encoder_weights)

    def forward(self, input_tensor, scale=2, pre_x=None):
        output = self.init(input_tensor)
        if pre_x is not None:
            output = output + F.interpolate(
                pre_x, scale_factor=scale, mode="bilinear", align_corners=True
            )
        outputs = [output]
        for index in range(1, self.depth + 1):
            output = getattr(self, "enc{}".format(index))(output)
            outputs.append(output)
        return tuple(outputs)


class DepthEncoder(_FeatureEncoder):
    def __init__(self, in_layers=2, layers=32, filter_size=3):
        super(DepthEncoder, self).__init__(in_layers, layers, filter_size, 2)


class RGBEncoder(_FeatureEncoder):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__(in_layers, layers, filter_size, 4)


def _decoder_stage(channels, filter_size, padding):
    return nn.Sequential(
        nn.ReLU(),
        nn.ConvTranspose2d(
            channels,
            channels,
            filter_size,
            stride=2,
            padding=padding,
            output_padding=padding,
        ),
        nn.ReLU(),
        nn.Conv2d(channels, channels, filter_size, stride=1, padding=padding),
    )


class DepthDecoder(nn.Module):
    def __init__(self, layers, filter_size, out_chan=1):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size - 1) / 2)

        self.dec2 = _decoder_stage(layers // 2, filter_size, padding)
        self.dec1 = _decoder_stage(layers // 2, filter_size, padding)

        self.prdct = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(layers // 2, out_chan, filter_size, stride=1, padding=padding),
        )

        # Init Weights
        for module in self.modules():
            if not isinstance(module, nn.Sequential):
                continue
            for parameter in module:
                if isinstance(parameter, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.xavier_normal_(parameter.weight)
                    nn.init.constant_(parameter.bias, 0.01)

    def forward(self, pre_dx, pre_cx):

        x2 = pre_dx[2] + pre_cx[2]  # torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]  # torch.cat((pre_dx[1], pre_cx[1]), 1)
        x0 = pre_dx[0] + pre_cx[0]

        x3 = self.dec2(x2)  # 1/2 input size
        x4 = self.dec1(x1 + x3)  # 1/1 input size

        # prediction
        output_d = self.prdct(x4 + x0)

        return x4, output_d


class DepthDecoder3chandep(DepthDecoder):
    def __init__(self, layers, filter_size, out_chan=3):
        super(DepthDecoder3chandep, self).__init__(layers, filter_size, out_chan)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

    def forward(self, input_d, input_rgb, c):

        enc_c = self.rgb_encoder(input_rgb)

        # for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(c, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        # for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(c, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(
            dcd_d14[1], scale_factor=2, mode="bilinear", align_corners=True
        )
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        # for the 1/1 res
        predict_d11 = F.interpolate(
            dcd_d12[1] + predict_d12,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[1] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(
            dcd_d14[1], scale_factor=4, mode="bilinear", align_corners=True
        )

        return (
            output_d11,
            output_d12,
            output_d14,
        )


class FinalNetwork(nn.Module):
    def __init__(self):
        super(FinalNetwork, self).__init__()
        self.module = Network()

    def forward(self, input_d, input_rgb, c):

        output_d11, output_d12, output_d14 = self.module(input_d, input_rgb, c)

        return output_d11, output_d12, output_d14


def _split_depth_prediction(decoded):
    channels = torch.split(decoded[1], 1, 1)
    confidence = torch.sigmoid(channels[2])
    fused = confidence * channels[0] + (1 - confidence) * channels[1]
    two_channel = torch.cat((channels[0], channels[1]), 1)
    return channels, fused, two_channel


def _merge_depth_prediction(decoded, residual):
    channels = torch.split(decoded[1], 1, 1)
    output = torch.cat((channels[0], channels[1]), 1) + residual
    return torch.cat((output, channels[2]), 1)


class MultiResNetworkDiffspatialsizes(nn.Module):
    def __init__(self):
        super(MultiResNetworkDiffspatialsizes, self).__init__()

        denc_layers = 64
        cenc_layers = 64
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder3chandep(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder3chandep(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder3chandep(ddcd_layers, 3)

    def forward(self, input_d, input_rgb):

        enc_c = self.rgb_encoder(input_rgb)

        # for the 1/4 res
        input_d14, input_d12 = self._downsample_depth(input_d)

        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])
        output_d14 = dcd_d14[1]  # predicted 3 chan dep in 1/4 res

        _, pred_depfus_d14, output_d14_2chan_interp = _split_depth_prediction(
            dcd_d14
        )
        # for the 1/2 res

        predict_d12 = F.interpolate(
            pred_depfus_d14, scale_factor=2, mode="bilinear", align_corners=True
        )  # fused dep scaled up to half res.
        output_d14_2chan_interp = F.interpolate(
            output_d14_2chan_interp, scale_factor=2, mode="bilinear", align_corners=True
        )  # predicted 2 chan dep scaled to half res
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        # for the 1/1 res
        dcd_d12_splitchannels, pred_depfus_d12, output_d12_2chan_interp = (
            _split_depth_prediction(dcd_d12)
        )
        output_d12_2chan_interp = output_d12_2chan_interp + output_d14_2chan_interp
        output_d12 = torch.cat((output_d12_2chan_interp, dcd_d12_splitchannels[2]), 1)

        predict_d11 = F.interpolate(
            pred_depfus_d12 + predict_d12,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        output_d12_2chan_interp = F.interpolate(
            output_d12_2chan_interp, scale_factor=2, mode="bilinear", align_corners=True
        )

        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = _merge_depth_prediction(dcd_d11, output_d12_2chan_interp)

        return output_d11, output_d12, output_d14

    def _downsample_depth(self, input_d):
        return F.max_pool2d(input_d, 4, 4), F.max_pool2d(input_d, 2, 2)


class MultiResNetworkAvgpoolDiffspatialsizes(MultiResNetworkDiffspatialsizes):
    def __init__(self):
        super(MultiResNetworkAvgpoolDiffspatialsizes, self).__init__()

    def _downsample_depth(self, input_d):
        confidence = (input_d > 0).float()
        pooled = []
        for scale in (4, 2):
            depth = F.avg_pool2d(input_d, scale, scale)
            depth = depth / (F.avg_pool2d(confidence, scale, scale) + 0.0001)
            depth[depth > input_d.max()] = 0
            pooled.append(depth)
        return tuple(pooled)


class MultiResNetworkSamespatialsizes(nn.Module):
    def __init__(self):
        super(MultiResNetworkSamespatialsizes, self).__init__()

        denc_layers = 64
        cenc_layers = 64
        ddcd_layers = denc_layers + cenc_layers

        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder3chandep(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(3, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder3chandep(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(3, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder3chandep(ddcd_layers, 3)

    def forward(self, input_d, input_rgb):

        enc_c = self.rgb_encoder(input_rgb)

        # for the 1/4 res
        input_d14 = F.max_pool2d(input_d, 4, 4)
        input_d12 = F.max_pool2d(input_d, 2, 2)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        dcd_d14_splitchannels = torch.split(dcd_d14[1], 1, 1)

        sig_d14 = torch.sigmoid(dcd_d14_splitchannels[2])
        output_d14_2chan_interp = torch.cat(
            (dcd_d14_splitchannels[0], dcd_d14_splitchannels[1]), 1
        )  # predicted 2 chan dep in 1/4 res
        # for the 1/2 res

        predict_d12 = F.interpolate(
            output_d14_2chan_interp, scale_factor=2, mode="bilinear", align_corners=True
        )  # fused dep scaled up to half res.
        predict_d14_scaled = F.interpolate(
            output_d14_2chan_interp, scale_factor=4, mode="bilinear", align_corners=True
        )  # fused dep scaled up to half res.
        output_d14_sigmoid_interp = F.interpolate(
            sig_d14, scale_factor=4, mode="nearest"
        )

        output_d14 = torch.cat((predict_d14_scaled, output_d14_sigmoid_interp), 1)

        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        # for the 1/1 res
        dcd_d12_splitchannels = torch.split(dcd_d12[1], 1, 1)
        sig_d12 = torch.sigmoid(dcd_d12_splitchannels[2])
        output_d12_2chan_interp = torch.cat(
            (dcd_d12_splitchannels[0], dcd_d12_splitchannels[1]), 1
        )

        predict_d12_scaled = F.interpolate(
            output_d12_2chan_interp + predict_d12,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        output_d12_sigmoid_interp = F.interpolate(
            sig_d12, scale_factor=2, mode="nearest"
        )

        output_d12 = torch.cat((predict_d12_scaled, output_d12_sigmoid_interp), 1)

        output_d12_2chan_interp = F.interpolate(
            output_d12_2chan_interp, scale_factor=2, mode="bilinear", align_corners=True
        )

        input_11 = torch.cat((input_d, predict_d12_scaled), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        dcd_d11_splitchannels = torch.split(dcd_d11[1], 1, 1)
        output_d11 = torch.cat((dcd_d11_splitchannels[0], dcd_d11_splitchannels[1]), 1)
        output_d11 = output_d11 + predict_d12_scaled
        output_d11 = torch.cat((output_d11, dcd_d11_splitchannels[2]), 1)

        return output_d11, output_d12, output_d14


class MaxPooling(nn.Module):

    def __init__(self, kernelsize, stride):
        super(MaxPooling, self).__init__()
        if stride == 1:
            pad = int((kernelsize - 1) / 2)
        else:
            pad = 0
        self.pool = nn.MaxPool2d(kernel_size=kernelsize, stride=stride, padding=pad)

    def forward(self, predlayer):

        predlayer = self.pool(predlayer)

        return predlayer


def _upproject_forward_807_part_1(_state):
    _state["out1_1"] = _state["self"].conv1_1(
        nn.functional.pad(_state["x"], (1, 1, 1, 1))
    )
    _state["out1_2"] = _state["self"].conv1_2(
        nn.functional.pad(_state["x"], (1, 1, 1, 0))
    )
    _state["out1_3"] = _state["self"].conv1_3(
        nn.functional.pad(_state["x"], (1, 0, 1, 1))
    )
    _state["out1_4"] = _state["self"].conv1_4(
        nn.functional.pad(_state["x"], (1, 0, 1, 0))
    )
    _state["out2_1"] = _state["self"].conv2_1(
        nn.functional.pad(_state["x"], (1, 1, 1, 1))
    )
    _state["out2_2"] = _state["self"].conv2_2(
        nn.functional.pad(_state["x"], (1, 1, 1, 0))
    )
    _state["out2_3"] = _state["self"].conv2_3(
        nn.functional.pad(_state["x"], (1, 0, 1, 1))
    )
    _state["out2_4"] = _state["self"].conv2_4(
        nn.functional.pad(_state["x"], (1, 0, 1, 0))
    )
    _state["height"] = _state["out1_1"].size()[2]
    _state["width"] = _state["out1_1"].size()[3]
    _state["out1_1_2"] = (
        torch.stack((_state["out1_1"], _state["out1_2"]), dim=-3)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .view(_state["self"].batch_size, -1, _state["height"], _state["width"] * 2)
    )
    _state["out1_3_4"] = (
        torch.stack((_state["out1_3"], _state["out1_4"]), dim=-3)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .view(_state["self"].batch_size, -1, _state["height"], _state["width"] * 2)
    )
    _state["out1_1234"] = (
        torch.stack((_state["out1_1_2"], _state["out1_3_4"]), dim=-3)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(_state["self"].batch_size, -1, _state["height"] * 2, _state["width"] * 2)
    )


def _upproject_forward_807_part_2(_state):
    _state["out2_1_2"] = (
        torch.stack((_state["out2_1"], _state["out2_2"]), dim=-3)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .view(_state["self"].batch_size, -1, _state["height"], _state["width"] * 2)
    )
    _state["out2_3_4"] = (
        torch.stack((_state["out2_3"], _state["out2_4"]), dim=-3)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .view(_state["self"].batch_size, -1, _state["height"], _state["width"] * 2)
    )
    _state["out2_1234"] = (
        torch.stack((_state["out2_1_2"], _state["out2_3_4"]), dim=-3)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(_state["self"].batch_size, -1, _state["height"] * 2, _state["width"] * 2)
    )
    _state["out1"] = _state["self"].bn1_1(_state["out1_1234"])
    _state["out1"] = _state["self"].relu(_state["out1"])
    _state["out1"] = _state["self"].conv3(_state["out1"])
    _state["out1"] = _state["self"].bn2(_state["out1"])
    _state["out2"] = _state["self"].bn1_2(_state["out2_1234"])
    _state["out"] = _state["out1"] + _state["out2"]
    _state["out"] = _state["self"].relu(_state["out"])


class UpProject(nn.Module):

    def __init__(self, in_channels, out_channels, batch_size, stats_flag=True):
        super(UpProject, self).__init__()
        self.batch_size = batch_size

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.bn1_1 = nn.BatchNorm2d(out_channels, track_running_stats=stats_flag)
        self.bn1_2 = nn.BatchNorm2d(out_channels, track_running_stats=stats_flag)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=stats_flag)

    def forward(self, x):
        _state = locals()
        _upproject_forward_807_part_1(_state)
        _upproject_forward_807_part_2(_state)

        return _state["out"]


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, stats_flag=True):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, statsFlag=stats_flag),
            nn.ReLU(inplace=True),
        )

        self.conv2 = convbn(planes, planes, 3, 1, statsFlag=stats_flag)

        self.ds = convbn(inplanes, planes, 1, stride, padding=0, statsFlag=stats_flag)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        x = self.ds(x)
        out += x
        out = self.relu(out)
        return out
