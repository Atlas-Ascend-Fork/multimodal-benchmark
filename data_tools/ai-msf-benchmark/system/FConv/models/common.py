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


import torch
import torch.nn as nn


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class Conv1dParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    i_c: object
    o_c: object
    k: object
    s: object
    p: object
    bn: object


class Conv2dParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    i_c: object
    o_c: object
    k: object
    s: object
    p: object
    bn: object


class Conv3dParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    i_c: object
    o_c: object
    k: object
    s: object
    p: object
    bn: object


class DeConv1dParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    i_c: object
    o_c: object
    k: object
    s: object
    p: object
    bn: object


class DeConv2dParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    i_c: object
    o_c: object
    k: object
    s: object
    p: object
    bn: object


class DeConv3dParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    i_c: object
    o_c: object
    k: object
    s: object
    p: object
    bn: object


class SigmoidFocalLossParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    prob: object
    target: object
    alpha: object
    gamma: object
    grad_scale: object
    weights: object


class SoftmaxFocalLossParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    prob: object
    target: object
    alpha: object
    gamma: object
    grad_scale: object
    weights: object


def init_params(m, method="constant"):
    """
    method: xavier_uniform, kaiming_normal, constant
    """
    if isinstance(m, list):
        for im in m:
            init_params(im, method)
    else:
        if method == "xavier_uniform":
            nn.init.xavier_uniform_(m.weight.data)
        elif method == "kaiming_normal":
            nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
        elif isinstance(method, (int, float)):
            m.weight.data.fill_(method)
        else:
            raise ValueError("unknown method.")
        if m.bias is not None:
            m.bias.data.zero_()


_CONV1D_PARAMETERS_SCHEMA = _ParameterSchema(
    Conv1dParameters,
    ("i_c", "o_c", "k", "s", "p", "bn"),
    {"s": 1, "p": 0, "bn": True},
    frozenset(()),
    6,
    None,
    None,
)
_CONV2D_PARAMETERS_SCHEMA = _ParameterSchema(
    Conv2dParameters,
    ("i_c", "o_c", "k", "s", "p", "bn"),
    {"s": 1, "p": 0, "bn": True},
    frozenset(()),
    6,
    None,
    None,
)
_CONV3D_PARAMETERS_SCHEMA = _ParameterSchema(
    Conv3dParameters,
    ("i_c", "o_c", "k", "s", "p", "bn"),
    {"s": 1, "p": 0, "bn": True},
    frozenset(()),
    6,
    None,
    None,
)
_DE_CONV1D_PARAMETERS_SCHEMA = _ParameterSchema(
    DeConv1dParameters,
    ("i_c", "o_c", "k", "s", "p", "bn"),
    {"s": 1, "p": 0, "bn": True},
    frozenset(()),
    6,
    None,
    None,
)
_DE_CONV2D_PARAMETERS_SCHEMA = _ParameterSchema(
    DeConv2dParameters,
    ("i_c", "o_c", "k", "s", "p", "bn"),
    {"s": 1, "p": 0, "bn": True},
    frozenset(()),
    6,
    None,
    None,
)
_DE_CONV3D_PARAMETERS_SCHEMA = _ParameterSchema(
    DeConv3dParameters,
    ("i_c", "o_c", "k", "s", "p", "bn"),
    {"s": 1, "p": 0, "bn": True},
    frozenset(()),
    6,
    None,
    None,
)
_SIGMOID_FOCAL_LOSS_PARAMETERS_SCHEMA = _ParameterSchema(
    SigmoidFocalLossParameters,
    ("prob", "target", "alpha", "gamma", "grad_scale", "weights"),
    {"alpha": 0.25, "gamma": 2, "grad_scale": None, "weights": None},
    frozenset(()),
    6,
    None,
    None,
)
_SOFTMAX_FOCAL_LOSS_PARAMETERS_SCHEMA = _ParameterSchema(
    SoftmaxFocalLossParameters,
    ("prob", "target", "alpha", "gamma", "grad_scale", "weights"),
    {"alpha": 0.25, "gamma": 2, "grad_scale": None, "weights": None},
    frozenset(()),
    6,
    None,
    None,
)


def conv1d(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CONV1D_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    i_c, o_c, k, s, p, bn = bound_parameters
    if bn:
        return nn.Sequential(
            nn.Conv1d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm1d(o_c), nn.ReLU(True)
        )
    else:
        return nn.Sequential(nn.Conv1d(i_c, o_c, k, s, p), nn.ReLU(True))


def conv2d(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CONV2D_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    i_c, o_c, k, s, p, bn = bound_parameters
    if bn:
        return nn.Sequential(
            nn.Conv2d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm2d(o_c), nn.ReLU(True)
        )
    else:
        return nn.Sequential(nn.Conv2d(i_c, o_c, k, s, p), nn.ReLU(True))


def conv3d(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _CONV3D_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    i_c, o_c, k, s, p, bn = bound_parameters
    if bn:
        return nn.Sequential(
            nn.Conv3d(i_c, o_c, k, s, p, bias=False), nn.BatchNorm3d(o_c), nn.ReLU(True)
        )
    else:
        return nn.Sequential(nn.Conv3d(i_c, o_c, k, s, p), nn.ReLU(True))


def de_conv1d(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _DE_CONV1D_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    i_c, o_c, k, s, p, bn = bound_parameters
    if bn:
        return nn.Sequential(
            nn.ConvTranspose1d(i_c, o_c, k, s, p, bias=False),
            nn.BatchNorm1d(o_c),
            nn.ReLU(True),
        )
    else:
        return nn.Sequential(nn.ConvTranspose1d(i_c, o_c, k, s, p), nn.ReLU(True))


def de_conv2d(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _DE_CONV2D_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    i_c, o_c, k, s, p, bn = bound_parameters
    if bn:
        return nn.Sequential(
            nn.ConvTranspose2d(i_c, o_c, k, s, p, bias=False),
            nn.BatchNorm2d(o_c),
            nn.ReLU(True),
        )
    else:
        return nn.Sequential(nn.ConvTranspose2d(i_c, o_c, k, s, p), nn.ReLU(True))


def de_conv3d(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _DE_CONV3D_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    i_c, o_c, k, s, p, bn = bound_parameters
    if bn:
        return nn.Sequential(
            nn.ConvTranspose3d(i_c, o_c, k, s, p, bias=False),
            nn.BatchNorm3d(o_c),
            nn.ReLU(True),
        )
    else:
        return nn.Sequential(nn.ConvTranspose3d(i_c, o_c, k, s, p), nn.ReLU(True))


def get_accuracy(output, target, ignore=None):

    if not (output.shape[0] == target.shape[0]):
        raise AssertionError
    if ignore is not None:
        if not (isinstance(ignore, int)):
            raise AssertionError
        keep = (target != ignore).nonzero().view(-1)
        output = output[keep]
        target = target[keep]

    pred = torch.argmax(output, -1)

    correct = (pred.view(-1) == target.view(-1)).float().sum()
    acc = correct * (1.0 / target.view(-1).shape[0])

    return acc


def _scatter_shape(x, y, shape):
    """
    Scatter updates into a new (initially zero) tensor according to indices.
    x (b, feat, k)
    y index LongTensor (b, k, 3)
    shape (n1, n2, n3)
    out (b, feat, n1, n2, n3)
    """
    if not (len(shape) == y.shape[-1]):
        raise AssertionError
    if not (x.dim() == 3 and x.shape[-1] == y.shape[1]):
        raise AssertionError
    if not (x.is_contiguous()):
        raise AssertionError

    batch_size = x.shape[0]
    num_feats = x.shape[1]
    output = x.new_zeros(batch_size, num_feats, *shape)

    # flat index to one dimension
    stride = torch.LongTensor(output[0][0].stride()).type_as(y.data)

    index = torch.sum(stride * y, dim=-1)  # b, k

    return output, index, batch_size, num_feats


def scatter_nd(x, y, shape):
    output, index, batch_size, num_feats = _scatter_shape(x, y, shape)
    flat = output.view(batch_size, num_feats, -1)
    flat.scatter_(2, index.unsqueeze(1).expand(-1, num_feats, -1).contiguous(), x)
    return flat.view(batch_size, num_feats, *shape)


def scatter_add_nd(x, y, shape):
    """
    Scatter updates into a new (initially zero) tensor according to indices.
    x (b, feat, k)
    y index LongTensor (b, k, 3)
    shape (n1, n2, n3)
    out (b, feat, n1, n2, n3)
    """
    output, index, batch_size, num_feats = _scatter_shape(x, y, shape)
    flat = output.view(batch_size, num_feats, -1)
    flat.scatter_add_(2, index.unsqueeze(1).expand(-1, num_feats, -1).contiguous(), x)
    return flat.view(batch_size, num_feats, *shape)


def gather_nd(x, y):
    """
    Scatter updates into a new (initially zero) tensor according to indices.
    x (b, feat, k)
    y index LongTensor (b, k, 3)
    shape (n1, n2, n3)
    out (b, feat, n1, n2, n3)
    """
    if not (x.is_contiguous()):
        raise AssertionError

    batch_size = x.shape[0]
    num_feats = x.shape[1]

    # flat index to one dimension
    stride = torch.LongTensor(x[0][0].stride()).type_as(y.data)

    index = torch.sum(stride * y, dim=-1)  # b, k

    x = x.view(batch_size, num_feats, -1)
    return torch.gather(x, 2, index.unsqueeze(1).expand(-1, num_feats, -1).contiguous())


def scatter_1d(x, y, shape):
    if not (len(shape) == 1):
        raise AssertionError
    output = x.new_zeros(x.shape[0], x.shape[1], *(shape))
    output.scatter_(2, y.unsqueeze(1).expand(-1, x.shape[1], -1), x)

    return output


def sigmoid_focal_loss(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SIGMOID_FOCAL_LOSS_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    prob, target, alpha, gamma, grad_scale, weights = bound_parameters
    pt = target * prob + (1 - target) * (1 - prob)

    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = -alpha_t * (1 - pt) ** gamma * torch.log(pt + 1e-14)

    if weights is not None:
        if not (weights.shape == loss.shape):
            raise AssertionError
        loss *= weights
    if grad_scale is None:
        grad_scale = 1.0 / loss.shape[0]

    loss = loss.sum() * grad_scale
    return loss


def softmax_focal_loss(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SOFTMAX_FOCAL_LOSS_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    prob, target, alpha, gamma, grad_scale, weights = bound_parameters
    alpha_t = (1 - alpha) * (target == 0).float() + alpha * (target >= 1).float()

    prob_t = prob[range(len(target)), target]
    loss = -alpha_t * (1 - prob_t) ** gamma * torch.log(prob_t + 1e-14)

    if weights is not None:
        if not (weights.shape == loss.shape):
            raise AssertionError
        loss *= weights

    if grad_scale is None:
        grad_scale = 1.0 / loss.shape[0]

    loss = loss.sum() * grad_scale

    return loss


def softmax_focal_loss_ignore(prob, target, alpha=0.25, gamma=2, ignore_idx=-1):
    keep = (target != ignore_idx).nonzero().view(-1)
    num_fg = (target > 0).data.sum()

    target = target[keep]
    prob = prob[keep, :]

    alpha_t = (1 - alpha) * (target == 0).float() + alpha * (target >= 1).float()

    prob_t = prob[range(len(target)), target]
    loss = -alpha_t * (1 - prob_t) ** gamma * torch.log(prob_t + 1e-14)

    loss = loss.sum() / (num_fg + 1e-14)

    return loss


def separable_conv2d(in_channels, out_channels, k, s=(1, 1), depth_multiplier=1):
    #
    conv = [
        nn.Conv2d(in_channels, in_channels * depth_multiplier, k, groups=in_channels)
    ]

    if out_channels is not None:
        conv += [nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, 1)]

    conv = nn.Sequential(*conv)

    return conv


class XConv(nn.Module):
    def __init__(self, k, c, depth_multiplier=1, with_x_transformation=True):

        super(XConv, self).__init__()

        # transform K*K
        self.conv_t0 = nn.Sequential(
            nn.Conv2d(3, k * k, (1, k)),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(k * k),
        )

        self.conv_t1 = nn.Sequential(
            separable_conv2d(k, None, (1, k), (1, 1), k),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(k * k),
        )

        self.conv_t2 = nn.Sequential(
            separable_conv2d(k, None, (1, k), (1, 1), k),
            nn.BatchNorm2d(k * k),
        )

        self.separable_conv2d = nn.Sequential(
            separable_conv2d(c, None, (1, k), (1, 1), depth_multiplier),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(c),
        )

        self.with_x_transformation = with_x_transformation

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, nn_pts_local, nn_fts_input):
        """
        pts: nn_pts_local (N, 3, P, K)
        nn_fts_input : (N, C, P, K)

        """
        n, c0, p, k = nn_pts_local.shape
        if not (c0 == 3):
            raise AssertionError

        if self.with_x_transformation:

            x_0 = self.conv_t0(nn_pts_local)  # N, K*K, P, 1
            x_0 = x_0.view(n, k, k, p).transpose(2, 3).contiguous()  # N, K, P, K
            x_1 = self.conv_t1(x_0)
            x_1 = x_1.view(n, k, k, p).transpose(2, 3).contiguous()  # N, K, P, K
            x_2 = self.conv_t2(x_1)
            x_2 = x_1.view(n, k, k, p).permute(0, 3, 1, 2).contiguous()  # N, K, P, K

            x = x_2.view(n * p, k, k)  #
            # N,C1,P,K -> N,P,C1,K -> N, P, K, C1
            nn_fts_input = (
                nn_fts_input.permute(0, 2, 3, 1).contiguous().view(n * p, k, -1)
            )
            # (N * P, K, K)  (N * P, K, C1)  -> (N * P, K, C1)
            fts_x = torch.bmm(x, nn_fts_input)
            # N, P, K, C1 -> N, C1, K, P -> N, C1, P, K
            fts_x = fts_x.view(n, p, k, -1).permute(0, 3, 1, 2).contiguous()

        else:
            fts_x = nn_fts_input

        fts = self.separable_conv2d(fts_x)  # (N, C, P, 1)

        return fts
