# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple

import torch.nn as nn


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters
from shared.parameters import bind_named as _bind_named


class SharedmlpInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    args: object
    bn: object
    activation: object
    preact: object
    first: object
    name: object
    instance_norm: object


class ConvbaseInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    in_size: object
    out_size: object
    kernel_size: object
    stride: object
    padding: object
    activation: object
    bn: object
    init: object
    conv: object
    batch_norm: object
    bias: object
    preact: object
    name: object
    instance_norm: object
    instance_norm_func: object


class Conv1dInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    in_size: object
    out_size: object
    kernel_size: object
    stride: object
    padding: object
    activation: object
    bn: object
    init: object
    bias: object
    preact: object
    name: object
    instance_norm: object


Conv2dInitParameters = Conv1dInitParameters


class FcInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    in_size: object
    out_size: object
    activation: object
    bn: object
    init: object
    preact: object
    name: object


_SHAREDMLP_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    SharedmlpInitParameters,
    ("args", "bn", "activation", "preact", "first", "name", "instance_norm"),
    {
        "bn": False,
        "activation": nn.ReLU(inplace=True),
        "preact": False,
        "first": False,
        "name": "",
        "instance_norm": False,
    },
    frozenset(()),
    1,
    None,
    None,
)
_CONVBASE_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    ConvbaseInitParameters,
    (
        "in_size",
        "out_size",
        "kernel_size",
        "stride",
        "padding",
        "activation",
        "bn",
        "init",
        "conv",
        "batch_norm",
        "bias",
        "preact",
        "name",
        "instance_norm",
        "instance_norm_func",
    ),
    {
        "conv": None,
        "batch_norm": None,
        "bias": True,
        "preact": False,
        "name": "",
        "instance_norm": False,
        "instance_norm_func": None,
    },
    frozenset(()),
    15,
    None,
    None,
)
_CONV_PARAMETER_NAMES = (
        "in_size",
        "out_size",
        "kernel_size",
        "stride",
        "padding",
        "activation",
        "bn",
        "init",
        "bias",
        "preact",
        "name",
        "instance_norm",
)


def _conv_schema(kernel_size, stride, padding):
    return _ParameterSchema(
        Conv1dInitParameters,
        _CONV_PARAMETER_NAMES,
        {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "activation": nn.ReLU(inplace=True),
        "bn": False,
        "init": nn.init.kaiming_normal_,
        "bias": True,
        "preact": False,
        "name": "",
        "instance_norm": False,
        },
        frozenset(()),
        2,
        None,
        None,
    )


_CONV1D_INIT_PARAMETERS_SCHEMA = _conv_schema(1, 1, 0)
_CONV2D_INIT_PARAMETERS_SCHEMA = _conv_schema((1, 1), (1, 1), (0, 0))
_FC_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    FcInitParameters,
    ("in_size", "out_size", "activation", "bn", "init", "preact", "name"),
    {
        "activation": nn.ReLU(inplace=True),
        "bn": False,
        "init": None,
        "preact": False,
        "name": "",
    },
    frozenset(()),
    2,
    None,
    None,
)


class SharedMLP(nn.Sequential):

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _SHAREDMLP_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        args, bn, activation, preact, first, name, instance_norm = bound_parameters
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=(
                        activation if (not first or not preact or (i != 0)) else None
                    ),
                    preact=preact,
                    instance_norm=instance_norm,
                ),
            )


def _normalization_unit(enabled, factory, in_size, out_size, preact, **kwargs):
    if not enabled:
        return None
    size = in_size if preact else out_size
    return factory(size, **kwargs)


def _add_conv_side_layers(*call_args, **call_kwargs):
    (
        module,
        add_before_conv,
        preact,
        name,
        bn_unit,
        activation,
        in_unit,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('module', 'add_before_conv', 'preact', 'name', 'bn_unit', 'activation', 'in_unit'),
    )
    if add_before_conv != preact:
        return
    if bn_unit is not None:
        module.add_module(name + "bn", bn_unit)
    if activation is not None:
        module.add_module(name + "activation", activation)
    if bn_unit is None and in_unit is not None:
        module.add_module(name + "in", in_unit)


class _ConvBase(nn.Sequential):

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _CONVBASE_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            in_size, out_size, kernel_size, stride, padding,
            activation, bn, init, conv, batch_norm,
            bias, preact, name, instance_norm, instance_norm_func,
        ) = bound_parameters
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        bn_unit = _normalization_unit(bn, batch_norm, in_size, out_size, preact)
        in_unit = _normalization_unit(
            instance_norm,
            instance_norm_func,
            in_size,
            out_size,
            preact,
            affine=False,
            track_running_stats=False,
        )
        _add_conv_side_layers(
            self, True, preact, name, bn_unit, activation, in_unit
        )
        self.add_module(name + "conv", conv_unit)
        _add_conv_side_layers(
            self, False, preact, name, bn_unit, activation, in_unit
        )


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConfiguredConv(_ConvBase):
    parameter_schema = None
    convolution = None
    batch_normalization = None
    instance_normalization = None

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            self.parameter_schema, call_args, call_kwargs
        )
        (
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            bias,
            preact,
            name,
            instance_norm,
        ) = bound_parameters
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=self.convolution,
            batch_norm=self.batch_normalization,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=self.instance_normalization,
        )


class Conv1d(_ConfiguredConv):
    parameter_schema = _CONV1D_INIT_PARAMETERS_SCHEMA
    convolution = nn.Conv1d
    batch_normalization = BatchNorm1d
    instance_normalization = nn.InstanceNorm1d


class Conv2d(_ConfiguredConv):
    parameter_schema = _CONV2D_INIT_PARAMETERS_SCHEMA
    convolution = nn.Conv2d
    batch_normalization = BatchNorm2d
    instance_normalization = nn.InstanceNorm2d


class FC(nn.Sequential):

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _FC_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        in_size, out_size, activation, bn, init, preact, name = bound_parameters
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "fc", fc)

        if not preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + "activation", activation)
