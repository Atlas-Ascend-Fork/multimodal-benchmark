# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Functions to build DetectionModel training optimizers."""

from functools import partial

from torch import nn
import torch

from torchplus.train.fastai_optim import OptimWrapper, FastAIMixedOptim


def children(m: nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))


def flatten_model(m):
    return sum(map(flatten_model, m.children()), []) if num_children(m) else [m]


def get_layer_groups(m):
    return [nn.Sequential(*flatten_model(m))]


def _module_build_39_part_1(_state):
    _state["optimizer_type"] = _state["optimizer_config"].WhichOneof("optimizer")
    _state["optimizer"] = None
    if _state["optimizer_type"] == "rms_prop_optimizer":
        _state["config"] = _state["optimizer_config"].rms_prop_optimizer
        _state["optimizer_func"] = partial(
            torch.optim.RMSprop,
            alpha=_state["config"].decay,
            momentum=_state["config"].momentum_optimizer_value,
            eps=_state["config"].epsilon,
        )
    if _state["optimizer_type"] == "momentum_optimizer":
        _state["config"] = _state["optimizer_config"].momentum_optimizer
        _state["optimizer_func"] = partial(
            torch.optim.SGD,
            momentum=_state["config"].momentum_optimizer_value,
            eps=_state["config"].epsilon,
        )
    if _state["optimizer_type"] == "adam_optimizer":
        _state["config"] = _state["optimizer_config"].adam_optimizer
        _state["optimizer_func"] = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=_state["config"].amsgrad
        )


def _module_build_39_part_2(_state):
    if _state["mixed"]:
        _state["optimizer"] = FastAIMixedOptim.create(
            _state["optimizer_func"],
            0.003,
            get_layer_groups(_state["net"]),
            _state["net"],
            loss_scale=_state["loss_scale"],
            wd=_state["config"].weight_decay,
            true_wd=_state["optimizer_config"].fixed_weight_decay,
            bn_wd=True,
        )
    else:
        _state["optimizer"] = OptimWrapper.create(
            _state["optimizer_func"],
            0.003,
            get_layer_groups(_state["net"]),
            wd=_state["config"].weight_decay,
            true_wd=_state["optimizer_config"].fixed_weight_decay,
            bn_wd=True,
        )
    if _state["optimizer"] is None:
        raise ValueError("Optimizer %s not supported." % _state["optimizer_type"])
    if _state["optimizer_config"].use_moving_average:
        raise ValueError("torch don't support moving average")
    if _state["name"] is None:
        _state["optimizer"].name = _state["optimizer_type"]
    else:
        _state["optimizer"].name = _state["name"]


def build(optimizer_config, net, name=None, mixed=False, loss_scale=512.0):
    """Create optimizer based on config.

    Args:
      optimizer_config: A Optimizer proto message.

    Returns:
      An optimizer and a list of variables for summary.

    Raises:
      ValueError: when using an unsupported input data type.
    """
    _state = locals()
    _module_build_39_part_1(_state)
    _module_build_39_part_2(_state)
    return _state["optimizer"]
