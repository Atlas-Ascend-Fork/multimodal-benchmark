# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import logging

import torch


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class MixedprecisionwrapperInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    optimizer: object
    scale: object
    auto_scale: object
    inc_factor: object
    dec_factor: object
    num_iters_be_stable: object


LOGGER = logging.getLogger(__name__)

required = object()


def param_fp32_copy(params):
    param_copy = [
        param.clone().type(torch.cuda.FloatTensor).detach() for param in params
    ]
    for param in param_copy:
        param.requires_grad = True
    return param_copy


def set_grad(params, params_with_grad, scale=1.0):
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(
                param.data.new().resize_(*param.data.size())
            )
        grad = param_w_grad.grad.data
        if scale is not None:
            grad /= scale
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            return True  # invalid grad
        param.grad.data.copy_(grad)
    return False


_MIXEDPRECISIONWRAPPER_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    MixedprecisionwrapperInitParameters,
    (
        "optimizer",
        "scale",
        "auto_scale",
        "inc_factor",
        "dec_factor",
        "num_iters_be_stable",
    ),
    {
        "scale": None,
        "auto_scale": True,
        "inc_factor": 2.0,
        "dec_factor": 0.5,
        "num_iters_be_stable": 500,
    },
    frozenset(()),
    6,
    None,
    None,
)


class MixedPrecisionWrapper(object):
    """mixed precision optimizer wrapper.
    Arguments:
        optimizer (torch.optim.Optimizer): an instance of
            :class:`torch.optim.Optimizer`
        scale: (float): a scalar for grad scale.
        auto_scale: (bool): whether enable auto scale.
            The algorihm of auto scale is discribled in
            http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
    """

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _MIXEDPRECISIONWRAPPER_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        optimizer, scale, auto_scale, inc_factor, dec_factor, num_iters_be_stable = (
            bound_parameters
        )
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("must provide a torch.optim.Optimizer")
        self.optimizer = optimizer
        if hasattr(self.optimizer, "name"):
            self.name = self.optimizer.name  # for ckpt system
        param_groups_copy = []
        for _, group in enumerate(optimizer.param_groups):
            group_copy = {n: v for n, v in group.items() if n != "params"}
            group_copy["params"] = param_fp32_copy(group["params"])
            param_groups_copy.append(group_copy)

        # switch param_groups, may be dangerous
        self.param_groups = optimizer.param_groups
        optimizer.param_groups = param_groups_copy
        self.grad_scale = scale
        self.auto_scale = auto_scale
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        self.stable_iter_count = 0
        self.num_iters_be_stable = num_iters_be_stable

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        return self.optimizer.__setstate__(state)

    def __repr__(self):
        return self.optimizer.__repr__()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        for g, g_copy in zip(self.param_groups, self.optimizer.param_groups):
            invalid = set_grad(g_copy["params"], g["params"], self.grad_scale)
            if invalid:
                if self.grad_scale is None or not (self.auto_scale):
                    raise ValueError("nan/inf detected but auto_scale disabled.")
                self.grad_scale *= self.dec_factor
                LOGGER.info("%s", "scale decay to {}".format(self.grad_scale))
                return
        if self.auto_scale:
            self.stable_iter_count += 1
            if self.stable_iter_count > self.num_iters_be_stable:
                if self.grad_scale is not None:
                    self.grad_scale *= self.inc_factor
                self.stable_iter_count = 0

        if closure is None:
            self.optimizer.step()
        else:
            self.optimizer.step(closure)
        for g, g_copy in zip(self.param_groups, self.optimizer.param_groups):
            for p_copy, p in zip(g_copy["params"], g["params"]):
                p.data.copy_(p_copy.data)
