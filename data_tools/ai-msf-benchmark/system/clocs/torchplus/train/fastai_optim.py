# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from collections.abc import Iterable

from torch import nn
from torch._utils import _unflatten_dense_tensors
from torch.nn.utils import parameters_to_vector
import torch


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class FastaimixedoptimCreateParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    opt_func: object
    lr: object
    layer_groups: object
    model: object
    flat_master: object
    loss_scale: object
    kwargs: object


bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def split_bn_bias(layer_groups):
    "Split the layers in `layer_groups` into batchnorm (`bn_types`) and non-batchnorm groups."
    split_groups = []
    for length_value in layer_groups:
        l1, l2 = [], []
        for c in length_value.children():
            if isinstance(c, bn_types):
                l2.append(c)
            else:
                l1.append(c)
        split_groups += [nn.Sequential(*l1), nn.Sequential(*l2)]
    return split_groups


def get_master(layer_groups, flat_master: bool = False):
    "Return two lists, one for the model parameters in FP16 and one for the master parameters in FP32."
    split_groups = split_bn_bias(layer_groups)
    model_params = [
        [param for param in lg.parameters() if param.requires_grad]
        for lg in split_groups
    ]
    if flat_master:
        master_params = []
        for lg in model_params:
            if len(lg) == 0:
                master_params.append([])
                continue
            mp = parameters_to_vector([param.data.float() for param in lg])
            mp = torch.nn.Parameter(mp, requires_grad=True)
            if mp.grad is None:
                mp.grad = mp.new(*mp.size())
            master_params.append([mp])
        return model_params, master_params
    else:
        master_params = [
            [param.clone().float().detach() for param in lg] for lg in model_params
        ]
        for mp in master_params:
            for param in mp:
                param.requires_grad = True
        return model_params, master_params


def model_g2master_g(model_params, master_params, flat_master: bool = False) -> None:
    "Copy the `model_params` gradients to `master_params` for the optimizer step."
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(master_group) != 0:
                master_group[0].grad.data.copy_(
                    parameters_to_vector([p.grad.data.float() for p in model_group])
                )
        return
    for model_group, master_group in zip(model_params, master_params):
        for model, master in zip(model_group, master_group):
            if model.grad is None:
                master.grad = None
                continue
            if master.grad is None:
                master.grad = master.data.new(*master.data.size())
            master.grad.data.copy_(model.grad.data)


def master2model(model_params, master_params, flat_master: bool = False) -> None:
    "Copy `master_params` to `model_params`."
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(model_group) == 0:
                continue
            for model, master in zip(
                model_group,
                _unflatten_dense_tensors(master_group[0].data, model_group),
            ):
                model.data.copy_(master)
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group):
                model.data.copy_(master.data)


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if isinstance(q, int) else len(p) if q is None else len(q)
    if len(p) == 1:
        p = p * n
    if not (len(p) == n):
        raise AssertionError(f"List len mismatch ({len(p)} vs {n})")
    return list(p)


def trainable_params(m: nn.Module):
    "Return list of trainable params in `m`."
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res


def is_tuple(x) -> bool:
    return isinstance(x, tuple)


# copy from fastai.
class OptimWrapper:
    "Basic wrapper around `opt` to simplify hyper-parameters changes."

    def __init__(self, opt, wd, true_wd: bool = False, bn_wd: bool = True):
        self.opt, self.true_wd, self.bn_wd = opt, true_wd, bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove("params")
        self.opt_func = self.opt.__class__
        self.model_params = []
        self.master_params = []
        self.flat_master = False
        self.loss_scale = 1.0
        self.model = None
        self._lr = []
        self._mom = []
        self._beta = None
        self._wd = []
        self.read_defaults()
        self.wd = wd

    def __repr__(self) -> str:
        return f"OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}"

    # Pytorch optimizer methods

    def __getattr__(self, k: str):
        return getattr(self.opt, k, None)

    @property
    def lr(self) -> float:
        return self._lr[-1]

    @property
    def learning_rates(self):
        return self._lr

    @lr.setter
    def lr(self, val: float) -> None:
        self._lr = self.set_val("lr", listify(val, self._lr))

    @property
    def mom(self) -> float:
        return self._mom[-1]

    @mom.setter
    def mom(self, val: float) -> None:
        if "momentum" in self.opt_keys:
            self.set_val("momentum", listify(val, self._mom))
        elif "betas" in self.opt_keys:
            self.set_val("betas", (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self) -> float:
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val: float) -> None:
        "Set beta (or alpha as makes sense for given optimizer)."
        if val is None:
            return
        if "betas" in self.opt_keys:
            self.set_val("betas", (self._mom, listify(val, self._beta)))
        elif "alpha" in self.opt_keys:
            self.set_val("alpha", listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self) -> float:
        return self._wd[-1]

    @wd.setter
    def wd(self, val: float) -> None:
        "Set weight decay."
        if not self.true_wd:
            self.set_val("weight_decay", listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    # Helper functions

    @classmethod
    def create(cls, opt_func, lr, layer_groups, **kwargs):
        "Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`."
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func(
            [
                {"params": trainable_params(length_value), "lr": 0}
                for length_value in split_groups
            ]
        )
        opt = cls(opt, **kwargs)
        opt.lr, opt.opt_func = listify(lr, layer_groups), opt_func
        return opt

    def new(self, layer_groups):
        "Create a new `OptimWrapper` from `self` with another `layer_groups` but the same hyper-parameters."
        opt_func = getattr(self, "opt_func", self.opt.__class__)
        split_groups = split_bn_bias(layer_groups)
        _ = opt_func(
            [
                {"params": trainable_params(length_value), "lr": 0}
                for length_value in split_groups
            ]
        )
        return self.create(
            opt_func,
            self.lr,
            layer_groups,
            wd=self.wd,
            true_wd=self.true_wd,
            bn_wd=self.bn_wd,
        )

    def step(self) -> None:
        "Set weight decay and step optimizer."
        # weight decay outside of optimizer step (AdamW)
        if self.true_wd:
            for lr, wd, pg1, pg2 in zip(
                self._lr,
                self._wd,
                self.opt.param_groups[::2],
                self.opt.param_groups[1::2],
            ):
                for p in pg1["params"]:
                    p.data.mul_(1 - wd * lr)
                if not self.bn_wd:
                    continue
                for p in pg2["params"]:
                    p.data.mul_(1 - wd * lr)
            self.set_val("weight_decay", listify(0, self._wd))
        self.opt.step()

    def zero_grad(self) -> None:
        "Clear optimizer gradients."
        self.opt.zero_grad()

    # Passthrough to the inner opt.

    def clear(self):
        "Reset the state of the inner optimizer."
        sd = self.state_dict()
        sd["state"] = {}
        self.load_state_dict(sd)

    # Hyperparameters as properties

    def read_defaults(self) -> None:
        "Read the values inside the optimizer for the hyper-parameters."
        self._beta = None
        if "lr" in self.opt_keys:
            self._lr = self.read_val("lr")
        if "momentum" in self.opt_keys:
            self._mom = self.read_val("momentum")
        if "alpha" in self.opt_keys:
            self._beta = self.read_val("alpha")
        if "betas" in self.opt_keys:
            self._mom, self._beta = self.read_val("betas")
        if "weight_decay" in self.opt_keys:
            self._wd = self.read_val("weight_decay")

    def set_val(self, key: str, val, bn_groups: bool = True):
        "Set `val` inside the optimizer dictionary at `key`."
        if is_tuple(val):
            val = [(v1, v2) for v1, v2 in zip(*val)]
        for v, pg1, pg2 in zip(
            val, self.opt.param_groups[::2], self.opt.param_groups[1::2]
        ):
            pg1[key] = v
            if bn_groups:
                pg2[key] = v
        return val

    def read_val(self, key: str):
        "Read a hyperparameter `key` in the optimizer dictionary."
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if is_tuple(val[0]):
            val = [o[0] for o in val], [o[1] for o in val]
        return val


_FASTAIMIXEDOPTIM_CREATE_PARAMETERS_SCHEMA = _ParameterSchema(
    FastaimixedoptimCreateParameters,
    ("opt_func", "lr", "layer_groups", "model", "flat_master", "loss_scale", "kwargs"),
    {"flat_master": False, "loss_scale": 512.0},
    frozenset(()),
    6,
    None,
    "kwargs",
)


class FastAIMixedOptim(OptimWrapper):
    @classmethod
    def create(cls, *call_args, **call_kwargs):
        "Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`."
        bound_parameters = _bind_parameters(
            _FASTAIMIXEDOPTIM_CREATE_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        opt_func, lr, layer_groups, model, flat_master, loss_scale, kwargs = (
            bound_parameters
        )
        opt = OptimWrapper.create(opt_func, lr, layer_groups, **kwargs)
        opt.model_params, opt.master_params = get_master(layer_groups, flat_master)
        opt.flat_master = flat_master
        opt.loss_scale = loss_scale
        opt.model = model
        # Changes the optimizer so that the optimization step is done in FP32.
        mom, wd, beta = opt.mom, opt.wd, opt.beta
        lrs = [lr for lr in opt.learning_rates for _ in range(2)]
        opt_params = [
            {"params": mp, "lr": lr} for mp, lr in zip(opt.master_params, lrs)
        ]
        opt.opt = opt_func(opt_params)
        opt.mom, opt.wd, opt.beta = mom, wd, beta
        return opt

    def step(self):
        model_g2master_g(self.model_params, self.master_params, self.flat_master)
        for group in self.master_params:
            for param in group:
                param.grad.div_(self.loss_scale)
        super(FastAIMixedOptim, self).step()
        self.model.zero_grad()
        # Update the params from master to model.
        master2model(self.model_params, self.master_params, self.flat_master)
