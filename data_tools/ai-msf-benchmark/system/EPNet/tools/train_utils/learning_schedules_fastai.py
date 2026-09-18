# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# This file is borrowed from https://github.com/traveller59/second.pytorch

from typing import NamedTuple
from functools import partial

import numpy as np

from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class OnecycleInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    fai_optimizer: object
    total_step: object
    lr_max: object
    moms: object
    div_factor: object
    pct_start: object


class LRSchedulerStep(object):
    def __init__(self, fai_optimizer, total_step, lr_phases, mom_phases):
        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                if not (self.lr_phases[-1][0] < start):
                    raise AssertionError
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append(
                    (
                        int(start * total_step),
                        int(lr_phases[i + 1][0] * total_step),
                        lambda_func,
                    )
                )
            else:
                self.lr_phases.append(
                    (int(start * total_step), total_step, lambda_func)
                )
        if not (self.lr_phases[0][0] == 0):
            raise AssertionError
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                if not (self.mom_phases[-1][0] < start):
                    raise AssertionError
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append(
                    (
                        int(start * total_step),
                        int(mom_phases[i + 1][0] * total_step),
                        lambda_func,
                    )
                )
            else:
                self.mom_phases.append(
                    (int(start * total_step), total_step, lambda_func)
                )
        if not (self.mom_phases[0][0] == 0):
            raise AssertionError

    def step(self, step):
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / (end - start))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


_ONECYCLE_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    OnecycleInitParameters,
    ("fai_optimizer", "total_step", "lr_max", "moms", "div_factor", "pct_start"),
    {},
    frozenset(()),
    6,
    None,
    None,
)


class OneCycle(LRSchedulerStep):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _ONECYCLE_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        fai_optimizer, total_step, lr_max, moms, div_factor, pct_start = (
            bound_parameters
        )
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        a1 = int(total_step * self.pct_start)
        _ = total_step - a1
        low_lr = self.lr_max / self.div_factor
        lr_phases = (
            (0, partial(annealing_cos, low_lr, self.lr_max)),
            (self.pct_start, partial(annealing_cos, self.lr_max, low_lr / 1e4)),
        )
        mom_phases = (
            (0, partial(annealing_cos, *self.moms)),
            (self.pct_start, partial(annealing_cos, *self.moms[::-1])),
        )
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)


class FakeOptim:
    def __init__(self):
        self.lr = 0
        self.mom = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    opt = FakeOptim()  # 3e-3, wd=0.4, div_factor=10
    schd = OneCycle(opt, 100, 3e-3, (0.95, 0.85), 10.0, 0.1)

    lrs = []
    moms = []
    for i in range(100):
        schd.step(i)
        lrs.append(opt.lr)
        moms.append(opt.mom)
    plt.plot(lrs)
    plt.show()
    plt.plot(moms)
    plt.show()
