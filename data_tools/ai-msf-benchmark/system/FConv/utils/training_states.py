# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections import OrderedDict

from utils.utils import AverageMeter


class TrainingStates(object):

    def __init__(self, state_names=None):

        if state_names is None:
            state_names = []
        self.states = OrderedDict([(key, 0) for key in state_names])

        self.average_meters = OrderedDict(
            [(key, AverageMeter()) for key in state_names]
        )

        self.state_names = state_names

    def update_states(self, states, batch_size=1):

        if len(self.states) == 0:
            state_names = states.keys()
            self.states = OrderedDict([(key, 0) for key in state_names])

            self.average_meters = OrderedDict(
                [(key, AverageMeter()) for key in state_names]
            )

        self.states.update(states)
        for key, meter in self.average_meters.items():
            meter.update(self.states[key], batch_size)

    def get_states(self, avg=True):
        states = OrderedDict()
        for key, meter in self.average_meters.items():
            if avg:
                states[key] = meter.avg
            else:
                states[key] = meter.val

        return states

    def format_states(self, states):
        output = ""
        for key, state in states.items():
            output += "%s: %.3f " % (key, state)
        return output
