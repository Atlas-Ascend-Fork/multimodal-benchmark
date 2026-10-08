# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""
import torch
import torch.nn as nn


#

#


class IMae(nn.Module):
    def __init__(self):
        super(IMae, self).__init__()

    def forward(self, outputs, target, *args):
        loss, count = _inverse_depth_error(outputs, target)
        return torch.mean(loss / count)


def _masked_depth_error(outputs, target, squared=False, require_output=True):
    valid = (target > 0).float()
    if require_output:
        valid *= (outputs > 0).float()
    valid = valid.cuda()
    difference = target * valid - outputs * valid
    error = difference ** 2 if squared else torch.abs(difference)
    loss = torch.sum(error.view(error.size(0), 1, -1), -1, keepdim=True)
    count = torch.sum(valid.view(valid.size(0), 1, -1), -1, keepdim=True)
    return loss, count


def _inverse_depth_error(outputs, target, squared=False):
    outputs, target = outputs / 1000.0, target / 1000.0
    outputs = torch.where(outputs == 0, outputs.new_tensor(-1), outputs).reciprocal()
    target = torch.where(target == 0, target.new_tensor(-1), target).reciprocal()
    outputs = torch.where(outputs == -1, outputs.new_tensor(0), outputs)
    target = torch.where(target == -1, target.new_tensor(0), target)
    return _masked_depth_error(outputs, target, squared, require_output=False)


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, outputs, target, *args):
        loss, count = _masked_depth_error(outputs, target)
        return torch.mean(loss / count) * 1000


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, outputs, target, *args):
        loss, count = _masked_depth_error(outputs, target, squared=True)
        return torch.mean(torch.sqrt(loss / count)) * 1000


class IRmse(nn.Module):
    def __init__(self):
        super(IRmse, self).__init__()

    def forward(self, outputs, target, *args):
        loss, count = _inverse_depth_error(outputs, target, squared=True)
        return torch.mean(torch.sqrt(loss / count))
