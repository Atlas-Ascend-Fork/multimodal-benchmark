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


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        return torch.sum(torch.abs(loss)) / torch.sum(val_pixels)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        return torch.sum(loss**2) / torch.sum(val_pixels)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, outputs, target, *args):
        return self.loss(outputs, target)


class BINLoss(nn.Module):
    def __init__(self):
        super(BINLoss, self).__init__()
        self.cross_entropy_loss = nn.cross_entropy_loss(ignore_index=-1)

    def forward(self, outputs, target_bin_cls, target_res_reg, *args):
        outputs_bin_cls, outputs_res_reg = torch.chunk(outputs, 2, 1)
        bin_cls_loss = self.cross_entropy_loss(outputs_bin_cls, target_bin_cls)

        val_pixels = torch.ge(target_bin_cls.unsqueeze(1), 0).float()
        outputs_res_reg = torch.gather(
            outputs_res_reg, 1, target_bin_cls.unsqueeze(1) * val_pixels.long()
        )
        res_reg_loss = outputs_res_reg * val_pixels - target_res_reg * val_pixels
        res_reg_loss = torch.sum(res_reg_loss**2) / torch.sum(val_pixels)
        return 5 * (bin_cls_loss + 10 * res_reg_loss)


if __name__ == "__main__":
    input_tensor = torch.randn([2, 20, 384, 1248])
    loss_function = BINLoss()
    target_bin_cls = torch.randint(low=-1, high=10, size=[2, 384, 1248])
    target_res_reg = torch.randn([2, 1, 384, 1248])

    loss = loss_function(input_tensor, target_bin_cls, target_res_reg)
