# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

from torch import nn
import torch

from torchplus.nn import Sequential

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.name = "fusion_layer"
        self.corner_points_feature = Sequential(
            nn.Conv2d(24, 48, 1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1),
            nn.ReLU(),
            nn.Conv2d(96, 4, 1),
        )
        self.fuse_2d_3d = Sequential(
            nn.Conv2d(4, 18, 1),
            nn.ReLU(),
            nn.Conv2d(18, 36, 1),
            nn.ReLU(),
            nn.Conv2d(36, 36, 1),
            nn.ReLU(),
            nn.Conv2d(36, 1, 1),
        )
        self.maxpool = Sequential(
            nn.MaxPool2d([200, 1], 1),
        )

    def forward(self, input_1, tensor_index):
        flag = -1
        if tensor_index[0, 0] == -1:
            out_1 = torch.zeros(
                1, 200, 70400, dtype=input_1.dtype, device=input_1.device
            )
            out_1[:, :, :] = -9999999
            flag = 0
        else:
            x = self.fuse_2d_3d(input_1)
            out_1 = torch.zeros(
                1, 200, 70400, dtype=input_1.dtype, device=input_1.device
            )
            out_1[:, :, :] = -9999999
            out_1[:, tensor_index[:, 0], tensor_index[:, 1]] = x[0, :, 0, :]
            flag = 1
        x = self.maxpool(out_1)
        x = x.squeeze().reshape(1, -1, 1)
        return x, flag
