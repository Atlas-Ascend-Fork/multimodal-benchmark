# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import sys

from query_depth_point.query_depth_point import QueryDepthPoint
import torch



sys.path.insert(0, "../")

LOGGER = logging.getLogger(__name__)

xyz1 = torch.rand(2, 3, 50) * 2 - 1

xyz2 = torch.index_select(xyz1, 2, torch.arange(10).long())

batch_size = xyz1.shape[0]
mask = torch.zeros(batch_size, xyz2.shape[2], xyz1.shape[2])
for i in range(batch_size):
    for j in range(xyz2.shape[2]):
        x1, y1, z1 = xyz2[i, :, j]
        inside = torch.abs(xyz1[i, 2] - z1) < 0.2
        mask[i, j] = inside

xyz1 = xyz1.float().cuda()
xyz2 = xyz2.float().cuda()

net = QueryDepthPoint(0.2, 4).cuda()

idx, pts_idx = net(xyz1, xyz2)
LOGGER.info("%s", idx[0])
LOGGER.info("%s", pts_idx)
LOGGER.info("%s", torch.nonzero(mask[0]))
