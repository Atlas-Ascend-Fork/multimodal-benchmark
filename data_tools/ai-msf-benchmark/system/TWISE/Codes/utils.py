# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch.nn.functional as F
from dataloaders.loader_common import ind2sub, sub2ind


def hard2chandep(chan_deps, params=None, device=None):
    if device is None:
        device = torch.device("cpu")
    split_deps = torch.split(chan_deps, 1, 1)
    split_deps = list(split_deps)
    b_sz, __, h_sz, w_sz = split_deps[0].size()
    alpha = torch.sigmoid(split_deps[2])
    max_dep = (alpha > 0.5).long()
    max_dep = max_dep.permute(0, 2, 3, 1).view(-1, 1)
    dep_cat = (
        torch.cat((F.relu(split_deps[1]), F.relu(split_deps[0])), 1)
        * params["depth_maxrange"]
    )
    dep_cat = dep_cat.permute(0, 2, 3, 1).view(-1, 2)
    final_dep = torch.gather(dep_cat, 1, max_dep)
    final_dep = final_dep.view(b_sz, h_sz, w_sz, 1).permute(0, 3, 1, 2)

    return final_dep


def smooth2chandep(chan_deps, params=None, device=None):
    if device is None:
        device = torch.device("cpu")
    split_deps = torch.split(chan_deps, 1, 1)
    split_deps = list(split_deps)

    alpha = torch.sigmoid(split_deps[2])
    final_dep = (
        alpha * F.relu(split_deps[0]) * params["depth_maxrange"]
        + (1 - alpha) * F.relu(split_deps[1]) * params["depth_maxrange"]
    )

    return final_dep
