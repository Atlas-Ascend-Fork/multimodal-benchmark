# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Normalization and convolution setup for sparse CLOCs models."""

import numpy as np
from shared.parameters import bind_named as _bind_named


def configure_sparse_state(*call_args, **call_kwargs):
    """Populate constructor state with norm-aware dense and sparse layers."""
    (
        state,
        change_defaults,
        neural_network,
        sparse_conv,
        empty_layer,
        logger,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('state', 'change_defaults', 'neural_network', 'sparse_conv', 'empty_layer', 'logger'),
    )
    state["self"].name = state["name"]
    if state["use_norm"]:
        state["batch_norm2d"] = change_defaults(eps=0.001, momentum=0.01)(
            neural_network.BatchNorm2d
        )
        state["batch_norm1d"] = change_defaults(eps=0.001, momentum=0.01)(
            neural_network.BatchNorm1d
        )
        bias = False
    else:
        state["batch_norm2d"] = empty_layer
        state["batch_norm1d"] = empty_layer
        bias = True
    state["conv2d"] = change_defaults(bias=bias)(neural_network.Conv2d)
    state["sp_conv3d"] = change_defaults(bias=bias)(sparse_conv.SparseConv3d)
    state["sub_m_conv3d"] = change_defaults(bias=bias)(sparse_conv.SubMConv3d)
    state["conv_transpose2d"] = change_defaults(bias=bias)(
        neural_network.ConvTranspose2d
    )
    state["sparse_shape"] = np.array(state["output_shape"][1:4]) + [1, 0, 0]
    logger.info("%s", state["sparse_shape"])
    state["self"].sparse_shape = state["sparse_shape"]
    state["self"].voxel_output_shape = state["output_shape"]
