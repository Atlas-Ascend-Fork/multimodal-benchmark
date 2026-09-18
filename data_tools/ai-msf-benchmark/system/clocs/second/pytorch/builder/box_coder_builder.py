# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from second.pytorch.core.box_coders import BevBoxCoderTorch, GroundBox3dCoderTorch


def build(box_coder_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    box_coder_type = box_coder_config.WhichOneof("box_coder")
    if box_coder_type == "ground_box3d_coder":
        cfg = box_coder_config.ground_box3d_coder
        return GroundBox3dCoderTorch(cfg.linear_dim, cfg.encode_angle_vector)
    elif box_coder_type == "bev_box_coder":
        cfg = box_coder_config.bev_box_coder
        return BevBoxCoderTorch(
            cfg.linear_dim, cfg.encode_angle_vector, cfg.z_fixed, cfg.h_fixed
        )
    else:
        raise ValueError("unknown box_coder type")
