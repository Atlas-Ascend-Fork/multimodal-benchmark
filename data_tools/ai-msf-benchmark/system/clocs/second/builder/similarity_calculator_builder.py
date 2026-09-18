# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from second.core import region_similarity


def build(similarity_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    similarity_type = similarity_config.WhichOneof("region_similarity")
    if similarity_type == "rotate_iou_similarity":
        return region_similarity.RotateIouSimilarity()
    elif similarity_type == "nearest_iou_similarity":
        return region_similarity.NearestIouSimilarity()
    elif similarity_type == "distance_similarity":
        cfg = similarity_config.distance_similarity
        return region_similarity.DistanceSimilarity(
            distance_norm=cfg.distance_norm,
            with_rotation=cfg.with_rotation,
            rotation_alpha=cfg.rotation_alpha,
        )
    else:
        raise ValueError("unknown similarity type")
