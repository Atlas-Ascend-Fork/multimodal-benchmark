# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from second.builder import anchor_generator_builder
from second.builder import similarity_calculator_builder
from second.core.target_assigner import TargetAssigner
from second.protos import target_pb2


def build(target_assigner_config, bv_range, box_coder):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(target_assigner_config, (target_pb2.TargetAssigner)):
        raise ValueError(
            "input_reader_config not of type " "input_reader_pb2.InputReader."
        )
    anchor_cfg = target_assigner_config.anchor_generators
    anchor_generators = []
    for a_cfg in anchor_cfg:
        anchor_generator = anchor_generator_builder.build(a_cfg)
        anchor_generators.append(anchor_generator)
    similarity_calc = similarity_calculator_builder.build(
        target_assigner_config.region_similarity_calculator
    )
    positive_fraction = target_assigner_config.sample_positive_fraction
    if positive_fraction < 0:
        positive_fraction = None
    target_assigner = TargetAssigner(
        box_coder=box_coder,
        anchor_generators=anchor_generators,
        region_similarity_calculator=similarity_calc,
        positive_fraction=positive_fraction,
        sample_size=target_assigner_config.sample_size,
    )
    return target_assigner
