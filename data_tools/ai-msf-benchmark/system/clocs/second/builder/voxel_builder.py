# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from spconv.utils import VoxelGenerator

from second.protos import voxel_generator_pb2


def build(voxel_config):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
        raise ValueError(
            "input_reader_config not of type " "input_reader_pb2.InputReader."
        )
    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_config.voxel_size),
        point_cloud_range=list(voxel_config.point_cloud_range),
        max_num_points=voxel_config.max_number_of_points_per_voxel,
        max_voxels=20000,
    )
    return voxel_generator
