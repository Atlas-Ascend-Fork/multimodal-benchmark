# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from torch.utils.data import Dataset

from second.builder import dataset_builder
from second.protos import input_reader_pb2


class DatasetWrapper(Dataset):
    """convert our dataset to Dataset class in pytorch."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def dataset(self):
        return self._dataset


def build(
    input_reader_config, model_config, training, voxel_generator, target_assigner=None
) -> DatasetWrapper:
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError(
            "input_reader_config not of type " "input_reader_pb2.InputReader."
        )
    dataset = dataset_builder.build(
        input_reader_config, model_config, training, voxel_generator, target_assigner
    )
    dataset = DatasetWrapper(dataset)
    return dataset
