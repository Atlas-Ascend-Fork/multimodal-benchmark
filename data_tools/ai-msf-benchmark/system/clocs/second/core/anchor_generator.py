# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import numpy as np

from second.core import box_np_ops


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class AnchorgeneratorstrideInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    sizes: object
    anchor_strides: object
    anchor_offsets: object
    rotations: object
    class_name: object
    match_threshold: object
    unmatch_threshold: object
    dtype: object


class AnchorgeneratorrangeInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    anchor_ranges: object
    sizes: object
    rotations: object
    class_name: object
    match_threshold: object
    unmatch_threshold: object
    dtype: object


class AnchorMetadata(NamedTuple):
    """Common metadata shared by all anchor generators."""

    dtype: object
    class_name: object
    match_threshold: object
    unmatch_threshold: object


_ANCHORGENERATORSTRIDE_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    AnchorgeneratorstrideInitParameters,
    (
        "sizes",
        "anchor_strides",
        "anchor_offsets",
        "rotations",
        "class_name",
        "match_threshold",
        "unmatch_threshold",
        "dtype",
    ),
    {
        "sizes": None,
        "anchor_strides": None,
        "anchor_offsets": None,
        "rotations": None,
        "class_name": None,
        "match_threshold": -1,
        "unmatch_threshold": -1,
        "dtype": np.float32,
    },
    frozenset(()),
    8,
    None,
    None,
)
_ANCHORGENERATORRANGE_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    AnchorgeneratorrangeInitParameters,
    (
        "anchor_ranges",
        "sizes",
        "rotations",
        "class_name",
        "match_threshold",
        "unmatch_threshold",
        "dtype",
    ),
    {
        "sizes": None,
        "rotations": None,
        "class_name": None,
        "match_threshold": -1,
        "unmatch_threshold": -1,
        "dtype": np.float32,
    },
    frozenset(()),
    7,
    None,
    None,
)


class _AnchorGeneratorBase:

    def __init__(self, sizes, rotations, metadata):
        self._sizes = sizes if sizes is not None else [1.6, 3.9, 1.56]
        self._rotations = rotations if rotations is not None else [0, np.pi / 2]
        self._dtype = metadata.dtype
        self._class_name = metadata.class_name
        self._match_threshold = metadata.match_threshold
        self._unmatch_threshold = metadata.unmatch_threshold

    @property
    def class_name(self):
        return self._class_name

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        rotation_count = len(self._rotations)
        size_count = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return rotation_count * size_count


class AnchorGeneratorStride(_AnchorGeneratorBase):

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _ANCHORGENERATORSTRIDE_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            sizes,
            anchor_strides,
            anchor_offsets,
            rotations,
            class_name,
            match_threshold,
            unmatch_threshold,
            dtype,
        ) = bound_parameters
        if anchor_strides is None:
            anchor_strides = [0.4, 0.4, 1.0]
        if anchor_offsets is None:
            anchor_offsets = [0.2, -39.8, -1.78]
        super().__init__(
            sizes,
            rotations,
            AnchorMetadata(dtype, class_name, match_threshold, unmatch_threshold),
        )
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets

    def generate(self, feature_map_size):
        return box_np_ops.create_anchors_3d_stride(
            feature_map_size,
            self._sizes,
            self._anchor_strides,
            self._anchor_offsets,
            self._rotations,
            self._dtype,
        )


class AnchorGeneratorRange(_AnchorGeneratorBase):

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _ANCHORGENERATORRANGE_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            anchor_ranges,
            sizes,
            rotations,
            class_name,
            match_threshold,
            unmatch_threshold,
            dtype,
        ) = bound_parameters
        super().__init__(
            sizes,
            rotations,
            AnchorMetadata(dtype, class_name, match_threshold, unmatch_threshold),
        )
        self._anchor_ranges = anchor_ranges

    def generate(self, feature_map_size):
        return box_np_ops.create_anchors_3d_range(
            feature_map_size,
            self._anchor_ranges,
            self._sizes,
            self._rotations,
            self._dtype,
        )
