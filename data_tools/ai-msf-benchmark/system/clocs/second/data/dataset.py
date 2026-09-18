# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from functools import partial
import logging
import pickle


from second.core import box_np_ops
from second.data.preprocess import _read_and_prep_v9


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class KittidatasetInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    info_path: object
    root_path: object
    num_point_features: object
    target_assigner: object
    feature_map_size: object
    prep_func: object


LOGGER = logging.getLogger(__name__)


class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


_KITTIDATASET_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    KittidatasetInitParameters,
    (
        "info_path",
        "root_path",
        "num_point_features",
        "target_assigner",
        "feature_map_size",
        "prep_func",
    ),
    {},
    frozenset(()),
    6,
    None,
    None,
)


class KittiDataset(Dataset):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIDATASET_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            info_path,
            root_path,
            num_point_features,
            target_assigner,
            feature_map_size,
            prep_func,
        ) = bound_parameters
        with open(info_path, "rb") as f:
            infos = pickle.load(f)
        self._root_path = root_path
        self._kitti_infos = infos
        self._num_point_features = num_point_features
        LOGGER.info("%s %s", "remain number of infos:", len(self._kitti_infos))
        # generate anchors cache
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
            "anchors_dict": anchors_dict,
        }
        self._prep_func = partial(prep_func, anchor_cache=anchor_cache)

    def __len__(self):
        return len(self._kitti_infos)

    def __getitem__(self, idx):
        return _read_and_prep_v9(
            info=self._kitti_infos[idx],
            root_path=self._root_path,
            num_point_features=self._num_point_features,
            prep_func=self._prep_func,
        )

    @property
    def kitti_infos(self):
        return self._kitti_infos
