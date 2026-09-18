# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

""" Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017

Modified by Zhixin Wang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import NamedTuple
import logging
import os
import pickle
import sys

from configs.config import cfg
from torch.utils.data.dataloader import default_collate
import numpy as np

from datasets.data_utils import rotate_pc_along_y, project_image_to_rect, compute_box_3d, extract_pc_in_box3d
from datasets.dataset_info import DATASET_INFO
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters
from datasets.provider_common import (
    LabelFormatResult,
    ProviderDatasetBase,
    augment_provider_sample,
    generate_point_labels,
    initialize_provider_sample,
    prepare_provider_training_state,
    provider_inputs,
)
from datasets.provider_geometry import CenterViewMixin

_HELPER_CONTINUE = object()


class ProviderdatasetGenerateLabelsParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    center: object
    dimension: object
    angle: object
    ref_xyz: object
    p: object


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


LOGGER = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


_PROVIDERDATASET_GENERATE_LABELS_PARAMETERS_SCHEMA = _ParameterSchema(
    ProviderdatasetGenerateLabelsParameters,
    ("center", "dimension", "angle", "ref_xyz", "p"),
    {},
    frozenset(()),
    5,
    None,
    None,
)


def _providerdataset_getitem_263_part_1(_state):
    initialize_provider_sample(
        _state,
        cfg,
        _state["self"].category_info.CLASSES,
        lambda count: np.random.permutation(len(count)),
        lambda class_name: "%s not in category_info" % class_name,
    )
    _state["p"] = _state["self"].calib_list[_state["index"]]["P2"].reshape(3, 4)
    _state["ref1"], _state["ref2"], _state["ref3"], _state["ref4"] = _state[
        "self"
    ].generate_ref(_state["box"], _state["p"])
    if _state["rotate_to_center"]:
        _state["ref1"] = _state["self"].get_center_view(_state["ref1"], _state["index"])
        _state["ref2"] = _state["self"].get_center_view(_state["ref2"], _state["index"])
        _state["ref3"] = _state["self"].get_center_view(_state["ref3"], _state["index"])
        _state["ref4"] = _state["self"].get_center_view(_state["ref4"], _state["index"])


def _providerdataset_getitem_263_part_2(_state):
    if _state["self"].from_rgb_detection:
        _state["data_inputs"] = provider_inputs(
            _state, ("ref1", "ref2", "ref3", "ref4"), rgb_detection=True
        )
        return _state["data_inputs"]
    prepare_provider_training_state(_state)
    return _HELPER_CONTINUE


def _providerdataset_getitem_263_part_3(_state):
    augment_provider_sample(
        _state, ("ref1", "ref2", "ref3", "ref4"), cfg.DATA.MAX_DEPTH
    )
    _state["labels_ref2"] = _state["self"].generate_labels(
        _state["box3d_center"],
        _state["box3d_size"],
        _state["heading_angle"],
        _state["ref2"],
        _state["p"],
    )


def _providerdataset_getitem_263_part_4(_state):
    _state["data_inputs"] = provider_inputs(
        _state, ("ref1", "ref2", "ref3", "ref4")
    )


def _provider_data_path(split, from_rgb_detection, overwritten_data_path):
    if overwritten_data_path:
        return overwritten_data_path
    category = "caronly" if cfg.DATA.CAR_ONLY else "pedcyc"
    if not (cfg.DATA.CAR_ONLY or cfg.DATA.PEOPLE_ONLY):
        category = "carpedcyc"
    suffix = "_rgb_detection" if from_rgb_detection else ""
    filename = "frustum_{}_{}{}.pickle".format(category, split, suffix)
    return os.path.join(cfg.DATA.DATA_ROOT, filename)


def _load_provider_fields(provider, data_path, from_rgb_detection):
    common_fields = ("id", "box2d", "input", "type", "frustum_angle")
    detection_fields = common_fields + ("prob", "calib")
    training_fields = common_fields[:2] + (
        "box3d", "input", "label", "type", "heading", "size",
        "frustum_angle", "gt_box2d", "calib",
    )
    fields = detection_fields if from_rgb_detection else training_fields
    with open(data_path, "rb") as fp:
        for field in fields:
            setattr(provider, field + "_list", pickle.load(fp))


def _extend_provider_fields(provider, data_path):
    extend_fields = (
        "id", "box2d", "box3d", "input", "label", "type", "heading",
        "size", "frustum_angle", "gt_box2d", "calib",
    )
    extend_path = data_path.replace(".", "_det.")
    if not os.path.exists(extend_path):
        raise AssertionError(extend_path)
    with open(extend_path, "rb") as fp:
        for field in extend_fields:
            getattr(provider, field + "_list").extend(pickle.load(fp))
    logger.info("load dataset from {}".format(extend_path))


class ProviderDataset(CenterViewMixin, ProviderDatasetBase):
    rotate_points = staticmethod(rotate_pc_along_y)
    config = cfg
    dataset_info = DATASET_INFO
    resolve_data_path = _provider_data_path
    load_fields = _load_provider_fields
    extend_fields = _extend_provider_fields
    logger = logger
    getitem_parts = (
        _providerdataset_getitem_263_part_1,
        _providerdataset_getitem_263_part_2,
        _providerdataset_getitem_263_part_3,
        _providerdataset_getitem_263_part_4,
    )
    continue_marker = _HELPER_CONTINUE

    def generate_labels(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _PROVIDERDATASET_GENERATE_LABELS_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        center, dimension, angle, ref_xyz, p = bound_parameters
        return generate_point_labels(
            center, dimension, angle, ref_xyz, (0.5, 1.0),
            compute_box_3d, extract_pc_in_box3d,
        )

    def generate_ref(self, box, p):

        s1, s2, s3, s4 = cfg.DATA.STRIDE
        max_depth = cfg.DATA.MAX_DEPTH

        z1 = np.arange(0, max_depth, s1) + s1 / 2.0
        z2 = np.arange(0, max_depth, s2) + s2 / 2.0
        z3 = np.arange(0, max_depth, s3) + s3 / 2.0
        z4 = np.arange(0, max_depth, s4) + s4 / 2.0

        cx, cy = (
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0,
        )

        xyz1 = np.zeros((len(z1), 3))
        xyz1[:, 0] = cx
        xyz1[:, 1] = cy
        xyz1[:, 2] = z1
        xyz1_rect = project_image_to_rect(xyz1, p)

        xyz2 = np.zeros((len(z2), 3))
        xyz2[:, 0] = cx
        xyz2[:, 1] = cy
        xyz2[:, 2] = z2
        xyz2_rect = project_image_to_rect(xyz2, p)

        xyz3 = np.zeros((len(z3), 3))
        xyz3[:, 0] = cx
        xyz3[:, 1] = cy
        xyz3[:, 2] = z3
        xyz3_rect = project_image_to_rect(xyz3, p)

        xyz4 = np.zeros((len(z4), 3))
        xyz4[:, 0] = cx
        xyz4[:, 1] = cy
        xyz4[:, 2] = z4
        xyz4_rect = project_image_to_rect(xyz4, p)

        return xyz1_rect, xyz2_rect, xyz3_rect, xyz4_rect


def from_prediction_to_label_format(center, angle, size, rot_angle, ref_center=None):
    """Convert predicted box parameters to label format."""
    length_value, w, h = size
    ry = angle + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()

    if ref_center is not None:
        tx = tx + ref_center[0]
        ty = ty + ref_center[1]
        tz = tz + ref_center[2]

    ty += h / 2.0
    return LabelFormatResult(h, w, length_value, tx, ty, tz, ry)


def compute_alpha(x, z, ry):

    beta = np.arctan2(z, x)
    alpha = -np.sign(beta) * np.pi / 2 + beta + ry

    return alpha


def collate_fn(batch):
    return default_collate(batch)


if __name__ == "__main__":

    cfg.DATA.DATA_ROOT = "kitti/data/pickle_data"
    cfg.DATA.RTC = True
    dataset = ProviderDataset(
        1024, split="val", random_flip=True, one_hot=True, random_shift=True
    )

    for i, _ in enumerate(dataset):
        data = dataset[i]

        for name, value in data.items():
            LOGGER.info("%s %s", name, value.shape)

        input()
