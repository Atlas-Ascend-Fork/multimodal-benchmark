# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Refined Frustum PointNet sample provider."""

from typing import NamedTuple
import logging
import os
import pickle
import sys

from configs.config import cfg
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch

from datasets.data_utils import rotate_pc_along_y, extract_pc_in_box3d, compute_box_3d
from datasets.dataset_info import DATASET_INFO
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters
from datasets.provider_common import LabelFormatResult, ProviderDatasetBase, generate_point_labels

_HELPER_CONTINUE = object()


class ProviderdatasetNormalizeInputParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    pc: object
    pred_box_center: object
    pred_box_angle: object
    gt_box_center: object
    gt_box_angle: object


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


LOGGER = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


_PROVIDERDATASET_NORMALIZE_INPUT_PARAMETERS_SCHEMA = _ParameterSchema(
    ProviderdatasetNormalizeInputParameters,
    ("pc", "pred_box_center", "pred_box_angle", "gt_box_center", "gt_box_angle"),
    {"gt_box_center": None, "gt_box_angle": None},
    frozenset(()),
    5,
    None,
    None,
)


def _providerdataset_getitem_278_part_1(_state):
    _state["rotate_to_center"] = cfg.DATA.RTC
    _state["with_extra_feat"] = cfg.DATA.WITH_EXTRA_FEAT
    _state["point_set"] = _state["self"].input_list[_state["index"]].copy()
    _state["pred_box3d"] = _state["self"].pred_box3d_list[_state["index"]].copy()
    _state["pred_box3d_center"] = (
        _state["pred_box3d"][0, :] + _state["pred_box3d"][6, :]
    ) / 2
    _state["pred_box3d_angle"] = _state["self"].pred_box3d_angle_list[_state["index"]]
    _state["pred_box3d_size"] = (
        _state["self"].pred_box3d_size_list[_state["index"]].copy()
    )
    _state["cls_type"] = _state["self"].type_list[_state["index"]]
    if not _state["cls_type"] in _state["self"].category_info.CLASSES:
        raise AssertionError("%s not in category_info" % _state["cls_type"])
    _state["size_class"] = _state["self"].category_info.CLASSES.index(
        _state["cls_type"]
    )
    if _state["self"].one_hot:
        _state["one_hot_vec"] = np.zeros(len(_state["self"].category_info.CLASSES))
        _state["one_hot_vec"][_state["size_class"]] = 1
    if _state["rotate_to_center"]:
        _state["point_set"][:, :3] = _state["self"].get_center_view_point(
            _state["point_set"][:, :3],
            _state["pred_box3d_center"],
            _state["pred_box3d_angle"],
        )
    if not _state["with_extra_feat"]:
        _state["point_set"] = _state["point_set"][:, :3]
    if _state["self"].npoints > 0:
        _state["choice"] = np.random.choice(
            _state["point_set"].shape[0],
            _state["self"].npoints,
            _state["point_set"].shape[0] < _state["self"].npoints,
        )
    else:
        _state["choice"] = np.random.permutation(len(_state["point_set"].shape[0]))
    _state["point_set"] = _state["point_set"][_state["choice"], :]
    if _state["rotate_to_center"]:
        _state["pred_box3d_center_rot"], _state["pred_box3d_angle_rot"] = _state[
            "self"
        ].get_center_view_box3d(
            _state["pred_box3d_center"],
            _state["pred_box3d_angle"],
            _state["pred_box3d_center"],
            _state["pred_box3d_angle"],
        )
    else:
        _state["pred_box3d_center_rot"] = _state["pred_box3d_center"]
        _state["pred_box3d_angle_rot"] = _state["pred_box3d_angle"]


def _providerdataset_getitem_278_part_2(_state):
    _state["pred_box3d"] = compute_box_3d(
        _state["pred_box3d_center_rot"],
        _state["pred_box3d_size"],
        _state["pred_box3d_angle_rot"],
    )
    _state["ref1"], _state["ref2"], _state["ref3"], _state["ref4"] = _state[
        "self"
    ].generate_ref(_state["pred_box3d"])
    if _state["self"].from_rgb_detection:
        _state["data_inputs"] = {
            "point_cloud": np.transpose(_state["point_set"], (1, 0)).astype(np.float32),
            "center_ref1": np.transpose(_state["ref1"], (1, 0)).astype(np.float32),
            "center_ref2": np.transpose(_state["ref2"], (1, 0)).astype(np.float32),
            "center_ref3": np.transpose(_state["ref3"], (1, 0)).astype(np.float32),
            "center_ref4": np.transpose(_state["ref4"], (1, 0)).astype(np.float32),
            "rgb_prob": np.array([_state["self"].prob_list[_state["index"]]]).astype(
                np.float32
            ),
            "rot_angle": np.array([_state["pred_box3d_angle"]]).astype(np.float32),
            "ref_center": _state["pred_box3d_center"].astype(np.float32),
        }
        if not _state["rotate_to_center"]:
            _state["data_inputs"].update({"rot_angle": torch.zeros(1)})
            _state["data_inputs"].update({"ref_center": torch.zeros(3)})
        if _state["self"].one_hot:
            _state["data_inputs"].update(
                {"one_hot": torch.FloatTensor(_state["one_hot_vec"])}
            )
        return _state["data_inputs"]
    _state["box3d"] = _state["self"].box3d_list[_state["index"]].copy()
    _state["heading_angle"] = _state["self"].heading_list[_state["index"]]
    _state["box3d_size"] = _state["self"].size_list[_state["index"]].copy()
    _state["box3d_center"] = (_state["box3d"][0, :] + _state["box3d"][6, :]) / 2
    if _state["rotate_to_center"]:
        _state["box3d_center"], _state["heading_angle"] = _state[
            "self"
        ].get_center_view_box3d(
            _state["box3d_center"],
            _state["heading_angle"],
            _state["pred_box3d_center"],
            _state["pred_box3d_angle"],
        )
    return _HELPER_CONTINUE


def _providerdataset_getitem_278_part_3(_state):
    if _state["self"].random_flip:
        if np.random.random() > 0.5:
            _state["point_set"][:, 0] *= -1
            _state["box3d_center"][0] *= -1
            _state["heading_angle"] = np.pi - _state["heading_angle"]
            _state["ref1"][:, 0] *= -1
            _state["ref2"][:, 0] *= -1
            _state["ref3"][:, 0] *= -1
            _state["ref4"][:, 0] *= -1
    if _state["self"].random_shift:
        _state["s1"] = cfg.DATA.STRIDE[0]
        _state["length_value"], _state["w"], _state["h"] = _state["self"].size_list[
            _state["index"]
        ]
        _state["dist"] = np.sqrt(np.sum(_state["length_value"] ** 2 + _state["w"] ** 2))
        _state["shift"] = np.clip(
            np.random.randn() * _state["dist"] * 0.1,
            -_state["s1"] * 2,
            2 * _state["s1"],
        )
        _state["point_set"][:, 2] += _state["shift"]
        _state["box3d_center"][2] += _state["shift"]
    _state["labels"] = _state["self"].generate_labels(
        _state["box3d_center"],
        _state["box3d_size"],
        _state["heading_angle"],
        _state["ref2"],
    )


def _providerdataset_getitem_278_part_4(_state):
    _state["data_inputs"] = {
        "point_cloud": np.transpose(_state["point_set"], (1, 0)).astype(np.float32),
        "cls_label": _state["labels"].astype(np.int64),
        "box3d_center": _state["box3d_center"].astype(np.float32),
        "box3d_heading": np.array([_state["heading_angle"]], dtype=np.float32),
        "box3d_size": _state["box3d_size"].astype(np.float32),
        "size_class": np.array([_state["size_class"]], dtype=np.int64),
        "center_ref1": np.transpose(_state["ref1"], (1, 0)).astype(np.float32),
        "center_ref2": np.transpose(_state["ref2"], (1, 0)).astype(np.float32),
        "center_ref3": np.transpose(_state["ref3"], (1, 0)).astype(np.float32),
        "center_ref4": np.transpose(_state["ref4"], (1, 0)).astype(np.float32),
        "rot_angle": np.array([_state["pred_box3d_angle"]]).astype(np.float32),
        "ref_center": _state["pred_box3d_center"].astype(np.float32),
    }
    if not _state["rotate_to_center"]:
        _state["data_inputs"].update({"rot_angle": torch.zeros(1)})
    if _state["self"].one_hot:
        _state["data_inputs"].update(
            {"one_hot": torch.FloatTensor(_state["one_hot_vec"])}
        )


def _provider_data_path(split, from_rgb_detection, overwritten_data_path):
    if overwritten_data_path:
        return overwritten_data_path
    if not from_rgb_detection and split == "val":
        split += "_det"
    category = "caronly" if cfg.DATA.CAR_ONLY else "pedcyc"
    if not (cfg.DATA.CAR_ONLY or cfg.DATA.PEOPLE_ONLY):
        category = "carpedcyc"
    suffix = "_rgb_detection_refine" if from_rgb_detection else ""
    filename = "frustum_{}_{}{}.pickle".format(category, split, suffix)
    return os.path.join(cfg.DATA.DATA_ROOT, filename)


def _load_provider_fields(provider, data_path, from_rgb_detection):
    detection_fields = (
        "id", "box2d", "input", "type", "frustum_angle", "prob", "calib",
        "pred_box3d", "pred_box3d_size", "pred_box3d_angle",
    )
    training_fields = (
        "id", "box3d", "input", "label", "type", "heading", "size",
        "frustum_angle", "box2d", "calib", "pred_box3d",
        "pred_box3d_size", "pred_box3d_angle",
    )
    fields = detection_fields if from_rgb_detection else training_fields
    with open(data_path, "rb") as fp:
        for field in fields:
            setattr(provider, field + "_list", pickle.load(fp))


def _extend_provider_fields(provider, data_path):
    fields = (
        "id", "box3d", "input", "label", "type", "heading", "size",
        "frustum_angle", "box2d", "calib", "pred_box3d",
        "pred_box3d_size", "pred_box3d_angle",
    )
    extend_path = data_path.replace(".", "_det.")
    if not os.path.exists(extend_path):
        raise AssertionError(extend_path)
    with open(extend_path, "rb") as fp:
        for field in fields:
            getattr(provider, field + "_list").extend(pickle.load(fp))
    logger.info("load dataset from {}".format(extend_path))


class ProviderDataset(ProviderDatasetBase):
    config = cfg
    dataset_info = DATASET_INFO
    resolve_data_path = _provider_data_path
    load_fields = _load_provider_fields
    extend_fields = _extend_provider_fields
    logger = logger
    getitem_parts = (
        _providerdataset_getitem_278_part_1,
        _providerdataset_getitem_278_part_2,
        _providerdataset_getitem_278_part_3,
        _providerdataset_getitem_278_part_4,
    )
    continue_marker = _HELPER_CONTINUE

    def get_center_view_box3d(self, box3d_center, box3d_angle, ref_center, ref_angle):

        box3d_center = box3d_center - ref_center
        box3d_angle = box3d_angle - ref_angle
        box3d_center = rotate_pc_along_y(
            box3d_center[np.newaxis, :], ref_angle
        ).squeeze(0)

        return box3d_center, box3d_angle

    def get_center_view_point(self, point, ref_center, ref_angle):

        point = point - ref_center
        point = rotate_pc_along_y(point, ref_angle)

        return point

    def normalize_input(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _PROVIDERDATASET_NORMALIZE_INPUT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        pc, pred_box_center, pred_box_angle, gt_box_center, gt_box_angle = (
            bound_parameters
        )
        ref_center = pred_box_center
        rot_angle = pred_box_angle

        pc = pc - ref_center
        pc = rotate_pc_along_y(pc, rot_angle)
        if gt_box_center is not None and gt_box_angle is not None:
            gt_box_center = gt_box_center - ref_center
            gt_box_angle = gt_box_angle - pred_box_angle
            gt_box_center = rotate_pc_along_y(
                gt_box_center[np.newaxis, :], rot_angle
            ).squeeze(0)

        return pc, gt_box_center, gt_box_angle

    def generate_labels(self, center, dimension, angle, ref_xyz):
        return generate_point_labels(
            center, dimension, angle, ref_xyz, (0.3, 0.6),
            compute_box_3d, extract_pc_in_box3d,
        )

    def generate_ref(self, pred_box3d):
        # 8*3
        s1, s2, s3, s4 = cfg.DATA.STRIDE

        cz = ((pred_box3d[0, :] + pred_box3d[6, :]) / 2)[2]
        z1 = np.min(pred_box3d[:, 2], 0)
        z2 = np.max(pred_box3d[:, 2], 0)
        front = pred_box3d[:, 2] < cz
        below = pred_box3d[:, 2] > cz

        cxcycz1 = np.mean(pred_box3d[front], 0)
        cxcycz2 = np.mean(pred_box3d[below], 0)

        delta = cxcycz2 - cxcycz1

        cz1 = np.arange(z1, z2, s1) + s1 / 2.0
        cz2 = np.arange(z1, z2, s2) + s2 / 2.0
        cz3 = np.arange(z1, z2, s3) + s3 / 2.0
        cz4 = np.arange(z1, z2, s4) + s4 / 2.0

        # line equation
        cx1 = (cz1 - cxcycz1[2]) / delta[2] * (delta[0]) + cxcycz1[0]
        cy1 = (cz1 - cxcycz1[2]) / delta[2] * (delta[1]) + cxcycz1[1]
        xyz1 = np.zeros((len(cz1), 3))
        xyz1[:, 0] = cx1
        xyz1[:, 1] = cy1
        xyz1[:, 2] = cz1

        cx2 = (cz2 - cxcycz1[2]) / delta[2] * (delta[0]) + cxcycz1[0]
        cy2 = (cz2 - cxcycz1[2]) / delta[2] * (delta[1]) + cxcycz1[1]
        xyz2 = np.zeros((len(cz2), 3))
        xyz2[:, 0] = cx2
        xyz2[:, 1] = cy2
        xyz2[:, 2] = cz2

        cx3 = (cz3 - cxcycz1[2]) / delta[2] * (delta[0]) + cxcycz1[0]
        cy3 = (cz3 - cxcycz1[2]) / delta[2] * (delta[1]) + cxcycz1[1]
        xyz3 = np.zeros((len(cz3), 3))
        xyz3[:, 0] = cx3
        xyz3[:, 1] = cy3
        xyz3[:, 2] = cz3

        cx4 = (cz4 - cxcycz1[2]) / delta[2] * (delta[0]) + cxcycz1[0]
        cy4 = (cz4 - cxcycz1[2]) / delta[2] * (delta[1]) + cxcycz1[1]
        xyz4 = np.zeros((len(cz4), 3))
        xyz4[:, 0] = cx4
        xyz4[:, 1] = cy4
        xyz4[:, 2] = cz4

        return xyz1, xyz2, xyz3, xyz4


def _field_length(item, key):
    if key not in item:
        return 0
    if key == "cls_label":
        if item[key].ndim != 1:
            raise ValueError("cls_label must be one-dimensional")
        return item[key].shape[0]
    return item[key].shape[1]


def _pad_field(item, key, target_length):
    length = _field_length(item, key)
    if length == 0 or length >= target_length:
        return
    pad = target_length - length
    padding = (0, pad) if key == "cls_label" else ((0, 0), (0, pad))
    item[key] = np.pad(item[key], padding, mode="edge")


def collate_fn(batch):
    names = ["center_ref1", "center_ref2", "center_ref3", "center_ref4", "cls_label"]
    max_l = {key: max(_field_length(item, key) for item in batch) for key in names}
    for item in batch:
        for key in names:
            _pad_field(item, key, max_l[key])

    return default_collate(batch)


def from_prediction_to_label_format(center, angle, size, rot_angle, ref_center):
    """Convert predicted box parameters to label format."""
    length_value, w, h = size
    ry = angle + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
    tx = tx + ref_center[0]
    ty = ty + ref_center[1]
    tz = tz + ref_center[2]
    ty += h / 2.0
    return LabelFormatResult(h, w, length_value, tx, ty, tz, ry)


def compute_alpha(box_lidar_center, ry):
    # box_lidar_center: (x, y, z) lidar coordinate
    # ry: estimated box orientation
    return -np.arctan2(-box_lidar_center[1], box_lidar_center[0]) + ry


if __name__ == "__main__":

    cfg.DATA.DATA_ROOT = "kitti/data/pickle_data_refine"
    cfg.DATA.RTC = True
    cfg.DATA.CAR_ONLY = True
    cfg.DATA.PEOPLE_ONLY = False

    dataset = ProviderDataset(
        512, split="val", random_flip=True, one_hot=True, random_shift=True
    )
    cfg.DATA.STRIDE = (0.1, 0.2, 0.4, 0.8)

    for i, _ in enumerate(dataset):
        data = dataset[i]

        for name, value in data.items():
            LOGGER.info("%s %s", name, value.shape)

        input()
