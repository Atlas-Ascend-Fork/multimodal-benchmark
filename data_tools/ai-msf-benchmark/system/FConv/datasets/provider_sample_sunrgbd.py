# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import os
import pickle
import sys

from configs.config import cfg
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np

from datasets.data_utils import rotate_pc_along_y, compute_box_3d, extract_pc_in_box3d
from datasets.dataset_info import SUNRGBDCategory
from shared.parameters import bind_parameters as _bind_parameters
from datasets.provider_common import (
    LabelFormatResult,
    PROVIDER_DATASET_INIT_SCHEMA,
    assign_provider_options,
    augment_provider_sample,
    generate_point_labels,
    initialize_provider_sample,
    prepare_provider_training_state,
    provider_inputs,
)
from datasets.provider_geometry import CenterViewMixin

_HELPER_CONTINUE = object()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


LOGGER = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def project_image_to_camera(uv_depth, k):
    n = uv_depth.shape[0]

    c_u, c_v = k[0, 2], k[1, 2]
    f_u, f_v = k[0, 0], k[1, 1]

    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v
    pts_3d_camera = np.zeros((n, 3))
    pts_3d_camera[:, 0] = x
    pts_3d_camera[:, 1] = y
    pts_3d_camera[:, 2] = uv_depth[:, 2]
    return pts_3d_camera


def project_image_to_upright_camera(uv_depth, k, rtilt):

    pts_3d_camera = project_image_to_camera(uv_depth, k)

    # X, Y, Z -> X, Z, -Y
    pts_3d_depth = pts_3d_camera[:, [0, 2, 1]] * np.array([1, 1, -1])

    pts_3d_upright_depth = np.transpose(np.dot(rtilt, np.transpose(pts_3d_depth)))

    # X, Y, Z -> X, -Z, Y
    pts_3d = pts_3d_upright_depth[:, [0, 2, 1]] * np.array([1, -1, 1])

    return pts_3d


def _providerdataset_getitem_226_part_1(_state):
    initialize_provider_sample(_state, cfg, SUNRGBDCategory.CLASSES, np.arange)
    _state["k"] = _state["self"].calib_k_list[_state["index"]]
    _state["r"] = _state["self"].calib_r_list[_state["index"]]
    _state["ref1"], _state["ref2"], _state["ref3"], _state["ref4"], _state["ref5"] = (
        _state["self"].generate_ref(_state["box"], _state["k"], _state["r"])
    )


def _providerdataset_getitem_226_part_2(_state):
    if _state["rotate_to_center"]:
        _state["ref1"] = _state["self"].get_center_view(_state["ref1"], _state["index"])
        _state["ref2"] = _state["self"].get_center_view(_state["ref2"], _state["index"])
        _state["ref3"] = _state["self"].get_center_view(_state["ref3"], _state["index"])
        _state["ref4"] = _state["self"].get_center_view(_state["ref4"], _state["index"])
        _state["ref5"] = _state["self"].get_center_view(_state["ref5"], _state["index"])
    if _state["self"].from_rgb_detection:
        _state["data_inputs"] = provider_inputs(
            _state,
            ("ref1", "ref2", "ref3", "ref4", "ref5"),
            rgb_detection=True,
        )
        return _state["data_inputs"]
    prepare_provider_training_state(_state)
    return _HELPER_CONTINUE


def _providerdataset_getitem_226_part_3(_state):
    augment_provider_sample(
        _state,
        ("ref1", "ref2", "ref3", "ref4", "ref5"),
        cfg.DATA.MAX_DEPTH,
        shift_height=True,
    )
    _state["labels_ref2"] = _state["self"].generate_labels(
        _state["box3d_center"],
        _state["self"].size_list[_state["index"]],
        _state["heading_angle"],
        _state["ref2"],
    )


def _providerdataset_getitem_226_part_4(_state):
    _state["data_inputs"] = provider_inputs(
        _state, ("ref1", "ref2", "ref3", "ref4", "ref5")
    )


def _assign_provider_fields(provider, data_dict, from_rgb_detection):
    common = {
        "id_list": "id", "input_list": "input", "box2d_list": "box2d",
        "type_list": "type", "frustum_angle_list": "frustum_angle",
        "calib_k_list": "calib_K", "calib_r_list": "calib_R",
    }
    detection = {"prob_list": "box2d_prob"}
    training = {
        "box3d_list": "box3d", "label_list": "label",
        "heading_list": "box3d_heading", "size_list": "box3d_size",
    }
    fields = dict(common)
    fields.update(detection if from_rgb_detection else training)
    for attribute, key in fields.items():
        setattr(provider, attribute, data_dict[key])


class ProviderDataset(CenterViewMixin, Dataset):
    rotate_points = staticmethod(rotate_pc_along_y)


    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            PROVIDER_DATASET_INIT_SCHEMA, call_args, call_kwargs
        )
        super(ProviderDataset, self).__init__()
        assign_provider_options(self, bound_parameters)

        split = bound_parameters.split
        from_rgb_detection = bound_parameters.from_rgb_detection
        overwritten_data_path = bound_parameters.overwritten_data_path
        if split not in ("train", "val"):
            raise AssertionError
        if not from_rgb_detection:
            suffix = "_aug5x" if split == "train" else ""
            filename = "sunrgbd_{}{}.pickle".format(split, suffix)
            overwritten_data_path = os.path.join(cfg.DATA.DATA_ROOT, filename)
        with open(overwritten_data_path, "rb") as f:
            data_dict = pickle.load(f)
        _assign_provider_fields(self, data_dict, from_rgb_detection)
        logger.info("load dataset from {}".format(overwritten_data_path))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):

        _state = locals()
        _providerdataset_getitem_226_part_1(_state)
        _helper_result = _providerdataset_getitem_226_part_2(_state)
        if _helper_result is not _HELPER_CONTINUE:
            return _helper_result
        _providerdataset_getitem_226_part_3(_state)
        _providerdataset_getitem_226_part_4(_state)

        return _state["data_inputs"]

    def generate_labels(self, center, dimension, angle, ref_xyz):
        box_corner1 = compute_box_3d(center, dimension * 0.5, angle)
        box_corner2 = compute_box_3d(center, dimension, angle)

        labels = np.zeros(len(ref_xyz))
        inside1 = extract_pc_in_box3d(ref_xyz, box_corner1)
        inside2 = extract_pc_in_box3d(ref_xyz, box_corner2)

        labels[inside2] = -1
        labels[inside1] = 1

        if inside1.sum() == 0:
            dis = np.sqrt(((ref_xyz - center) ** 2).sum(1))
            argmin = np.argmin(dis)
            labels[argmin] = 1
        return labels

    def generate_ref(self, box, k, r):

        s1, s2, s3, s4, s5 = cfg.DATA.STRIDE
        max_depth = cfg.DATA.MAX_DEPTH

        z1 = np.arange(0, max_depth, s1) + s1 / 2.0
        z2 = np.arange(0, max_depth, s2) + s2 / 2.0
        z3 = np.arange(0, max_depth, s3) + s3 / 2.0
        z4 = np.arange(0, max_depth, s4) + s4 / 2.0
        z5 = np.arange(0, max_depth, s5) + s5 / 2.0

        cx, cy = (
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0,
        )

        xyz1 = np.zeros((len(z1), 3))
        xyz1[:, 0] = cx
        xyz1[:, 1] = cy
        xyz1[:, 2] = z1
        xyz1_rect = project_image_to_upright_camera(xyz1, k, r)

        xyz2 = np.zeros((len(z2), 3))
        xyz2[:, 0] = cx
        xyz2[:, 1] = cy
        xyz2[:, 2] = z2
        xyz2_rect = project_image_to_upright_camera(xyz2, k, r)

        xyz3 = np.zeros((len(z3), 3))
        xyz3[:, 0] = cx
        xyz3[:, 1] = cy
        xyz3[:, 2] = z3
        xyz3_rect = project_image_to_upright_camera(xyz3, k, r)

        xyz4 = np.zeros((len(z4), 3))
        xyz4[:, 0] = cx
        xyz4[:, 1] = cy
        xyz4[:, 2] = z4
        xyz4_rect = project_image_to_upright_camera(xyz4, k, r)

        xyz5 = np.zeros((len(z5), 3))
        xyz5[:, 0] = cx
        xyz5[:, 1] = cy
        xyz5[:, 2] = z5
        xyz5_rect = project_image_to_upright_camera(xyz5, k, r)

        return xyz1_rect, xyz2_rect, xyz3_rect, xyz4_rect, xyz5_rect


def from_prediction_to_label_format(center, angle, size, rot_angle, ref_center=None):
    """Convert predicted box parameters to label format."""
    length_value, w, h = size
    ry = angle + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()

    if ref_center is not None:
        tx = tx + ref_center[0]
        ty = ty + ref_center[1]
        tz = tz + ref_center[2]

    return LabelFormatResult(tx, ty, tz, length_value, w, h, ry)


def collate_fn(batch):
    return default_collate(batch)


if __name__ == "__main__":

    cfg.DATA.DATA_ROOT = "./sunrgbd/data/pickle_data"
    cfg.DATA.RTC = True
    cfg.DATA.STRIDE = (0.05, 0.1, 0.2, 0.4, 0.8)
    dataset = ProviderDataset(
        1024, split="val", random_flip=False, one_hot=True, random_shift=False
    )

    for i, _ in enumerate(dataset):
        data = dataset[i]

        for name, value in data.items():
            LOGGER.info("%s %s", name, value.shape)

        input()
