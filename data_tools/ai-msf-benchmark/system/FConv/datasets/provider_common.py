# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Parameter and result types for FConv dataset providers."""

from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from shared.parameters import ParameterSchema, bind_parameters
from shared.parameters import bind_named as _bind_named


class ProviderDatasetInitParameters(NamedTuple):
    """Parameters shared by the standard and refinement providers."""

    npoints: object
    split: object
    random_flip: object
    random_shift: object
    one_hot: object
    from_rgb_detection: object
    overwritten_data_path: object
    extend_from_det: object


class LabelFormatResult(NamedTuple):
    """Parsed KITTI label fields."""

    value_0: object
    value_1: object
    value_2: object
    value_3: object
    value_4: object
    value_5: object
    value_6: object


def assign_provider_options(provider, parameters):
    """Copy common dataset options from a bound parameter record."""
    provider.npoints = parameters.npoints
    provider.split = parameters.split
    provider.random_flip = parameters.random_flip
    provider.random_shift = parameters.random_shift
    provider.one_hot = parameters.one_hot
    provider.from_rgb_detection = parameters.from_rgb_detection


PROVIDER_DATASET_INIT_SCHEMA = ParameterSchema(
    ProviderDatasetInitParameters,
    (
        "npoints",
        "split",
        "random_flip",
        "random_shift",
        "one_hot",
        "from_rgb_detection",
        "overwritten_data_path",
        "extend_from_det",
    ),
    {
        "random_flip": False,
        "random_shift": False,
        "one_hot": True,
        "from_rgb_detection": False,
        "overwritten_data_path": "",
        "extend_from_det": False,
    },
    frozenset(),
    8,
    None,
    None,
)


def initialize_provider_sample(
    state, config, classes, full_set_choice, invalid_message=None
):
    provider = state["self"]
    index = state["index"]
    state["rotate_to_center"] = config.DATA.RTC
    state["with_extra_feat"] = config.DATA.WITH_EXTRA_FEAT
    state["rot_angle"] = provider.get_center_view_rot_angle(index)
    state["cls_type"] = provider.type_list[index]
    if state["cls_type"] not in classes:
        message = state["cls_type"]
        if invalid_message is not None:
            message = invalid_message(state["cls_type"])
        raise AssertionError(message)
    state["size_class"] = classes.index(state["cls_type"])
    if provider.one_hot:
        state["one_hot_vec"] = np.zeros(len(classes))
        state["one_hot_vec"][state["size_class"]] = 1
    state["point_set"] = (
        provider.get_center_view_point_set(index)
        if state["rotate_to_center"]
        else provider.input_list[index]
    )
    if not state["with_extra_feat"]:
        state["point_set"] = state["point_set"][:, :3]
    point_count = state["point_set"].shape[0]
    if provider.npoints > 0:
        state["choice"] = np.random.choice(
            point_count, provider.npoints, point_count < provider.npoints
        )
    else:
        state["choice"] = full_set_choice(point_count)
    state["point_set"] = state["point_set"][state["choice"], :]
    state["box"] = provider.box2d_list[index]


def prepare_provider_training_state(state):
    provider = state["self"]
    index = state["index"]
    state["seg"] = provider.label_list[index].astype(np.int64)[state["choice"]]
    state["box3d_center"] = (
        provider.get_center_view_box3d_center(index)
        if state["rotate_to_center"]
        else provider.get_box3d_center(index)
    )
    state["heading_angle"] = provider.heading_list[index]
    if state["rotate_to_center"]:
        state["heading_angle"] -= state["rot_angle"]
    state["box3d_size"] = provider.size_list[index]


def augment_provider_sample(state, reference_names, max_depth, shift_height=False):
    provider = state["self"]
    if provider.random_flip and np.random.random() > 0.5:
        state["point_set"][:, 0] *= -1
        state["box3d_center"][0] *= -1
        state["heading_angle"] = np.pi - state["heading_angle"]
        for name in reference_names:
            state[name][:, 0] *= -1
    if not provider.random_shift:
        return
    length_value, width, _ = provider.size_list[state["index"]]
    distance = np.sqrt(np.sum(length_value ** 2 + width ** 2))
    shift = np.clip(
        np.random.randn() * distance * 0.2, -0.5 * distance, 0.5 * distance
    )
    shift = (
        np.clip(shift + state["box3d_center"][2], 0, max_depth)
        - state["box3d_center"][2]
    )
    state["point_set"][:, 2] += shift
    state["box3d_center"][2] += shift
    if shift_height:
        height_shift = np.random.random() * 0.4 - 0.2
        state["point_set"][:, 1] += height_shift
        state["box3d_center"][1] += height_shift


def provider_inputs(state, reference_names, rgb_detection=False):
    provider = state["self"]
    result = {
        "point_cloud": torch.FloatTensor(state["point_set"]).transpose(1, 0),
        "rot_angle": torch.FloatTensor([state["rot_angle"]]),
    }
    result.update(
        {
            "center_ref{}".format(index + 1): torch.FloatTensor(state[name]).transpose(1, 0)
            for index, name in enumerate(reference_names)
        }
    )
    if rgb_detection:
        result["rgb_prob"] = torch.FloatTensor([provider.prob_list[state["index"]]])
    else:
        result.update(
            {
                "cls_label": torch.LongTensor(state["labels_ref2"]),
                "box3d_center": torch.FloatTensor(state["box3d_center"]),
                "box3d_heading": torch.FloatTensor([state["heading_angle"]]),
                "box3d_size": torch.FloatTensor(state["box3d_size"]),
                "size_class": torch.LongTensor([state["size_class"]]),
                "seg_label": torch.LongTensor(state["seg"].astype(np.int64)),
            }
        )
    if not state["rotate_to_center"]:
        result["rot_angle"] = torch.zeros(1)
    if provider.one_hot:
        result["one_hot"] = torch.FloatTensor(state["one_hot_vec"])
    return result


def generate_point_labels(*call_args, **call_kwargs):
    (
        center,
        dimensions,
        angle,
        reference_points,
        scales,
        box_corners,
        points_in_box,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('center', 'dimensions', 'angle', 'reference_points', 'scales', 'box_corners', 'points_in_box'),
    )
    inner_box = box_corners(center, dimensions * scales[0], angle)
    outer_box = box_corners(center, dimensions * scales[1], angle)
    labels = np.zeros(len(reference_points))
    inside_inner = points_in_box(reference_points, inner_box)
    inside_outer = points_in_box(reference_points, outer_box)
    labels[inside_outer] = -1
    labels[inside_inner] = 1
    if inside_inner.sum() == 0:
        closest = np.argmin(np.sqrt(((reference_points - center) ** 2).sum(1)))
        labels[closest] = 1
    return labels


class ProviderDatasetBase(Dataset):
    """Shared setup for standard and refinement FConv providers."""

    config = None
    dataset_info = None
    resolve_data_path = None
    load_fields = None
    extend_fields = None
    logger = None
    getitem_parts = ()
    continue_marker = None

    def __init__(self, *call_args, **call_kwargs):
        parameters = bind_parameters(
            PROVIDER_DATASET_INIT_SCHEMA, call_args, call_kwargs
        )
        super().__init__()
        assign_provider_options(self, parameters)

        provider_type = type(self)
        dataset_name = provider_type.config.DATA.DATASET_NAME
        category_info = provider_type.dataset_info.get(dataset_name)
        if category_info is None:
            raise AssertionError
        self.category_info = category_info
        data_path = provider_type.resolve_data_path(
            parameters.split, parameters.from_rgb_detection,
            parameters.overwritten_data_path,
        )
        provider_type.load_fields(self, data_path, parameters.from_rgb_detection)
        if not parameters.from_rgb_detection and parameters.extend_from_det:
            provider_type.extend_fields(self, data_path)
        provider_type.logger.info("load dataset from %s", data_path)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        state = {"self": self, "index": index}
        first, second, third, fourth = type(self).getitem_parts
        first(state)
        result = second(state)
        if result is not type(self).continue_marker:
            return result
        third(state)
        fourth(state)
        return state["data_inputs"]
