# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from collections import OrderedDict
import abc
import logging

import numba
import numpy as np

from second.core import box_np_ops
from second.core.geometry import (
    points_in_convex_polygon_3d_jit,
    points_in_convex_polygon_jit,
)
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class BatchsamplerInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    sampled_list: object
    name: object
    epoch: object
    shuffle: object
    drop_reminder: object


class RandomCropFrustumParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    bboxes: object
    rect: object
    trv2c: object
    p2: object
    max_crop_height: object
    max_crop_width: object


class NoisePerObjectV3Parameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    gt_boxes: object
    points: object
    valid_mask: object
    rotation_perturb: object
    center_noise_std: object
    global_random_rot_range: object
    num_try: object
    group_ids: object


class NoisePerObjectV2Parameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    gt_boxes: object
    points: object
    valid_mask: object
    rotation_perturb: object
    center_noise_std: object
    global_random_rot_range: object
    num_try: object


LOGGER = logging.getLogger(__name__)


_RANDOM_CROP_FRUSTUM_PARAMETERS_SCHEMA = _ParameterSchema(
    RandomCropFrustumParameters,
    ("bboxes", "rect", "trv2c", "p2", "max_crop_height", "max_crop_width"),
    {"max_crop_height": 1.0, "max_crop_width": 0.9},
    frozenset(()),
    6,
    None,
    None,
)
_NOISE_PER_OBJECT_V3_PARAMETERS_SCHEMA = _ParameterSchema(
    NoisePerObjectV3Parameters,
    (
        "gt_boxes",
        "points",
        "valid_mask",
        "rotation_perturb",
        "center_noise_std",
        "global_random_rot_range",
        "num_try",
        "group_ids",
    ),
    {
        "points": None,
        "valid_mask": None,
        "rotation_perturb": np.pi / 4,
        "center_noise_std": 1.0,
        "global_random_rot_range": np.pi / 4,
        "num_try": 100,
        "group_ids": None,
    },
    frozenset(()),
    8,
    None,
    None,
)
_NOISE_PER_OBJECT_V2_PARAMETERS_SCHEMA = _ParameterSchema(
    NoisePerObjectV2Parameters,
    (
        "gt_boxes",
        "points",
        "valid_mask",
        "rotation_perturb",
        "center_noise_std",
        "global_random_rot_range",
        "num_try",
    ),
    {
        "points": None,
        "valid_mask": None,
        "rotation_perturb": np.pi / 4,
        "center_noise_std": 1.0,
        "global_random_rot_range": np.pi / 4,
        "num_try": 100,
    },
    frozenset(()),
    7,
    None,
    None,
)
_BATCHSAMPLER_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    BatchsamplerInitParameters,
    ("sampled_list", "name", "epoch", "shuffle", "drop_reminder"),
    {"name": None, "epoch": None, "shuffle": True, "drop_reminder": False},
    frozenset(()),
    5,
    None,
    None,
)


class BatchSampler:
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _BATCHSAMPLER_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        sampled_list, name, epoch, shuffle, drop_reminder = bound_parameters
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx: self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._name is not None:
            LOGGER.info("%s %s", "reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0


class DataBasePreprocessing:
    def __call__(self, db_infos):
        return self._preprocess(db_infos)

    @abc.abstractmethod
    def _preprocess(self, db_infos):
        pass


class DBFilterByDifficulty(DataBasePreprocessing):
    def __init__(self, removed_difficulties):
        self._removed_difficulties = removed_difficulties
        LOGGER.info("%s", removed_difficulties)

    def _preprocess(self, db_infos):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info
                for info in dinfos
                if info["difficulty"] not in self._removed_difficulties
            ]
        return new_db_infos


class DBFilterByMinNumPoint(DataBasePreprocessing):
    def __init__(self, min_gt_point_dict):
        self._min_gt_point_dict = min_gt_point_dict
        LOGGER.info("%s", min_gt_point_dict)

    def _preprocess(self, db_infos):
        for name, min_num in self._min_gt_point_dict.items():
            if min_num <= 0:
                continue
            db_infos[name] = [
                info for info in db_infos[name]
                if info["num_points_in_gt"] >= min_num
            ]
        return db_infos


class DataBasePreprocessor:
    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def __call__(self, db_infos):
        for prepor in self._preprocessors:
            db_infos = prepor(db_infos)
        return db_infos


def random_crop_frustum(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _RANDOM_CROP_FRUSTUM_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    bboxes, rect, trv2c, p2, max_crop_height, max_crop_width = bound_parameters
    num_gt = bboxes.shape[0]
    crop_minxy = np.random.uniform(
        [1 - max_crop_width, 1 - max_crop_height], [0.3, 0.3], size=[num_gt, 2]
    )
    crop_maxxy = np.ones([num_gt, 2], dtype=bboxes.dtype)
    crop_bboxes = np.concatenate([crop_minxy, crop_maxxy], axis=1)
    left = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if left:
        crop_bboxes[:, [0, 2]] -= crop_bboxes[:, 0:1]
    # crop_relative_bboxes to real bboxes
    crop_bboxes *= np.tile(bboxes[:, 2:] - bboxes[:, :2], [1, 2])
    crop_bboxes += np.tile(bboxes[:, :2], [1, 2])
    c, r, t = box_np_ops.projection_matrix_to_crt_kitti(p2)
    frustums = box_np_ops.get_frustum_v2(crop_bboxes, c)
    frustums -= t
    frustums = np.einsum("ij, akj->aki", np.linalg.inv(r), frustums)
    frustums = box_np_ops.camera_to_lidar(frustums, rect, trv2c)

    return frustums


def filter_gt_box_outside_range(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_boxes_bv = box_np_ops.center_to_corner_box2d(
        gt_boxes[:, [0, 1]], gt_boxes[:, [3, 3 + 1]], gt_boxes[:, 6]
    )
    bounding_box = box_np_ops.minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...]
    )
    ret = points_in_convex_polygon_jit(gt_boxes_bv.reshape(-1, 2), bounding_box)
    return np.any(ret.reshape(-1, 4), axis=1)


def filter_gt_box_outside_range_by_center(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_box_centers = gt_boxes[:, :2]
    bounding_box = box_np_ops.minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...]
    )
    ret = points_in_convex_polygon_jit(gt_box_centers, bounding_box)
    return ret.reshape(-1)


def filter_gt_low_points(gt_boxes, points, num_gt_points, point_num_threshold=2):
    points_mask = np.ones([points.shape[0]], np.bool)
    gt_boxes_mask = np.ones([gt_boxes.shape[0]], np.bool)
    for i, num in enumerate(num_gt_points):
        if num <= point_num_threshold:
            masks = box_np_ops.points_in_rbbox(points, gt_boxes[i: i + 1])
            masks = masks.reshape([-1])
            points_mask &= np.logical_not(masks)
            gt_boxes_mask[i] = False
    return gt_boxes[gt_boxes_mask], points[points_mask]


def remove_points_in_boxes(points, boxes):
    masks = box_np_ops.points_in_rbbox(points, boxes)
    points = points[np.logical_not(masks.any(-1))]
    return points


def remove_points_outside_boxes(points, boxes):
    masks = box_np_ops.points_in_rbbox(points, boxes)
    points = points[masks.any(-1)]
    return points


def mask_points_in_corners(points, box_corners):
    surfaces = box_np_ops.corner_to_surfaces_3d(box_corners)
    mask = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return mask


@numba.njit
def _rotation_matrix_3d_(rot_mat_t, angle, axis):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_t[:] = np.eye(3)
    if axis == 1:
        rot_mat_t[0, 0] = rot_cos
        rot_mat_t[0, 2] = -rot_sin
        rot_mat_t[2, 0] = rot_sin
        rot_mat_t[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_t[0, 0] = rot_cos
        rot_mat_t[0, 1] = -rot_sin
        rot_mat_t[1, 0] = rot_sin
        rot_mat_t[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_t[1, 1] = rot_cos
        rot_mat_t[1, 2] = -rot_sin
        rot_mat_t[2, 1] = rot_sin
        rot_mat_t[2, 2] = rot_cos


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_t):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_t[0, 0] = rot_cos
    rot_mat_t[0, 1] = -rot_sin
    rot_mat_t[1, 0] = rot_sin
    rot_mat_t[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_t


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    for i in range(num_boxes):
        if not valid_mask[i]:
            continue
        for j in range(num_tests):
            current_corners[:] = box_corners[i]
            current_corners -= boxes[i, :2]
            _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_t)
            current_corners += boxes[i, :2] + loc_noises[i, j, :2]
            coll_mat = box_collision_test(
                current_corners.reshape(1, 4, 2), box_corners
            )
            coll_mat[0, i] = False
            if not coll_mat.any():
                success_mask[i] = j
                box_corners[i] = current_corners
                break
    return success_mask


@numba.njit
def noise_per_box_group(boxes, valid_mask, loc_noises, rot_noises, group_nums):
    # WARNING: this function need boxes to be sorted by group id.
    _ = group_nums.shape[0]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    idx = 0
    for num in group_nums:
        if not valid_mask[idx]:
            idx += num
            continue
        for j in range(num_tests):
            for i in range(num):
                current_corners[i] = box_corners[i + idx]
                current_corners[i] -= boxes[i + idx, :2]
                _rotation_box2d_jit_(
                    current_corners[i], rot_noises[idx + i, j], rot_mat_t
                )
                current_corners[i] += boxes[i + idx, :2] + loc_noises[i + idx, j, :2]
            coll_mat = box_collision_test(
                current_corners[:num].reshape(num, 4, 2), box_corners
            )
            for i in range(num):  # remove self-coll
                coll_mat[i, idx: idx + num] = False
            if coll_mat.any():
                continue
            for i in range(num):
                success_mask[i + idx] = j
                box_corners[i + idx] = current_corners[i]
            break
        idx += num
    return success_mask


def _module_noise_per_box_group_v2_487_part_1(_state):
    _state["loc_noises"], _state["rot_noises"], _state["global_rot_noises"] = _state[
        "noises"
    ]
    _state["num_boxes"] = _state["boxes"].shape[0]
    _state["num_tests"] = _state["loc_noises"].shape[1]
    _state["box_corners"] = box_np_ops.box2d_to_corner_jit(_state["boxes"])
    _state["max_group_num"] = _state["group_nums"].max()
    _state["current_box"] = np.zeros((1, 5), dtype=_state["boxes"].dtype)
    _state["current_corners"] = np.zeros(
        (_state["max_group_num"], 4, 2), dtype=_state["boxes"].dtype
    )
    _state["dst_pos"] = np.zeros(
        (_state["max_group_num"], 2), dtype=_state["boxes"].dtype
    )
    _state["current_grot"] = np.zeros(
        (_state["max_group_num"],), dtype=_state["boxes"].dtype
    )
    _state["dst_grot"] = np.zeros(
        (_state["max_group_num"],), dtype=_state["boxes"].dtype
    )
    _state["rot_mat_t"] = np.zeros((2, 2), dtype=_state["boxes"].dtype)
    _state["success_mask"] = -np.ones((_state["num_boxes"],), dtype=np.int64)
    _state["corners_norm"] = np.zeros((4, 2), dtype=_state["boxes"].dtype)
    _state["corners_norm"][1, 1] = 1.0
    _state["corners_norm"][2] = 1.0
    _state["corners_norm"][3, 0] = 1.0
    _state["corners_norm"] -= np.array([0.5, 0.5], dtype=_state["boxes"].dtype)
    _state["corners_norm"] = _state["corners_norm"].reshape(4, 2)
    _state["idx"] = 0


def _prepare_group_noise_corners(_state):
    for _state["i"] in range(_state["num"]):
        box_index = _state["i"] + _state["idx"]
        _state["current_box"][0, :] = _state["boxes"][box_index]
        _state["current_radius"] = np.sqrt(
            _state["current_box"][0, 0] ** 2 + _state["current_box"][0, 1] ** 2
        )
        _state["current_grot"][_state["i"]] = np.arctan2(
            _state["current_box"][0, 0], _state["current_box"][0, 1]
        )
        _state["dst_grot"][_state["i"]] = (
            _state["current_grot"][_state["i"]]
            + _state["global_rot_noises"][box_index, _state["j"]]
        )
        _state["dst_pos"][_state["i"], 0] = _state["current_radius"] * np.sin(
            _state["dst_grot"][_state["i"]]
        )
        _state["dst_pos"][_state["i"], 1] = _state["current_radius"] * np.cos(
            _state["dst_grot"][_state["i"]]
        )
        _state["current_box"][0, :2] = _state["dst_pos"][_state["i"]]
        _state["current_box"][0, -1] += (
            _state["dst_grot"][_state["i"]] - _state["current_grot"][_state["i"]]
        )
        _state["rot_sin"] = np.sin(_state["current_box"][0, -1])
        _state["rot_cos"] = np.cos(_state["current_box"][0, -1])
        _state["rot_mat_t"][:] = (
            (_state["rot_cos"], -_state["rot_sin"]),
            (_state["rot_sin"], _state["rot_cos"]),
        )
        current_corner = (
            _state["current_box"][0, 2:4]
            * _state["corners_norm"] @ _state["rot_mat_t"]
        )
        _rotation_box2d_jit_(
            current_corner,
            _state["rot_noises"][box_index, _state["j"]],
            _state["rot_mat_t"],
        )
        current_corner += _state["current_box"][0, :2]
        current_corner += _state["loc_noises"][box_index, _state["j"], :2]
        _state["current_corners"][_state["i"]] = current_corner


def _module_noise_per_box_group_v2_487_part_2(_state):
    for _state["num"] in _state["group_nums"]:
        if not _state["valid_mask"][_state["idx"]]:
            _state["idx"] += _state["num"]
            continue
        for _state["j"] in range(_state["num_tests"]):
            _prepare_group_noise_corners(_state)
            _state["coll_mat"] = box_collision_test(
                _state["current_corners"][: _state["num"]].reshape(_state["num"], 4, 2),
                _state["box_corners"],
            )
            for _state["i"] in range(_state["num"]):
                _state["coll_mat"][
                    _state["i"], _state["idx"]: _state["idx"] + _state["num"]
                ] = False
            if _state["coll_mat"].any():
                continue
            for _state["i"] in range(_state["num"]):
                _state["success_mask"][_state["i"] + _state["idx"]] = _state["j"]
                _state["box_corners"][_state["i"] + _state["idx"]] = _state[
                    "current_corners"
                ][_state["i"]]
                _state["loc_noises"][
                    _state["i"] + _state["idx"], _state["j"], :2
                ] += (
                    _state["dst_pos"][_state["i"]]
                    - _state["boxes"][_state["i"] + _state["idx"], :2]
                )
                _state["rot_noises"][_state["i"] + _state["idx"], _state["j"]] += (
                    _state["dst_grot"][_state["i"]]
                    - _state["current_grot"][_state["i"]]
                )
            break
        _state["idx"] += _state["num"]


@numba.njit
def noise_per_box_group_v2_(boxes, valid_mask, noises, group_nums):
    _state = locals()
    _module_noise_per_box_group_v2_487_part_1(_state)
    _module_noise_per_box_group_v2_487_part_2(_state)
    return _state["success_mask"]


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises, global_rot_noises):
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_t = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if not valid_mask[i]:
            continue
        for j in range(num_tests):
            current_box[0, :] = boxes[i]
            current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
            current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
            dst_grot = current_grot + global_rot_noises[i, j]
            dst_pos[0] = current_radius * np.sin(dst_grot)
            dst_pos[1] = current_radius * np.cos(dst_grot)
            current_box[0, :2] = dst_pos
            current_box[0, -1] += dst_grot - current_grot

            rot_sin = np.sin(current_box[0, -1])
            rot_cos = np.cos(current_box[0, -1])
            rot_mat_t[0, 0] = rot_cos
            rot_mat_t[0, 1] = -rot_sin
            rot_mat_t[1, 0] = rot_sin
            rot_mat_t[1, 1] = rot_cos
            current_corners[:] = (
                current_box[0, 2:4] * corners_norm @ rot_mat_t + current_box[0, :2]
            )
            current_corners -= current_box[0, :2]
            _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_t)
            current_corners += current_box[0, :2] + loc_noises[i, j, :2]
            coll_mat = box_collision_test(
                current_corners.reshape(1, 4, 2), box_corners
            )
            coll_mat[0, i] = False
            if not coll_mat.any():
                success_mask[i] = j
                box_corners[i] = current_corners
                loc_noises[i, j, :2] += dst_pos - boxes[i, :2]
                rot_noises[i, j] += dst_grot - current_grot
                break
    return success_mask


@numba.njit
def points_transform_(points, centers, point_masks, transforms, valid_mask):
    loc_transform, rot_transform = transforms
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_t = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_t[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j] and point_masks[i, j] == 1:
                points[i, :3] -= centers[j, :3]
                points[i: i + 1, :3] = points[i: i + 1, :3] @ rot_mat_t[j]
                points[i, :3] += centers[j, :3]
                points[i, :3] += loc_transform[j]
                break  # only apply first box's transform


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def _select_transform(transform, indices):
    result = np.zeros((transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@numba.njit
def _group_transform_core(noises, locs, group_center, valid_mask):
    loc_noise, rot_noise, base_rotation = noises
    # loc_noise: [N, M, 3], locs: [N, 3]
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x**2 + y**2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j] + base_rotation[i, j])
                    - np.sin(rot_center + base_rotation[i, j])
                )
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j] + base_rotation[i, j])
                    - np.cos(rot_center + base_rotation[i, j])
                )


@numba.njit
def group_transform_(loc_noise, rot_noise, locs, group_center, valid_mask):
    noises = loc_noise, rot_noise, np.zeros_like(rot_noise)
    _group_transform_core(noises, locs, group_center, valid_mask)


@numba.njit
def group_transform_v2_(noises, locs, group_center, valid_mask):
    loc_noise, rot_noise, group_rotation = noises
    _group_transform_core(noises, locs, group_center, valid_mask)


def set_group_noise_same_(loc_noise, rot_noise, group_ids):
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict.get(group_ids[i])]
        rot_noise[i] = rot_noise[gid_to_index_dict.get(group_ids[i])]


def set_group_noise_same_v2_(loc_noise, rot_noise, grot_noise, group_ids):
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict.get(group_ids[i])]
        rot_noise[i] = rot_noise[gid_to_index_dict.get(group_ids[i])]
        grot_noise[i] = grot_noise[gid_to_index_dict.get(group_ids[i])]


def get_group_center(locs, group_ids):
    num_groups = 0
    group_centers = np.zeros_like(locs)
    group_centers_ret = np.zeros_like(locs)
    group_id_dict = {}
    group_id_num_dict = OrderedDict()
    for i, gid in enumerate(group_ids):
        if gid >= 0:
            if gid in group_id_dict:
                group_centers[group_id_dict[gid]] += locs[i]
                group_id_num_dict[gid] += 1
            else:
                group_id_dict[gid] = num_groups
                num_groups += 1
                group_id_num_dict[gid] = 1
                group_centers[group_id_dict[gid]] = locs[i]
    for i, gid in enumerate(group_ids):
        group_centers_ret[i] = (
            group_centers[group_id_dict[gid]] / group_id_num_dict[gid]
        )
    return group_centers_ret, group_id_num_dict


def _module_noise_per_object_v3_721_part_1_chunk_1(_state):
    _state["bound_parameters"] = _bind_parameters(
        _NOISE_PER_OBJECT_V3_PARAMETERS_SCHEMA,
        _state["call_args"],
        _state["call_kwargs"],
    )
    (
        _state["gt_boxes"],
        _state["points"],
        _state["valid_mask"],
        _state["rotation_perturb"],
        _state["center_noise_std"],
        _state["global_random_rot_range"],
        _state["num_try"],
        _state["group_ids"],
    ) = _state["bound_parameters"]
    _state["num_boxes"] = _state["gt_boxes"].shape[0]
    if not isinstance(_state["rotation_perturb"], (list, tuple, np.ndarray)):
        _state["rotation_perturb"] = [
            -_state["rotation_perturb"],
            _state["rotation_perturb"],
        ]
    if not isinstance(_state["global_random_rot_range"], (list, tuple, np.ndarray)):
        _state["global_random_rot_range"] = [
            -_state["global_random_rot_range"],
            _state["global_random_rot_range"],
        ]
    _state["enable_grot"] = (
        np.abs(
            _state["global_random_rot_range"][0] - _state["global_random_rot_range"][1]
        )
        >= 0.001
    )
    if not isinstance(_state["center_noise_std"], (list, tuple, np.ndarray)):
        _state["center_noise_std"] = [
            _state["center_noise_std"],
            _state["center_noise_std"],
            _state["center_noise_std"],
        ]
    if _state["valid_mask"] is None:
        _state["valid_mask"] = np.ones((_state["num_boxes"],), dtype=np.bool_)
    _state["center_noise_std"] = np.array(
        _state["center_noise_std"], dtype=_state["gt_boxes"].dtype
    )


def _module_noise_per_object_v3_721_part_1_chunk_2(_state):
    _state["loc_noises"] = np.random.normal(
        scale=_state["center_noise_std"],
        size=[_state["num_boxes"], _state["num_try"], 3],
    )
    _state["rot_noises"] = np.random.uniform(
        _state["rotation_perturb"][0],
        _state["rotation_perturb"][1],
        size=[_state["num_boxes"], _state["num_try"]],
    )
    _state["gt_grots"] = np.arctan2(_state["gt_boxes"][:, 0], _state["gt_boxes"][:, 1])
    _state["grot_lowers"] = _state["global_random_rot_range"][0] - _state["gt_grots"]
    _state["grot_uppers"] = _state["global_random_rot_range"][1] - _state["gt_grots"]
    _state["global_rot_noises"] = np.random.uniform(
        _state["grot_lowers"][..., np.newaxis],
        _state["grot_uppers"][..., np.newaxis],
        size=[_state["num_boxes"], _state["num_try"]],
    )


def _module_noise_per_object_v3_721_part_1(_state):
    _module_noise_per_object_v3_721_part_1_chunk_1(_state)
    _module_noise_per_object_v3_721_part_1_chunk_2(_state)


def _module_noise_per_object_v3_721_part_2(_state):
    if _state["group_ids"] is not None:
        if _state["enable_grot"]:
            set_group_noise_same_v2_(
                _state["loc_noises"],
                _state["rot_noises"],
                _state["global_rot_noises"],
                _state["group_ids"],
            )
        else:
            set_group_noise_same_(
                _state["loc_noises"], _state["rot_noises"], _state["group_ids"]
            )
        _state["group_centers"], _state["group_id_num_dict"] = get_group_center(
            _state["gt_boxes"][:, :3], _state["group_ids"]
        )
        if _state["enable_grot"]:
            group_transform_v2_(
                (
                    _state["loc_noises"],
                    _state["rot_noises"],
                    _state["global_rot_noises"],
                ),
                _state["gt_boxes"][:, :3],
                _state["group_centers"],
                _state["valid_mask"],
            )
        else:
            group_transform_(
                _state["loc_noises"],
                _state["rot_noises"],
                _state["gt_boxes"][:, :3],
                _state["group_centers"],
                _state["valid_mask"],
            )
        _state["group_nums"] = np.array(
            list(_state["group_id_num_dict"].values()), dtype=np.int64
        )
    _state["origin"] = [0.5, 0.5, 0]
    _state["gt_box_corners"] = box_np_ops.center_to_corner_box3d(
        _state["gt_boxes"][:, :3],
        _state["gt_boxes"][:, 3:6],
        _state["gt_boxes"][:, 6],
        origin=_state["origin"],
        axis=2,
    )


def _module_noise_per_object_v3_721_part_3_chunk_1(_state):
    if _state["group_ids"] is not None:
        if not _state["enable_grot"]:
            _state["selected_noise"] = noise_per_box_group(
                _state["gt_boxes"][:, [0, 1, 3, 4, 6]],
                _state["valid_mask"],
                _state["loc_noises"],
                _state["rot_noises"],
                _state["group_nums"],
            )
        else:
            _state["selected_noise"] = noise_per_box_group_v2_(
                _state["gt_boxes"][:, [0, 1, 3, 4, 6]],
                _state["valid_mask"],
                (
                    _state["loc_noises"],
                    _state["rot_noises"],
                    _state["global_rot_noises"],
                ),
                _state["group_nums"],
            )
    elif not _state["enable_grot"]:
        _state["selected_noise"] = noise_per_box(
            _state["gt_boxes"][:, [0, 1, 3, 4, 6]],
            _state["valid_mask"],
            _state["loc_noises"],
            _state["rot_noises"],
        )
    else:
        _state["selected_noise"] = noise_per_box_v2_(
            _state["gt_boxes"][:, [0, 1, 3, 4, 6]],
            _state["valid_mask"],
            _state["loc_noises"],
            _state["rot_noises"],
            _state["global_rot_noises"],
        )
    _state["loc_transforms"] = _select_transform(
        _state["loc_noises"], _state["selected_noise"]
    )
    _state["rot_transforms"] = _select_transform(
        _state["rot_noises"], _state["selected_noise"]
    )
    _state["surfaces"] = box_np_ops.corner_to_surfaces_3d_jit(_state["gt_box_corners"])


def _module_noise_per_object_v3_721_part_3_chunk_2(_state):
    if _state["points"] is not None:
        _state["point_masks"] = points_in_convex_polygon_3d_jit(
            _state["points"][:, :3], _state["surfaces"]
        )
        points_transform_(
            _state["points"],
            _state["gt_boxes"][:, :3],
            _state["point_masks"],
            (_state["loc_transforms"], _state["rot_transforms"]),
            _state["valid_mask"],
        )
    box3d_transform_(
        _state["gt_boxes"],
        _state["loc_transforms"],
        _state["rot_transforms"],
        _state["valid_mask"],
    )


def _module_noise_per_object_v3_721_part_3(_state):
    _module_noise_per_object_v3_721_part_3_chunk_1(_state)
    _module_noise_per_object_v3_721_part_3_chunk_2(_state)


def noise_per_object_v3_(*call_args, **call_kwargs):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    _state = locals()
    _module_noise_per_object_v3_721_part_1(_state)
    _module_noise_per_object_v3_721_part_2(_state)
    _module_noise_per_object_v3_721_part_3(_state)


def _object_noise_arrays(gt_boxes, num_try, center_noise_std,
                         rotation_perturb, global_random_rot_range):
    num_boxes = gt_boxes.shape[0]
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try]
    )
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    global_rot_noises = np.random.uniform(
        (global_random_rot_range[0] - gt_grots)[..., np.newaxis],
        (global_random_rot_range[1] - gt_grots)[..., np.newaxis],
        size=[num_boxes, num_try],
    )
    return loc_noises, rot_noises, global_rot_noises


def noise_per_object_v2_(*call_args, **call_kwargs):
    """Randomly rotate or remove each ground-truth box independently."""
    bound_parameters = _bind_parameters(
        _NOISE_PER_OBJECT_V2_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (gt_boxes, points, valid_mask, rotation_perturb, center_noise_std,
     global_random_rot_range, num_try) = bound_parameters
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range, global_random_rot_range]

    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    loc_noises, rot_noises, global_rot_noises = _object_noise_arrays(
        gt_boxes, num_try, center_noise_std, rotation_perturb, global_random_rot_range
    )

    origin = [0.5, 0.5, 0]
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2
    )
    if np.abs(global_random_rot_range[0] - global_random_rot_range[1]) < 1e-3:
        selected_noise = noise_per_box(
            gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises
        )
    else:
        selected_noise = noise_per_box_v2_(
            gt_boxes[:, [0, 1, 3, 4, 6]],
            valid_mask,
            loc_noises,
            rot_noises,
            global_rot_noises,
        )
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    if points is not None:
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(
            points,
            gt_boxes[:, :3],
            point_masks,
            (loc_transforms, rot_transforms),
            valid_mask,
        )

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


def global_scaling(gt_boxes, points, scale=0.05):
    if not isinstance(scale, list):
        scale = [-scale, scale]
    noise_scale = np.random.uniform(scale[0] + 1, scale[1] + 1)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def global_rotation(gt_boxes, points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points


def random_flip(gt_boxes, points, probability=0.5):
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
        points[:, 1] = -points[:, 1]
    return gt_boxes, points


def global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def global_rotation_v2(gt_boxes, points, min_rad=-np.pi / 4, max_rad=np.pi / 4):
    noise_rotation = np.random.uniform(min_rad, max_rad)
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points


@numba.jit(nopython=True)
def _line_sets_intersect(lines, query_lines):
    for line_index in range(4):
        for query_index in range(4):
            a = lines[line_index, 0]
            b = lines[line_index, 1]
            c = query_lines[query_index, 0]
            d = query_lines[query_index, 1]
            acd = (d[1] - a[1]) * (c[0] - a[0]) > (c[1] - a[1]) * (d[0] - a[0])
            bcd = (d[1] - b[1]) * (c[0] - b[0]) > (c[1] - b[1]) * (d[0] - b[0])
            if acd == bcd:
                continue
            cross_abc = (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
            abd = (d[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (d[0] - a[0])
            if cross_abc != abd:
                return True
    return False


@numba.jit(nopython=True)
def _contains_all_points(box, points, clockwise):
    for point_index in range(4):
        for corner_index in range(4):
            vector = box[corner_index] - box[(corner_index + 1) % 4]
            if clockwise:
                vector = -vector
            cross = vector[1] * (box[corner_index, 0] - points[point_index, 0])
            cross -= vector[0] * (box[corner_index, 1] - points[point_index, 1])
            if cross >= 0:
                return False
    return True


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    n = boxes.shape[0]
    k_value = qboxes.shape[0]
    ret = np.zeros((n, k_value), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2
    )  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(n):
        for j in range(k_value):
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if iw <= 0:
                continue
            ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                boxes_standup[i, 1], qboxes_standup[j, 1]
            )
            if ih <= 0:
                continue
            intersects = _line_sets_intersect(lines_boxes[i], lines_qboxes[j])
            box_contains_query = _contains_all_points(boxes[i], qboxes[j], clockwise)
            query_contains_box = _contains_all_points(qboxes[j], boxes[i], clockwise)
            ret[i, j] = intersects or box_contains_query or query_contains_box
    return ret
