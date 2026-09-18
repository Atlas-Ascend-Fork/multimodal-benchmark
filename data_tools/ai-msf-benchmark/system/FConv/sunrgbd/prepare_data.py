# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

""" Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: October 2017

Modified by Zhixin Wang
"""
from typing import NamedTuple
import argparse
import logging
import os
import pickle
import sys

from sunrgbd_utils import random_shift_box2d, extract_pc_in_box3d
from SunrgbdObject import SunrgbdObject
from tqdm import tqdm
import numpy as np
from shared.dataset_geometry import SUNRGBD_CLASSES_10
import sunrgbd_utils as utils
from utils.frustum_data import add_generation_arguments


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class ExtractFrustumDataParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    sunrgbd_dir: object
    idx_filename: object
    split: object
    output_filename: object
    type_whitelist: object
    perturb_box2d: object
    augment_x: object
    with_down_sample: object


class ExtractFrustumDataFromRgbDetectionParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    sunrgbd_dir: object
    det_file: object
    split: object
    output_filename: object
    type_whitelist: object
    valid_id_list: object
    with_down_sample: object


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


LOGGER = logging.getLogger(__name__)

SUNRGBD_CLASSES_19 = (
    "bathtub",
    "bed",
    "bookshelf",
    "box",
    "chair",
    "counter",
    "desk",
    "door",
    "dresser",
    "garbage_bin",
    "lamp",
    "monitor",
    "night_stand",
    "pillow",
    "sink",
    "sofa",
    "table",
    "tv",
    "toilet",
)


def ravel_hash(coord):
    if not (coord.ndim == 2):
        raise AssertionError

    coord -= coord.min(0)
    coord_max = coord.max(0) + 1

    keys = np.zeros(coord.shape[0], dtype=np.int64)

    for i in range(coord.shape[1] - 1):
        keys += coord[:, i]
        keys *= coord_max[i + 1]
    keys += coord[:, -1]

    return keys


def down_sample(x, voxel_size=(0.05,)):

    if isinstance(voxel_size, float):
        voxel_size = (voxel_size,)

    if len(voxel_size) == 1:
        voxel_size = voxel_size * 3

    voxel_size = np.array(voxel_size, dtype=np.float32)
    voxel_index = np.floor(x / voxel_size).astype(np.int64, copy=False)
    hash_keys = ravel_hash(voxel_index)
    _, idx = np.unique(hash_keys, return_index=True)

    return idx


def get_box3d_dim_statistics(my_sunrgbd_dir, idx_filename, type_whitelist):
    dataset = SunrgbdObject(my_sunrgbd_dir)
    dimension_list = []
    type_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        LOGGER.info("%s %s", "------------- ", data_idx)
        objects = dataset.get_label_objects(data_idx)
        for obj_idx, _ in enumerate(objects):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist:
                continue
            dimension_list.append(np.array([obj.length_value, obj.w, obj.h]))
            type_list.append(obj.classname)

    LOGGER.info("%s", "number of objects: {} ".format(len(type_list)))
    LOGGER.info("%s %s", "categories:", list(sorted(type_whitelist)))

    # Get average box size for different categories
    for class_type in sorted(set(type_list)):
        cnt = 0
        box3d_list = []
        for i, _ in enumerate(dimension_list):
            if type_list[i] == class_type:
                cnt += 1
                box3d_list.append(dimension_list[i])

        median_box3d = np.median(box3d_list, 0)
        LOGGER.info(
            "%s",
            "'%s': np.array([%f,%f,%f]),"
            % (
                class_type,
                median_box3d[0] * 2,
                median_box3d[1] * 2,
                median_box3d[2] * 2,
            ),
        )


def read_det_file(det_file):
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    # data_idx, type_list, prob, box2d
    with open(det_file, "rt") as f:
        for line in f:
            t = line.rstrip().split(" ")
            id_list.append(int(t[0]))
            type_list.append(t[1])
            prob_list.append(float(t[2]))
            box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))

    return id_list, type_list, box2d_list, prob_list


def read_det_pkl_file(det_file):
    classes = ("__background__",) + SUNRGBD_CLASSES_19
    with open(det_file, "rb") as f:
        dets = pickle.load(f)

    num_classes = len(dets)
    num_images = len(dets[0])
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []

    for i in range(num_images):
        for c in range(1, num_classes):
            det = dets[c][i]
            for j, _ in enumerate(det):
                id_list.append((i + 1))
                type_list.append(classes[c])
                prob_list.append(det[j][4])
                box2d_list.append(det[j][:4])

    return id_list, type_list, box2d_list, prob_list


_EXTRACT_FRUSTUM_DATA_PARAMETERS_SCHEMA = _ParameterSchema(
    ExtractFrustumDataParameters,
    (
        "sunrgbd_dir",
        "idx_filename",
        "split",
        "output_filename",
        "type_whitelist",
        "perturb_box2d",
        "augment_x",
        "with_down_sample",
    ),
    {"perturb_box2d": False, "augment_x": 1, "with_down_sample": False},
    frozenset(()),
    8,
    None,
    None,
)
_EXTRACT_FRUSTUM_DATA_FROM_RGB_DETECTION_PARAMETERS_SCHEMA = _ParameterSchema(
    ExtractFrustumDataFromRgbDetectionParameters,
    (
        "sunrgbd_dir",
        "det_file",
        "split",
        "output_filename",
        "type_whitelist",
        "valid_id_list",
        "with_down_sample",
    ),
    {"valid_id_list": None, "with_down_sample": False},
    frozenset(()),
    7,
    None,
    None,
)


def _load_sunrgbd_scene(dataset, data_index, with_down_sample):
    calibration = dataset.get_calibration(data_index)
    depth_points = dataset.get_pointcloud(data_index)
    camera_points = np.zeros_like(depth_points)
    camera_points[:, :3] = calibration.project_upright_depth_to_upright_camera(
        depth_points[:, :3]
    )
    camera_points[:, 3:] = depth_points[:, 3:]
    if with_down_sample:
        selected = down_sample(camera_points[:, :3], 0.01)
        camera_points, depth_points = camera_points[selected], depth_points[selected]
    image_coordinates, _ = calibration.project_upright_depth_to_image(depth_points)
    return calibration, camera_points, image_coordinates


def _crop_sunrgbd_frustum(camera_points, image_coordinates, box):
    xmin, ymin, xmax, ymax = box
    selected = (
        (image_coordinates[:, 0] < xmax) & (image_coordinates[:, 0] >= xmin)
        & (image_coordinates[:, 1] < ymax) & (image_coordinates[:, 1] >= ymin)
    )
    return camera_points[selected], image_coordinates[selected]


def _sunrgbd_frustum_angle(calibration, box):
    xmin, ymin, xmax, ymax = box
    image_depth = np.array([[(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, 20]])
    center = calibration.project_image_to_upright_camera(image_depth)
    return -np.arctan2(center[0, 2], center[0, 0])


def _subsample_sunrgbd_frustum(points, coordinates, labels=None):
    if len(points) <= 2048:
        return points, coordinates, labels
    selected = np.random.choice(len(points), 2048, replace=False)
    labels = labels[selected] if labels is not None else None
    return points[selected], coordinates[selected], labels


def _new_sunrgbd_output(labeled):
    keys = ["id", "box2d", "input", "type", "frustum_angle", "calib_K", "calib_R"]
    if labeled:
        keys += ["box3d", "box3d_size", "box3d_heading", "label"]
    else:
        keys += ["box2d_prob"]
    return {key: [] for key in keys}


def _labeled_sunrgbd_sample(calibration, obj, camera_points, image_coordinates, box):
    points, coordinates = _crop_sunrgbd_frustum(
        camera_points, image_coordinates, box
    )
    _, box_corners = utils.compute_box_3d(obj, calibration)
    box_corners = calibration.project_upright_depth_to_upright_camera(box_corners)
    try:
        _, indices = extract_pc_in_box3d(points, box_corners)
    except Exception as error:
        LOGGER.info("%s", error)
        return None
    labels = np.zeros(len(points))
    labels[indices] = 1
    points, coordinates, labels = _subsample_sunrgbd_frustum(
        points, coordinates, labels
    )
    if np.sum(labels) < 5:
        return None
    return {
        "box3d": box_corners,
        "box3d_size": np.array([2 * obj.length_value, 2 * obj.w, 2 * obj.h]),
        "input": points.astype(np.float32),
        "label": labels.astype(np.bool),
        "frustum_angle": _sunrgbd_frustum_angle(calibration, box),
        "calib_K": calibration.K, "calib_R": calibration.Rtilt,
    }


def _append_sunrgbd_sample(output, common, sample):
    for key, value in common.items():
        output[key].append(value)
    for key, value in sample.items():
        output[key].append(value)


def _save_sunrgbd_output(output, output_filename):
    with open(output_filename, "wb") as file_handle:
        pickle.dump(output, file_handle, -1)
    LOGGER.info("%s", "save in {}".format(output_filename))


def _collect_sunrgbd_augmentations(context, obj):
    positive_count = point_count = 0
    for _ in range(context["augment_x"]):
        box = random_shift_box2d(obj.box2d) if context["perturb_box2d"] else obj.box2d
        sample = _labeled_sunrgbd_sample(
            context["calibration"], obj, context["camera_points"],
            context["image_coordinates"], box,
        )
        if sample is None:
            continue
        _append_sunrgbd_sample(
            context["output"],
            {
                "id": context["data_index"],
                "box2d": np.array(box, dtype=np.float32),
                "type": obj.classname,
            },
            sample,
        )
        positive_count += np.sum(sample["label"])
        point_count += len(sample["input"])
    return positive_count, point_count


def extract_frustum_data(*call_args, **call_kwargs):
    parameters = _bind_parameters(
        _EXTRACT_FRUSTUM_DATA_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    dataset = SunrgbdObject(parameters.sunrgbd_dir, parameters.split)
    data_indices = [int(line.rstrip()) for line in open(parameters.idx_filename)]
    output = _new_sunrgbd_output(labeled=True)
    positive_count, point_count = 0, 0
    for data_index in data_indices:
        LOGGER.info("%s %s", "------------- ", data_index)
        calibration, camera_points, image_coordinates = _load_sunrgbd_scene(
            dataset, data_index, parameters.with_down_sample
        )
        for obj in dataset.get_label_objects(data_index):
            if obj.classname not in parameters.type_whitelist:
                continue
            counts = _collect_sunrgbd_augmentations(
                {
                    "augment_x": parameters.augment_x,
                    "perturb_box2d": parameters.perturb_box2d,
                    "calibration": calibration,
                    "camera_points": camera_points,
                    "image_coordinates": image_coordinates,
                    "output": output,
                    "data_index": data_index,
                },
                obj,
            )
            positive_count += counts[0]
            point_count += counts[1]
    LOGGER.info("%s %s", "Average pos ratio: ", positive_count / float(point_count))
    LOGGER.info("%s %s", "Average npoints: ", float(point_count) / len(output["id"]))
    _save_sunrgbd_output(output, parameters.output_filename)


def extract_frustum_data_from_rgb_detection(*call_args, **call_kwargs):
    parameters = _bind_parameters(
        _EXTRACT_FRUSTUM_DATA_FROM_RGB_DETECTION_PARAMETERS_SCHEMA,
        call_args, call_kwargs,
    )
    dataset = SunrgbdObject(parameters.sunrgbd_dir, parameters.split)
    reader = read_det_file if parameters.det_file.endswith(".txt") else read_det_pkl_file
    detection_values = reader(parameters.det_file)
    output = _new_sunrgbd_output(labeled=False)
    cached_index, cached_scene = -1, None
    for data_index, object_type, box, probability in tqdm(zip(*detection_values)):
        if parameters.valid_id_list is not None and data_index not in parameters.valid_id_list:
            continue
        if object_type not in parameters.type_whitelist:
            continue
        if cached_index != data_index:
            cached_scene = _load_sunrgbd_scene(
                dataset, data_index, parameters.with_down_sample
            )
            cached_index = data_index
        calibration, camera_points, image_coordinates = cached_scene
        points, coordinates = _crop_sunrgbd_frustum(
            camera_points, image_coordinates, box
        )
        points, coordinates, _ = _subsample_sunrgbd_frustum(points, coordinates)
        if len(points) < 5:
            continue
        _append_sunrgbd_sample(
            output,
            {"id": data_index, "type": object_type, "box2d": box,
             "box2d_prob": probability},
            {"input": points.astype(np.float32),
             "frustum_angle": _sunrgbd_frustum_angle(calibration, box),
             "calib_K": calibration.K, "calib_R": calibration.Rtilt},
        )
    _save_sunrgbd_output(output, parameters.output_filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_generation_arguments(parser)
    parser.add_argument(
        "--num_classes", default=10, type=int, help="19 or 10 categories, default 10"
    )
    parser.add_argument(
        "--save_dir",
        default="sunrgbd/data/pickle_data",
        type=str,
        help="directory to save data, default[sunrgbd/data/pickle_data]",
    )
    parser.add_argument(
        "--gen_avg_dim", action="store_true", help="get average dimension of each class"
    )

    args = parser.parse_args()

    my_sunrgbd_dir = "sunrgbd/mysunrgbd"  # change if you do not set default path

    if args.num_classes == 10:
        type_whitelist = SUNRGBD_CLASSES_10
    elif args.num_classes == 19:
        type_whitelist = SUNRGBD_CLASSES_19
    else:
        if not (False):
            raise AssertionError("please set correct num_classes")

    type_whitelist = set(type_whitelist)

    if args.gen_avg_dim:
        get_box3d_dim_statistics(
            my_sunrgbd_dir, "sunrgbd/image_sets/train.txt", type_whitelist
        )

    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.gen_train:
        extract_frustum_data(
            my_sunrgbd_dir,
            "sunrgbd/image_sets/train.txt",
            "training",
            output_filename=os.path.join(save_dir, "sunrgbd_train_aug5x.pickle"),
            type_whitelist=type_whitelist,
            perturb_box2d=True,
            augmentX=5,
            with_down_sample=False,
        )

    if args.gen_val:
        extract_frustum_data(
            my_sunrgbd_dir,
            "sunrgbd/image_sets/val.txt",
            "training",
            output_filename=os.path.join(save_dir, "sunrgbd_val.pickle"),
            type_whitelist=type_whitelist,
            perturb_box2d=False,
            augmentX=1,
            with_down_sample=False,
        )

    if args.gen_val_rgb_detection:
        extract_frustum_data_from_rgb_detection(
            my_sunrgbd_dir,
            "./sunrgbd/rgb_detections/sunrgbd_rgb_det_val_classes19_mAP50.2.txt",
            "training",
            os.path.join(save_dir, "sunrgbd_rgb_det_val.pickle"),
            type_whitelist=type_whitelist,
        )
