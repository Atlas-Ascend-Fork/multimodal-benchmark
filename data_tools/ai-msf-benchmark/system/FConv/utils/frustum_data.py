# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Geometry and export helpers for FConv frustum preparation."""

import os
import pickle

import numpy as np


DETECTION_ID_TO_TYPE = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}

GENERATION_FLAG_HELP = {
    "gen_train": "Generate train split frustum data with perturbed GT 2D boxes",
    "gen_val": "Generate val split frustum data with GT 2D boxes",
    "gen_val_rgb_detection": "Generate val split frustum data with RGB detection 2D boxes",
}


def add_generation_arguments(parser):
    for name, help_text in GENERATION_FLAG_HELP.items():
        parser.add_argument("--" + name, action="store_true", help=help_text)


def extract_kitti_split(extractor, paths, split, type_whitelist, **options):
    base_dir, save_dir, output_prefix = paths
    extractor(
        os.path.join(base_dir, "image_sets/{}.txt".format(split)),
        "training",
        os.path.join(save_dir, output_prefix + split + ".pickle"),
        type_whitelist=type_whitelist,
        **options
    )


def run_default_kitti_splits(extractor, paths, type_whitelist, arguments, val_options=None):
    if arguments.gen_train:
        extract_kitti_split(
            extractor, paths, "train", type_whitelist,
            perturb_box2d=True, augmentX=5,
        )
    if arguments.gen_val:
        options = {"perturb_box2d": False, "augmentX": 1}
        options.update(val_options or {})
        extract_kitti_split(extractor, paths, "val", type_whitelist, **options)


def _detection_type(value, allow_named_types):
    try:
        return DETECTION_ID_TO_TYPE.get(int(value))
    except ValueError as error:
        if not allow_named_types or value not in DETECTION_ID_TO_TYPE.values():
            raise AssertionError from error
        return value


def read_detection_file(filename, allow_named_types=False):
    ids, types, probabilities, boxes = [], [], [], []
    with open(filename, "r") as detection_file:
        for line in detection_file:
            fields = line.rstrip().split(" ")
            ids.append(int(os.path.basename(fields[0]).rstrip(".png")))
            types.append(_detection_type(fields[1], allow_named_types))
            probabilities.append(float(fields[2]))
            boxes.append(np.array([float(fields[index]) for index in range(3, 7)]))
    return ids, types, boxes, probabilities


def read_detection_pickle(filename):
    with open(filename, "r") as detection_file:
        results = pickle.load(detection_file)
    return (
        results["id_list"],
        results["type_list"],
        results["box2d_list"],
        results["prob_list"],
    )


def make_2d_detection_writer(reader):
    def writer(det_filename, split, result_dir):
        write_2d_rgb_detection(det_filename, split, result_dir, reader)

    return writer


def collect_box_dimensions(dataset, index_filename, progress):
    dimensions, types, rotations = [], [], []
    indices = [int(line.rstrip()) for line in open(index_filename)]
    for data_index in progress(indices):
        dataset.get_calibration(data_index)
        for obj in dataset.get_label_objects(data_index):
            if obj.type == "DontCare":
                continue
            dimensions.append(np.array([obj.length_value, obj.w, obj.h]))
            types.append(obj.type)
            rotations.append(obj.ry)
    return dimensions, types, rotations


def in_hull(points, hull):
    from scipy.spatial import Delaunay

    triangulation = hull if isinstance(hull, Delaunay) else Delaunay(hull)
    return triangulation.find_simplex(points) >= 0


def extract_pc_in_box3d(point_cloud, box):
    selected = in_hull(point_cloud[:, :3], box)
    return point_cloud[selected], selected


def extract_pc_in_box2d(point_cloud, box):
    xmin, ymin, xmax, ymax = box
    corners = np.array(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
    selected = in_hull(point_cloud[:, :2], corners)
    return point_cloud[selected], selected


def random_shift_box2d(box, image_height, image_width, shift_ratio=0.1,
                       require_clipped_valid=True):
    """Randomly shift and scale a 2D box within image bounds."""
    xmin, ymin, xmax, ymax = box
    if not (xmin < xmax and ymin < ymax):
        raise AssertionError
    height, width = ymax - ymin, xmax - xmin
    center_x, center_y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    while True:
        shifted_x = center_x + width * shift_ratio * (np.random.random() * 2 - 1)
        shifted_y = center_y + height * shift_ratio * (np.random.random() * 2 - 1)
        scaled_height = height * (
            1 + np.random.random() * 2 * shift_ratio - shift_ratio
        )
        scaled_width = width * (
            1 + np.random.random() * 2 * shift_ratio - shift_ratio
        )
        shifted = np.array(
            [
                shifted_x - scaled_width / 2,
                shifted_y - scaled_height / 2,
                shifted_x + scaled_width / 2,
                shifted_y + scaled_height / 2,
            ]
        )
        shifted[[0, 2]] = np.clip(shifted[[0, 2]], 0, image_width - 1)
        shifted[[1, 3]] = np.clip(shifted[[1, 3]], 0, image_height - 1)
        clipped_valid = shifted[0] < shifted[2] and shifted[1] < shifted[3]
        scaled_valid = scaled_width > 0 and scaled_height > 0
        if clipped_valid if require_clipped_valid else scaled_valid:
            return shifted


def write_2d_detections(det_filename, result_dir, reader):
    ids, types, boxes, probabilities = reader(det_filename)
    results = {}
    for index, object_type, box, probability in zip(ids, types, boxes, probabilities):
        coordinates = "%f %f %f %f" % tuple(box)
        line = "{} -1 -1 -10 {} -1 -1 -1 -1000 -1000 -1000 -10 %f".format(
            object_type, coordinates
        ) % probability
        results.setdefault(index, []).append(line)
    output_dir = os.path.join(result_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    for index, lines in results.items():
        filename = os.path.join(output_dir, "%06d.txt" % index)
        with open(filename, "w") as output_file:
            output_file.write("\n".join(lines) + "\n")


def write_2d_rgb_detection(det_filename, split, result_dir, reader):
    """Export RGB detections while retaining the legacy split argument."""
    del split
    write_2d_detections(det_filename, result_dir, reader)


def prepare_generation_output(arguments, default_directory):
    """Initialize deterministic output settings for KITTI data generation."""
    np.random.seed(3)
    save_directory = arguments.save_dir or default_directory
    os.makedirs(save_directory, exist_ok=True)
    if arguments.car_only:
        return save_directory, ["Car"], "frustum_caronly_"
    if arguments.people_only:
        return save_directory, ["Pedestrian", "Cyclist"], "frustum_pedcyc_"
    return (
        save_directory,
        ["Car", "Pedestrian", "Cyclist"],
        "frustum_carpedcyc_",
    )
