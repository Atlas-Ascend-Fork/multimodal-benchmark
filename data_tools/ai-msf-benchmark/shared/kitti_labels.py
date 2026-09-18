# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Parser for KITTI text annotations."""

import pathlib
import re

import numpy as np


def assign_object_label(obj, line):
    """Populate the common fields of an object parsed from one KITTI label."""
    fields = line.strip().split(" ")
    obj.cls_type = fields[0]
    obj.truncation = float(fields[1])
    obj.occlusion = float(fields[2])
    obj.alpha = float(fields[3])
    obj.box2d = np.asarray(fields[4:8], dtype=np.float32)
    obj.h, obj.w, obj.length_value = map(float, fields[8:11])
    obj.pos = np.asarray(fields[11:14], dtype=np.float32)
    obj.dis_to_cam = np.linalg.norm(obj.pos)
    obj.ry = float(fields[14])
    obj.score = float(fields[15]) if len(fields) == 16 else -1.0


def format_object_label(obj, integer_truncation=False):
    """Format common object fields in KITTI label order."""
    truncation = int(obj.truncation) if integer_truncation else obj.truncation
    prefix = "%s %d %d" if integer_truncation else "%s %.2f %d"
    values = (
        obj.cls_type, truncation, int(obj.occlusion), obj.alpha, *obj.box2d,
        obj.h, obj.w, obj.length_value, *obj.pos, obj.ry,
    )
    return prefix % values[:3] + " " + " ".join("%.2f" % value for value in values[3:])


def filter_annotations_by_score(image_annotations, threshold):
    """Return annotation dictionaries containing only scores above a threshold."""
    filtered_images = []
    for annotation in image_annotations:
        indices = [
            index for index, score in enumerate(annotation["score"])
            if score >= threshold
        ]
        filtered_images.append(
            {key: values[indices] for key, values in annotation.items()}
        )
    return filtered_images


def get_label_annotation(label_path, object_type=None, include_metadata=True):
    """Parse one KITTI label file, optionally retaining one object type."""
    with open(label_path, "r") as label_file:
        content = [line.strip().split(" ") for line in label_file]
    if object_type is not None:
        content = [fields for fields in content if fields[0] == object_type]
    object_count = sum(fields[0] != "DontCare" for fields in content)
    annotations = {
        "name": np.array([fields[0] for fields in content]),
        "truncated": np.array([float(fields[1]) for fields in content]),
        "occluded": np.array([int(fields[2]) for fields in content]),
        "alpha": np.array([float(fields[3]) for fields in content]),
        "bbox": np.array(
            [[float(value) for value in fields[4:8]] for fields in content]
        ).reshape(-1, 4),
        "dimensions": np.array(
            [[float(value) for value in fields[8:11]] for fields in content]
        ).reshape(-1, 3)[:, [2, 0, 1]],
        "location": np.array(
            [[float(value) for value in fields[11:14]] for fields in content]
        ).reshape(-1, 3),
        "rotation_y": np.array(
            [float(fields[14]) for fields in content]
        ).reshape(-1),
    }
    ground_truth_count = len(annotations["name"])
    if content and len(content[0]) == 16:
        annotations["score"] = np.array([float(fields[15]) for fields in content])
    else:
        annotations["score"] = np.zeros((annotations["bbox"].shape[0],))
    if include_metadata:
        index = list(range(object_count)) + [-1] * (ground_truth_count - object_count)
        annotations["index"] = np.array(index, dtype=np.int32)
        annotations["group_ids"] = np.arange(ground_truth_count, dtype=np.int32)
    return annotations


def get_label_annotations(
    label_folder, image_ids=None, include_image_index=False, include_metadata=True
):
    folder = pathlib.Path(label_folder)
    if image_ids is None:
        filename_pattern = re.compile(r"^\d{6}.txt$")
        paths = filter(lambda path: filename_pattern.match(path.name), folder.glob("*.txt"))
        image_ids = sorted(int(path.stem) for path in paths)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annotations = []
    for image_index in image_ids:
        annotation = get_label_annotation(
            folder / ("%06d.txt" % int(image_index)),
            include_metadata=include_metadata,
        )
        if include_image_index:
            count = annotation["name"].shape[0]
            annotation["image_idx"] = np.full(count, image_index, dtype=np.int64)
        annotations.append(annotation)
    return annotations
