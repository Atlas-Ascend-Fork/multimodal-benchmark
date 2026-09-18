# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""I/O and indexing helpers for TWISE KITTI loaders."""

import os

from PIL import Image
import numpy as np
from shared.parameters import bind_named as _bind_named


def sub2ind(array_shape, rows, cols):
    indices = rows * array_shape[1] + cols
    indices[indices < 0] = -1
    indices[indices >= array_shape[0] * array_shape[1]] = -1
    return indices


def ind2sub(array_shape, indices):
    indices[indices < 0] = -1
    indices[indices >= array_shape[0] * array_shape[1]] = -1
    rows = indices.astype("int") / array_shape[1]
    cols = indices % array_shape[1]
    return rows, cols


def load_calib():
    """Load the fixed KITTI camera calibration adjusted for center cropping."""
    with open("dataloaders/calib_cam_to_cam.txt", "r") as calibration:
        projection_line = calibration.readlines()[25]
    values = projection_line.split(":")[1].split(" ")[1:]
    projection = np.reshape(
        np.array([float(value) for value in values]), (3, 4)
    ).astype(np.float32)
    camera = projection[:3, :3]
    camera[0, 2] -= 13
    camera[1, 2] -= 11.5
    return camera


def rgb_read(filename, convert_rgb=False):
    if not os.path.exists(filename):
        raise AssertionError("file not found: {}".format(filename))
    with Image.open(filename) as image:
        source = image.convert("RGB") if convert_rgb else image
        return np.array(source, dtype="uint8")


def depth_read(filename):
    if not os.path.exists(filename):
        raise AssertionError("file not found: {}".format(filename))
    with Image.open(filename) as image:
        depth_png = np.array(image, dtype=int)
    return np.expand_dims(depth_png.astype(np.float) / 256.0, -1)


def drop_depth_measurements(depth, probability_to_keep):
    depth *= np.random.binomial(1, probability_to_keep, depth.shape)
    return depth


def initialize_kitti_depth(dataset, split, args, trainvalsplit,
                           path_resolver):
    dataset.args = args
    dataset.split = split
    dataset.trainvalsplit = trainvalsplit
    dataset.paths, dataset.transform = path_resolver(
        split, args, trainvalsplit=trainvalsplit
    )
    dataset.k = load_calib()
    dataset.threshold_translation = 0.1


def read_kitti_sample(dataset, index, rgb_reader=rgb_read):
    rgb_path = dataset.paths["rgb"][index]
    depth_path = dataset.paths["d"][index]
    target_path = dataset.paths["gt"][index]
    rgb = (
        rgb_reader(rgb_path)
        if rgb_path is not None and (dataset.args.use_rgb or dataset.args.use_g)
        else None
    )
    sparse = (
        depth_read(depth_path)
        if depth_path is not None and dataset.args.use_d
        else None
    )
    target = depth_read(target_path) if target_path is not None else None
    return rgb, sparse, target


def train_transform(*call_args, **call_kwargs):
    """Apply the shared KITTI training crop, flip, and color jitter."""
    (
        rgb,
        sparse,
        target,
        rgb_near,
        args,
        transforms,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('rgb', 'sparse', 'target', 'rgb_near', 'args', 'transforms'),
    )
    do_flip = np.random.uniform(0.0, 1.0) < 0.5
    crop_left = np.random.randint(1, size=())
    geometric = transforms.Compose(
        [
            transforms.BottomCrop((256, 1216)),
            transforms.HorizontalFlip(do_flip),
            transforms.RandomCrop([crop_left, crop_left + 1216, 0, 256]),
        ]
    )
    if sparse is not None:
        sparse = geometric(sparse)
    target = geometric(target)
    if rgb is not None:
        jitter = [
            np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
            for _ in range(3)
        ]
        rgb_transform = transforms.Compose(
            [transforms.ColorJitter(*jitter, 0), geometric]
        )
        rgb = rgb_transform(rgb)
        if rgb_near is not None:
            rgb_near = rgb_transform(rgb_near)
    return rgb, sparse, target, rgb_near


def evaluation_transform(*call_args, **call_kwargs):
    """Apply the shared validation or test crop."""
    (
        rgb,
        sparse,
        target,
        rgb_near,
        transforms,
        random_crop,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('rgb', 'sparse', 'target', 'rgb_near', 'transforms', 'random_crop'),
    )
    operations = [transforms.BottomCrop((256, 1216))]
    if random_crop:
        crop_left = np.random.randint(1, size=())
        operations.append(
            transforms.RandomCrop([crop_left, crop_left + 1216, 0, 256])
        )
    transform = transforms.Compose(operations)
    values = [rgb, sparse, target, rgb_near]
    return tuple(transform(value) if value is not None else None for value in values)
