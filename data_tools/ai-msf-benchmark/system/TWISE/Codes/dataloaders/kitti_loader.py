# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import glob
import os
import os.path

from dataloaders import transforms
import numpy as np
import torch.utils.data as data
from dataloaders.loader_common import depth_read, drop_depth_measurements
from dataloaders.loader_common import evaluation_transform as _evaluation_transform
from dataloaders.loader_common import ind2sub, initialize_kitti_depth, load_calib
from dataloaders.loader_common import read_kitti_sample, rgb_read as _rgb_read, sub2ind
from dataloaders.loader_common import train_transform as _train_transform


input_options = ["d", "rgb", "rgbd", "g", "gd"]

to_tensor = transforms.ToTensor()


def to_float_tensor(x):
    return to_tensor(x).float()


def _selection_globs(split, data_folder):
    if split == "val_selection":
        glob_d = glob.glob(
            os.path.join(
                data_folder,
                "Data",
                "depth_selection/val_selection_cropped/velodyne_raw/*.png",
            )
        )
        glob_gt = glob.glob(
            os.path.join(
                data_folder,
                "Data",
                "depth_selection/val_selection_cropped/groundtruth_depth/*.png",
            )
        )
        glob_rgb = glob.glob(
            os.path.join(
                data_folder,
                "Data",
                "depth_selection/val_selection_cropped/image/*.png",
            )
        )
        return glob_d, glob_gt, glob_rgb
    if split == "test_completion":
        glob_d = glob.glob(
            os.path.join(
                data_folder,
                "Data",
                "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png",
            )
        )
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = glob.glob(
            os.path.join(
                data_folder,
                "Data",
                "depth_selection/test_depth_completion_anonymous/image/*.png",
            )
        )
        return glob_d, glob_gt, glob_rgb
    raise ValueError("Unrecognized split " + str(split))


def _validate_paths(paths, runtime_args, source):
    paths_rgb, paths_d, paths_gt = paths
    if not paths_d and not paths_rgb and not paths_gt:
        raise RuntimeError("Found 0 images under {}".format(source))
    if not paths_d and runtime_args.use_d:
        raise RuntimeError("Requested sparse depth but none was found")
    if not paths_rgb and runtime_args.use_rgb:
        raise RuntimeError("Requested rgb images but none was found")
    if not paths_rgb and runtime_args.use_g:
        raise RuntimeError("Requested gray images but no rgb was found")
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise RuntimeError("Produced different sizes for datasets")


def _split_paths(paths, split, fraction):
    paths_rgb, paths_d, paths_gt = paths
    boundary = int(round(fraction * len(paths_rgb)))
    if split == "train":
        slices = slice(0, boundary)
    elif split == "val":
        slices = slice(boundary + 1, None)
    else:
        slices = slice(None)
    return {"rgb": paths_rgb[slices], "d": paths_d[slices], "gt": paths_gt[slices]}


def get_paths_and_transform(split, args, trainvalsplit=0.9):
    if not (args.use_d or args.use_rgb or args.use_g):
        raise ValueError("no proper input selected")
    glob_d, glob_gt, glob_rgb = _selection_globs(split, args.data_folder)

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob_d)
        paths_gt = sorted(glob_gt)
        if glob_rgb:
            paths_rgb = sorted(glob_rgb)
        else:
            paths_rgb = [
                path.replace("proj_depth/groundtruth_256", "color") for path in paths_gt
            ]
    else:
        # test only has d or rgb
        paths_rgb = sorted(glob_rgb)
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob_d)

    path_groups = (paths_rgb, paths_d, paths_gt)
    _validate_paths(path_groups, args, glob_gt)
    return _split_paths(path_groups, split, trainvalsplit), test_transform


def rgb_read(filename):
    return _rgb_read(filename, convert_rgb=True)


oheight, owidth, cwidth = 256, 1216, 1216


def train_transform(rgb, sparse, target, args, params=None):
    del params
    return _train_transform(rgb, sparse, target, None, args, transforms)[:3]


def val_transform(rgb, sparse, target, args):
    del args
    return _evaluation_transform(
        rgb, sparse, target, None, transforms, random_crop=True
    )[:3]


def test_transform(rgb, sparse, target, args):
    del args
    return _evaluation_transform(
        rgb, sparse, target, None, transforms, random_crop=False
    )[:3]


def no_transform(rgb, sparse, target, args):
    return (
        rgb,
        sparse,
        target,
    )


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset"""

    def __init__(self, split, args, trainvalsplit=0.9):
        initialize_kitti_depth(
            self, split, args, trainvalsplit, get_paths_and_transform
        )

    def __getitem__(self, index):
        (
            rgb,
            sparse,
            target,
        ) = self._get_raw(index)
        _ = np.shape(sparse)
        (
            rgb,
            sparse,
            target,
        ) = self.transform(rgb, sparse, target, self.args)

        if rgb is not None:
            rgb = rgb / 255.0

        candidates = {"rgb": rgb, "d": sparse, "gt": target}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items()
            if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths["gt"])

    def _get_raw(self, index):
        return read_kitti_sample(self, index, rgb_read)
