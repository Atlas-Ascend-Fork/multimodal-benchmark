# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""
import logging
import os

from torch.utils.data import DataLoader
from torchvision import transforms


#

#


LOGGER = logging.getLogger(__name__)


def _module_kitti_data_loader_35_part_1(_state):
    _state["norm_factor"] = _state["params"]["data_normalize_factor"]
    _state["invert_depth"] = _state["params"]["invert_depth"]
    _state["ds_dir"] = _state["params"]["depth_dir"]
    _state["rgb_dir"] = (
        _state["params"]["rgb_dir"] if "rgb_dir" in _state["params"] else None
    )
    _state["rgb2gray"] = (
        _state["params"]["rgb2gray"] if "rgb2gray" in _state["params"] else False
    )
    _state["fill_depth"] = (
        _state["params"]["fill_depth"] if "fill_depth" in _state["params"] else False
    )
    _state["flip"] = _state["params"]["flip"] if "flip" in _state["params"] else False
    _state["dataset"] = "KittiDepthDataset"
    _state["num_worker"] = _state["params"]["num_worker"]
    _state["image_datasets"] = {}
    _state["dataloaders"] = {}
    _state["dataset_sizes"] = {}
    _state["train_data_path"] = os.path.join(
        _state["ds_dir"], "data_depth_velodyne/train"
    )
    _state["train_gt_path"] = os.path.join(
        _state["ds_dir"], "data_depth_annotated/train"
    )
    if _state["params"]["transform_type"] == "center":
        _state["train_transform"] = transforms.Compose(
            [transforms.CenterCrop((352, 1216))]
        )
    else:
        _state["train_transform"] = None
    _state["image_datasets"]["train"] = eval(_state["dataset"])(
        _state["train_data_path"],
        _state["train_gt_path"],
        setname="train",
        transform=_state["train_transform"],
        norm_factor=_state["norm_factor"],
        invert_depth=_state["invert_depth"],
        rgb_dir=_state["rgb_dir"],
        rgb2gray=_state["rgb2gray"],
        fill_depth=_state["fill_depth"],
        flip=_state["flip"],
    )


def _module_kitti_data_loader_35_part_2(_state):
    if _state["params"]["train_on"] != "full":
        _state["image_datasets"]["train"].data = _state["image_datasets"]["train"].data[
            0: _state["params"]["train_on"]
        ]
        _state["image_datasets"]["train"].gt = _state["image_datasets"]["train"].gt[
            0: _state["params"]["train_on"]
        ]
    _state["dataloaders"]["train"] = DataLoader(
        _state["image_datasets"]["train"],
        shuffle=False,
        batch_size=_state["params"]["train_batch_sz"],
        num_workers=_state["num_worker"],
    )
    _state["dataset_sizes"]["train"] = {len(_state["image_datasets"]["train"])}
    _state["val_data_path"] = os.path.join(_state["ds_dir"], "data_depth_velodyne/val")
    _state["val_gt_path"] = os.path.join(_state["ds_dir"], "data_depth_annotated/val")
    _state["val_transform"] = transforms.Compose([transforms.CenterCrop((352, 1216))])
    _state["image_datasets"]["val"] = eval(_state["dataset"])(
        _state["val_data_path"],
        _state["val_gt_path"],
        setname="val",
        transform=_state["val_transform"],
        norm_factor=_state["norm_factor"],
        invert_depth=_state["invert_depth"],
        rgb_dir=_state["rgb_dir"],
        rgb2gray=_state["rgb2gray"],
        fill_depth=_state["fill_depth"],
        flip=_state["flip"],
    )
    _state["dataloaders"]["val"] = DataLoader(
        _state["image_datasets"]["val"],
        shuffle=False,
        batch_size=_state["params"]["val_batch_sz"],
        num_workers=_state["num_worker"],
    )
    _state["dataset_sizes"]["val"] = {len(_state["image_datasets"]["val"])}
    _state["selval_data_path"] = os.path.join(
        _state["ds_dir"], "depth_selection/val_selection_cropped/velodyne_raw"
    )
    _state["selval_gt_path"] = os.path.join(
        _state["ds_dir"], "depth_selection/val_selection_cropped/groundtruth_depth"
    )


def _module_kitti_data_loader_35_part_3(_state):
    _state["image_datasets"]["selval"] = eval(_state["dataset"])(
        _state["selval_data_path"],
        _state["selval_gt_path"],
        setname="selval",
        transform=None,
        norm_factor=_state["norm_factor"],
        invert_depth=_state["invert_depth"],
        rgb_dir=_state["rgb_dir"],
        rgb2gray=_state["rgb2gray"],
        fill_depth=_state["fill_depth"],
        flip=_state["flip"],
    )
    _state["dataloaders"]["selval"] = DataLoader(
        _state["image_datasets"]["selval"],
        shuffle=False,
        batch_size=_state["params"]["test_batch_sz"],
        num_workers=_state["num_worker"],
    )
    _state["dataset_sizes"]["selval"] = {len(_state["image_datasets"]["selval"])}
    _state["test_data_path"] = os.path.join(
        _state["ds_dir"], "depth_selection/test_depth_completion_anonymous/velodyne_raw"
    )
    _state["test_gt_path"] = os.path.join(
        _state["ds_dir"], "depth_selection/test_depth_completion_anonymous/velodyne_raw"
    )
    _state["image_datasets"]["test"] = eval(_state["dataset"])(
        _state["test_data_path"],
        _state["test_gt_path"],
        setname="test",
        transform=None,
        norm_factor=_state["norm_factor"],
        invert_depth=_state["invert_depth"],
        rgb_dir=_state["rgb_dir"],
        rgb2gray=_state["rgb2gray"],
        fill_depth=_state["fill_depth"],
    )


def _module_kitti_data_loader_35_part_4(_state):
    _state["dataloaders"]["test"] = DataLoader(
        _state["image_datasets"]["test"],
        shuffle=False,
        batch_size=_state["params"]["test_batch_sz"],
        num_workers=_state["num_worker"],
    )
    _state["dataset_sizes"]["test"] = {len(_state["image_datasets"]["test"])}
    LOGGER.info("%s %s", "==", _state["dataset_sizes"])


def kitti_data_loader(params):
    # Input images are 16-bit, but only 15 bits are used. Normalize them to [0, 1].
    _state = locals()
    _module_kitti_data_loader_35_part_1(_state)
    _module_kitti_data_loader_35_part_2(_state)
    _module_kitti_data_loader_35_part_3(_state)
    _module_kitti_data_loader_35_part_4(_state)

    return (_state["dataloaders"], _state["dataset_sizes"])
