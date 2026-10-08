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
from typing import NamedTuple
import glob
import random

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class KittidepthdatasetInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    data_path: object
    gt_path: object
    setname: object
    transform: object
    norm_factor: object
    invert_depth: object
    rgb_dir: object
    rgb2gray: object
    fill_depth: object
    flip: object
    blind: object


#
#


_KITTIDEPTHDATASET_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    KittidepthdatasetInitParameters,
    (
        "data_path",
        "gt_path",
        "setname",
        "transform",
        "norm_factor",
        "invert_depth",
        "rgb_dir",
        "rgb2gray",
        "fill_depth",
        "flip",
        "blind",
    ),
    {
        "setname": "train",
        "transform": None,
        "norm_factor": 256,
        "invert_depth": False,
        "rgb_dir": None,
        "rgb2gray": False,
        "fill_depth": False,
        "flip": False,
        "blind": False,
    },
    frozenset(()),
    11,
    None,
    None,
)


def _kittidepthdataset_getitem_158_part_1(_state):
    if _state["self"].setname == "train" or _state["self"].setname == "val":
        _state["data_path"] = (
            _state["self"].data[_state["item"]].split(_state["self"].setname)[1]
        )
        _state["gt_path"] = (
            _state["self"].gt[_state["item"]].split(_state["self"].setname)[1]
        )
        _state["data_path"] = _state["data_path"].split("image")[1]
        _state["gt_path"] = _state["gt_path"].split("image")[1]
        if not _state["data_path"][:-4] == _state["gt_path"][:-4]:
            raise AssertionError
    elif _state["self"].setname == "selval":
        _state["data_path"] = _state["self"].data[_state["item"]].split("00000")[1]
        _state["sep"] = str(_state["self"].data[_state["item"]]).split("/velodyne_raw/")
    with Image.open(str(_state["self"].data[_state["item"]])) as _state["image"]:
        _state["data"] = _state["image"].copy()
    with Image.open(str(_state["self"].gt[_state["item"]])) as _state["image"]:
        _state["gt"] = _state["image"].copy()


def _kittidepthdataset_getitem_158_part_2(_state):
    if _state["self"].setname == "train" or _state["self"].setname == "val":
        _state["gt_path"] = str(_state["self"].gt[_state["item"]])
        _state["idx"] = _state["gt_path"].find("2011")
        _state["day_dir"] = _state["gt_path"][_state["idx"]: _state["idx"] + 10]
        _state["idx2"] = _state["gt_path"].find("groundtruth")
        _state["fname"] = _state["gt_path"][_state["idx2"] + 12:]
        _state["rgb_path"] = (
            _state["self"].rgb_dir
            + "/"
            + _state["day_dir"]
            + "/"
            + _state["gt_path"][_state["idx"]: _state["idx"] + 26]
            + "/"
            + _state["fname"][:8]
            + "/data/"
            + _state["fname"][9:]
        )
        _state["rgb_path"] = _state["rgb_path"][:-4] + ".png"
        with Image.open(_state["rgb_path"]) as _state["image"]:
            _state["rgb"] = _state["image"].copy()
    elif _state["self"].setname == "selval":
        _state["data_path"] = str(_state["self"].data[_state["item"]])
        _state["idx"] = _state["data_path"].find("velodyne_raw")
        _state["fname"] = _state["data_path"][_state["idx"] + 12:]
        _state["idx2"] = _state["fname"].find("velodyne_raw")
        _state["rgb_path"] = (
            _state["data_path"][: _state["idx"]]
            + "image"
            + _state["fname"][: _state["idx2"]]
            + "image"
            + _state["fname"][_state["idx2"] + 12:]
        )
        with Image.open(_state["rgb_path"]) as _state["image"]:
            _state["rgb"] = _state["image"].convert("RGB")
    elif _state["self"].setname == "test":
        _state["data_path"] = str(_state["self"].data[_state["item"]])
        _state["idx"] = _state["data_path"].find("velodyne_raw")
        _state["fname"] = _state["data_path"][_state["idx"] + 12:]
        _state["rgb_path"] = (
            _state["data_path"][: _state["idx"]] + "image/" + _state["fname"]
        )
        with Image.open(_state["rgb_path"]) as _state["image"]:
            _state["rgb"] = _state["image"].copy()
    if _state["self"].rgb2gray:
        _state["t"] = torchvision.transforms.Grayscale(1)
        _state["rgb"] = _state["t"](_state["rgb"])


def _kittidepthdataset_getitem_158_part_3_chunk_1(_state):
    _state["w"], _state["h"] = _state["data"].size
    if _state["self"].transform is not None:
        _state["data"] = _state["self"].transform(_state["data"])
        _state["gt"] = _state["self"].transform(_state["gt"])
        _state["rgb"] = _state["self"].transform(_state["rgb"])
    if _state["self"].transform is None and _state["self"].setname == "train":
        _state["crop_lt_u"] = random.randint(0, _state["w"] - 1216)
        _state["crop_lt_v"] = random.randint(0, _state["h"] - 352)
        _state["data"] = _state["data"].crop(
            (
                _state["crop_lt_u"],
                _state["crop_lt_v"],
                _state["crop_lt_u"] + 1216,
                _state["crop_lt_v"] + 352,
            )
        )
        _state["gt"] = _state["gt"].crop(
            (
                _state["crop_lt_u"],
                _state["crop_lt_v"],
                _state["crop_lt_u"] + 1216,
                _state["crop_lt_v"] + 352,
            )
        )
        _state["rgb"] = _state["rgb"].crop(
            (
                _state["crop_lt_u"],
                _state["crop_lt_v"],
                _state["crop_lt_u"] + 1216,
                _state["crop_lt_v"] + 352,
            )
        )
    if (
        _state["self"].flip
        and random.randint(0, 1)
        and (_state["self"].setname == "train")
    ):
        _state["data"] = _state["data"].transpose(Image.FLIP_LEFT_RIGHT)
        _state["gt"] = _state["gt"].transpose(Image.FLIP_LEFT_RIGHT)
        _state["rgb"] = _state["rgb"].transpose(Image.FLIP_LEFT_RIGHT)
    _state["data"] = np.array(_state["data"], dtype=np.float16)
    _state["gt"] = np.array(_state["gt"], dtype=np.float16)
    if _state["self"].blind and _state["self"].setname == "train":
        _state["blind_start"] = random.randint(100, _state["h"] - 50)
        _state["data"][_state["blind_start"]: _state["blind_start"] + 50, :] = 0


def _kittidepthdataset_getitem_158_part_3_chunk_2(_state):
    _state["c"] = (_state["data"] > 0).astype(float)
    _state["data"] = _state["data"] / _state["self"].norm_factor
    _state["gt"] = _state["gt"] / _state["self"].norm_factor
    _state["data"] = np.expand_dims(_state["data"], 0)
    _state["gt"] = np.expand_dims(_state["gt"], 0)
    _state["c"] = np.expand_dims(_state["c"], 0)
    _state["data"] = torch.tensor(_state["data"], dtype=torch.float)
    _state["gt"] = torch.tensor(_state["gt"], dtype=torch.float)
    _state["c"] = torch.tensor(_state["c"], dtype=torch.float)


def _kittidepthdataset_getitem_158_part_3(_state):
    _kittidepthdataset_getitem_158_part_3_chunk_1(_state)
    _kittidepthdataset_getitem_158_part_3_chunk_2(_state)


def _kittidepthdataset_getitem_158_part_4(_state):
    if _state["self"].invert_depth:
        if not False:
            raise AssertionError("DO NOT SUPPORT 'invert_depth'")
        _state["data"][_state["data"] == 0] = -1
        _state["data"] = 1 / _state["data"]
        _state["data"][_state["data"] == -1] = 0
        _state["gt"][_state["gt"] == 0] = -1
        _state["gt"] = 1 / _state["gt"]
        _state["gt"][_state["gt"] == -1] = 0
    _state["rgb"] = np.array(_state["rgb"], dtype=np.float16)
    _state["rgb"] /= 255
    if _state["self"].rgb2gray:
        _state["rgb"] = np.expand_dims(_state["rgb"], 0)
    else:
        _state["rgb"] = np.transpose(_state["rgb"], (2, 0, 1))
    _state["rgb"] = torch.tensor(_state["rgb"], dtype=torch.float)


class KittiDepthDataset(Dataset):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIDEPTHDATASET_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            data_path,
            gt_path,
            setname,
            transform,
            norm_factor,
            invert_depth,
            rgb_dir,
            rgb2gray,
            fill_depth,
            flip,
            blind,
        ) = bound_parameters
        self.data_path = data_path
        self.gt_path = gt_path
        self.setname = setname
        self.transform = transform
        self.norm_factor = norm_factor
        self.invert_depth = invert_depth
        self.rgb_dir = rgb_dir
        self.rgb2gray = rgb2gray
        self.fill_depth = fill_depth
        self.flip = False
        self.blind = blind

        self.data = list(
            sorted(glob.iglob(self.data_path + "/**/*.png", recursive=True))
        )
        self.gt = list(sorted(glob.iglob(self.gt_path + "/**/*.png", recursive=True)))
        if not (len(self.data) == len(self.gt)):
            raise AssertionError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Check if Data filename is equal to GT filename
        _state = locals()
        _kittidepthdataset_getitem_158_part_1(_state)
        _kittidepthdataset_getitem_158_part_2(_state)
        _kittidepthdataset_getitem_158_part_3(_state)
        _kittidepthdataset_getitem_158_part_4(_state)

        return (
            _state["data"],
            _state["c"],
            _state["item"],
            _state["rgb"],
            _state["gt"],
        )
