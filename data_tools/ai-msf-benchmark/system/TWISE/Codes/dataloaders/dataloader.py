# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import logging
import os
import os.path

import dataloaders.transforms as transforms
import h5py
import numpy as np
import torch.utils.data as data


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class MydataloaderInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    root: object
    entry_type: object
    sparsifier: object
    modality: object
    loader: object


LOGGER = logging.getLogger(__name__)

IMG_EXTENSIONS = [
    ".h5",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(directory):
    classes = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def _image_items(root, filenames, class_index):
    return [
        (os.path.join(root, filename), class_index)
        for filename in sorted(filenames)
        if is_image_file(filename)
    ]


def make_dataset(directory, class_to_idx):
    images = []
    directory = os.path.expanduser(directory)
    for target in sorted(os.listdir(directory)):
        d = os.path.join(directory, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            images.extend(_image_items(root, fnames, class_to_idx[target]))
    return images


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f["rgb"])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f["depth"])
    return rgb, depth


to_tensor = transforms.ToTensor()


_MYDATALOADER_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    MydataloaderInitParameters,
    ("root", "entry_type", "sparsifier", "modality", "loader"),
    {"sparsifier": None, "modality": "rgb", "loader": h5_loader},
    frozenset(()),
    5,
    None,
    None,
)


class MyDataloader(data.Dataset):
    modality_names = ["rgb", "rgbd", "d"]  # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _MYDATALOADER_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        root, entry_type, sparsifier, modality, loader = bound_parameters
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if not (len(imgs) > 0):
            raise AssertionError("Found 0 images in subfolders of: " + root + "\n")
        LOGGER.info("%s", "Found {} images in {} folder.".format(len(imgs), entry_type))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        if entry_type == "train":
            self.transform = self.train_transform
        elif entry_type == "val":
            self.transform = self.val_transform
        else:
            raise (
                RuntimeError(
                    "Invalid dataset type: " + entry_type + "\n"
                    "Supported dataset types are: train, val"
                )
            )
        self.loader = loader
        self.sparsifier = sparsifier

        if not (modality in self.modality_names):
            raise AssertionError(
                "Invalid modality type: "
                + modality
                + "\n"
                + "Supported dataset types are: "
                + "".join(self.modality_names)
            )
        self.modality = modality

    def __getitem__(self, index):
        rgb, depth = self._get_raw(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise (RuntimeError("transform not defined"))

        # color normalization

        if self.modality == "rgb":
            input_np = rgb_np
        elif self.modality == "rgbd":
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == "d":
            input_np = self.create_sparse_depth(rgb_np, depth_np)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    @staticmethod
    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def _get_raw(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)
        return rgb, depth
