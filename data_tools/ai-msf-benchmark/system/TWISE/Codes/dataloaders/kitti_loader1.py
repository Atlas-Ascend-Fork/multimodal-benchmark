# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import glob
import os
import os.path

from dataloaders import transforms
from PIL import Image
import numpy as np
import torch.utils.data as data
from dataloaders.loader_common import depth_read, drop_depth_measurements
from dataloaders.loader_common import evaluation_transform as _evaluation_transform
from dataloaders.loader_common import ind2sub, initialize_kitti_depth, load_calib
from dataloaders.loader_common import read_kitti_sample, rgb_read, sub2ind
from dataloaders.loader_common import train_transform as _train_transform


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class TrainTransformParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    rgb: object
    sparse: object
    target: object
    rgb_near: object
    args: object
    params: object


input_options = ["d", "rgb", "rgbd", "g", "gd"]
to_tensor = transforms.ToTensor()


def to_float_tensor(value):
    return to_tensor(value).float()


def get_rgb_near(path, _args):
    """Load the nearest available adjacent RGB frame."""
    frame = os.path.basename(path)
    stem, suffix = os.path.splitext(frame)
    if not stem.isdigit():
        return None
    for offset in (-1, 1):
        candidate = os.path.join(
            os.path.dirname(path), f"{int(stem) + offset:010d}{suffix}"
        )
        if os.path.exists(candidate):
            return rgb_read(candidate)
    return None


def _depth_glob(root, pattern):
    return glob.glob(os.path.join(root, "KITTI_Depth", pattern))


def _validation_globs(args, sparse_depth):
    if args.val == "full":
        sparse = _depth_glob(
            args.data_folder,
            f"KITTI_final_origcrop/val/*_sync/proj_depth/{sparse_depth}/image_0[2,3]/*.png",
        )
        ground_truth = _depth_glob(
            args.data_folder,
            "KITTI_final_origcrop/val/*_sync/proj_depth/groundtruth_256/image_0[2,3]/*.png",
        )
        rgb = _depth_glob(
            args.data_folder,
            "KITTI_final_origcrop/val/*_sync/color/image_0[2,3]/*.png",
        )
        return sparse, ground_truth, rgb
    if args.val in ("select", "test_selectsplit"):
        base = "KITTI_final_origcrop/val_temp/val_shortened"
        sparse = _depth_glob(
            args.data_folder, f"{base}/proj_depth/{sparse_depth}/*.png"
        )
        ground_truth = _depth_glob(
            args.data_folder, f"{base}/proj_depth/groundtruth_256/*.png"
        )
        rgb = _depth_glob(args.data_folder, f"{base}/image/*.png")
        return sparse, ground_truth, rgb
    raise ValueError("Unrecognized validation set " + str(args.val))


def _selection_globs(args, completion=False):
    subset = (
        "test_depth_completion_anonymous" if completion else "val_selection_cropped"
    )
    sparse = _depth_glob(
        args.data_folder, f"depth_selection/{subset}/velodyne_raw/*.png"
    )
    rgb = _depth_glob(args.data_folder, f"depth_selection/{subset}/image/*.png")
    ground_truth = None
    if not completion:
        ground_truth = _depth_glob(
            args.data_folder, f"depth_selection/{subset}/groundtruth_depth/*.png"
        )
    return sparse, ground_truth, rgb


def _resolve_globs(split, args):
    if split == "test":
        return _validation_globs(args, "4x0_nSKips")
    if split == "val_selection":
        return _selection_globs(args)
    if split == "test_completion":
        return _selection_globs(args, completion=True)
    raise ValueError("Unrecognized split " + str(split))


def _rgb_from_ground_truth(path, args):
    if args.val != "full":
        return path.replace("proj_depth/groundtruth_256", "image")
    prefix = "/".join(path.split("/")[4:8])
    suffix = "/".join(path.split("/")[-2:])
    return os.path.join(args.data_folder, prefix, "color", suffix)


def _validate_paths(paths, args, source):
    rgb, sparse, ground_truth = paths
    if not sparse and not rgb and not ground_truth:
        raise RuntimeError("Found 0 images under {}".format(source))
    if not sparse and args.use_d:
        raise RuntimeError("Requested sparse depth but none was found")
    if not rgb and args.use_rgb:
        raise RuntimeError("Requested rgb images but none was found")
    if not rgb and args.use_g:
        raise RuntimeError("Requested gray images but no rgb was found")
    if len(rgb) != len(sparse) or len(rgb) != len(ground_truth):
        raise RuntimeError("Produced different sizes for datasets")


def _slice_paths(paths, split, fraction):
    rgb, sparse, ground_truth = paths
    boundary = int(round(fraction * len(rgb)))
    selected = slice(None)
    if split == "train":
        selected = slice(0, boundary)
    elif split == "val":
        selected = slice(boundary + 1, None)
    return {"rgb": rgb[selected], "d": sparse[selected], "gt": ground_truth[selected]}


def get_paths_and_transform(split, args, trainvalsplit=0.9):
    if not (args.use_d or args.use_rgb or args.use_g):
        raise ValueError("no proper input selected")
    glob_d, glob_gt, glob_rgb = _resolve_globs(split, args)
    if glob_gt is not None:
        paths_d = sorted(glob_d)
        paths_gt = sorted(glob_gt)
        paths_rgb = (
            sorted(glob_rgb)
            if glob_rgb
            else [_rgb_from_ground_truth(path, args) for path in paths_gt]
        )
    else:
        paths_rgb = sorted(glob_rgb)
        paths_gt = [None] * len(paths_rgb)
        paths_d = sorted(glob_d)
    paths = (paths_rgb, paths_d, paths_gt)
    _validate_paths(paths, args, glob_gt)
    return _slice_paths(paths, split, trainvalsplit), test_transform


oheight, owidth, cwidth = 256, 1216, 1216


def _three_hot_weights(delta):
    return np.concatenate(
        ((0.5 - delta) / 2, 0.5 * np.ones(np.shape(delta)), (0.5 + delta) / 2),
        axis=0,
    )


def _gaussian_three_hot_weights(depth, depthind, good_indices, dstep):
    sigma = 0.6 * dstep
    scale = np.power(dstep / sigma, 2) / 2
    weights = [
        np.exp(-np.power(depthind[good_indices] + offset - depth[good_indices], 2) * scale)
        for offset in (-1, 0, 1)
    ]
    norm = sum(weights)
    return np.concatenate([weight / norm for weight in weights], axis=0)


def depth2dc(depth, method="gaussian3hot", params=None, sigma=None):

    nchan = params["dce_nChannels"] + 2
    depthbins = np.arange(1, nchan)
    dstep = np.float64(params["dce_dstep"])
    hgt, wid, _ = np.shape(depth)
    depth = depth.flatten() / dstep
    depthind = np.round(depth)
    delta = depth - depthind
    goodpix = np.where(
        np.logical_and(depthind > 0, depthind <= params["dce_nChannels"])
    )
    pix_indics = np.concatenate((goodpix[0], goodpix[0], goodpix[0]), axis=0)
    depthbins = np.int64(
        np.concatenate(
            (depthind[goodpix[0]] - 1, depthind[goodpix[0]], depthind[goodpix[0]] + 1),
            0,
        )
    )
    if method == "3hot":
        vals = _three_hot_weights(delta[goodpix[0]])
    elif method == "gaussian3hot":
        vals = _gaussian_three_hot_weights(depth, depthind, goodpix[0], dstep)
    else:
        ValueError("depth2dc method does not exist")

    dcc_fin = np.zeros((hgt * wid * nchan))
    ind = sub2ind((hgt * wid, nchan), pix_indics, depthbins)
    dcc_fin[ind] = vals
    dcc_fin = np.reshape(dcc_fin, (hgt, wid, nchan))

    return dcc_fin


def dc2depth(dcpred, spatialdim, params=None):

    nchan = params["dce_nChannels"] + 2
    dstep = params["dce_dstep"]
    dcpred = np.reshape(dcpred, [spatialdim[0] * spatialdim[1], nchan])
    depthchan = np.float32(np.arange(0, nchan)) * dstep

    dvals = np.ones((spatialdim[0] * spatialdim[1], 1), np.float32) * depthchan
    sum_probs = np.maximum(1e-5, np.sum(dcpred, 1))  # avoid divide by zero
    depth = np.zeros((spatialdim[0] * spatialdim[1]))
    goodpix = np.where(sum_probs > 1e-5)
    depth[goodpix[0]] = (
        np.sum(dvals[goodpix[0], :] * dcpred[goodpix[0], :], 1) / sum_probs[goodpix[0]]
    )
    depth = np.reshape(depth, (spatialdim[0], spatialdim[1], 1))
    return depth


def dc2max3coeffdep(dcpred, spatialdim, params=None):
    n_channels = params["dce_nChannels"] + 2
    dstep = params["dce_dstep"]
    dcpred = np.reshape(dcpred, (-1, n_channels))
    max_bin = np.argmax(dcpred, 1)
    goodpixind = np.logical_and(max_bin > 0, max_bin <= (n_channels + 1))

    max_binfilt = max_bin[goodpixind]
    near_binfilt = max_binfilt - 1
    far_binfilt = max_binfilt + 1

    max_wtfilt = dcpred[goodpixind, max_binfilt]
    near_wtfilt = dcpred[goodpixind, near_binfilt]
    far_wtfilt = dcpred[goodpixind, far_binfilt]

    depth_vals = (
        max_wtfilt * max_binfilt * dstep
        + near_wtfilt * near_binfilt * dstep
        + far_binfilt * far_wtfilt * dstep
    ) / (max_wtfilt + near_wtfilt + far_wtfilt)

    depth = np.zeros((spatialdim[0] * spatialdim[1], 1), np.float32)
    depth[goodpixind] = depth_vals
    depth = np.reshape(depth, (spatialdim[0], spatialdim[1], 1))

    return depth


def full2sparse(depth, method="gaussian3hot", params=None):

    nchan = params["dce_nChannels"] + 2
    spatial_dim = depth.shape
    depthbins = np.arange(1, nchan)
    dstep = np.float64(params["dce_dstep"])
    depth = depth.flatten() / dstep
    depthind = np.round(depth)
    delta = depth - depthind
    goodpix = np.where(
        np.logical_and(depthind > 0, depthind <= params["dce_nChannels"])
    )
    y, x = np.tile(np.asarray(ind2sub(spatial_dim, goodpix[0])), (1, 3)).astype(
        np.int32
    )

    _ = np.concatenate((goodpix[0], goodpix[0], goodpix[0]), axis=0)
    depthbins = np.int32(
        np.concatenate(
            (depthind[goodpix[0]] - 1, depthind[goodpix[0]], depthind[goodpix[0]] + 1),
            0,
        )
    )
    depthbins = depthbins[:, np.newaxis].T
    coords = np.vstack((y, x, depthbins)).T
    if method == "3hot":
        vals = _three_hot_weights(delta[goodpix[0]])
    elif method == "gaussian3hot":
        vals = _gaussian_three_hot_weights(depth, depthind, goodpix[0], dstep)
    else:
        ValueError("depth2dc method does not exist")
    vals = vals[:, np.newaxis]
    return coords, vals


def full2sparse_nonuniform(depth_map, depthbins, binsigmas, params=None):

    sz = depth_map.shape()
    depth_map = np.flatten(depth_map)
    n = sz[0] * sz[1] * sz[2]
    depthbins = depthbins.astype(np.float32)
    binsigmas = binsigmas.astype(np.float32)
    depthbins = np.ones((n, 1)) * depthbins
    binlen = depthbins.shape()[1]
    goodmask = depth_map > 0
    validpix = np.where(goodmask)
    deltas = np.pow(depthbins - depth_map[:, np.newaxis], 2) / (
        2 * (binsigmas[np.newaxis, :] ** 2)
    )
    gauss = np.exp(-deltas)
    depth_mask = depth_map == 0
    depth_mask = depth_mask[:, np.newaxis]
    depth_mask = np.tile(depth_mask, (1, binlen))
    gauss[depth_mask] = 0

    mainind = np.argmax(gauss, 1)
    mainind = mainind * goodmask.astype(np.float32)
    mainind = mainind[:, np.newaxis]
    closeind = mainind - 1
    farind = mainind + 1
    pixind = np.concatenate((validpix, validpix, validpix), 0)
    depthbins = np.concatenate((closeind, mainind, farind), 0)
    sparseind = sub2ind((n, binlen), pixind, depthbins)
    vals = gauss[sparseind]
    y, x = np.tile(np.asarray(ind2sub((sz[0] * sz[1]), validpix)), (1, 3)).astype(
        np.int32
    )
    depthbins = depthbins[:, np.newaxis].T
    coords = np.vstack((y, x, depthbins)).T

    vals = vals[:, np.newaxis]
    return coords, vals


_TRAIN_TRANSFORM_PARAMETERS_SCHEMA = _ParameterSchema(
    TrainTransformParameters,
    ("rgb", "sparse", "target", "rgb_near", "args", "params"),
    {"params": None},
    frozenset(()),
    6,
    None,
    None,
)


def train_transform(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _TRAIN_TRANSFORM_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    rgb, sparse, target, rgb_near, args, params = bound_parameters
    del params
    return _train_transform(rgb, sparse, target, rgb_near, args, transforms)


def val_transform(rgb, sparse, target, rgb_near, args):
    del args
    return _evaluation_transform(
        rgb, sparse, target, rgb_near, transforms, random_crop=True
    )


def test_transform(rgb, sparse, target, rgb_near, args):
    del args
    return _evaluation_transform(
        rgb, sparse, target, rgb_near, transforms, random_crop=False
    )


def no_transform(rgb, sparse, target, rgb_near, args):
    return rgb, sparse, target, rgb_near


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert("L"))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset"""

    def __init__(self, split, args, trainvalsplit=0.9):
        initialize_kitti_depth(
            self, split, args, trainvalsplit, get_paths_and_transform
        )

    def __getitem__(self, index):
        rgb, sparse, target, rgb_near = self._get_raw(index)
        orig_sz = np.shape(sparse)
        rgb, sparse, target, rgb_near = self.transform(
            rgb, sparse, target, rgb_near, self.args
        )

        rgb, gray = rgb, None
        if rgb is not None:
            rgb = rgb / 255.0
        if gray is not None:
            gray = gray / 255.0

        candidates = {
            "rgb": rgb,
            "d": sparse,
            "gt": target,
            "g": gray,
            "rgb_near": rgb_near,
            "orig_sz": np.zeros(orig_sz),
        }
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items()
            if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths["gt"])

    def _get_raw(self, index):
        rgb, sparse, target = read_kitti_sample(self, index)
        rgb_near = (
            get_rgb_near(self.paths["rgb"][index], self.args)
            if self.split == "train"
            else None
        )
        return rgb, sparse, target, rgb_near
