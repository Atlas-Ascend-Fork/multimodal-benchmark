# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Second-stage preprocessing built on the canonical CLOCs pipeline."""

__all__ = ["merge_second_batch", "prep_pointcloud"]

import pathlib

import numpy as np
from PIL import Image

from second.data import kitti_common as kitti
from second.data import preprocess as base_preprocess


merge_second_batch = base_preprocess.merge_second_batch


def _add_transient_2d_boxes(call_args, call_kwargs):
    """Supply the base pipeline's filtered field without exposing it in output."""
    source = call_args[0] if call_args else call_kwargs["input_dict"]
    if "gt_names" not in source or "gt_2d_bbox" in source:
        return call_args, call_kwargs
    source = source.copy()
    source["gt_2d_bbox"] = np.zeros((len(source["gt_names"]), 4), dtype=np.float32)
    if call_args:
        return (source,) + call_args[1:], call_kwargs
    call_kwargs = call_kwargs.copy()
    call_kwargs["input_dict"] = source
    return call_args, call_kwargs


def prep_pointcloud(*call_args, **call_kwargs):
    """Run base preprocessing and remove fields unused by the fusion stage."""
    call_args, call_kwargs = _add_transient_2d_boxes(call_args, call_kwargs)
    example = base_preprocess.prep_pointcloud(*call_args, **call_kwargs)
    example.pop("d3_gt_boxes", None)
    example.pop("gt_2d_boxes", None)
    return example


def _read_and_prep_v9(info, root_path, num_point_features, prep_func):
    """Read point-cloud and image inputs before applying second-stage prep."""
    velocity_path = pathlib.Path(root_path) / info["velodyne_path"]
    velocity_path = (
        velocity_path.parent.parent
        / (velocity_path.parent.stem + "_reduced")
        / velocity_path.name
    )
    image_path = pathlib.Path(root_path) / info["img_path"]
    points = np.fromfile(str(velocity_path), dtype=np.float32).reshape(
        [-1, num_point_features]
    )
    with Image.open(str(image_path)) as image:
        image_data = np.array(image.convert("RGB"))[:, :, [2, 1, 0]]
    image_idx = info["image_idx"]
    input_dict = {
        "points": points,
        "rect": info["calib/R0_rect"].astype(np.float32),
        "Trv2c": info["calib/Tr_velo_to_cam"].astype(np.float32),
        "P2": info["calib/P2"].astype(np.float32),
        "image_shape": np.array(info["img_shape"], dtype=np.int32),
        "image_idx": image_idx,
        "image_path": info["img_path"],
        "images": image_data,
    }
    if "annos" in info:
        annos = kitti.remove_dontcare(info["annos"])
        input_dict.update(
            {
                "gt_boxes": np.concatenate(
                    [
                        annos["location"],
                        annos["dimensions"],
                        annos["rotation_y"][..., None],
                    ],
                    axis=1,
                ).astype(np.float32),
                "gt_names": annos["name"],
                "difficulty": annos["difficulty"],
            }
        )
        if "group_ids" in annos:
            input_dict["group_ids"] = annos["group_ids"]
    return base_preprocess.finalize_example(prep_func, input_dict, image_idx)
