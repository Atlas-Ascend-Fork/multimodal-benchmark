# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections import namedtuple
from typing import NamedTuple
import logging
import os
import pickle

from lib.config import cfg
from lib.datasets.kitti_dataset import KittiDataset
from torch.nn.functional import grid_sample
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import numpy as np
from lib.datasets.generation_utils import (
    filter_lidar_sample, load_lidar_sample, populate_point_sample, sample_point_indices,
)


DatabaseGroundTruthResult = namedtuple(
    "DatabaseGroundTruthResult", "box points features object enlarged corners"
)
from shared.dataset_geometry import ConfiguredGeometryMixin, filter_objects
import torch
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters

_HELPER_CONTINUE = object()


class KittircnndatasetInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    root_dir: object
    npoints: object
    split: object
    classes: object
    mode: object
    random_select: object
    logger: object
    rcnn_training_roi_dir: object
    rcnn_training_feature_dir: object
    rcnn_eval_roi_dir: object
    rcnn_eval_feature_dir: object
    gt_database_dir: object


class KittircnndatasetDataAugmentationParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    aug_pts_rect: object
    aug_gt_boxes3d: object
    gt_alpha: object
    sample_id: object
    mustaug: object
    stage: object


LOGGER = logging.getLogger(__name__)


def interpolate_img_by_xy(img, xy, normal_shape):
    """
    :param img:(H,W,c)
    :param xy:(N,2) (x,y)->(w,h)
    :param normal_shape:(2),H_size and W_size
    :return:interpolated features (N,3)
    """
    channel = img.shape[-1]
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    xy = xy * 2 / (normal_shape - 1.0) - 1.0

    xy = torch.from_numpy(xy).view(1, 1, -1, 2)
    ret_img = grid_sample(img, xy, padding_mode="zeros", mode="bilinear")
    ret_img = ret_img.view(channel, -1).permute(1, 0).numpy()

    return ret_img


_KITTIRCNNDATASET_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    KittircnndatasetInitParameters,
    (
        "root_dir",
        "npoints",
        "split",
        "classes",
        "mode",
        "random_select",
        "logger",
        "rcnn_training_roi_dir",
        "rcnn_training_feature_dir",
        "rcnn_eval_roi_dir",
        "rcnn_eval_feature_dir",
        "gt_database_dir",
    ),
    {
        "npoints": 16384,
        "split": "train",
        "classes": "Car",
        "mode": "TRAIN",
        "random_select": True,
        "logger": None,
        "rcnn_training_roi_dir": None,
        "rcnn_training_feature_dir": None,
        "rcnn_eval_roi_dir": None,
        "rcnn_eval_feature_dir": None,
        "gt_database_dir": None,
    },
    frozenset(()),
    12,
    None,
    None,
)
_KITTIRCNNDATASET_DATA_AUGMENTATION_PARAMETERS_SCHEMA = _ParameterSchema(
    KittircnndatasetDataAugmentationParameters,
    ("aug_pts_rect", "aug_gt_boxes3d", "gt_alpha", "sample_id", "mustaug", "stage"),
    {"sample_id": None, "mustaug": False, "stage": 1},
    frozenset(()),
    6,
    None,
    None,
)


def _kittircnndataset_data_augmentation_992_part_1_chunk_1(_state):
    _state["bound_parameters"] = _bind_parameters(
        _KITTIRCNNDATASET_DATA_AUGMENTATION_PARAMETERS_SCHEMA,
        _state["call_args"],
        _state["call_kwargs"],
    )
    (
        _state["aug_pts_rect"],
        _state["aug_gt_boxes3d"],
        _state["gt_alpha"],
        _state["sample_id"],
        _state["mustaug"],
        _state["stage"],
    ) = _state["bound_parameters"]
    _state["aug_list"] = cfg.AUG_METHOD_LIST
    _state["aug_enable"] = 1 - np.random.rand(3)
    if _state["mustaug"]:
        _state["aug_enable"][0] = -1
        _state["aug_enable"][1] = -1
    _state["aug_method"] = []


def _kittircnndataset_data_augmentation_992_part_1_chunk_2(_state):
    if (
        "rotation" in _state["aug_list"]
        and _state["aug_enable"][0] < cfg.AUG_METHOD_PROB[0]
    ):
        _state["angle"] = np.random.uniform(
            -np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE
        )
        _state["aug_pts_rect"] = kitti_utils.rotate_pc_along_y(
            _state["aug_pts_rect"], rot_angle=_state["angle"]
        )
        if _state["stage"] == 1:
            _state["aug_gt_boxes3d"] = kitti_utils.rotate_pc_along_y(
                _state["aug_gt_boxes3d"], rot_angle=_state["angle"]
            )
            _state["x"], _state["z"] = (
                _state["aug_gt_boxes3d"][:, 0],
                _state["aug_gt_boxes3d"][:, 2],
            )
            _state["beta"] = np.arctan2(_state["z"], _state["x"])
            _state["new_ry"] = (
                np.sign(_state["beta"]) * np.pi / 2
                + _state["gt_alpha"]
                - _state["beta"]
            )
            _state["aug_gt_boxes3d"][:, 6] = _state["new_ry"]
        elif _state["stage"] == 2:
            if not _state["aug_gt_boxes3d"].shape[0] == 2:
                raise AssertionError
            _state["aug_gt_boxes3d"][0] = _state["self"].rotate_box3d_along_y(
                _state["aug_gt_boxes3d"][0], _state["angle"]
            )
            _state["aug_gt_boxes3d"][1] = _state["self"].rotate_box3d_along_y(
                _state["aug_gt_boxes3d"][1], _state["angle"]
            )
        else:
            raise NotImplementedError
        _state["aug_method"].append(["rotation", _state["angle"]])


def _kittircnndataset_data_augmentation_992_part_1(_state):
    _kittircnndataset_data_augmentation_992_part_1_chunk_1(_state)
    _kittircnndataset_data_augmentation_992_part_1_chunk_2(_state)


def _kittircnndataset_data_augmentation_992_part_2(_state):
    if (
        "scaling" in _state["aug_list"]
        and _state["aug_enable"][1] < cfg.AUG_METHOD_PROB[1]
    ):
        _state["scale"] = np.random.uniform(0.95, 1.05)
        _state["aug_pts_rect"] = _state["aug_pts_rect"] * _state["scale"]
        _state["aug_gt_boxes3d"][:, 0:6] = (
            _state["aug_gt_boxes3d"][:, 0:6] * _state["scale"]
        )
        _state["aug_method"].append(["scaling", _state["scale"]])
    if (
        "flip" in _state["aug_list"]
        and _state["aug_enable"][2] < cfg.AUG_METHOD_PROB[2]
    ):
        _state["aug_pts_rect"][:, 0] = -_state["aug_pts_rect"][:, 0]
        _state["aug_gt_boxes3d"][:, 0] = -_state["aug_gt_boxes3d"][:, 0]
        if _state["stage"] == 1:
            _state["aug_gt_boxes3d"][:, 6] = (
                np.sign(_state["aug_gt_boxes3d"][:, 6]) * np.pi
                - _state["aug_gt_boxes3d"][:, 6]
            )
        elif _state["stage"] == 2:
            if not _state["aug_gt_boxes3d"].shape[0] == 2:
                raise AssertionError
            _state["aug_gt_boxes3d"][0, 6] = (
                np.sign(_state["aug_gt_boxes3d"][0, 6]) * np.pi
                - _state["aug_gt_boxes3d"][0, 6]
            )
            _state["aug_gt_boxes3d"][1, 6] = (
                np.sign(_state["aug_gt_boxes3d"][1, 6]) * np.pi
                - _state["aug_gt_boxes3d"][1, 6]
            )
        else:
            raise NotImplementedError
        _state["aug_method"].append("flip")


def _kittircnndataset_get_proposal_from_file_1188_part_1(_state):
    _state["sample_id"] = int(_state["self"].image_idx_list[_state["index"]])
    _state["proposal_file"] = os.path.join(
        _state["self"].rcnn_eval_roi_dir, "%06d.txt" % _state["sample_id"]
    )
    _state["roi_obj_list"] = kitti_utils.get_objects_from_label(_state["proposal_file"])
    (
        _state["rpn_xyz"],
        _state["rpn_features"],
        _state["rpn_intensity"],
        _state["seg_mask"],
    ) = _state["self"].get_rpn_features(
        _state["self"].rcnn_eval_feature_dir, _state["sample_id"]
    )
    _state["pts_rect"], _state["pts_rpn_features"], _state["pts_intensity"] = (
        _state["rpn_xyz"],
        _state["rpn_features"],
        _state["rpn_intensity"],
    )
    _state["roi_box3d_list"], _state["roi_scores"] = ([], [])
    for _state["obj"] in _state["roi_obj_list"]:
        _state["box3d"] = np.array(
            [
                _state["obj"].pos[0],
                _state["obj"].pos[1],
                _state["obj"].pos[2],
                _state["obj"].h,
                _state["obj"].w,
                _state["obj"].length_value,
                _state["obj"].ry,
            ],
            dtype=np.float32,
        )
        _state["roi_box3d_list"].append(_state["box3d"].reshape(1, 7))
        _state["roi_scores"].append(_state["obj"].score)
    _state["roi_boxes3d"] = np.concatenate(_state["roi_box3d_list"], axis=0)
    _state["roi_scores"] = np.array(_state["roi_scores"], dtype=np.float32)


def _kittircnndataset_get_proposal_from_file_1188_part_2(_state):
    if cfg.RCNN.ROI_SAMPLE_JIT:
        _state["sample_dict"] = {
            "sample_id": _state["sample_id"],
            "rpn_xyz": _state["rpn_xyz"],
            "rpn_features": _state["rpn_features"],
            "seg_mask": _state["seg_mask"],
            "roi_boxes3d": _state["roi_boxes3d"],
            "roi_scores": _state["roi_scores"],
            "pts_depth": np.linalg.norm(_state["rpn_xyz"], ord=2, axis=1),
        }
        if _state["self"].mode != "TEST":
            _state["gt_obj_list"] = _state["self"].filtrate_objects(
                _state["self"].get_label(_state["sample_id"])
            )
            _state["gt_boxes3d"] = kitti_utils.objs_to_boxes3d(_state["gt_obj_list"])
            _state["roi_corners"] = kitti_utils.boxes3d_to_corners3d(
                _state["roi_boxes3d"]
            )
            _state["gt_corners"] = kitti_utils.boxes3d_to_corners3d(
                _state["gt_boxes3d"]
            )
            _state["iou3d"] = kitti_utils.get_iou3d(
                _state["roi_corners"], _state["gt_corners"]
            )
            if _state["gt_boxes3d"].shape[0] > 0:
                _state["gt_iou"] = _state["iou3d"].max(axis=1)
            else:
                _state["gt_iou"] = np.zeros(_state["roi_boxes3d"].shape[0]).astype(
                    np.float32
                )
            _state["sample_dict"]["gt_boxes3d"] = _state["gt_boxes3d"]
            _state["sample_dict"]["gt_iou"] = _state["gt_iou"]
        return _state["sample_dict"]
    if cfg.RCNN.USE_INTENSITY:
        _state["pts_extra_input_list"] = [
            _state["pts_intensity"].reshape(-1, 1),
            _state["seg_mask"].reshape(-1, 1),
        ]
    else:
        _state["pts_extra_input_list"] = [_state["seg_mask"].reshape(-1, 1)]
    return _HELPER_CONTINUE


def _kittircnndataset_get_proposal_from_file_1188_part_3(_state):
    if cfg.RCNN.USE_DEPTH:
        _state["cur_depth"] = np.linalg.norm(_state["pts_rect"], axis=1, ord=2)
        _state["cur_depth_norm"] = _state["cur_depth"] / 70.0 - 0.5
        _state["pts_extra_input_list"].append(_state["cur_depth_norm"].reshape(-1, 1))
    _state["pts_extra_input"] = np.concatenate(_state["pts_extra_input_list"], axis=1)
    _state["pts_input"], _state["pts_features"] = roipool3d_utils.roipool3d_cpu(
        _state["roi_boxes3d"],
        _state["pts_rect"],
        _state["pts_rpn_features"],
        _state["pts_extra_input"],
        cfg.RCNN.POOL_EXTRA_WIDTH,
        sampled_pt_num=cfg.RCNN.NUM_POINTS,
    )
    _state["sample_dict"] = {
        "sample_id": _state["sample_id"],
        "pts_input": _state["pts_input"],
        "pts_features": _state["pts_features"],
        "roi_boxes3d": _state["roi_boxes3d"],
        "roi_scores": _state["roi_scores"],
        "roi_size": _state["roi_boxes3d"][:, 3:6],
    }
    if _state["self"].mode == "TEST":
        return _state["sample_dict"]
    _state["gt_obj_list"] = _state["self"].filtrate_objects(
        _state["self"].get_label(_state["sample_id"])
    )
    _state["gt_boxes3d"] = np.zeros(
        (_state["gt_obj_list"].__len__(), 7), dtype=np.float32
    )
    return _HELPER_CONTINUE


def _kittircnndataset_get_proposal_from_file_1188_part_4(_state):
    for _state["k"], _state["obj"] in enumerate(_state["gt_obj_list"]):
        (
            _state["gt_boxes3d"][_state["k"], 0:3],
            _state["gt_boxes3d"][_state["k"], 3],
            _state["gt_boxes3d"][_state["k"], 4],
            _state["gt_boxes3d"][_state["k"], 5],
            _state["gt_boxes3d"][_state["k"], 6],
        ) = (
            _state["obj"].pos,
            _state["obj"].h,
            _state["obj"].w,
            _state["obj"].length_value,
            _state["obj"].ry,
        )
    if _state["gt_boxes3d"].__len__() == 0:
        _state["gt_iou"] = np.zeros(_state["roi_boxes3d"].shape[0], dtype=np.float32)
    else:
        _state["roi_corners"] = kitti_utils.boxes3d_to_corners3d(_state["roi_boxes3d"])
        _state["gt_corners"] = kitti_utils.boxes3d_to_corners3d(_state["gt_boxes3d"])
        _state["iou3d"] = kitti_utils.get_iou3d(
            _state["roi_corners"], _state["gt_corners"]
        )
        _state["gt_iou"] = _state["iou3d"].max(axis=1)
    _state["sample_dict"]["gt_boxes3d"] = _state["gt_boxes3d"]
    _state["sample_dict"]["gt_iou"] = _state["gt_iou"]


def _kittircnndataset_get_rcnn_sample_info_1057_part_1_chunk_1(_state):
    _state["sample_id"], _state["gt_box3d"] = (
        _state["roi_info"]["sample_id"],
        _state["roi_info"]["gt_box3d"],
    )
    (
        _state["rpn_xyz"],
        _state["rpn_features"],
        _state["rpn_intensity"],
        _state["seg_mask"],
    ) = _state["self"].rpn_feature_list[_state["sample_id"]]
    _state["roi_box3d"] = _state["self"].aug_roi_by_noise(_state["roi_info"])
    _state["pooled_boxes3d"] = kitti_utils.enlarge_box3d(
        _state["roi_box3d"].reshape(1, 7), cfg.RCNN.POOL_EXTRA_WIDTH
    )
    _state["boxes_pts_mask_list"] = roipool3d_utils.pts_in_boxes3d_cpu(
        torch.from_numpy(_state["rpn_xyz"]), torch.from_numpy(_state["pooled_boxes3d"])
    )
    _state["pt_mask_flag"] = _state["boxes_pts_mask_list"][0].numpy() == 1
    _state["cur_pts"] = _state["rpn_xyz"][_state["pt_mask_flag"]].astype(np.float32)
    _state["aug_pts"] = _state["cur_pts"].copy()
    _state["aug_gt_box3d"] = _state["gt_box3d"].copy().astype(np.float32)
    _state["aug_roi_box3d"] = _state["roi_box3d"].copy()


def _augment_roi_and_gt(_state):
    boxes = np.stack((_state["aug_roi_box3d"], _state["aug_gt_box3d"]))
    beta = np.arctan2(boxes[:, 2], boxes[:, 0]).astype(np.float64)
    alpha = -np.sign(beta) * np.pi / 2 + beta + boxes[:, 6]
    _state["aug_pts"], augmented_boxes, _state["aug_method"] = _state[
        "self"
    ].data_augmentation(
        _state["aug_pts"], boxes, alpha, mustaug=True, stage=2
    )
    _state["aug_roi_box3d"], _state["aug_gt_box3d"] = augmented_boxes


def _kittircnndataset_get_rcnn_sample_info_1057_part_1_chunk_2(_state):
    if cfg.AUG_DATA and _state["self"].mode == "TRAIN":
        _augment_roi_and_gt(_state)
        _state["aug_gt_box3d"] = _state["aug_gt_box3d"].astype(_state["gt_box3d"].dtype)
    _state["valid_mask"] = 1


def _kittircnndataset_get_rcnn_sample_info_1057_part_1(_state):
    _kittircnndataset_get_rcnn_sample_info_1057_part_1_chunk_1(_state)
    _kittircnndataset_get_rcnn_sample_info_1057_part_1_chunk_2(_state)


def _kittircnndataset_get_rcnn_sample_info_1057_part_2_chunk_1(_state):
    if _state["aug_pts"].shape[0] == 0:
        _state["pts_features"] = np.zeros((1, 128), dtype=np.float32)
        _state["input_channel"] = (
            3
            + int(cfg.RCNN.USE_INTENSITY)
            + int(cfg.RCNN.USE_MASK)
            + int(cfg.RCNN.USE_DEPTH)
        )
        _state["pts_input"] = np.zeros((1, _state["input_channel"]), dtype=np.float32)
        _state["valid_mask"] = 0
    else:
        _state["pts_features"] = _state["rpn_features"][_state["pt_mask_flag"]].astype(
            np.float32
        )
        _state["pts_intensity"] = _state["rpn_intensity"][
            _state["pt_mask_flag"]
        ].astype(np.float32)
        _state["pts_input_list"] = [
            _state["aug_pts"],
            _state["pts_intensity"].reshape(-1, 1),
        ]
        if cfg.RCNN.USE_INTENSITY:
            _state["pts_input_list"] = [
                _state["aug_pts"],
                _state["pts_intensity"].reshape(-1, 1),
            ]
        else:
            _state["pts_input_list"] = [_state["aug_pts"]]
        if cfg.RCNN.USE_MASK:
            if cfg.RCNN.MASK_TYPE == "seg":
                _state["pts_mask"] = _state["seg_mask"][_state["pt_mask_flag"]].astype(
                    np.float32
                )
            elif cfg.RCNN.MASK_TYPE == "roi":
                _state["pts_mask"] = roipool3d_utils.pts_in_boxes3d_cpu(
                    torch.from_numpy(_state["aug_pts"]),
                    torch.from_numpy(_state["aug_roi_box3d"].reshape(1, 7)),
                )
                _state["pts_mask"] = (_state["pts_mask"][0].numpy() == 1).astype(
                    np.float32
                )
            else:
                raise NotImplementedError
            _state["pts_input_list"].append(_state["pts_mask"].reshape(-1, 1))
        if cfg.RCNN.USE_DEPTH:
            _state["pts_depth"] = np.linalg.norm(_state["aug_pts"], axis=1, ord=2)
            _state["pts_depth_norm"] = _state["pts_depth"] / 70.0 - 0.5
            _state["pts_input_list"].append(_state["pts_depth_norm"].reshape(-1, 1))
        _state["pts_input"] = np.concatenate(_state["pts_input_list"], axis=1)


def _kittircnndataset_get_rcnn_sample_info_1057_part_2_chunk_2(_state):
    _state["aug_gt_corners"] = kitti_utils.boxes3d_to_corners3d(
        _state["aug_gt_box3d"].reshape(-1, 7)
    )


def _kittircnndataset_get_rcnn_sample_info_1057_part_2(_state):
    _kittircnndataset_get_rcnn_sample_info_1057_part_2_chunk_1(_state)
    _kittircnndataset_get_rcnn_sample_info_1057_part_2_chunk_2(_state)


def _kittircnndataset_get_rcnn_sample_info_1057_part_3(_state):
    _state["aug_roi_corners"] = kitti_utils.boxes3d_to_corners3d(
        _state["aug_roi_box3d"].reshape(-1, 7)
    )
    _state["iou3d"] = kitti_utils.get_iou3d(
        _state["aug_roi_corners"], _state["aug_gt_corners"]
    )
    _state["cur_iou"] = _state["iou3d"][0][0]
    _state["reg_valid_mask"] = (
        1
        if _state["cur_iou"] >= cfg.RCNN.REG_FG_THRESH and _state["valid_mask"] == 1
        else 0
    )
    _state["cls_label"] = 1 if _state["cur_iou"] > cfg.RCNN.CLS_FG_THRESH else 0
    if (
        cfg.RCNN.CLS_BG_THRESH < _state["cur_iou"] < cfg.RCNN.CLS_FG_THRESH
        or _state["valid_mask"] == 0
    ):
        _state["cls_label"] = -1
    _state["pts_input_ct"], _state["gt_box3d_ct"] = _state["self"].canonical_transform(
        _state["pts_input"], _state["aug_roi_box3d"], _state["aug_gt_box3d"]
    )
    _state["pts_input_ct"], _state["pts_features"] = _state["self"].rcnn_input_sample(
        _state["pts_input_ct"], _state["pts_features"]
    )
    _state["sample_info"] = {
        "sample_id": _state["sample_id"],
        "pts_input": _state["pts_input_ct"],
        "pts_features": _state["pts_features"],
        "cls_label": _state["cls_label"],
        "reg_valid_mask": _state["reg_valid_mask"],
        "gt_boxes3d_ct": _state["gt_box3d_ct"],
        "roi_boxes3d": _state["aug_roi_box3d"],
        "roi_size": _state["aug_roi_box3d"][3:6],
        "gt_boxes3d": _state["aug_gt_box3d"],
    }


def _kittircnndataset_get_rcnn_training_sample_batch_1290_part_1_chunk_1(_state):
    _state["sample_id"] = int(_state["self"].sample_id_list[_state["index"]])
    (
        _state["rpn_xyz"],
        _state["rpn_features"],
        _state["rpn_intensity"],
        _state["seg_mask"],
    ) = _state["self"].get_rpn_features(
        _state["self"].rcnn_training_feature_dir, _state["sample_id"]
    )
    _state["roi_file"] = os.path.join(
        _state["self"].rcnn_training_roi_dir, "%06d.txt" % _state["sample_id"]
    )
    _state["roi_obj_list"] = kitti_utils.get_objects_from_label(_state["roi_file"])
    _state["roi_boxes3d"] = kitti_utils.objs_to_boxes3d(_state["roi_obj_list"])
    _state["gt_obj_list"] = _state["self"].filtrate_objects(
        _state["self"].get_label(_state["sample_id"])
    )
    _state["gt_boxes3d"] = kitti_utils.objs_to_boxes3d(_state["gt_obj_list"])
    _state["iou3d"] = kitti_utils.get_iou3d(
        kitti_utils.boxes3d_to_corners3d(_state["roi_boxes3d"]),
        kitti_utils.boxes3d_to_corners3d(_state["gt_boxes3d"]),
    )
    _state["max_overlaps"], _state["gt_assignment"] = (
        _state["iou3d"].max(axis=1),
        _state["iou3d"].argmax(axis=1),
    )
    _state["max_iou_of_gt"], _state["roi_assignment"] = (
        _state["iou3d"].max(axis=0),
        _state["iou3d"].argmax(axis=0),
    )
    _state["roi_assignment"] = _state["roi_assignment"][
        _state["max_iou_of_gt"] > 0
    ].reshape(-1)
    _state["fg_rois_per_image"] = int(
        np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE)
    )
    _state["fg_thresh"] = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
    _state["fg_inds"] = np.nonzero(_state["max_overlaps"] >= _state["fg_thresh"])[0]
    _state["fg_inds"] = np.concatenate(
        (_state["fg_inds"], _state["roi_assignment"]), axis=0
    )
    _state["easy_bg_inds"] = np.nonzero(
        _state["max_overlaps"] < cfg.RCNN.CLS_BG_THRESH_LO
    )[0]


def _kittircnndataset_get_rcnn_training_sample_batch_1290_part_1_chunk_2(_state):
    _state["hard_bg_inds"] = np.nonzero(
        (_state["max_overlaps"] < cfg.RCNN.CLS_BG_THRESH)
        & (_state["max_overlaps"] >= cfg.RCNN.CLS_BG_THRESH_LO)
    )[0]
    _state["fg_num_rois"] = _state["fg_inds"].size
    _state["bg_num_rois"] = _state["hard_bg_inds"].size + _state["easy_bg_inds"].size


def _kittircnndataset_get_rcnn_training_sample_batch_1290_part_1(_state):
    _kittircnndataset_get_rcnn_training_sample_batch_1290_part_1_chunk_1(_state)
    _kittircnndataset_get_rcnn_training_sample_batch_1290_part_1_chunk_2(_state)


def _kittircnndataset_get_rcnn_training_sample_batch_1290_part_2(_state):
    if _state["fg_num_rois"] > 0 and _state["bg_num_rois"] > 0:
        _state["fg_rois_per_this_image"] = min(
            _state["fg_rois_per_image"], _state["fg_num_rois"]
        )
        _state["rand_num"] = np.random.permutation(_state["fg_num_rois"])
        _state["fg_inds"] = _state["fg_inds"][
            _state["rand_num"][: _state["fg_rois_per_this_image"]]
        ]
        _state["bg_rois_per_this_image"] = (
            cfg.RCNN.ROI_PER_IMAGE - _state["fg_rois_per_this_image"]
        )
        _state["bg_inds"] = _state["self"].sample_bg_inds(
            _state["hard_bg_inds"],
            _state["easy_bg_inds"],
            _state["bg_rois_per_this_image"],
        )
    elif _state["fg_num_rois"] > 0 and _state["bg_num_rois"] == 0:
        _state["rand_num"] = np.floor(
            np.random.rand(cfg.RCNN.ROI_PER_IMAGE) * _state["fg_num_rois"]
        )
        _state["rand_num"] = (
            torch.from_numpy(_state["rand_num"]).type_as(_state["gt_boxes3d"]).long()
        )
        _state["fg_inds"] = _state["fg_inds"][_state["rand_num"]]
        _state["fg_rois_per_this_image"] = cfg.RCNN.ROI_PER_IMAGE
        _state["bg_rois_per_this_image"] = 0
    elif _state["bg_num_rois"] > 0 and _state["fg_num_rois"] == 0:
        _state["bg_rois_per_this_image"] = cfg.RCNN.ROI_PER_IMAGE
        _state["bg_inds"] = _state["self"].sample_bg_inds(
            _state["hard_bg_inds"],
            _state["easy_bg_inds"],
            _state["bg_rois_per_this_image"],
        )
        _state["fg_rois_per_this_image"] = 0
    else:
        raise RuntimeError("no foreground or background regions available")
    _state["roi_list"], _state["roi_iou_list"], _state["roi_gt_list"] = ([], [], [])
    if _state["fg_rois_per_this_image"] > 0:
        _state["fg_rois_src"] = _state["roi_boxes3d"][_state["fg_inds"]].copy()
        _state["gt_of_fg_rois"] = _state["gt_boxes3d"][
            _state["gt_assignment"][_state["fg_inds"]]
        ]
        _state["fg_rois"], _state["fg_iou3d"] = _state["self"].aug_roi_by_noise_batch(
            _state["fg_rois_src"], _state["gt_of_fg_rois"], aug_times=10
        )
        _state["roi_list"].append(_state["fg_rois"])
        _state["roi_iou_list"].append(_state["fg_iou3d"])
        _state["roi_gt_list"].append(_state["gt_of_fg_rois"])


def _kittircnndataset_get_rcnn_training_sample_batch_1290_part_3(_state):
    if _state["bg_rois_per_this_image"] > 0:
        _state["bg_rois_src"] = _state["roi_boxes3d"][_state["bg_inds"]].copy()
        _state["gt_of_bg_rois"] = _state["gt_boxes3d"][
            _state["gt_assignment"][_state["bg_inds"]]
        ]
        _state["bg_rois"], _state["bg_iou3d"] = _state["self"].aug_roi_by_noise_batch(
            _state["bg_rois_src"], _state["gt_of_bg_rois"], aug_times=1
        )
        _state["roi_list"].append(_state["bg_rois"])
        _state["roi_iou_list"].append(_state["bg_iou3d"])
        _state["roi_gt_list"].append(_state["gt_of_bg_rois"])
    _state["rois"] = np.concatenate(_state["roi_list"], axis=0)
    _state["iou_of_rois"] = np.concatenate(_state["roi_iou_list"], axis=0)
    _state["gt_of_rois"] = np.concatenate(_state["roi_gt_list"], axis=0)
    if cfg.RCNN.USE_INTENSITY:
        _state["pts_extra_input_list"] = [
            _state["rpn_intensity"].reshape(-1, 1),
            _state["seg_mask"].reshape(-1, 1),
        ]
    else:
        _state["pts_extra_input_list"] = [_state["seg_mask"].reshape(-1, 1)]
    if cfg.RCNN.USE_DEPTH:
        _state["pts_depth"] = (
            np.linalg.norm(_state["rpn_xyz"], ord=2, axis=1) / 70.0 - 0.5
        )
        _state["pts_extra_input_list"].append(_state["pts_depth"].reshape(-1, 1))
    _state["pts_extra_input"] = np.concatenate(_state["pts_extra_input_list"], axis=1)
    _state["pts_input"], _state["pts_features"], _state["pts_empty_flag"] = (
        roipool3d_utils.roipool3d_cpu(
            _state["rois"],
            _state["rpn_xyz"],
            _state["rpn_features"],
            _state["pts_extra_input"],
            cfg.RCNN.POOL_EXTRA_WIDTH,
            sampled_pt_num=cfg.RCNN.NUM_POINTS,
            canonical_transform=False,
        )
    )


def _kittircnndataset_get_rcnn_training_sample_batch_1290_part_4_chunk_1(_state):
    if cfg.AUG_DATA and _state["self"].mode == "TRAIN":
        for _state["k"] in range(_state["rois"].__len__()):
            _state["aug_pts"] = _state["pts_input"][_state["k"], :, 0:3].copy()
            _state["aug_gt_box3d"] = _state["gt_of_rois"][_state["k"]].copy()
            _state["aug_roi_box3d"] = _state["rois"][_state["k"]].copy()
            _augment_roi_and_gt(_state)
            _state["pts_input"][_state["k"], :, 0:3] = _state["aug_pts"]
            _state["rois"][_state["k"]] = _state["aug_roi_box3d"]
            _state["gt_of_rois"][_state["k"]] = _state["aug_gt_box3d"]
    _state["valid_mask"] = (_state["pts_empty_flag"] == 0).astype(np.int32)
    _state["reg_valid_mask"] = (_state["iou_of_rois"] > cfg.RCNN.REG_FG_THRESH).astype(
        np.int32
    ) & _state["valid_mask"]
    _state["cls_label"] = (_state["iou_of_rois"] > cfg.RCNN.CLS_FG_THRESH).astype(
        np.int32
    )


def _kittircnndataset_get_rcnn_training_sample_batch_1290_part_4_chunk_2(_state):
    _state["invalid_mask"] = (_state["iou_of_rois"] > cfg.RCNN.CLS_BG_THRESH) & (
        _state["iou_of_rois"] < cfg.RCNN.CLS_FG_THRESH
    )
    _state["cls_label"][_state["invalid_mask"]] = -1
    _state["cls_label"][_state["valid_mask"] == 0] = -1
    _state["pts_input_ct"], _state["gt_boxes3d_ct"] = _state[
        "self"
    ].canonical_transform_batch(
        _state["pts_input"], _state["rois"], _state["gt_of_rois"]
    )


def _kittircnndataset_get_rcnn_training_sample_batch_1290_part_4(_state):
    _kittircnndataset_get_rcnn_training_sample_batch_1290_part_4_chunk_1(_state)
    _kittircnndataset_get_rcnn_training_sample_batch_1290_part_4_chunk_2(_state)


def _kittircnndataset_get_rcnn_training_sample_batch_1290_part_5(_state):
    _state["sample_info"] = {
        "sample_id": _state["sample_id"],
        "pts_input": _state["pts_input_ct"],
        "pts_features": _state["pts_features"],
        "cls_label": _state["cls_label"],
        "reg_valid_mask": _state["reg_valid_mask"],
        "gt_boxes3d_ct": _state["gt_boxes3d_ct"],
        "roi_boxes3d": _state["rois"],
        "roi_size": _state["rois"][:, 3:6],
        "gt_boxes3d": _state["gt_of_rois"],
    }


def _kittircnndataset_get_rpn_sample_733_part_1(_state):
    _state["sample_id"] = int(_state["self"].sample_id_list[_state["index"]])
    if _state["sample_id"] < 10000:
        (_state["calib"], _state["img_shape"], _state["pts_rect"],
         _state["pts_intensity"]) = load_lidar_sample(
            _state["self"], _state["sample_id"]
        )
        _state["img"] = _state["self"].get_image_rgb_with_normal(_state["sample_id"])
    else:
        _state["calib"] = _state["self"].get_calib(_state["sample_id"] % 10000)
        _state["img"] = _state["self"].get_image_rgb_with_normal(
            _state["sample_id"] % 10000
        )
        _state["img_shape"] = _state["self"].get_image_shape(
            _state["sample_id"] % 10000
        )
        _state["pts_file"] = os.path.join(
            _state["self"].aug_pts_dir, "%06d.bin" % _state["sample_id"]
        )
        if not os.path.exists(_state["pts_file"]):
            raise AssertionError("%s" % _state["pts_file"])
        _state["aug_pts"] = np.fromfile(_state["pts_file"], dtype=np.float32).reshape(
            -1, 4
        )
        _state["pts_rect"], _state["pts_intensity"] = (
            _state["aug_pts"][:, 0:3],
            _state["aug_pts"][:, 3],
        )
    filtered = filter_lidar_sample(
        _state["self"], _state["calib"], _state["img_shape"],
        _state["pts_rect"], _state["pts_intensity"],
    )
    _state["pts_rect"], _state["pts_intensity"] = filtered[:2]
    _state["pts_in_img"] = filtered.image_points[filtered.valid]
    _state["shape"] = _state["self"].image_hw_with_padding_np
    _state["pts_rgb"] = interpolate_img_by_xy(
        _state["img"], _state["pts_in_img"], _state["shape"]
    )


def _kittircnndataset_get_rpn_sample_733_part_2(_state):
    if cfg.GT_AUG_ENABLED and _state["self"].mode == "TRAIN":
        _state["pts_features"] = np.concatenate(
            [_state["pts_intensity"].reshape(-1, 1), _state["pts_rgb"]], axis=1
        )
        _state["all_gt_obj_list"] = _state["self"].filtrate_dc_objects(
            _state["self"].get_label(_state["sample_id"])
        )
        _state["all_gt_boxes3d"] = kitti_utils.objs_to_boxes3d(
            _state["all_gt_obj_list"]
        )
        _state["gt_aug_flag"] = False
        if np.random.rand() < cfg.GT_AUG_APPLY_PROB:
            (
                _state["gt_aug_flag"],
                _state["pts_rect"],
                _state["pts_features"],
                _state["extra_gt_boxes3d"],
                _state["extra_gt_obj_list"],
            ) = _state["self"].apply_gt_aug_to_one_scene(
                _state["sample_id"],
                _state["pts_rect"],
                _state["pts_features"],
                _state["all_gt_boxes3d"],
            )
        _state["pts_intensity"], _state["pts_rgb"] = (
            _state["pts_features"][:, 0],
            _state["pts_features"][:, 1:],
        )


def _kittircnndataset_get_rpn_sample_733_part_3(_state):
    if _state["self"].mode == "TRAIN" or _state["self"].random_select:
        _state["choice"] = sample_point_indices(
            _state["pts_rect"], _state["self"].npoints
        )
        _state["ret_pts_rect"] = _state["pts_rect"][_state["choice"], :]
        _state["ret_pts_intensity"] = _state["pts_intensity"][_state["choice"]] - 0.5
        _state["ret_pts_rgb"] = _state["pts_rgb"][_state["choice"], :]
    else:
        _state["ret_pts_rect"] = _state["pts_rect"]
        _state["ret_pts_intensity"] = _state["pts_intensity"] - 0.5
        _state["ret_pts_rgb"] = _state["pts_rgb"]
    _state["pts_features"] = []
    if cfg.RPN.USE_INTENSITY:
        _state["pts_features"].append(_state["ret_pts_intensity"].reshape(-1, 1))


def _kittircnndataset_get_rpn_sample_733_part_4(_state):
    if cfg.RPN.USE_RGB:
        _state["pts_features"].append(_state["ret_pts_rgb"])
    _state["ret_pts_features"] = None
    if _state["pts_features"].__len__() >= 1:
        _state["ret_pts_features"] = np.concatenate(_state["pts_features"], axis=1)
    _state["sample_info"] = {
        "sample_id": _state["sample_id"],
        "random_select": _state["self"].random_select,
        "rgb": _state["ret_pts_rgb"],
    }
    if _state["self"].mode == "TEST":
        return populate_point_sample(
            _state["sample_info"], _state["ret_pts_rect"], _state["ret_pts_features"]
        )
    _state["gt_obj_list"] = _state["self"].filtrate_objects(
        _state["self"].get_label(_state["sample_id"])
    )
    if cfg.GT_AUG_ENABLED and _state["self"].mode == "TRAIN" and _state["gt_aug_flag"]:
        _state["gt_obj_list"].extend(_state["extra_gt_obj_list"])
    _state["gt_boxes3d"] = kitti_utils.objs_to_boxes3d(_state["gt_obj_list"])
    _state["gt_alpha"] = np.zeros(_state["gt_obj_list"].__len__(), dtype=np.float32)
    for _state["k"], _state["obj"] in enumerate(_state["gt_obj_list"]):
        _state["gt_alpha"][_state["k"]] = _state["obj"].alpha
    _state["aug_pts_rect"] = _state["ret_pts_rect"].copy()
    _state["aug_gt_boxes3d"] = _state["gt_boxes3d"].copy()
    if cfg.AUG_DATA and _state["self"].mode == "TRAIN":
        _state["aug_pts_rect"], _state["aug_gt_boxes3d"], _state["aug_method"] = _state[
            "self"
        ].data_augmentation(
            _state["aug_pts_rect"],
            _state["aug_gt_boxes3d"],
            _state["gt_alpha"],
            _state["sample_id"],
        )
        _state["sample_info"]["aug_method"] = _state["aug_method"]
    return _HELPER_CONTINUE


def _kittircnndataset_get_rpn_sample_733_part_5(_state):
    if cfg.RPN.USE_INTENSITY or cfg.RPN.USE_RGB:
        _state["pts_input"] = np.concatenate(
            (_state["aug_pts_rect"], _state["ret_pts_features"]), axis=1
        )
    else:
        _state["pts_input"] = _state["aug_pts_rect"]
    if cfg.RPN.FIXED:
        _state["sample_info"]["pts_input"] = _state["pts_input"]
        _state["sample_info"]["pts_rect"] = _state["aug_pts_rect"]
        _state["sample_info"]["pts_features"] = _state["ret_pts_features"]
        _state["sample_info"]["gt_boxes3d"] = _state["aug_gt_boxes3d"]
        return _state["sample_info"]
    _state["rpn_cls_label"], _state["rpn_reg_label"] = _state[
        "self"
    ].generate_rpn_training_labels(_state["aug_pts_rect"], _state["aug_gt_boxes3d"])
    _state["sample_info"]["pts_input"] = _state["pts_input"]
    _state["sample_info"]["pts_rect"] = _state["aug_pts_rect"]
    _state["sample_info"]["pts_features"] = _state["ret_pts_features"]
    _state["sample_info"]["rpn_cls_label"] = _state["rpn_cls_label"]
    _state["sample_info"]["rpn_reg_label"] = _state["rpn_reg_label"]
    _state["sample_info"]["gt_boxes3d"] = _state["aug_gt_boxes3d"]
    return _HELPER_CONTINUE


class KittiRCNNDataset(ConfiguredGeometryMixin, KittiDataset):
    geometry_config = cfg
    geometry_box_utils = kitti_utils

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIRCNNDATASET_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            root_dir,
            npoints,
            split,
            classes,
            mode,
            random_select,
            logger,
            rcnn_training_roi_dir,
            rcnn_training_feature_dir,
            rcnn_eval_roi_dir,
            rcnn_eval_feature_dir,
            gt_database_dir,
        ) = bound_parameters
        super().__init__(root_dir=root_dir, split=split)
        aug_scene_root_dir = self._configure_classes(root_dir, classes)
        self.num_class = len(self.classes)

        self.npoints = npoints
        self.sample_id_list = []
        self.random_select = random_select
        self.logger = logger

        self.aug_label_dir = os.path.join(aug_scene_root_dir, "training", "aug_label")
        self.aug_pts_dir = os.path.join(
            aug_scene_root_dir, "training", "rectified_data"
        )

        # for rcnn training
        self.rcnn_training_bbox_list = []
        self.rpn_feature_list = {}
        self.pos_bbox_list = []
        self.neg_bbox_list = []
        self.far_neg_bbox_list = []
        self.rcnn_eval_roi_dir = rcnn_eval_roi_dir
        self.rcnn_eval_feature_dir = rcnn_eval_feature_dir
        self.rcnn_training_roi_dir = rcnn_training_roi_dir
        self.rcnn_training_feature_dir = rcnn_training_feature_dir

        self.gt_database = None

        if not self.random_select:
            self.logger.warning("random select is False")

        if mode not in ["TRAIN", "EVAL", "TEST"]:
            raise ValueError("Invalid mode: %s" % mode)
        self.mode = mode

        if cfg.RPN.ENABLED:
            self._load_gt_database(gt_database_dir, logger)
        self._initialize_sample_ids(mode)

    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.sample_id_list)
        elif cfg.RCNN.ENABLED:
            if self.mode == "TRAIN":
                return len(self.sample_id_list)
            else:
                return len(self.image_idx_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if cfg.LI_FUSION.ENABLED:
            return self.get_rpn_with_li_fusion(index)

        if cfg.RPN.ENABLED:
            return self.get_rpn_sample(index)
        elif cfg.RCNN.ENABLED:
            if self.mode == "TRAIN":
                if cfg.RCNN.ROI_SAMPLE_JIT:
                    return self.get_rcnn_sample_jit(index)
                else:
                    return self.get_rcnn_training_sample_batch(index)
            else:
                return self.get_proposal_from_file(index)
        else:
            raise NotImplementedError

    @staticmethod
    def get_rpn_features(rpn_feature_dir, idx):
        rpn_feature_file = os.path.join(rpn_feature_dir, "%06d.npy" % idx)
        rpn_xyz_file = os.path.join(rpn_feature_dir, "%06d_xyz.npy" % idx)
        rpn_intensity_file = os.path.join(rpn_feature_dir, "%06d_intensity.npy" % idx)
        if cfg.RCNN.USE_SEG_SCORE:
            rpn_seg_file = os.path.join(rpn_feature_dir, "%06d_rawscore.npy" % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
            rpn_seg_score = torch.sigmoid(torch.from_numpy(rpn_seg_score)).numpy()
        else:
            rpn_seg_file = os.path.join(rpn_feature_dir, "%06d_seg.npy" % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
        return (
            np.load(rpn_xyz_file),
            np.load(rpn_feature_file),
            np.load(rpn_intensity_file).reshape(-1),
            rpn_seg_score,
        )

    @staticmethod
    def filtrate_dc_objects(obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type in ["DontCare"]:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    @staticmethod
    def canonical_transform(pts_input, roi_box3d, gt_box3d):
        roi_ry = roi_box3d[6] % (2 * np.pi)  # 0 ~ 2pi
        roi_center = roi_box3d[0:3]
        # shift to center
        pts_input[:, [0, 1, 2]] = pts_input[:, [0, 1, 2]] - roi_center
        gt_box3d_ct = np.copy(gt_box3d)
        gt_box3d_ct[0:3] = gt_box3d_ct[0:3] - roi_center
        # rotate to the direction of head
        gt_box3d_ct = kitti_utils.rotate_pc_along_y(
            gt_box3d_ct.reshape(1, 7), roi_ry
        ).reshape(7)
        gt_box3d_ct[6] = gt_box3d_ct[6] - roi_ry
        pts_input = kitti_utils.rotate_pc_along_y(pts_input, roi_ry)

        return pts_input, gt_box3d_ct

    @staticmethod
    def canonical_transform_batch(pts_input, roi_boxes3d, gt_boxes3d):
        """
        :param pts_input: (N, npoints, 3 + C)
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        roi_ry = roi_boxes3d[:, 6] % (2 * np.pi)  # 0 ~ 2pi
        roi_center = roi_boxes3d[:, 0:3]
        # shift to center
        pts_input[:, :, [0, 1, 2]] = pts_input[:, :, [0, 1, 2]] - roi_center.reshape(
            -1, 1, 3
        )
        gt_boxes3d_ct = np.copy(gt_boxes3d)
        gt_boxes3d_ct[:, 0:3] = gt_boxes3d_ct[:, 0:3] - roi_center
        # rotate to the direction of head
        gt_boxes3d_ct = (
            kitti_utils.rotate_pc_along_y_torch(
                torch.from_numpy(gt_boxes3d_ct.reshape(-1, 1, 7)),
                torch.from_numpy(roi_ry),
            )
            .numpy()
            .reshape(-1, 7)
        )
        gt_boxes3d_ct[:, 6] = gt_boxes3d_ct[:, 6] - roi_ry
        pts_input = kitti_utils.rotate_pc_along_y_torch(
            torch.from_numpy(pts_input), torch.from_numpy(roi_ry)
        ).numpy()

        return pts_input, gt_boxes3d_ct

    @staticmethod
    def rcnn_input_sample(pts_input, pts_features):
        choice = np.random.choice(pts_input.shape[0], cfg.RCNN.NUM_POINTS, replace=True)

        if pts_input.shape[0] < cfg.RCNN.NUM_POINTS:
            choice[: pts_input.shape[0]] = np.arange(pts_input.shape[0])
            np.random.shuffle(choice)
        pts_input = pts_input[choice]
        pts_features = pts_features[choice]

        return pts_input, pts_features

    @staticmethod
    def random_aug_box3d(box3d):
        """Randomly shift, scale, and rotate one 3D box."""
        method = cfg.RCNN.REG_AUG_METHOD
        if method == "single":
            return KittiRCNNDataset._scaled_aug_box3d(box3d, 0.5, 0.15, np.pi / 12)
        if method == "multiple":
            ranges = (
                (0.2, 0.1, np.pi / 12), (0.3, 0.15, np.pi / 12),
                (0.5, 0.15, np.pi / 9), (0.8, 0.15, np.pi / 6),
                (1.0, 0.15, np.pi / 3),
            )
            return KittiRCNNDataset._scaled_aug_box3d(
                box3d, *ranges[np.random.randint(len(ranges))]
            )
        if method == "normal":
            return KittiRCNNDataset._normal_aug_box3d(box3d)
        raise NotImplementedError

    @staticmethod
    def _scaled_aug_box3d(box3d, position_range, size_range, angle_range):
        position = ((np.random.rand(3) - 0.5) / 0.5) * position_range
        size = ((np.random.rand(3) - 0.5) / 0.5) * size_range + 1.0
        angle = ((np.random.rand(1) - 0.5) / 0.5) * angle_range
        return np.concatenate(
            [box3d[0:3] + position, box3d[3:6] * size, box3d[6:7] + angle]
        )

    @staticmethod
    def _normal_aug_box3d(box3d):
        shifts = np.array(
            [
                np.random.normal(0, 0.3), np.random.normal(0, 0.2),
                np.random.normal(0, 0.3), np.random.normal(0, 0.25),
                np.random.normal(0, 0.15), np.random.normal(0, 0.5),
                ((np.random.rand() - 0.5) / 0.5) * np.pi / 12,
            ]
        )
        return box3d + shifts

    @staticmethod
    def _rpn_point_input(points, features):
        if cfg.RPN.USE_INTENSITY:
            return np.concatenate((points, features), axis=1)
        return points

    @staticmethod
    def _append_augmented_gt(state, prepared, pts_rect):
        box, points, features, obj, enlarged, corners = prepared
        removal_box = box.copy()
        removal_box[3] += 2
        point_masks = roipool3d_utils.pts_in_boxes3d_cpu(
            torch.from_numpy(pts_rect), torch.from_numpy(removal_box.reshape(1, 7))
        )
        state["src_pts_flag"][point_masks[0].numpy() == 1] = 0
        state["new_points"].append(points)
        state["new_features"].append(features)
        state["boxes"] = np.concatenate((state["boxes"], enlarged.reshape(1, 7)))
        state["corners"] = np.concatenate((state["corners"], corners))
        state["extra_boxes"].append(box.reshape(1, 7))
        state["extra_objects"].append(obj)

    @staticmethod
    def _finish_gt_augmentation(state, pts_rect, point_features):
        if not state["new_points"]:
            return False, pts_rect, point_features, None, None
        keep = state["src_pts_flag"] == 1
        pts_rect = np.concatenate((pts_rect[keep], np.concatenate(state["new_points"])))
        point_features = np.concatenate(
            (point_features[keep], np.concatenate(state["new_features"]))
        )
        return (
            True,
            pts_rect,
            point_features,
            np.concatenate(state["extra_boxes"], axis=0),
            state["extra_objects"],
        )

    @staticmethod
    def _pad_batch_boxes(batch, key):
        max_boxes = max(item[key].__len__() for item in batch)
        boxes = np.zeros((len(batch), max_boxes, 7), dtype=np.float32)
        for index, item in enumerate(batch):
            boxes[index, : item[key].__len__(), :] = item[key]
        return boxes

    @staticmethod
    def _collate_batch_value(batch, key):
        first_value = batch[0][key]
        if isinstance(first_value, np.ndarray):
            return np.concatenate(
                [item[key][np.newaxis, ...] for item in batch], axis=0
            )
        values = [item[key] for item in batch]
        if isinstance(first_value, int):
            return np.array(values, dtype=np.int32)
        if isinstance(first_value, float):
            return np.array(values, dtype=np.float32)
        return values

    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        self.logger.info("Loading %s samples from %s ..." % (self.mode, self.label_dir))
        for idx in range(0, self.num_sample):
            sample_id = int(self.image_idx_list[idx])
            obj_list = self.filtrate_objects(self.get_label(sample_id))
            if len(obj_list) == 0:
                continue
            self.sample_id_list.append(sample_id)

        self.logger.info(
            "Done: filter %s results: %d / %d\n"
            % (self.mode, len(self.sample_id_list), len(self.image_idx_list))
        )

    def get_label(self, idx):
        if idx < 10000:
            label_file = os.path.join(self.label_dir, "%06d.txt" % idx)
        else:
            label_file = os.path.join(self.aug_label_dir, "%06d.txt" % idx)

        if not (os.path.exists(label_file)):
            raise AssertionError
        return kitti_utils.get_objects_from_label(label_file)

    def get_image(self, idx):
        return super().get_image(idx % 10000)

    def get_image_shape(self, idx):
        return super().get_image_shape(idx % 10000)

    def get_calib(self, idx):
        return super().get_calib(idx % 10000)

    def get_road_plane(self, idx):
        return super().get_road_plane(idx % 10000)

    def filtrate_objects(self, obj_list):
        return filter_objects(obj_list, self.classes, self.mode, cfg, self.check_pc_range)

    def get_rpn_with_li_fusion(self, index):
        sample_id = int(self.sample_id_list[index])
        calib, img, img_shape, pts_rect, pts_intensity = self._load_li_fusion_points(
            sample_id
        )
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(
            pts_rect, pts_img, pts_rect_depth, img_shape
        )
        pts_rect = pts_rect[pts_valid_flag][:, :3]
        pts_intensity = pts_intensity[pts_valid_flag]
        pts_origin_xy = pts_img[pts_valid_flag]
        ret_pts_rect, ret_intensity, ret_pts_origin_xy = self._select_li_fusion_points(
            pts_rect, pts_intensity, pts_origin_xy
        )
        ret_pts_features = ret_intensity.reshape(-1, 1)
        sample_info = {
            "sample_id": sample_id,
            "random_select": self.random_select,
            "img": img,
            "pts_origin_xy": ret_pts_origin_xy,
        }
        if self.mode == "TEST":
            sample_info.update(
                {
                    "pts_input": self._rpn_point_input(ret_pts_rect, ret_pts_features),
                    "pts_rect": ret_pts_rect,
                    "pts_features": ret_pts_features,
                }
            )
            return sample_info
        return self._training_li_fusion_sample(
            sample_info, sample_id, ret_pts_rect, ret_pts_features
        )

    def get_rpn_sample(self, index):
        _state = locals()
        _kittircnndataset_get_rpn_sample_733_part_1(_state)
        _kittircnndataset_get_rpn_sample_733_part_2(_state)
        _kittircnndataset_get_rpn_sample_733_part_3(_state)
        _helper_result = _kittircnndataset_get_rpn_sample_733_part_4(_state)
        if _helper_result is not _HELPER_CONTINUE:
            return _helper_result
        _helper_result = _kittircnndataset_get_rpn_sample_733_part_5(_state)
        if _helper_result is not _HELPER_CONTINUE:
            return _helper_result
        return _state["sample_info"]

    def apply_gt_aug_to_one_scene(
        self, sample_id, pts_rect, pts_features_offline, all_gt_boxes3d
    ):
        """Insert collision-free database objects into one scene."""
        if self.gt_database is None:
            raise AssertionError
        extra_count = (
            np.random.randint(10, cfg.GT_EXTRA_NUM)
            if cfg.GT_AUG_RAND_NUM else cfg.GT_EXTRA_NUM
        )
        boxes = all_gt_boxes3d.copy()
        boxes[:, 4:6] += 0.5
        state = {
            "boxes": boxes,
            "corners": kitti_utils.boxes3d_to_corners3d(boxes),
            "extra_objects": [],
            "extra_boxes": [],
            "new_points": [],
            "new_features": [],
            "src_pts_flag": np.ones(pts_rect.shape[0], dtype=np.int32),
        }
        road_plane = self.get_road_plane(sample_id)
        try_times, attempted = 100, 0
        while try_times > 0:
            if attempted > extra_count:
                break
            try_times -= 1
            prepared = self._prepare_database_gt(
                self._random_gt_database_entry(), road_plane
            )
            if prepared is None:
                continue
            attempted += 1
            if kitti_utils.get_iou3d(prepared[-1], state["corners"]).max() >= 1e-8:
                continue
            self._append_augmented_gt(state, prepared, pts_rect)
        return self._finish_gt_augmentation(state, pts_rect, pts_features_offline)

    def data_augmentation(self, *call_args, **call_kwargs):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return:
        """
        _state = locals()
        _kittircnndataset_data_augmentation_992_part_1(_state)
        _kittircnndataset_data_augmentation_992_part_2(_state)

        return (_state["aug_pts_rect"], _state["aug_gt_boxes3d"], _state["aug_method"])

    def get_rcnn_sample_info(self, roi_info):
        _state = locals()
        _kittircnndataset_get_rcnn_sample_info_1057_part_1(_state)
        _kittircnndataset_get_rcnn_sample_info_1057_part_2(_state)
        _kittircnndataset_get_rcnn_sample_info_1057_part_3(_state)

        return _state["sample_info"]

    def aug_roi_by_noise(self, roi_info):
        """
                add noise to original roi to get aug_box3d
        :param roi_info:
        :return:
        """
        roi_box3d, gt_box3d = roi_info["roi_box3d"], roi_info["gt_box3d"]
        original_iou = roi_info["iou3d"]
        temp_iou = cnt = 0
        pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_box3d.reshape(-1, 7))
        aug_box3d = roi_box3d
        while temp_iou < pos_thresh and cnt < 10:
            if roi_info["type"] == "gt":
                aug_box3d = self.random_aug_box3d(roi_box3d)  # GT, must random
            else:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
            aug_corners = kitti_utils.boxes3d_to_corners3d(aug_box3d.reshape(-1, 7))
            iou3d = kitti_utils.get_iou3d(aug_corners, gt_corners)
            temp_iou = iou3d[0][0]
            cnt += 1
            if original_iou < pos_thresh:  # original bg, break
                break
        return aug_box3d

    def get_proposal_from_file(self, index):
        _state = locals()
        _kittircnndataset_get_proposal_from_file_1188_part_1(_state)
        _helper_result = _kittircnndataset_get_proposal_from_file_1188_part_2(_state)
        if _helper_result is not _HELPER_CONTINUE:
            return _helper_result
        _helper_result = _kittircnndataset_get_proposal_from_file_1188_part_3(_state)
        if _helper_result is not _HELPER_CONTINUE:
            return _helper_result
        _kittircnndataset_get_proposal_from_file_1188_part_4(_state)

        return _state["sample_dict"]

    def get_rcnn_training_sample_batch(self, index):
        _state = locals()
        _kittircnndataset_get_rcnn_training_sample_batch_1290_part_1(_state)
        _kittircnndataset_get_rcnn_training_sample_batch_1290_part_2(_state)
        _kittircnndataset_get_rcnn_training_sample_batch_1290_part_3(_state)
        _kittircnndataset_get_rcnn_training_sample_batch_1290_part_4(_state)
        _kittircnndataset_get_rcnn_training_sample_batch_1290_part_5(_state)

        return _state["sample_info"]

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        if hard_bg_inds.size > 0 and easy_bg_inds.size > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * cfg.RCNN.HARD_BG_RATIO)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_num = np.floor(
                np.random.rand(hard_bg_rois_num) * hard_bg_inds.size
            ).astype(np.int32)
            hard_bg_inds = hard_bg_inds[rand_num]
            # sampling easy bg
            rand_num = np.floor(
                np.random.rand(easy_bg_rois_num) * easy_bg_inds.size
            ).astype(np.int32)
            easy_bg_inds = easy_bg_inds[rand_num]

            bg_inds = np.concatenate([hard_bg_inds, easy_bg_inds], axis=0)
        elif hard_bg_inds.size > 0 and easy_bg_inds.size == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_num = np.floor(
                np.random.rand(hard_bg_rois_num) * hard_bg_inds.size
            ).astype(np.int32)
            bg_inds = hard_bg_inds[rand_num]
        elif hard_bg_inds.size == 0 and easy_bg_inds.size > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_num = np.floor(
                np.random.rand(easy_bg_rois_num) * easy_bg_inds.size
            ).astype(np.int32)
            bg_inds = easy_bg_inds[rand_num]
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise_batch(self, roi_boxes3d, gt_boxes3d, aug_times=10):
        """
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        iou_of_rois = np.zeros(roi_boxes3d.shape[0], dtype=np.float32)
        for k in range(roi_boxes3d.__len__()):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]
            gt_box3d = gt_boxes3d[k]
            pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_box3d.reshape(1, 7))
            aug_box3d = roi_box3d
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                aug_corners = kitti_utils.boxes3d_to_corners3d(aug_box3d.reshape(1, 7))
                iou3d = kitti_utils.get_iou3d(aug_corners, gt_corners)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d
            iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    def get_rcnn_sample_jit(self, index):
        sample_id = int(self.sample_id_list[index])
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = self.get_rpn_features(
            self.rcnn_training_feature_dir, sample_id
        )

        # load rois and gt_boxes3d for this sample
        roi_file = os.path.join(self.rcnn_training_roi_dir, "%06d.txt" % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(roi_file)
        roi_boxes3d = kitti_utils.objs_to_boxes3d(roi_obj_list)

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        sample_info = {
            "sample_id": sample_id,
            "rpn_xyz": rpn_xyz,
            "rpn_features": rpn_features,
            "rpn_intensity": rpn_intensity,
            "seg_mask": seg_mask,
            "roi_boxes3d": roi_boxes3d,
            "gt_boxes3d": gt_boxes3d,
            "pts_depth": np.linalg.norm(rpn_xyz, ord=2, axis=1),
        }

        return sample_info

    def collate_batch(self, batch):
        if self.mode != "TRAIN" and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            if not (batch.__len__() == 1):
                raise AssertionError
            return batch[0]

        ans_dict = {}

        for key in batch[0].keys():
            should_pad_boxes = (
                cfg.RPN.ENABLED
                and key == "gt_boxes3d"
                or (
                    cfg.RCNN.ENABLED
                    and cfg.RCNN.ROI_SAMPLE_JIT
                    and key in ["gt_boxes3d", "roi_boxes3d"]
                )
            )
            ans_dict[key] = (
                self._pad_batch_boxes(batch, key)
                if should_pad_boxes
                else self._collate_batch_value(batch, key)
            )

        return ans_dict

    def _configure_classes(self, root_dir, classes):
        class_groups = {
            "Car": (("Background", "Car"), "aug_scene"),
            "People": (("Background", "Pedestrian", "Cyclist"), "aug_scene"),
            "Pedestrian": (("Background", "Pedestrian"), "aug_scene_ped"),
            "Cyclist": (("Background", "Cyclist"), "aug_scene_cyclist"),
        }
        if classes not in class_groups:
            raise ValueError("Invalid classes: %s" % classes)
        self.classes, scene_dir = class_groups[classes]
        return os.path.join(root_dir, "KITTI", scene_dir)

    def _load_gt_database(self, database_path, logger):
        if database_path is None:
            return
        with open(database_path, "rb") as database_file:
            self.gt_database = pickle.load(database_file)
        if cfg.GT_AUG_HARD_RATIO <= 0:
            logger.info(
                "Loading gt_database(%d) from %s"
                % (len(self.gt_database), database_path)
            )
            return
        easy_list, hard_list = [], []
        for item in self.gt_database:
            target = easy_list if item["points"].shape[0] > 100 else hard_list
            target.append(item)
        self.gt_database = [easy_list, hard_list]
        logger.info(
            "Loading gt_database(easy(pt_num>100): %d, hard(pt_num<=100): %d) from %s"
            % (len(easy_list), len(hard_list), database_path)
        )

    def _initialize_sample_ids(self, mode):
        if cfg.RPN.ENABLED:
            if mode == "TRAIN":
                self.preprocess_rpn_training_data()
                return
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        elif cfg.RCNN.ENABLED:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        else:
            return
        self.logger.info("Load testing samples from %s" % self.imageset_dir)
        self.logger.info("Done: total test samples %d" % len(self.sample_id_list))

    def _load_li_fusion_points(self, sample_id):
        if sample_id >= 10000:
            raise ValueError("unable to use augmented data with image alignment")
        calibration = self.get_calib(sample_id)
        image = self.get_image_rgb_with_normal(sample_id)
        image_shape = self.get_image_shape(sample_id)
        lidar = self.get_lidar(sample_id)
        points = calibration.lidar_to_rect(lidar[:, :3])
        return calibration, image, image_shape, points, lidar[:, 3]

    def _point_selection(self, points):
        if self.npoints < len(points):
            near = points[:, 2] < 40.0
            far_indices = np.where(~near)[0]
            near_indices = np.where(near)[0]
            selected_near = np.random.choice(
                near_indices, self.npoints - len(far_indices), replace=False
            )
            selection = (
                np.concatenate((selected_near, far_indices))
                if len(far_indices) > 0
                else selected_near
            )
        else:
            selection = np.arange(len(points), dtype=np.int32)
            deficit = self.npoints - len(points)
            if 0 < deficit < len(selection):
                extra = np.random.choice(selection, deficit, replace=False)
                selection = np.concatenate((selection, extra))
            elif deficit >= len(selection) and deficit > 0:
                LOGGER.info(
                    "cannot pad %d points from a set of %d", deficit, len(points)
                )
        np.random.shuffle(selection)
        return selection

    def _select_li_fusion_points(self, points, intensity, image_points):
        if self.mode != "TRAIN" and not self.random_select:
            return points, intensity - 0.5, image_points
        selection = self._point_selection(points)
        if cfg.fid:
            selection = np.arange(len(points), dtype=np.int32)
            np.random.seed(0)
            np.random.shuffle(selection)
        return (
            points[selection],
            intensity[selection] - 0.5,
            image_points[selection],
        )

    def _training_li_fusion_sample(self, sample_info, sample_id, points, features):
        objects = self.filtrate_objects(self.get_label(sample_id))
        boxes = kitti_utils.objs_to_boxes3d(objects)
        alpha = np.array([obj.alpha for obj in objects], dtype=np.float32)
        augmented_points, augmented_boxes = points.copy(), boxes.copy()
        if cfg.AUG_DATA and self.mode == "TRAIN":
            augmented_points, augmented_boxes, method = self.data_augmentation(
                augmented_points, augmented_boxes, alpha, sample_id
            )
            sample_info["aug_method"] = method
        sample_info.update(
            {
                "pts_input": self._rpn_point_input(augmented_points, features),
                "pts_rect": augmented_points,
                "pts_features": features,
                "gt_boxes3d": augmented_boxes,
            }
        )
        if not cfg.RPN.FIXED:
            labels = self.generate_rpn_training_labels(
                augmented_points, augmented_boxes
            )
            sample_info["rpn_cls_label"], sample_info["rpn_reg_label"] = labels
        return sample_info

    def _random_gt_database_entry(self):
        if cfg.GT_AUG_HARD_RATIO <= 0:
            return self.gt_database[np.random.randint(0, len(self.gt_database))]
        difficulty = 0 if np.random.rand() > cfg.GT_AUG_HARD_RATIO else 1
        entries = self.gt_database[difficulty]
        return entries[np.random.randint(0, len(entries))]

    def _prepare_database_gt(self, entry, road_plane):
        box = entry["gt_box3d"].copy()
        points = entry["points"].copy()
        center = box[:3]
        if cfg.PC_REDUCE_BY_RANGE and not self.check_pc_range(center):
            return None
        if len(points) < 5:
            return None
        intensity = entry["intensity"].copy().reshape(-1, 1)
        features = np.concatenate((intensity, entry["rgb"].copy()), axis=1)
        obj = entry["obj"]
        a, b, c, d = road_plane
        road_height = (-d - a * center[0] - c * center[2]) / b
        move_height = box[1] - road_height
        box[1] -= move_height
        points[:, 1] -= move_height
        obj.pos[1] -= move_height
        enlarged = box.copy()
        enlarged[4:6] += 0.5
        corners = kitti_utils.boxes3d_to_corners3d(enlarged.reshape(1, 7))
        return DatabaseGroundTruthResult(box, points, features, obj, enlarged, corners)


if __name__ == "__main__":
    img = np.array([0, 0, 0, 0, 5, 2.0]).reshape(3, 2, 1)
    LOGGER.info("%s", img[2, 0])

    xy = np.array([2.0, 0.5]).reshape(1, 2)
    y = interpolate_img_by_xy(img, xy, np.array([3.0, 2.0]))
    LOGGER.info("%s", y)
