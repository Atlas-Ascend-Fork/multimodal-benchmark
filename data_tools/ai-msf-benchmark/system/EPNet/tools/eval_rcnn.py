# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections import namedtuple
from typing import NamedTuple
from datetime import datetime, timezone
from decimal import Decimal
import argparse
import glob
import logging
import os
import re
import time
import warnings

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.net.point_rcnn import PointRCNN
from lib.utils.bbox_transform import decode_bbox_target
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaPerformanceWarning,
    NumbaWarning,
)
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import _init_path  # noqa: F401
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate
import tools.train_utils.train_utils as train_utils


from shared.parameters import (
    ParameterSchema as _ParameterSchema,
    add_rcnn_output_arguments,
    bind_parameters as _bind_parameters,
)
from shared.kitti_detection import (
    format_detection_line,
    observation_angle,
    project_detection_boxes,
)


RepeatEvaluationSetup = namedtuple(
    "RepeatEvaluationSetup",
    "result_dir logger test_loader model checkpoint_record tensorboard_log",
)


class SaveKittiFormatParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    sample_id: object
    calib: object
    bbox3d: object
    kitti_output_dir: object
    scores: object
    img_shape: object


class SaveRpnFeaturesParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    seg_result: object
    rpn_scores_raw: object
    pts_features: object
    backbone_xyz: object
    backbone_features: object
    kitti_features_dir: object
    sample_id: object


class EvalOneEpochJointParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    model: object
    dataloader: object
    epoch_id: object
    result_dir: object
    logger: object
    is_get_feature: object


class EvalOneEpochParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    model: object
    dataloader: object
    epoch_id: object
    result_dir: object
    logger: object
    is_get_feature: object


LOGGER = logging.getLogger(__name__)

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.simplefilter("ignore", category=NumbaWarning)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument(
    "--cfg_file",
    type=str,
    default="cfgs/default.yml",
    help="specify the config for evaluation",
)
parser.add_argument(
    "--eval_mode",
    type=str,
    default="rpn",
    required=True,
    help="specify the evaluation mode",
)

parser.add_argument(
    "--eval_all",
    action="store_true",
    default=False,
    help="whether to evaluate all checkpoints",
)
parser.add_argument(
    "--test", action="store_true", default=False, help="evaluate without ground truth"
)
parser.add_argument(
    "--ckpt", type=str, default=None, help="specify a checkpoint to be evaluated"
)
parser.add_argument(
    "--rpn_ckpt",
    type=str,
    default=None,
    help="specify the checkpoint of rpn if trained separated",
)
parser.add_argument(
    "--rcnn_ckpt",
    type=str,
    default=None,
    help="specify the checkpoint of rcnn if trained separated",
)

parser.add_argument(
    "--batch_size", type=int, default=1, help="batch size for evaluation"
)
parser.add_argument(
    "--workers", type=int, default=4, help="number of workers for dataloader"
)
parser.add_argument(
    "--extra_tag", type=str, default="default", help="extra tag for multiple evaluation"
)
parser.add_argument(
    "--output_dir", type=str, default=None, help="specify an output directory if needed"
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default=None,
    help="specify a ckpt directory to be evaluated if needed",
)

parser.add_argument(
    "--save_result",
    action="store_true",
    default=False,
    help="save evaluation results to files",
)
parser.add_argument(
    "--save_rpn_feature",
    action="store_true",
    default=False,
    help="save features for separately rcnn training and evaluation",
)

parser.add_argument(
    "--random_select",
    action="store_true",
    default=True,
    help="sample to the same number of points",
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    help="ignore the checkpoint smaller than this epoch",
)
parser.add_argument(
    "--max_waiting_mins", type=int, default=30, help="max waiting minutes"
)
add_rcnn_output_arguments(parser)
parser.add_argument("--fid", type=bool, default=False, help="cal features")
args = parser.parse_args()


def create_logger(log_file):
    log_format = "%(asctime)s  %(levelname)5s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


_SAVE_KITTI_FORMAT_PARAMETERS_SCHEMA = _ParameterSchema(
    SaveKittiFormatParameters,
    ("sample_id", "calib", "bbox3d", "kitti_output_dir", "scores", "img_shape"),
    {},
    frozenset(()),
    6,
    None,
    None,
)
_SAVE_RPN_FEATURES_PARAMETERS_SCHEMA = _ParameterSchema(
    SaveRpnFeaturesParameters,
    (
        "seg_result",
        "rpn_scores_raw",
        "pts_features",
        "backbone_xyz",
        "backbone_features",
        "kitti_features_dir",
        "sample_id",
    ),
    {},
    frozenset(()),
    7,
    None,
    None,
)
_EVAL_ONE_EPOCH_JOINT_PARAMETERS_SCHEMA = _ParameterSchema(
    EvalOneEpochJointParameters,
    ("model", "dataloader", "epoch_id", "result_dir", "logger", "is_get_feature"),
    {"is_get_feature": False},
    frozenset(()),
    6,
    None,
    None,
)
_EVAL_ONE_EPOCH_PARAMETERS_SCHEMA = _ParameterSchema(
    EvalOneEpochParameters,
    ("model", "dataloader", "epoch_id", "result_dir", "logger", "is_get_feature"),
    {"is_get_feature": False},
    frozenset(()),
    6,
    None,
    None,
)


def save_kitti_format(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SAVE_KITTI_FORMAT_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape = bound_parameters
    img_boxes, box_valid_mask = project_detection_boxes(
        bbox3d, calib, img_shape, kitti_utils
    )

    kitti_output_file = os.path.join(kitti_output_dir, "%06d.txt" % sample_id)
    with open(kitti_output_file, "w") as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            f.write(format_detection_line(
                cfg.CLASSES, observation_angle(bbox3d[k]),
                img_boxes[k], bbox3d[k], {"score": scores[k]}
            ) + "\n")


def save_rpn_features(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SAVE_RPN_FEATURES_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        seg_result,
        rpn_scores_raw,
        pts_features,
        backbone_xyz,
        backbone_features,
        kitti_features_dir,
        sample_id,
    ) = bound_parameters
    pts_intensity = pts_features[:, 0]

    output_file = os.path.join(kitti_features_dir, "%06d.npy" % sample_id)
    xyz_file = os.path.join(kitti_features_dir, "%06d_xyz.npy" % sample_id)
    seg_file = os.path.join(kitti_features_dir, "%06d_seg.npy" % sample_id)
    intensity_file = os.path.join(kitti_features_dir, "%06d_intensity.npy" % sample_id)
    np.save(output_file, backbone_features)
    np.save(xyz_file, backbone_xyz)
    np.save(seg_file, seg_result)
    np.save(intensity_file, pts_intensity)
    rpn_scores_raw_file = os.path.join(
        kitti_features_dir, "%06d_rawscore.npy" % sample_id
    )
    np.save(rpn_scores_raw_file, rpn_scores_raw)


def _trim_padded_boxes(boxes):
    last = len(boxes) - 1
    while last > 0 and boxes[last].sum() == 0:
        last -= 1
    return boxes[: last + 1]


def _rpn_sample_metrics(boxes, ground_truth, labels, segmentation, thresholds):
    ground_truth = _trim_padded_boxes(ground_truth)
    recalled = [0] * len(thresholds)
    if ground_truth.shape[0] > 0:
        overlaps = iou3d_utils.boxes_iou3d_gpu(boxes, ground_truth[:, 0:7])
        maximum_iou = overlaps.max(dim=0)[0]
        recalled = [int((maximum_iou > threshold).sum()) for threshold in thresholds]
    foreground = labels > 0
    correct = ((segmentation == labels) & foreground).sum().float()
    union = foreground.sum().float() + (segmentation > 0).sum().float() - correct
    return recalled, len(ground_truth), (correct / torch.clamp(union, min=1.0)).item()


def _save_rpn_sample(sample, directories, dataset):
    feature_dir, segmentation_dir, detection_dir = directories
    if args.save_rpn_feature:
        save_rpn_features(
            sample["segmentation"].float().cpu().numpy(),
            sample["scores"].float().cpu().numpy(),
            sample["features"],
            sample["xyz"].cpu().numpy(),
            sample["backbone"].cpu().numpy().transpose(1, 0),
            feature_dir,
            sample["id"],
        )
    if not (args.save_result or args.save_rpn_feature):
        return
    predicted = sample["segmentation"].cpu().numpy()
    columns = [sample["points"].reshape(-1, 3)]
    if not args.test:
        columns.append(sample["labels"].cpu().numpy().reshape(-1, 1))
    columns.append(predicted.reshape(-1, 1))
    output = np.concatenate(columns, axis=1)
    np.save(
        os.path.join(segmentation_dir, "%06d.npy" % sample["id"]),
        output.astype(np.float16),
    )
    save_kitti_format(
        sample["id"],
        dataset.get_calib(sample["id"]),
        sample["boxes"].cpu().numpy(),
        detection_dir,
        sample["scores"],
        dataset.get_image_shape(sample["id"]),
    )


def _rpn_model_input(data):
    result = {
        "pts_input": torch.from_numpy(data["pts_input"]).cuda(non_blocking=True).float()
    }
    if cfg.LI_FUSION.ENABLED:
        result["pts_origin_xy"] = (
            torch.from_numpy(data["pts_origin_xy"]).cuda(non_blocking=True).float()
        )
        result["img"] = (
            torch.from_numpy(data["img"]).cuda(non_blocking=True).float().permute((0, 3, 1, 2))
        )
    if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
        result["pts_rgb"] = torch.from_numpy(data["rgb"]).cuda(non_blocking=True).float()
    return result


def _rpn_ground_truth(data):
    labels = torch.from_numpy(data["rpn_cls_label"]).cuda(non_blocking=True).long()
    boxes = data["gt_boxes3d"]
    if boxes.shape[1] != 0:
        boxes = torch.from_numpy(boxes).cuda(non_blocking=True).float()
    return labels, boxes


def _rpn_batch_predictions(model, data):
    result = model(_rpn_model_input(data))
    raw_scores = result["rpn_cls"][:, :, 0]
    segmentation = (torch.sigmoid(raw_scores) > cfg.RPN.SCORE_THRESH).long()
    rois, roi_scores = model.rpn.proposal_layer(
        raw_scores, result["rpn_reg"], result["backbone_xyz"]
    )
    return result, segmentation, rois, roi_scores


def _rpn_evaluation_result(logger, context, stats):
    epoch_id, result_dir = context
    thresholds = stats["thresholds"]
    recalled = stats["recalled"]
    gt_count = stats["ground_truth"]
    iou, count = stats["iou"], stats["count"]
    logger.info("%s", datetime.now(timezone.utc))
    logger.info("-------------------performance of epoch %s---------------------" % epoch_id)
    logger.info("max number of objects: %d" % 0)
    logger.info("rpn iou avg: %f" % (iou / max(count, 1.0)))
    result = {"max_obj_num": 0, "rpn_iou": iou / count}
    for index, threshold in enumerate(thresholds):
        recall = recalled[index] / max(gt_count, 1.0)
        logger.info(
            "total bbox recall(thresh=%.3f): %d / %d = %f"
            % (threshold, recalled[index], gt_count, recall)
        )
        result["rpn_recall(thresh=%.2f)" % threshold] = recall
    logger.info("result is saved to: %s" % result_dir)
    return result


def _evaluate_rpn_samples(data, predictions, ground_truth, stats, output):
    labels, gt_boxes = ground_truth
    directories, dataset = output
    result, segmentation, rois, roi_scores = predictions
    for index, sample_id in enumerate(data["sample_id"]):
        sample = {
            "id": sample_id, "scores": roi_scores[index], "boxes": rois[index],
            "segmentation": segmentation[index], "points": data["pts_rect"][index],
            "features": data["pts_features"][index],
            "xyz": result["backbone_xyz"][index],
            "backbone": result["backbone_features"][index], "labels": None,
        }
        if not args.test:
            sample["labels"] = labels[index]
            recalled, gt_count, sample_iou = _rpn_sample_metrics(
                sample["boxes"], gt_boxes[index], sample["labels"],
                sample["segmentation"], stats["thresholds"],
            )
            stats["ground_truth"] += gt_count
            stats["iou"] += sample_iou
            stats["recalled"] = [
                total + value for total, value in zip(stats["recalled"], recalled)
            ]
        _save_rpn_sample(sample, directories, dataset)


def eval_one_epoch_rpn(model, dataloader, epoch_id, result_dir, logger):
    np.random.seed(1024)
    mode = "TEST" if args.test else "EVAL"
    save_outputs = any((args.save_result, args.save_rpn_feature))
    feature_dir = os.path.join(result_dir, "features") if args.save_rpn_feature else None
    detection_dir = os.path.join(result_dir, "detections", "data") if save_outputs else None
    segmentation_dir = os.path.join(result_dir, "seg_result") if save_outputs else None
    for directory in (feature_dir, detection_dir, segmentation_dir):
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
    logger.info("---- EPOCH %s RPN EVALUATION ----" % epoch_id)
    model.eval()
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    stats = {"thresholds": thresholds, "recalled": [0] * 5,
             "ground_truth": 0, "iou": 0, "count": 0}
    dataset = dataloader.dataset
    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc="eval")
    for data in dataloader:
        stats["count"] += len(data["sample_id"])
        labels, gt_boxes = (None, None) if args.test else _rpn_ground_truth(data)
        predictions = _rpn_batch_predictions(model, data)
        _evaluate_rpn_samples(
            data, predictions, (labels, gt_boxes), stats,
            ((feature_dir, segmentation_dir, detection_dir), dataset),
        )
        progress_bar.set_postfix({
            "mode": mode,
            "recall": "%d/%d" % (stats["recalled"][3], stats["ground_truth"]),
            "rpn_iou": stats["iou"] / max(stats["count"], 1.0),
        })
        progress_bar.update()
    progress_bar.close()
    return _rpn_evaluation_result(
        logger, (epoch_id, result_dir), stats,
    )


def _rcnn_output_directories(result_dir):
    final_output_dir = os.path.join(result_dir, "final_result", "data")
    os.makedirs(final_output_dir, exist_ok=True)
    roi_output_dir = refine_output_dir = None
    if args.save_result:
        roi_output_dir = os.path.join(result_dir, "roi_result", "data")
        refine_output_dir = os.path.join(result_dir, "refine_result", "data")
        os.makedirs(roi_output_dir, exist_ok=True)
        os.makedirs(refine_output_dir, exist_ok=True)
    return final_output_dir, roi_output_dir, refine_output_dir


def _prepare_rcnn_input(data):
    if args.batch_size != 1:
        raise AssertionError("Only support bs=1 here")
    input_data = {
        key: torch.from_numpy(value).contiguous().cuda(non_blocking=True).float()
        for key, value in data.items()
        if key != "sample_id"
    }
    roi_boxes3d, roi_scores = input_data.get("roi_boxes3d"), input_data.get(
        "roi_scores"
    )
    if cfg.RCNN.ROI_SAMPLE_JIT:
        for key in input_data:
            if key not in ["gt_iou", "gt_boxes3d"]:
                input_data[key] = input_data[key].unsqueeze(dim=0)
    else:
        input_data["pts_input"] = torch.cat(
            (input_data.get("pts_input"), input_data.get("pts_features")), dim=-1
        )
    if cfg.LI_FUSION.ENABLED:
        input_data["pts_origin_xy"] = (
            torch.from_numpy(data["pts_origin_xy"]).cuda(non_blocking=True).float()
        )
        input_data["img"] = (
            torch.from_numpy(data["img"])
            .cuda(non_blocking=True)
            .float()
            .permute((0, 3, 1, 2))
        )
    if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
        input_data["pts_rgb"] = (
            torch.from_numpy(data["rgb"]).cuda(non_blocking=True).float()
        )
    return input_data, roi_boxes3d, roi_scores


def _rcnn_predictions(model, input_data, roi_boxes3d, mean_size):
    outputs = model(input_data)
    rcnn_cls, rcnn_reg = outputs["rcnn_cls"], outputs["rcnn_reg"]
    anchor_size = input_data.get("roi_size") if cfg.RCNN.SIZE_RES_ON_ROI else mean_size
    pred_boxes3d = decode_bbox_target(
        roi_boxes3d,
        rcnn_reg,
        anchor_size=anchor_size,
        loc_scope=cfg.RCNN.LOC_SCOPE,
        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
        get_xz_fine=True,
        get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
        loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
        get_ry_fine=True,
    )
    if rcnn_cls.shape[1] == 1:
        raw_scores = rcnn_cls.view(-1)
        norm_scores = torch.sigmoid(raw_scores)
        pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
    else:
        pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
        normalized = F.softmax(rcnn_cls, dim=1)
        raw_scores = rcnn_cls[:, pred_classes]
        norm_scores = normalized[:, pred_classes]
    return pred_boxes3d, raw_scores, norm_scores, pred_classes


def _update_rcnn_metrics(stats, input_data, predictions, roi_boxes3d, thresholds):
    pred_boxes3d, _, _, pred_classes = predictions
    gt_boxes3d, gt_iou = input_data.get("gt_boxes3d"), input_data.get("gt_iou")
    gt_num = gt_boxes3d.shape[0]
    if gt_num > 0:
        predicted_iou = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d, gt_boxes3d)
        gt_max_iou, _ = predicted_iou.max(dim=0)
        roi_iou = iou3d_utils.boxes_iou3d_gpu(roi_boxes3d, gt_boxes3d)
        gt_max_roi_iou, _ = roi_iou.max(dim=0)
        for index, threshold in enumerate(thresholds):
            stats["recall"][index] += (gt_max_iou > threshold).sum().item()
            stats["roi_recall"][index] += (
                gt_max_roi_iou > threshold
            ).sum().item()
        stats["gt_count"] += gt_num
    cls_label = (gt_iou > cfg.RCNN.CLS_FG_THRESH).float()
    valid_mask = (
        (gt_iou >= cfg.RCNN.CLS_FG_THRESH) | (gt_iou <= cfg.RCNN.CLS_BG_THRESH)
    ).float()
    accuracy = ((pred_classes == cls_label.long()).float() * valid_mask).sum() / max(
        valid_mask.sum(), 1.0
    )
    iou_threshold = 0.7 if cfg.CLASSES == "Car" else 0.5
    refined_label = (gt_iou >= iou_threshold).float()
    refined_accuracy = (pred_classes == refined_label.long()).float().sum() / max(
        refined_label.shape[0], 1.0
    )
    stats["accuracy"] += accuracy.item()
    stats["refined_accuracy"] += refined_accuracy.item()
    return {
        "mode": "EVAL",
        "recall": "%d/%d" % (stats["recall"][3], stats["gt_count"]),
        "cls_acc_refined": "%.2f" % refined_accuracy.item(),
    }


def _save_rcnn_sample(context, predictions, roi_boxes3d, roi_scores):
    pred_boxes3d, raw_scores, norm_scores, _ = predictions
    dataset, sample_id = context["dataset"], context["sample_id"]
    image_shape, calibration = (
        dataset.get_image_shape(sample_id),
        dataset.get_calib(sample_id),
    )
    if args.save_result:
        save_kitti_format(
            sample_id,
            calibration,
            roi_boxes3d.cpu().numpy(),
            context["roi_output_dir"],
            roi_scores,
            image_shape,
        )
        save_kitti_format(
            sample_id,
            calibration,
            pred_boxes3d.cpu().numpy(),
            context["refine_output_dir"],
            raw_scores.cpu().numpy(),
            image_shape,
        )
    indices = norm_scores > cfg.RCNN.SCORE_THRESH
    if indices.sum() == 0:
        return 0
    selected_boxes, selected_scores = pred_boxes3d[indices], raw_scores[indices]
    keep_indices = iou3d_utils.nms_gpu(
        kitti_utils.boxes3d_to_bev_torch(selected_boxes),
        selected_scores,
        cfg.RCNN.NMS_THRESH,
    )
    selected_boxes = selected_boxes[keep_indices].cpu().numpy()
    selected_scores = selected_scores[keep_indices].cpu().numpy()
    save_kitti_format(
        sample_id,
        calibration,
        selected_boxes,
        context["final_output_dir"],
        selected_scores,
        image_shape,
    )
    return selected_boxes.shape[0]


def _dump_empty_rcnn_files(dataset, output_dir, logger):
    split_file = os.path.abspath(
        os.path.join(
            dataset.imageset_dir, "..", "..", "ImageSets", dataset.split + ".txt"
        )
    )
    with open(split_file) as split_stream:
        image_indices = [line.strip() for line in split_stream]
    empty_count = 0
    for image_index in image_indices:
        result_file = os.path.join(output_dir, "%s.txt" % image_index)
        if not os.path.exists(result_file):
            open(result_file, "w").close()
            empty_count += 1
            logger.info("empty_cnt=%d: dump empty file %s" % (empty_count, result_file))
    return split_file, empty_count


def _log_rcnn_recall(logger, result, stats, thresholds, metric):
    source, result_prefix = metric
    for index, threshold in enumerate(thresholds):
        recall = stats[source][index] / max(stats["gt_count"], 1.0)
        logger.info(
            "total %s bbox recall(thresh=%.3f): %d / %d = %f"
            % (
                "roi" if source == "roi_recall" else "",
                threshold,
                stats[source][index],
                stats["gt_count"],
                recall,
            )
        )
        result["%s_recall(thresh=%.2f)" % (result_prefix, threshold)] = recall


def _summarize_rcnn_evaluation(context, stats, thresholds):
    logger, dataset = context["logger"], context["dataset"]
    split_file, empty_count = _dump_empty_rcnn_files(
        dataset, context["output_dir"], logger
    )
    logger.info(
        "-------------------performance of epoch %s---------------------"
        % context["epoch_id"]
    )
    logger.info("%s", datetime.now(timezone.utc))
    sample_count = max(stats["sample_count"], 1.0)
    result = {
        "empty_cnt": empty_count,
        "rcnn_cls_acc": stats["accuracy"] / sample_count,
        "rcnn_cls_acc_refined": stats["refined_accuracy"] / sample_count,
        "rcnn_avg_num": stats["detection_count"] / sample_count,
    }
    logger.info("final average detections: %.3f" % result["rcnn_avg_num"])
    logger.info("final average cls acc: %.3f" % result["rcnn_cls_acc"])
    logger.info(
        "final average cls acc refined: %.3f" % result["rcnn_cls_acc_refined"]
    )
    _log_rcnn_recall(logger, result, stats, thresholds, ("roi_recall", "rpn"))
    _log_rcnn_recall(logger, result, stats, thresholds, ("recall", "rcnn"))
    if cfg.TEST.SPLIT != "test":
        logger.info("Averate Precision:")
        class_indices = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        ap_result, ap_values = kitti_evaluate(
            dataset.label_dir,
            context["output_dir"],
            label_split_file=split_file,
            current_class=class_indices.get(cfg.CLASSES),
        )
        logger.info(ap_result)
        result.update(ap_values)
    logger.info("result is saved to: %s" % context["result_dir"])
    return result


def _new_rcnn_stats():
    return {
        "recall": [0] * 5,
        "roi_recall": [0] * 5,
        "gt_count": 0,
        "accuracy": 0,
        "refined_accuracy": 0,
        "sample_count": 0,
        "detection_count": 0,
    }


def eval_one_epoch_rcnn(model, dataloader, epoch_id, result_dir, logger):
    np.random.seed(1024)
    mean_size = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    output_dirs = _rcnn_output_directories(result_dir)
    logger.info("---- EPOCH %s RCNN EVALUATION ----" % epoch_id)
    model.eval()
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    stats = _new_rcnn_stats()
    dataset = dataloader.dataset
    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc="eval")
    for data in dataloader:
        stats["sample_count"] += 1
        input_data, roi_boxes3d, roi_scores = _prepare_rcnn_input(data)
        predictions = _rcnn_predictions(model, input_data, roi_boxes3d, mean_size)
        display = {"mode": "TEST"}
        if not args.test:
            display = _update_rcnn_metrics(
                stats, input_data, predictions, roi_boxes3d, thresholds
            )
        sample_context = {
            "dataset": dataset,
            "sample_id": data["sample_id"],
            "final_output_dir": output_dirs[0],
            "roi_output_dir": output_dirs[1],
            "refine_output_dir": output_dirs[2],
        }
        stats["detection_count"] += _save_rcnn_sample(
            sample_context, predictions, roi_boxes3d, roi_scores
        )
        progress_bar.set_postfix(display)
        progress_bar.update()
    progress_bar.close()
    return _summarize_rcnn_evaluation(
        {
            "logger": logger,
            "dataset": dataset,
            "output_dir": output_dirs[0],
            "epoch_id": epoch_id,
            "result_dir": result_dir,
        },
        stats,
        thresholds,
    )


def _joint_model_input(data):
    model_input = {
        "pts_input": torch.from_numpy(data["pts_input"]).cuda(non_blocking=True).float()
    }
    if cfg.LI_FUSION.ENABLED:
        model_input["pts_origin_xy"] = (
            torch.from_numpy(data["pts_origin_xy"]).cuda(non_blocking=True).float()
        )
        model_input["img"] = (
            torch.from_numpy(data["img"])
            .cuda(non_blocking=True)
            .float()
            .permute((0, 3, 1, 2))
        )
    if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
        model_input["pts_rgb"] = (
            torch.from_numpy(data["rgb"]).cuda(non_blocking=True).float()
        )
    return model_input


def _joint_predictions(model, model_input, batch_size, mean_size):
    output = model(model_input)
    rois = output["rois"]
    classes = output["rcnn_cls"].view(batch_size, -1, output["rcnn_cls"].shape[1])
    regression = output["rcnn_reg"].view(batch_size, -1, output["rcnn_reg"].shape[1])
    if cfg.USE_IOU_BRANCH:
        iou_scores = output["rcnn_iou_branch"].view(
            batch_size, -1, output["rcnn_iou_branch"].shape[1]
        )
        classes *= torch.clamp(iou_scores, min=1e-4)
    if cfg.RCNN.SIZE_RES_ON_ROI:
        raise ValueError("SIZE_RES_ON_ROI is not supported by joint evaluation")
    boxes = decode_bbox_target(
        rois.view(-1, 7),
        regression.view(-1, regression.shape[-1]),
        anchor_size=mean_size,
        loc_scope=cfg.RCNN.LOC_SCOPE,
        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
        get_xz_fine=True,
        get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
        loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
        get_ry_fine=True,
    ).view(batch_size, -1, 7)
    if classes.shape[2] == 1:
        raw_scores, scores = classes, torch.sigmoid(classes)
    else:
        predicted = torch.argmax(classes, dim=1).view(-1)
        raw_scores = classes[:, predicted]
        scores = F.softmax(classes, dim=1)[:, predicted]
    return {
        "output": output,
        "rois": rois,
        "roi_scores": output["roi_scores_raw"],
        "segmentation": output["seg_result"].long(),
        "boxes": boxes,
        "raw_scores": raw_scores,
        "scores": scores,
    }


def _joint_update_recall(stats, data, predictions, thresholds):
    labels = None
    if not cfg.RPN.FIXED:
        labels = torch.from_numpy(data["rpn_cls_label"]).cuda(non_blocking=True).long()
    for index, boxes in enumerate(data["gt_boxes3d"]):
        ground_truth = _trim_padded_boxes(boxes)
        if len(ground_truth) > 0:
            ground_truth = (
                torch.from_numpy(ground_truth).cuda(non_blocking=True).float()
            )
            refined = iou3d_utils.boxes_iou3d_gpu(
                predictions["boxes"][index], ground_truth
            ).max(dim=0)[0]
            roi = iou3d_utils.boxes_iou3d_gpu(
                predictions["rois"][index], ground_truth
            ).max(dim=0)[0]
            for threshold_index, threshold in enumerate(thresholds):
                stats["refined"][threshold_index] += int((refined > threshold).sum())
                stats["roi"][threshold_index] += int((roi > threshold).sum())
            stats["ground_truth"] += len(ground_truth)
        if labels is not None:
            foreground = labels[index] > 0
            segmentation = predictions["segmentation"][index]
            correct = ((segmentation == labels[index]) & foreground).sum().float()
            union = (
                foreground.sum().float() + (segmentation > 0).sum().float() - correct
            )
            stats["rpn_iou"] += (correct / torch.clamp(union, min=1.0)).item()


def _save_joint_intermediate(data, predictions, dataset, directories):
    roi_dir, refine_dir, rpn_dir = directories
    batch_size = len(data["sample_id"])
    output = predictions["output"]
    rpn_data = np.concatenate(
        (
            output["backbone_xyz"].cpu().numpy(),
            output["rpn_cls"].cpu().numpy().reshape(batch_size, -1, 1),
            predictions["segmentation"].cpu().numpy().reshape(batch_size, -1, 1),
        ),
        axis=2,
    )
    for index, sample_id in enumerate(data["sample_id"]):
        calibration = dataset.get_calib(sample_id)
        image_shape = dataset.get_image_shape(sample_id)
        save_kitti_format(
            sample_id,
            calibration,
            predictions["rois"][index].cpu().numpy(),
            roi_dir,
            predictions["roi_scores"][index].cpu().numpy(),
            image_shape,
        )
        save_kitti_format(
            sample_id,
            calibration,
            predictions["boxes"][index].cpu().numpy(),
            refine_dir,
            predictions["raw_scores"][index].cpu().numpy(),
            image_shape,
        )
        path = os.path.join(rpn_dir, "%06d.npy" % sample_id)
        np.save(path, rpn_data[index].astype(np.float32))


def _save_joint_final(data, predictions, dataset, output_dir):
    selected = predictions["scores"] > cfg.RCNN.SCORE_THRESH
    saved = 0
    for index, sample_id in enumerate(data["sample_id"]):
        mask = selected[index].view(-1)
        if mask.sum() == 0:
            continue
        boxes = predictions["boxes"][index, mask]
        raw_scores = predictions["raw_scores"][index, mask]
        bev_boxes = kitti_utils.boxes3d_to_bev_torch(boxes)
        keep = iou3d_utils.nms_gpu(bev_boxes, raw_scores, cfg.RCNN.NMS_THRESH).view(-1)
        boxes, scores = boxes[keep].cpu().numpy(), raw_scores[keep].cpu().numpy()
        save_kitti_format(
            sample_id,
            dataset.get_calib(sample_id),
            boxes,
            output_dir,
            scores,
            dataset.get_image_shape(sample_id),
        )
        saved += len(boxes)
    return saved


def _save_joint_feature(model, model_input, batch_index):
    features = model.forward_features(model_input)
    base_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/fusion_feature/temp"
    np.save(os.path.join(base_path, f"{batch_index}.npy"), features.cpu())


def _dump_empty_joint_results(dataset, output_dir, logger):
    split_file = os.path.abspath(os.path.join(
        dataset.imageset_dir, "..", "..", "ImageSets", dataset.split + ".txt"
    ))
    with open(split_file, encoding="utf-8") as split_stream:
        image_indices = [line.strip() for line in split_stream]
    empty_count = 0
    for image_index in image_indices:
        filename = os.path.join(output_dir, "%s.txt" % image_index)
        if os.path.exists(filename):
            continue
        with open(filename, "w"):
            pass
        empty_count += 1
        logger.info("empty_cnt=%d: dump empty file %s" % (empty_count, filename))
    return split_file, empty_count


def _joint_metric_summary(stats, dataset, thresholds, logger):
    average_iou = float(
        Decimal(str(stats["rpn_iou"])) / Decimal(str(max(stats["batches"], 1)))
    )
    average_detections = float(
        Decimal(str(stats["detections"])) / Decimal(str(max(len(dataset), 1)))
    )
    logger.info("final average detections: %.3f" % average_detections)
    logger.info("final average rpn_iou refined: %.3f" % average_iou)
    logger.info("final average cls acc: %.3f" % 0.0)
    logger.info("final average cls acc refined: %.3f" % 0.0)
    result = {
        "rpn_iou": average_iou, "rcnn_cls_acc": 0.0,
        "rcnn_cls_acc_refined": 0.0, "rcnn_avg_num": average_detections,
    }
    for label, values, key in (
        ("roi", stats["roi"], "rpn_recall"),
        ("bbox", stats["refined"], "rcnn_recall"),
    ):
        for index, threshold in enumerate(thresholds):
            recall = values[index] / max(stats["ground_truth"], 1.0)
            logger.info(
                "total %s recall(thresh=%.3f): %d / %d = %f"
                % (label, threshold, values[index], stats["ground_truth"], recall)
            )
            result["%s(thresh=%.2f)" % (key, threshold)] = recall
    return result


def _joint_average_precision(dataloader, dataset, output_dir, split_file, logger):
    if cfg.TEST.SPLIT == "test":
        return {}
    logger.info("Averate Precision:")
    if len(dataloader) == 1:
        LOGGER.info("%s", "data_num < 50 skip eval")
        return {}
    class_index = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}.get(cfg.CLASSES)
    result_string, result = kitti_evaluate(
        dataset.label_dir, output_dir,
        label_split_file=split_file, current_class=class_index,
    )
    logger.info(result_string)
    return result


def _joint_directories(result_dir):
    final_dir = os.path.join(result_dir, "final_result", "data")
    os.makedirs(final_dir, exist_ok=True)
    if not args.save_result:
        return final_dir, (None, None, None)
    directories = tuple(
        os.path.join(result_dir, name, "data")
        for name in ("roi_result", "refine_result", "rpn_result")
    )
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return final_dir, directories


def _run_joint_batches(model, dataloader, mean_size, get_feature, context):
    directories, stats, progress_bar = context
    dataset, final_output_dir, intermediate_dirs = (
        dataloader.dataset, directories[0], directories[1]
    )
    for data in dataloader:
        stats["batches"] += 1
        model_input = _joint_model_input(data)
        if get_feature:
            _save_joint_feature(model, model_input, stats["batches"] - 1)
            continue
        predictions = _joint_predictions(
            model, model_input, len(data["sample_id"]), mean_size
        )
        if not args.test:
            _joint_update_recall(stats, data, predictions, stats["thresholds"])
        progress_bar.set_postfix({
            "mode": stats["mode"],
            "recall": "%d/%d" % (stats["refined"][3], stats["ground_truth"]),
        })
        progress_bar.update()
        if args.save_result:
            _save_joint_intermediate(data, predictions, dataset, intermediate_dirs)
        stats["detections"] += _save_joint_final(
            data, predictions, dataset, final_output_dir
        )


def eval_one_epoch_joint(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _EVAL_ONE_EPOCH_JOINT_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    model, dataloader, epoch_id, result_dir, logger, is_get_feature = bound_parameters
    np.random.seed(666)
    mean_size = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    mode = "TEST" if args.test else "EVAL"
    final_output_dir, intermediate_dirs = _joint_directories(result_dir)
    logger.info("---- EPOCH %s JOINT EVALUATION ----" % epoch_id)
    logger.info("==> Output file: %s" % result_dir)
    model.eval()

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    dataset = dataloader.dataset
    stats = {
        "refined": [0] * 5,
        "roi": [0] * 5,
        "ground_truth": 0,
        "rpn_iou": 0,
        "detections": 0,
        "batches": 0,
        "thresholds": thresh_list,
        "mode": mode,
    }
    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc="eval")
    _run_joint_batches(
        model, dataloader, mean_size, is_get_feature,
        ((final_output_dir, intermediate_dirs), stats, progress_bar),
    )
    progress_bar.close()
    split_file, empty_count = _dump_empty_joint_results(
        dataset, final_output_dir, logger
    )
    logger.info(
        "-------------------performance of epoch %s---------------------" % epoch_id
    )
    logger.info("%s", datetime.now(timezone.utc))
    ret_dict = {"empty_cnt": empty_count}
    ret_dict.update(_joint_metric_summary(stats, dataset, thresh_list, logger))
    ret_dict.update(_joint_average_precision(
        dataloader, dataset, final_output_dir, split_file, logger
    ))
    logger.info("result is saved to: %s" % result_dir)
    return ret_dict


def eval_one_epoch(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _EVAL_ONE_EPOCH_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    model, dataloader, epoch_id, result_dir, logger, is_get_feature = bound_parameters
    if is_get_feature:
        eval_one_epoch_joint(
            model, dataloader, epoch_id, result_dir, logger, is_get_feature=True
        )
        return None
    if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_rpn(model, dataloader, epoch_id, result_dir, logger)
    elif not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_rcnn(model, dataloader, epoch_id, result_dir, logger)
    elif cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger)
    else:
        raise NotImplementedError
    return ret_dict


def load_part_ckpt(model, filename, logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint["model_state"]

        update_model_state = {
            key: val for key, val in model_state.items() if key in model.state_dict()
        }
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


def load_ckpt_based_on_args(model, logger):
    if args.ckpt is not None:
        train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)

    total_keys = model.state_dict().keys().__len__()
    if cfg.RPN.ENABLED and args.rpn_ckpt is not None:
        load_part_ckpt(
            model, filename=args.rpn_ckpt, logger=logger, total_keys=total_keys
        )

    if cfg.RCNN.ENABLED and args.rcnn_ckpt is not None:
        load_part_ckpt(
            model, filename=args.rcnn_ckpt, logger=logger, total_keys=total_keys
        )


def _evaluation_runtime_setup(root_result_dir, log_file):
    logger = create_logger(log_file)
    logger.info("**********************Start logging**********************")
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger=logger)
    test_loader = create_dataloader(logger)
    if args.model_type == "base":
        model = PointRCNN(
            num_classes=test_loader.dataset.num_class, use_xyz=True, mode="TEST"
        )
    model.cuda()
    backup_dir = os.path.join(root_result_dir, "backup_files")
    os.makedirs(backup_dir, exist_ok=True)
    os.system("cp *.py %s/" % backup_dir)
    os.system("cp ../lib/net/*.py %s/" % backup_dir)
    os.system("cp ../lib/datasets/kitti_rcnn_dataset.py %s/" % backup_dir)
    return logger, test_loader, model


def eval_single_ckpt(root_result_dir, is_get_feature=False):
    root_result_dir = os.path.join(root_result_dir, "eval")
    # set epoch_id and output dir
    num_list = re.findall(r"\d+", args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else "no_number"
    root_result_dir = os.path.join(
        root_result_dir, "epoch_%s" % epoch_id, cfg.TEST.SPLIT
    )
    root_result_dir = os.path.join(root_result_dir, "eval")
    if args.test:
        root_result_dir = os.path.join(root_result_dir, "test_mode")

    if args.extra_tag != "default":
        root_result_dir = os.path.join(root_result_dir, args.extra_tag)
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, "log_eval_one.txt")
    logger, test_loader, model = _evaluation_runtime_setup(root_result_dir, log_file)

    # load checkpoint
    load_ckpt_based_on_args(model, logger)

    # start evaluation
    eval_one_epoch(
        model,
        test_loader,
        epoch_id,
        root_result_dir,
        logger,
        is_get_feature=is_get_feature,
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, "*checkpoint_epoch_*.pth"))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [
        float(x.strip()) for x in open(ckpt_record_file, "r").readlines()
    ]

    for cur_ckpt in ckpt_list:
        num_list = re.findall("checkpoint_epoch_(.*).pth", cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if (
            float(epoch_id) not in evaluated_ckpt_list
            and int(float(epoch_id)) >= args.start_epoch
        ):
            return epoch_id, cur_ckpt
    return -1, None


def _repeat_eval_setup(root_result_dir):
    root_result_dir = os.path.join(
        root_result_dir, "eval", "eval_all_" + args.extra_tag
    )
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, "log_eval_all_%s.txt" % cfg.TEST.SPLIT)
    logger, test_loader, model = _evaluation_runtime_setup(root_result_dir, log_file)

    ckpt_record_file = os.path.join(
        root_result_dir, "eval_list_%s.txt" % cfg.TEST.SPLIT
    )
    with open(ckpt_record_file, "a"):
        pass

    tb_log = SummaryWriter(
        logdir=os.path.join(root_result_dir, "tensorboard_%s" % cfg.TEST.SPLIT)
    )
    return RepeatEvaluationSetup(
        root_result_dir, logger, test_loader, model, ckpt_record_file, tb_log
    )


def repeat_eval_ckpt(root_result_dir, ckpt_dir):
    setup = _repeat_eval_setup(root_result_dir)
    root_result_dir, logger, test_loader, model, ckpt_record_file, tb_log = setup

    total_time = 0
    first_eval = True
    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            LOGGER.info(
                "%s", "Wait %s second for next check: %s" % (wait_second, ckpt_dir)
            )
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (not (first_eval)):
                break
            continue

        total_time = 0
        first_eval = False

        # load checkpoint
        train_utils.load_checkpoint(model, filename=cur_ckpt)

        # start evaluation
        cur_result_dir = os.path.join(
            root_result_dir, "epoch_%s" % cur_epoch_id, cfg.TEST.SPLIT
        )
        tb_dict = eval_one_epoch(
            model, test_loader, cur_epoch_id, cur_result_dir, logger
        )

        step = int(float(cur_epoch_id))
        if step == float(cur_epoch_id):
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, step)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, "a") as f:
            f.write(str("%s" % cur_epoch_id) + "\n")
        logger.info("Epoch %s has been evaluated" % cur_epoch_id)


def create_dataloader(logger):
    mode = "TEST" if args.test else "EVAL"
    data_path = os.path.join("../", "data")
    # create dataloader
    test_set = KittiRCNNDataset(
        root_dir=data_path,
        npoints=cfg.RPN.NUM_POINTS,
        split=cfg.TEST.SPLIT,
        mode=mode,
        random_select=args.random_select,
        rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
        rcnn_eval_feature_dir=args.rcnn_eval_feature_dir,
        classes=cfg.CLASSES,
        logger=logger,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=test_set.collate_batch,
    )

    return test_loader


if __name__ == "__main__":
    # merge config and log to file
    is_get_feature = args.fid
    cfg.fid = False
    if is_get_feature:
        LOGGER.info("%s %s", cfg.RPN.USE_INTENSITY, cfg.RPN.NUM_POINTS)
        cfg.RPN.USE_INTENSITY = False
        cfg.fid = True
        LOGGER.info("%s %s", cfg.RPN.USE_INTENSITY, cfg.fid)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.eval_mode == "rpn":
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join("../", "output", "rpn", cfg.TAG)
        ckpt_dir = os.path.join("../", "output", "rpn", cfg.TAG, "ckpt")
    elif args.eval_mode == "rcnn":
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join("../", "output", "rcnn", cfg.TAG)
        ckpt_dir = os.path.join("../", "output", "rcnn", cfg.TAG, "ckpt")
    elif args.eval_mode == "rcnn_online":
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = True
        cfg.RPN.FIXED = False
        root_result_dir = os.path.join("../", "output", "rcnn", cfg.TAG)
        ckpt_dir = os.path.join("../", "output", "rcnn", cfg.TAG, "ckpt")
    elif args.eval_mode == "rcnn_offline":
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        root_result_dir = os.path.join("../", "output", "rcnn", cfg.TAG)
        ckpt_dir = os.path.join("../", "output", "rcnn", cfg.TAG, "ckpt")
        if not (
            args.rcnn_eval_roi_dir is not None
            and args.rcnn_eval_feature_dir is not None
        ):
            raise AssertionError
    else:
        raise NotImplementedError

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir

    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok=True)

    with torch.no_grad():
        if args.eval_all:
            if not (os.path.exists(ckpt_dir)):
                raise AssertionError("%s" % ckpt_dir)
            repeat_eval_ckpt(root_result_dir, ckpt_dir)
        else:
            eval_single_ckpt(root_result_dir, is_get_feature=is_get_feature)
