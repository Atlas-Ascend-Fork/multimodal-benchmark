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
import argparse
import logging
import os
import re
import time

from torch.utils.data import DataLoader
import numpy as np
import torch
import tqdm

from jmodt.config import cfg, VALID_SEQ_ID
from jmodt.detection.datasets.kitti_dataset import KittiDataset
from jmodt.detection.evaluation.evaluate import evaluate as evaluate_detection
from jmodt.detection.modeling.point_rcnn import PointRCNN
from jmodt.ops.iou3d import iou3d_utils
from jmodt.tracking import tracker
from jmodt.tracking.kitti_evaluate import evaluate as evaluate_tracking
from jmodt.utils import train_utils, kitti_utils
from jmodt.utils.bbox_transform import decode_bbox_target
from jmodt.utils.object3d import Object3d


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters
from shared.kitti_detection import project_detection_boxes


DetectionPredictions = namedtuple(
    "DetectionPredictions",
    "regions boxes raw_scores scores features segmentation",
)


class SaveKittiDetectionFormatParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    sample_id: object
    calib: object
    bbox3d: object
    kitti_output_dir: object
    scores: object
    img_shape: object
    feat: object
    feat_output_dir: object


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument(
    "--data_root", type=str, default="data/KITTI", help="the ground truth data root"
)
parser.add_argument(
    "--det_output",
    type=str,
    default="output/det",
    help="the detection output directory",
)
parser.add_argument(
    "--output_dir", type=str, default="output", help="the tracking output directory"
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="checkpoints/jmodt.pth",
    help="the pretrained model path",
)
parser.add_argument(
    "--tag", type=str, default="mot_data", help="the tag for tracking results"
)
parser.add_argument(
    "--hungarian", action="store_true", help="whether to use hungarian algorithm"
)
parser.add_argument(
    "--only_tracking",
    action="store_true",
    help="whether to evaluate tracking without detection",
)
parser.add_argument(
    "--test", action="store_true", help="whether to use the test split data"
)
args = parser.parse_args()

# global random seed can be specified here
np.random.seed(2333)


def _create_detection_runtime(logger):
    mode = "TEST" if args.test else "EVAL"
    split = cfg.TEST.SPLIT if args.test else cfg.EVAL.SPLIT
    dataset = KittiDataset(
        root_dir=args.data_root,
        npoints=cfg.RPN.NUM_POINTS,
        split=split,
        mode=mode,
        classes=cfg.CLASSES,
        challenge="tracking",
        logger=logger,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        collate_fn=dataset.collate_batch,
        timeout=600,
        persistent_workers=True,
    )
    model = PointRCNN(num_classes=dataset.num_class, use_xyz=True, mode=mode)
    model.eval()
    model.cuda()
    train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)
    return mode, dataset, loader, model


def _detection_output_dirs():
    text_dir = os.path.join(args.det_output, "txt")
    feature_dir = os.path.join(args.det_output, "feat")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    return text_dir, feature_dir


def _detection_inputs(data):
    inputs = torch.from_numpy(data["pts_input"]).cuda(non_blocking=True).float()
    model_input = {"pts_input": inputs}
    if cfg.LI_FUSION.ENABLED:
        model_input["pts_xy"] = (
            torch.from_numpy(data["pts_xy"]).cuda(non_blocking=True).float()
        )
        model_input["img"] = (
            torch.from_numpy(data["img"])
            .cuda(non_blocking=True)
            .float()
            .permute((0, 3, 1, 2))
        )
    return model_input


def _detection_predictions(model, data):
    batch_size = len(data["sample_id"])
    output = model(_detection_inputs(data))
    rois = output["rois"]
    classes = output["rcnn_cls"].view(batch_size, -1, output["rcnn_cls"].shape[1])
    regression = output["rcnn_reg"].view(batch_size, -1, output["rcnn_reg"].shape[1])
    features = output["rcnn_feat"].view(batch_size, -1, output["rcnn_feat"].shape[1])
    if cfg.USE_IOU_BRANCH:
        iou_scores = output["rcnn_iou_branch"].view(
            batch_size, -1, output["rcnn_iou_branch"].shape[1]
        )
        classes *= torch.clamp(iou_scores, min=1e-4)
    boxes = decode_bbox_target(
        rois.view(-1, 7),
        regression.view(-1, regression.shape[-1]),
        anchor_size=torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda(),
        loc_scope=cfg.RCNN.LOC_SCOPE,
        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
        get_xz_fine=True,
        get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
        loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
        get_ry_fine=True,
    ).view(batch_size, -1, 7)
    raw_scores, scores = _normalized_detection_scores(classes)
    return DetectionPredictions(
        rois, boxes, raw_scores, scores, features, output["seg_result"].long()
    )


def _normalized_detection_scores(classes):
    if classes.shape[2] == 1:
        return classes, torch.sigmoid(classes)
    predicted = torch.argmax(classes, dim=1).view(-1)
    normalized = torch.softmax(classes, dim=1)
    return classes[:, predicted], normalized[:, predicted]


def _valid_ground_truth(boxes):
    last = len(boxes) - 1
    while last >= 0 and boxes[last].sum() == 0:
        last -= 1
    if last < 0:
        return None
    return torch.from_numpy(boxes[: last + 1]).cuda(non_blocking=True).float()


def _add_recall_counts(counts, maximum_iou, thresholds):
    for index, threshold in enumerate(thresholds):
        counts[index] += (maximum_iou > threshold).sum().item()


def _update_detection_recall(stats, data, predictions, thresholds):
    rois, boxes, _, _, _, segmentation = predictions
    rpn_labels = None
    if not cfg.RPN.FIXED:
        rpn_labels = (
            torch.from_numpy(data["rpn_cls_label"]).cuda(non_blocking=True).long()
        )
    for index, ground_truth in enumerate(data["gt_boxes3d"]):
        ground_truth = _valid_ground_truth(ground_truth)
        if ground_truth is not None:
            refined_iou = iou3d_utils.boxes_iou3d_gpu(boxes[index], ground_truth)
            roi_iou = iou3d_utils.boxes_iou3d_gpu(rois[index], ground_truth)
            _add_recall_counts(stats["refined"], refined_iou.max(dim=0)[0], thresholds)
            _add_recall_counts(stats["roi"], roi_iou.max(dim=0)[0], thresholds)
            stats["ground_truth"] += ground_truth.shape[0]
        if rpn_labels is not None:
            foreground = rpn_labels > 0
            correct = ((segmentation == rpn_labels) & foreground).sum().float()
            union = (
                foreground.sum().float() + (segmentation > 0).sum().float() - correct
            )
            stats["rpn_iou"] += (correct / torch.clamp(union, min=1.0)).item()


def _save_detection_batch(dataset, data, predictions, output_dirs):
    _, boxes, raw_scores, scores, features, _ = predictions
    text_dir, feature_dir = output_dirs
    selected = scores > cfg.RCNN.SCORE_THRESH
    saved = 0
    for index, sample_id in enumerate(data["sample_id"]):
        mask = selected[index].view(-1)
        if mask.sum() == 0:
            continue
        batch_boxes = boxes[index, mask]
        batch_raw_scores = raw_scores[index, mask]
        batch_features = features[index, mask]
        batch_scores = scores[index, mask]
        bev_boxes = kitti_utils.boxes3d_to_bev_torch(batch_boxes)
        keep = iou3d_utils.nms_gpu(
            bev_boxes, batch_raw_scores, cfg.RCNN.NMS_THRESH
        ).view(-1)
        selected_boxes = batch_boxes[keep].cpu().numpy()
        save_kitti_detection_format(
            sample_id,
            dataset.get_calib(sample_id),
            selected_boxes,
            text_dir,
            batch_scores[keep].cpu().numpy(),
            dataset.get_image_shape(sample_id),
            batch_features[keep].cpu().numpy(),
            feature_dir,
        )
        saved += selected_boxes.shape[0]
    return saved


def _create_empty_detection_files(dataset, text_dir, logger):
    empty_count = 0
    for image_index in dataset.image_idx_list:
        path = os.path.join(text_dir, "%06d.txt" % int(image_index))
        if os.path.exists(path):
            continue
        open(path, "w").close()
        empty_count += 1
        logger.info("empty_cnt=%d: dump empty file %s", empty_count, path)


def _log_detection_performance(logger, dataset, text_dir, stats, thresholds):
    logger.info(
        "-------------------performance of epoch %s---------------------",
        stats["epoch_id"],
    )
    count = max(stats["batches"], 1)
    logger.info(
        "final average detections: %.3f", stats["detections"] / max(len(dataset), 1)
    )
    logger.info("final average rpn_iou refined: %.3f", stats["rpn_iou"] / count)
    for name, label in (("roi", "roi bbox"), ("refined", "bbox")):
        for index, threshold in enumerate(thresholds):
            recalled = stats[name][index]
            recall = recalled / max(stats["ground_truth"], 1)
            logger.info(
                "total %s recall(thresh=%.3f): %d / %d = %f",
                label,
                threshold,
                recalled,
                stats["ground_truth"],
                recall,
            )
    class_ids = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    result, _ = evaluate_detection(
        dataset.label_dir,
        text_dir,
        image_idx_list=dataset.image_idx_list,
        current_class=class_ids.get(cfg.CLASSES),
    )
    logger.info("Averate Precision:\n%s", result)


@torch.no_grad()
def eval_joint_detection(logger):
    det_output = args.det_output
    # set epoch_id and output dir
    num_list = re.findall(r"\d+", args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else "no_number"

    logger.info("**********************Start evaluate detection**********************")

    mode, dataset, dataloader, model = _create_detection_runtime(logger)
    output_dirs = _detection_output_dirs()
    logger.info("==> Detection output dir: %s", det_output)
    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    stats = {
        "refined": [0] * len(thresh_list),
        "roi": [0] * len(thresh_list),
        "ground_truth": 0,
        "rpn_iou": 0,
        "batches": 0,
        "detections": 0,
        "epoch_id": epoch_id,
    }
    progress_bar = tqdm.tqdm(
        total=len(dataloader), leave=True, desc="test" if args.test else "eval"
    )
    for data in dataloader:
        stats["batches"] += 1
        predictions = _detection_predictions(model, data)
        if not args.test:
            _update_detection_recall(stats, data, predictions, thresh_list)
        stats["detections"] += _save_detection_batch(
            dataset, data, predictions, output_dirs
        )
        disp_dict = {
            "mode": mode,
            "recall": "%d/%d" % (stats["refined"][3], stats["ground_truth"]),
        }
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()
    progress_bar.close()
    _create_empty_detection_files(dataset, output_dirs[0], logger)
    if not args.test:
        _log_detection_performance(logger, dataset, output_dirs[0], stats, thresh_list)
    logger.info("result is saved to: %s", det_output)


_SAVE_KITTI_DETECTION_FORMAT_PARAMETERS_SCHEMA = _ParameterSchema(
    SaveKittiDetectionFormatParameters,
    (
        "sample_id",
        "calib",
        "bbox3d",
        "kitti_output_dir",
        "scores",
        "img_shape",
        "feat",
        "feat_output_dir",
    ),
    {"feat": None, "feat_output_dir": None},
    frozenset(()),
    8,
    None,
    None,
)


def _format_kitti_detection(box, image_box, score):
    x, z, ry = box[0], box[2], box[6]
    beta = np.arctan2(z, x)
    alpha = -np.sign(beta) * np.pi / 2 + beta + ry
    values = (
        cfg.CLASSES, alpha, image_box[0], image_box[1], image_box[2], image_box[3],
        box[3], box[4], box[5], box[0], box[1], box[2], box[6], score,
    )
    template = "%s -1 -1 " + " ".join(["%.4f"] * 13)
    return template % values + "\n"


def save_kitti_detection_format(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _SAVE_KITTI_DETECTION_FORMAT_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        sample_id,
        calib,
        bbox3d,
        kitti_output_dir,
        scores,
        img_shape,
        feat,
        feat_output_dir,
    ) = bound_parameters
    img_boxes, box_valid_mask = project_detection_boxes(
        bbox3d, calib, img_shape, kitti_utils
    )

    kitti_output_file = os.path.join(kitti_output_dir, "%06d.txt" % sample_id)
    with open(kitti_output_file, "w") as f:
        for k in range(bbox3d.shape[0]):
            if not box_valid_mask[k]:
                continue
            f.write(_format_kitti_detection(bbox3d[k], img_boxes[k], scores[k]))
    if feat is not None:
        output_file = os.path.join(feat_output_dir, "%06d.npy" % sample_id)
        np.save(output_file, feat[box_valid_mask].astype(np.float32))


def convert_det_sample_to_seq_frame(seq2sample_path, sample2frame_path):
    seq2sample_dict = {}
    sample2frame_dict = {}
    with open(seq2sample_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split()
        seq_id = split_line[0]
        seq2sample_dict[seq_id] = split_line[1:]
    with open(sample2frame_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split()
        sample2frame_dict[split_line[0]] = split_line[2]
    return seq2sample_dict, sample2frame_dict


def _build_tracking_tracker(logger):
    params = {
        "t_miss": 2, "t_hit": 0, "w_cls": 100, "w_app": 2,
        "w_iou": 10, "w_dis": 10, "w_se": 1,
    }
    logger.info("**********************Start evaluate tracking**********************")
    logger.info(
        "t_miss={t_miss}, t_hit={t_hit}, w_cls={w_cls}, w_app={w_app}, "
        "w_iou={w_iou}, w_dis={w_dis}, w_se={w_se}".format(**params)
    )
    model = PointRCNN(num_classes=2, mode="TEST" if args.test else "EVAL")
    model.eval()
    model.cuda()
    train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)
    return tracker.Tracker(
        link_model=model.rcnn_net.link_layer,
        se_model=model.rcnn_net.se_layer,
        hungarian=args.hungarian,
        **params,
    )


def _tracking_index_maps():
    split = "testing" if args.test else "training"
    mapping_root = os.path.join(args.data_root, "tracking_object", split)
    return convert_det_sample_to_seq_frame(
        os.path.join(mapping_root, "seq2sample.txt"),
        os.path.join(mapping_root, "sample2frame.txt"),
    )


def _load_tracking_frame(det_res_dir, sample_id, cls_threshold):
    with open(os.path.join(det_res_dir, "txt", f"{sample_id}.txt"), "r") as file_handle:
        object_lines = file_handle.readlines()
    if not object_lines:
        return None
    objects = np.array([Object3d(line) for line in object_lines])
    features = np.load(os.path.join(det_res_dir, "feat", f"{sample_id}.npy"))
    if len(features) != len(objects):
        raise AssertionError(f"obj {len(objects)} != feat {len(features)}")
    features = torch.from_numpy(features).cuda(non_blocking=True)
    boxes = np.empty((len(objects), 7), dtype=np.float32)
    for index, obj in enumerate(objects):
        boxes[index] = (*obj.pos, obj.h, obj.w, obj.length_value, obj.ry)
    scores = np.array([obj.score for obj in objects], dtype=np.float32)
    mask = scores > cls_threshold
    return boxes[mask], scores[mask], features[mask], objects[mask]


def _run_tracking_evaluation(logger, part):
    if args.test:
        return
    evaluate_tracking(
        result_sha=args.tag,
        result_root=args.output_dir,
        part=part,
        gt_path=os.path.join(args.data_root, "tracking", "training"),
        logger=logger,
    )


def _evaluate_tracking_sequence(context, sequence_id, sample_ids):
    tracker = context["tracker"]
    tracker.reset()
    total_time = total_frames = 0
    output_path = os.path.join(context["output_dir"], f"{sequence_id}.txt")
    progress = tqdm.tqdm(
        total=len(sample_ids), desc=sequence_id, dynamic_ncols=True, leave=True
    )
    with open(output_path, "w+") as output_file, torch.no_grad():
        for sample_id in sample_ids:
            frame_data = _load_tracking_frame(
                context["detection_dir"], sample_id, context["threshold"]
            )
            if frame_data is None:
                progress.update()
                continue
            boxes_3d, scores, car_features, car_objects = frame_data
            frame_id = int(context["sample_to_frame"][sample_id])
            start_time = time.time()
            results = tracker.update(
                frame_id, boxes_3d, scores, car_features, car_objects
            )
            frame_time = time.time() - start_time
            total_time += frame_time
            total_frames += 1
            progress.set_postfix({"time": frame_time})
            progress.update()
            save_kitti_tracking_format(results, frame_id, output_file)
    progress.close()
    return total_time, total_frames


def eval_tracking(logger):
    part = "test" if args.test else "val"
    tracking_res_dir = os.path.join(args.output_dir, args.tag, part)
    os.makedirs(tracking_res_dir, exist_ok=True)
    det_res_dir = args.det_output

    cls_threshold = 0.85
    car_tracker = _build_tracking_tracker(logger)

    total_time = total_frames = 0

    seq2sample_dict, sample2frame_dict = _tracking_index_maps()

    seq_list = [str(i).zfill(4) for i in range(29)] if args.test else VALID_SEQ_ID
    context = {
        "tracker": car_tracker,
        "output_dir": tracking_res_dir,
        "detection_dir": det_res_dir,
        "threshold": cls_threshold,
        "sample_to_frame": sample2frame_dict,
    }
    for seq_id in seq_list:
        elapsed, frame_count = _evaluate_tracking_sequence(
            context, seq_id, seq2sample_dict[seq_id]
        )
        total_time += elapsed
        total_frames += frame_count
    logger.info(
        f"total frames: {total_frames}, total time: {total_time}, frames per second: {total_frames / total_time}"
    )

    _run_tracking_evaluation(logger, part)


def save_kitti_tracking_format(results, frame_id, out_file):
    for tid, info, _ in results:
        out_file.write(
            "%d %d %s %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n"
            % (
                frame_id,
                tid,
                info.cls_type,
                int(info.truncation),
                int(info.occlusion),
                info.alpha,
                info.box2d[0],
                info.box2d[1],
                info.box2d[2],
                info.box2d[3],
                info.h,
                info.w,
                info.length_value,
                info.pos[0],
                info.pos[1],
                info.pos[2],
                info.ry,
                info.score,
            )
        )


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("%(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level=logging.INFO)
    logger.addHandler(ch)

    if not args.only_tracking:
        os.makedirs(args.det_output, exist_ok=True)
        det_log_path = os.path.join(
            args.det_output,
            f'{datetime.now(timezone.utc).strftime("%Y-%m-%d-%S-%M-%H")}.log',
        )
        det_fh = logging.FileHandler(det_log_path)
        det_fh.setFormatter(formatter)
        det_fh.setLevel(level=logging.INFO)
        logger.addHandler(det_fh)

        # start eval detection
        eval_joint_detection(logger)

        logger.removeHandler(det_fh)

    os.makedirs(os.path.join(args.output_dir, args.tag), exist_ok=True)
    trk_log_path = os.path.join(
        args.output_dir,
        args.tag,
        f'{datetime.now(timezone.utc).strftime("%Y-%m-%d-%S-%M-%H")}.log',
    )
    trk_fh = logging.FileHandler(trk_log_path)
    trk_fh.setFormatter(formatter)
    trk_fh.setLevel(level=logging.INFO)
    logger.addHandler(trk_fh)

    # start eval tracking
    eval_tracking(logger)


if __name__ == "__main__":
    main()
