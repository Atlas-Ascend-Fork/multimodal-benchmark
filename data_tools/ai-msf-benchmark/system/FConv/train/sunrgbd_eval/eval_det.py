# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

""" Generic Code for Object Detection Evaluation

    Input:
    For each class:
        For each image:
            Predictions: box, score
            Groundtruths: box

    Output:
    For each class:
        precision-recal and average precision

    Author: Charles R. Qi
    Date: Oct 4th 2017

    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py

Author: Charles R. Qi
Date: October, 2017
"""
import logging
import os

from ops.pybind11 import box_ops_cc
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils.box_util import box3d_iou


matplotlib.use("Agg")

mpl.rc("axes", linewidth=2)


logger = logging.getLogger(__name__)


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_iou(bb1, bb2):
    """Compute IoU of two bounding boxes.
    ** Define your bod IoU function HERE **
    """

    iou3d, iou2d = box3d_iou(bb1, bb2)
    return iou3d


def get_iou_cc(bb1, bb2):
    ious = box_ops_cc.rbbox_iou_3d_pair(bb1[np.newaxis, ...], bb2[np.newaxis, ...])
    return ious[0, 1]


def _build_class_records(pred, gt):
    class_recs = {}
    npos = 0
    for img_id in gt:
        bbox = np.array(gt[img_id])
        npos += len(bbox)
        class_recs[img_id] = {"bbox": bbox, "det": [False] * len(bbox)}
    for img_id in pred:
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "det": []}
    return class_recs, npos


def _sorted_detections(pred):
    image_ids = []
    confidence = []
    bb_value = []
    for img_id in pred:
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            bb_value.append(box)
    confidence = np.array(confidence)
    bb_value = np.array(bb_value)
    sorted_ind = np.argsort(-confidence)
    return [image_ids[x] for x in sorted_ind], bb_value[sorted_ind, ...]


def _best_detection_overlap(box, ground_truth_boxes):
    maximum, maximum_index = -np.inf, -1
    for index in range(ground_truth_boxes.shape[0]):
        overlap = get_iou_cc(box, ground_truth_boxes[index])
        if overlap > maximum:
            maximum, maximum_index = overlap, index
    return maximum, maximum_index


def _match_detections(image_ids, bb_value, class_recs, ovthresh):
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        r = class_recs.get(image_ids[d])
        bb = bb_value[d, :].astype(float)
        bbgt = r["bbox"].astype(float)
        ovmax, jmax = _best_detection_overlap(bb, bbgt)

        if ovmax > ovthresh:
            if not r["det"][jmax]:
                tp[d] = 1.0
                r["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0
    return tp, fp


def eval_det_cls(pred, gt, ovthresh=0.25, use_07_metric=False):
    """Compute precision and recall for one object-detection class."""
    class_recs, npos = _build_class_records(pred, gt)
    image_ids, bb_value = _sorted_detections(pred)
    tp, fp = _match_detections(image_ids, bb_value, class_recs, ovthresh)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap


def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, result_dir=None):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    rec = {}
    prec = {}
    ap = {}
    for classname in gt_all.keys():
        rec[classname], prec[classname], ap[classname] = eval_det_cls(
            pred_all[classname], gt_all[classname], ovthresh, use_07_metric
        )

    plot_folder = os.path.join(result_dir, "ap_curves")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    for classname in sorted(ap.keys()):
        logging.info("%s: %.5f" % (classname, ap.get(classname)))
        plt.plot(rec.get(classname), prec.get(classname), lw=3)
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=24)
        plt.ylabel("Precision", fontsize=24)
        plt.title(classname, fontsize=24)
        plt.savefig(os.path.join(plot_folder, "%s.png" % classname))
        plt.cla()

    logging.info("mean_AP: %.5f" % (np.mean([ap.get(classname) for classname in ap])))
    return rec, prec, ap
