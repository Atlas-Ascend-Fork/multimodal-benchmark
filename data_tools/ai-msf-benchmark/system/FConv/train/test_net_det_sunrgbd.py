# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
import sys
import time

from configs.config import cfg
from ops.pybind11.rbbox_iou import rotate_nms_3d_cc as cube_nms
from train.sunrgbd_eval.compare_matlab_python_eval import get_gt_all
from train.sunrgbd_eval.eval_det import eval_det
import numpy as np
import torch

from datasets.data_utils import compute_box_3d
from datasets.provider_sample_sunrgbd import from_prediction_to_label_format
from train.test_runner import run_detection_test
from utils.utils import AverageMeter
from utils.model_helpers import run_detection_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


LOGGER = logging.getLogger(__name__)


def write_detection_results(output_dir, det_results):
    all_results = (
        {}
    )  # map from idx to list of strings, each string is a line (without \n)
    for idx in det_results:
        for class_type in det_results[idx]:
            dets = det_results[idx][class_type]

            if not isinstance(dets, np.ndarray):
                dets = np.array(dets, dtype=np.float32)

            for i, _ in enumerate(dets):
                box_center = dets[i][4: 4 + 3]
                box_size = dets[i][4 + 3: 4 + 6]
                heading = dets[i][4 + 6]
                score = dets[i][-1]
                box3d_corners = compute_box_3d(box_center, box_size, heading)

                dets_scores = (box3d_corners, score)

                class_results = all_results.setdefault(class_type, {})
                class_results.setdefault(idx, []).append(dets_scores)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return all_results


def write_detection_results_nms(output_dir, det_results, threshold=cfg.TEST.THRESH):

    _ = time.time()
    nms_results = {}
    for idx in det_results:
        for class_type in det_results[idx]:
            dets = np.array(det_results[idx][class_type], dtype=np.float32)
            if len(dets) > 1:
                dets_for_nms = dets[:, 4:]
                keep = cube_nms(dets_for_nms, threshold)
                dets_keep = dets[keep]
            else:
                dets_keep = dets
            if idx not in nms_results:
                nms_results[idx] = {}
            nms_results[idx][class_type] = dets_keep

    return write_detection_results(output_dir, nms_results)


def _prepare_detection_batch(data_dicts):
    point_clouds = data_dicts["point_cloud"]
    rot_angles = data_dicts["rot_angle"]
    ref_centers = data_dicts.get("ref_center")
    rgb_probs = data_dicts.get("rgb_prob")
    if rgb_probs is None:
        rgb_probs = torch.ones_like(rot_angles)
    if ref_centers is None:
        ref_centers = torch.zeros((point_clouds.shape[0], 3))
    if "box3d_center" in data_dicts:
        data_dicts.pop("box3d_center")
    inputs = {key: value.cuda() for key, value in data_dicts.items()}
    metadata = tuple(
        value.numpy() for value in (rot_angles.view(-1), rgb_probs.view(-1), ref_centers)
    )
    return inputs, metadata


def _foreground_indices(class_probabilities, batch_index):
    if cfg.TEST.METHOD != "nms":
        return np.array([np.argmax(class_probabilities[batch_index, :, 1])])
    indices = (class_probabilities[batch_index, :, 1] > 0.5).nonzero()[0]
    if indices.size == 0:
        indices = np.array([np.argmax(class_probabilities[batch_index, :, 1])])
    return indices


def _append_batch_detections(context, batch_index):
    class_probs, centers, headings, sizes, _, size_probs = context["outputs"]
    indices = _foreground_indices(class_probs, batch_index)
    sample_index = context["batch_offset"] + batch_index
    dataset = context["dataset"]
    data_idx, class_type = dataset.id_list[sample_index], dataset.type_list[sample_index]
    destination = context["predictions"].setdefault(data_idx, {}).setdefault(
        class_type, []
    )
    rot_angles, rgb_probs, ref_centers = context["metadata"]
    for prediction_index in indices:
        box = from_prediction_to_label_format(
            centers[batch_index, prediction_index],
            headings[batch_index, prediction_index],
            sizes[batch_index, prediction_index],
            rot_angles[batch_index],
            ref_centers[batch_index],
        )
        tx, ty, tz, length_value, width, height, rotation = box
        if min(height, width, length_value) < 0.01:
            continue
        x1, y1, x2, y2 = dataset.box2d_list[sample_index]
        score = rgb_probs[batch_index] + size_probs[batch_index, prediction_index].max()
        destination.append(
            [
                x1,
                y1,
                x2,
                y2,
                tx,
                ty,
                tz,
                length_value,
                width,
                height,
                rotation,
                score,
            ]
        )


def _finish_detection_test(
    test_loader, time_meter, det_results, result_dir, output_filename
):
    num_images = len(det_results)
    logging.info("Average time:")
    logging.info("batch:%0.3f" % time_meter.avg)
    logging.info("avg_per_object:%0.3f" % (time_meter.avg / test_loader.batch_size))
    logging.info(
        "avg_per_image:%.3f" % (time_meter.avg * len(test_loader) / num_images)
    )
    with open(output_filename, "wb") as f:
        pickle.dump(det_results, f, -1)
    tic = time.time()
    if cfg.TEST.METHOD == "nms":
        all_results = write_detection_results_nms(
            result_dir, det_results, cfg.TEST.THRESH
        )
    else:
        all_results = write_detection_results(result_dir, det_results)

    LOGGER.info("%s", "post: %.3f" % (time.time() - tic))

    tic = time.time()

    gt_all = get_gt_all()
    pred_all = all_results
    eval_det(pred_all, gt_all, result_dir=result_dir)
    LOGGER.info("%s", "eval: %.3f" % (time.time() - tic))


def test(model, test_dataset, test_loader, output_filename, result_dir=None):
    model.eval()
    time_meter = AverageMeter()
    det_results = {}
    for batch_index, data_dicts in enumerate(test_loader):
        inputs, metadata = _prepare_detection_batch(data_dicts)
        outputs = run_detection_model(model, inputs, time_meter)
        LOGGER.info(
            "%s", "%d/%d %.3f" % (batch_index, len(test_loader), time_meter.val)
        )
        context = {
            "dataset": test_dataset,
            "predictions": det_results,
            "outputs": outputs,
            "metadata": metadata,
            "batch_offset": test_loader.batch_size * batch_index,
        }
        for item_index in range(data_dicts["point_cloud"].shape[0]):
            _append_batch_detections(context, item_index)
    _finish_detection_test(
        test_loader, time_meter, det_results, result_dir, output_filename
    )


if __name__ == "__main__":
    run_detection_test(test)
