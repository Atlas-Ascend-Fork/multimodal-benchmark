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
import shutil
import subprocess
import sys

from configs.config import cfg
from ops.pybind11.rbbox_iou import rotate_nms_3d_cc as cube_nms
from tqdm import tqdm
import numpy as np
import torch

from datasets.provider_sample import from_prediction_to_label_format, compute_alpha
from train.test_runner import run_detection_test
from utils.utils import AverageMeter
from utils.model_helpers import run_detection_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


LOGGER = logging.getLogger(__name__)


def fill_files(output_dir, to_fill_filename_list):
    """Create empty files if not exist for the filelist."""
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, "w")
            fout.close()


def write_detection_results(output_dir, det_results):
    results = {}  # map from idx to list of strings, each string is a line (without \n)
    for idx in det_results:
        for class_type in det_results[idx]:
            dets = det_results[idx][class_type]
            for i, _ in enumerate(dets):
                box2d = dets[i][:4]
                tx, ty, tz, h, w, length_value, ry = dets[i][4:-1]
                score = dets[i][-1]
                alpha = compute_alpha(tx, tz, ry)
                output_str = class_type + " -1 -1 "
                output_str += "%.4f " % alpha
                output_str += "%.4f %.4f %.4f %.4f " % (
                    box2d[0],
                    box2d[1],
                    box2d[2],
                    box2d[3],
                )
                output_str += "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %f" % (
                    h,
                    w,
                    length_value,
                    tx,
                    ty,
                    tz,
                    ry,
                    score,
                )
                results.setdefault(idx, []).append(output_str)

    result_dir = os.path.join(output_dir, "data")
    shutil.rmtree(result_dir, ignore_errors=True)
    os.mkdir(result_dir)

    for idx in results:
        pred_filename = os.path.join(result_dir, "%06d.txt" % (idx))
        fout = open(pred_filename, "w")
        for line in results[idx]:
            fout.write(line + "\n")
        fout.close()

    # Make sure for each frame (no matter if we have measurement for that frame),
    # there is a TXT file
    idx_path = "kitti/image_sets/%s.txt" % cfg.TEST.DATASET

    to_fill_filename_list = [line.rstrip() + ".txt" for line in open(idx_path)]
    fill_files(result_dir, to_fill_filename_list)


def write_detection_results_nms(output_dir, det_results, threshold):
    nms_results = {}
    for idx in det_results:
        for class_type in det_results[idx]:
            dets = np.array(det_results[idx][class_type], dtype=np.float32)
            if len(dets) > 1:
                # (tx, ty, tz, h, w, l, ry, score) -> (tx, ty, tz, l, w, h, ry, score)
                dets_for_nms = dets[:, 4:][:, [0, 1, 2, 5, 4, 3, 6, 7]]
                keep = cube_nms(dets_for_nms, threshold)
                # (tx, ty, tz, h, w, l, ry, score) -> (tx, tz, l, w, ry, score)
                dets_keep = dets[keep]
            else:
                dets_keep = dets
            if idx not in nms_results:
                nms_results[idx] = {}
            nms_results[idx][class_type] = dets_keep

    write_detection_results(output_dir, nms_results)


def evaluate_py_wrapper(output_dir, async_eval=False):
    # official evaluation
    gt_dir = "data/kitti/training/label_2/"
    command = ["./train/kitti_eval/evaluate_object_3d_offline", gt_dir, output_dir]
    LOGGER.info("%s", command)
    with open(
        os.path.join(output_dir, "log_test.txt"), "a", encoding="utf-8"
    ) as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        if not async_eval and process.wait() != 0:
            raise RuntimeError("official KITTI evaluation failed")


def evaluate_cuda_wrapper(result_dir, image_set="val", async_eval=False):
    # https://github.com/traveller59/kitti-object-eval-python
    # Sometime we can not get the same result as official evaluation for car BEV HARD detection (+-6%)
    if cfg.DATA.CAR_ONLY:
        classes_idx = "0"
    elif cfg.DATA.PEOPLE_ONLY:
        classes_idx = "1,2"
    else:
        classes_idx = "0,1,2"

    gt_dir = "data/kitti/training/label_2/"
    label_split_file = "./data/kitti/%s.txt" % image_set
    command = [
        sys.executable,
        "../kitti-object-eval-python/evaluate.py",
        "evaluate",
        f"--label_path={gt_dir}",
        f"--result_path={result_dir}",
        f"--label_split_file={label_split_file}",
        f"--current_class={classes_idx}",
        "--coco=False",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    LOGGER.info("%s", command)
    log_path = os.path.join(result_dir, "..", "log_test_new.txt")
    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, env=environment
        )
        if not async_eval and process.wait() != 0:
            raise RuntimeError("CUDA KITTI evaluation failed")


def _prepare_detection_batch(data_dicts):
    point_clouds = data_dicts["point_cloud"]
    rot_angles = data_dicts["rot_angle"]
    ref_centers = data_dicts.get("ref_center")
    rgb_probs = data_dicts.get("rgb_prob")
    if rgb_probs is None:
        rgb_probs = torch.ones_like(rot_angles)
    if ref_centers is None:
        ref_centers = torch.zeros((point_clouds.shape[0], 3))
    rot_angles = rot_angles.view(-1)
    rgb_probs = rgb_probs.view(-1)
    if "box3d_center" in data_dicts:
        data_dicts.pop("box3d_center")
    inputs = {key: value.cuda() for key, value in data_dicts.items()}
    return inputs, tuple(
        value.numpy() for value in (rot_angles, rgb_probs, ref_centers)
    )


def _foreground_indices(class_probabilities, batch_index):
    if cfg.TEST.METHOD != "nms":
        return np.array([np.argmax(class_probabilities[batch_index, :, 1])])
    indices = (
        class_probabilities[batch_index, :, 0]
        < class_probabilities[batch_index, :, 1]
    ).nonzero()[0]
    if indices.size == 0:
        indices = np.array([np.argmax(class_probabilities[batch_index, :, 1])])
    return indices


def _append_batch_detections(context, batch_index):
    predictions = context["predictions"]
    class_probs, centers, headings, sizes = context["outputs"][:4]
    indices = _foreground_indices(class_probs, batch_index)
    sample_index = context["batch_offset"] + batch_index
    data_idx = context["dataset"].id_list[sample_index]
    class_type = context["dataset"].type_list[sample_index]
    box2d = context["dataset"].box2d_list[sample_index]
    destination = predictions.setdefault(data_idx, {}).setdefault(class_type, [])
    rot_angles, rgb_probs, ref_centers = context["metadata"]
    for prediction_index in indices:
        box = from_prediction_to_label_format(
            centers[batch_index, prediction_index],
            headings[batch_index, prediction_index],
            sizes[batch_index, prediction_index],
            rot_angles[batch_index],
            ref_centers[batch_index],
        )
        h, w, length_value, tx, ty, tz, ry = box
        if min(h, w, length_value) < 0.01:
            continue
        x1, y1, x2, y2 = box2d
        score = class_probs[batch_index, prediction_index, 1] + rgb_probs[batch_index]
        destination.append(
            [x1, y1, x2, y2, tx, ty, tz, h, w, length_value, ry, score]
        )


def _finish_detection_test(test_loader, time_meter, det_results, result_dir):
    num_images = len(det_results)
    logging.info("Average time:")
    logging.info("batch:%0.3f" % time_meter.avg)
    logging.info("avg_per_object:%0.3f" % (time_meter.avg / test_loader.batch_size))
    if num_images != 0:
        logging.info(
            "avg_per_image:%.3f" % (time_meter.avg * len(test_loader) / num_images)
        )
    if cfg.TEST.METHOD == "nms":
        write_detection_results_nms(result_dir, det_results, threshold=cfg.TEST.THRESH)
    else:
        write_detection_results(result_dir, det_results)
    output_dir = os.path.join(result_dir, "data")
    if "test" not in cfg.TEST.DATASET:
        if os.path.exists("../kitti-object-eval-python"):
            evaluate_cuda_wrapper(output_dir, cfg.TEST.DATASET)
        else:
            evaluate_py_wrapper(result_dir)
    else:
        LOGGER.info("results file save in  {}".format(result_dir))
        shutil.make_archive(
            os.path.join(result_dir, "..", "results"), "zip", result_dir
        )


def test(model, test_dataset, test_loader, output_filename, result_dir=None):
    model.eval()
    time_meter = AverageMeter()
    det_results = {}
    for batch_index, data_dicts in tqdm(enumerate(test_loader)):
        inputs, metadata = _prepare_detection_batch(data_dicts)
        outputs = run_detection_model(model, inputs, time_meter)
        context = {
            "dataset": test_dataset,
            "predictions": det_results,
            "outputs": outputs,
            "metadata": metadata,
            "batch_offset": test_loader.batch_size * batch_index,
        }
        for item_index in range(data_dicts["point_cloud"].shape[0]):
            _append_batch_detections(context, item_index)
    _finish_detection_test(test_loader, time_meter, det_results, result_dir)


if __name__ == "__main__":
    run_detection_test(test)
