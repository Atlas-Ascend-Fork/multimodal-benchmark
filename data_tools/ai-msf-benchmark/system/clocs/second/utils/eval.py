# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import io as sysio

import numba
import numpy as np

from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class EvalClassParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    gt_annos: object
    dt_annos: object
    current_class: object
    difficulty: object
    metric: object
    min_overlap: object
    compute_aos: object
    num_parts: object


class EvalClassV3Parameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    gt_annos: object
    dt_annos: object
    current_classes: object
    difficultys: object
    metric: object
    min_overlaps: object
    compute_aos: object
    num_parts: object


class DoEvalV2Parameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    gt_annos: object
    dt_annos: object
    current_classes: object
    min_overlaps: object
    compute_aos: object
    difficultys: object


class PreparedDataResult(NamedTuple):
    """Named result returned without changing tuple unpacking behavior."""

    value_0: object
    value_1: object
    value_2: object
    value_3: object
    value_4: object
    value_5: object
    value_6: object


def get_m_ap(prec):
    sums = 0
    for value in prec[::4]:
        sums += value
    return sums / 11 * 100


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if ((r_recall - current_recall) < (current_recall - l_recall)) and (
            i < (len(scores) - 1)
        ):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def _module_clean_data_137_part_1(_state):
    _state["class_names"] = [
        "car",
        "pedestrian",
        "cyclist",
        "van",
        "person_sitting",
        "car",
        "tractor",
        "trailer",
    ]
    _state["min_height"] = [40, 25, 25]
    _state["max_occlusion"] = [0, 1, 2]
    _state["max_truncation"] = [0.15, 0.3, 0.5]
    _state["dc_bboxes"], _state["ignored_gt"], _state["ignored_dt"] = ([], [], [])
    _state["current_cls_name"] = _state["class_names"][_state["current_class"]].lower()
    _state["num_gt"] = len(_state["gt_anno"]["name"])
    _state["num_dt"] = len(_state["dt_anno"]["name"])
    _state["num_valid_gt"] = 0


def _ground_truth_class_status(gt_name, current_class_name):
    if gt_name == current_class_name:
        return 1
    neighboring_classes = {"pedestrian": "person_sitting", "car": "van"}
    return 0 if neighboring_classes.get(current_class_name) == gt_name else -1


def _module_clean_data_137_part_2(_state):
    for _state["i"] in range(_state["num_gt"]):
        _state["bbox"] = _state["gt_anno"]["bbox"][_state["i"]]
        _state["gt_name"] = _state["gt_anno"]["name"][_state["i"]].lower()
        _state["height"] = _state["bbox"][3] - _state["bbox"][1]
        _state["valid_class"] = _ground_truth_class_status(
            _state["gt_name"], _state["current_cls_name"]
        )
        _state["ignore"] = False
        if (
            _state["gt_anno"]["occluded"][_state["i"]]
            > _state["max_occlusion"][_state["difficulty"]]
            or _state["gt_anno"]["truncated"][_state["i"]]
            > _state["max_truncation"][_state["difficulty"]]
            or _state["height"] <= _state["min_height"][_state["difficulty"]]
        ):
            _state["ignore"] = True
        if _state["valid_class"] == 1 and (not _state["ignore"]):
            _state["ignored_gt"].append(0)
            _state["num_valid_gt"] += 1
        elif _state["valid_class"] == 0 or (
            _state["ignore"] and _state["valid_class"] == 1
        ):
            _state["ignored_gt"].append(1)
        else:
            _state["ignored_gt"].append(-1)
        if _state["gt_anno"]["name"][_state["i"]] == "DontCare":
            _state["dc_bboxes"].append(_state["gt_anno"]["bbox"][_state["i"]])


def _module_clean_data_137_part_3(_state):
    for _state["i"] in range(_state["num_dt"]):
        if _state["dt_anno"]["name"][_state["i"]].lower() == _state["current_cls_name"]:
            _state["valid_class"] = 1
        else:
            _state["valid_class"] = -1
        _state["height"] = abs(
            _state["dt_anno"]["bbox"][_state["i"], 3]
            - _state["dt_anno"]["bbox"][_state["i"], 1]
        )
        if _state["height"] < _state["min_height"][_state["difficulty"]]:
            _state["ignored_dt"].append(1)
        elif _state["valid_class"] == 1:
            _state["ignored_dt"].append(0)
        else:
            _state["ignored_dt"].append(-1)


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    _state = locals()
    _module_clean_data_137_part_1(_state)
    _module_clean_data_137_part_2(_state)
    _module_clean_data_137_part_3(_state)

    return (
        _state["num_valid_gt"],
        _state["ignored_gt"],
        _state["ignored_dt"],
        _state["dc_bboxes"],
    )


@numba.jit(nopython=True)
def _intersection_denominator(
    first_area, second_area, intersection, criterion, fallback
):
    if criterion == -1:
        return first_area + second_area - intersection
    if criterion == 0:
        return first_area
    if criterion == 1:
        return second_area
    return fallback


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    n_value = boxes.shape[0]
    k_value = query_boxes.shape[0]
    overlaps = np.zeros((n_value, k_value), dtype=boxes.dtype)
    for k in range(k_value):
        qbox_area = (query_boxes[k, 2] - query_boxes[k, 0]) * (
            query_boxes[k, 3] - query_boxes[k, 1]
        )
        for n in range(n_value):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]
            )
            if iw <= 0:
                continue
            ih = min(boxes[n, 3], query_boxes[k, 3]) - max(
                boxes[n, 1], query_boxes[k, 1]
            )
            if ih <= 0:
                continue
            box_area = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
            intersection = iw * ih
            ua = _intersection_denominator(
                box_area, qbox_area, intersection, criterion, 1.0
            )
            overlaps[n, k] = intersection / ua
    return overlaps


# pang added to build the tensor for the second stage of training
@numba.jit(nopython=True)
def _stage2_overlap(box, query_box, intersection, criterion):
    intersection_width, intersection_height = intersection
    query_area = (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if criterion == -1:
        union_area = box_area + query_area - intersection_width * intersection_height
    elif criterion == 0:
        union_area = box_area
    elif criterion == 1:
        union_area = query_area
    else:
        union_area = 1.0
    return intersection_width * intersection_height / union_area


@numba.jit(nopython=True, parallel=True)
def build_stage2_training(box_data, score_data, outputs, criterion):
    boxes, query_boxes = box_data
    scores_3d, scores_2d, dis_to_lidar_3d = score_data
    overlaps, tensor_index = outputs
    n_value = boxes.shape[0]  # 70400
    k_value = query_boxes.shape[0]  # 30
    ind = 0
    for k in range(k_value):
        for n in range(n_value):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]
            )
            ih = min(boxes[n, 3], query_boxes[k, 3]) - max(
                boxes[n, 1], query_boxes[k, 1]
            )
            if iw > 0 and ih > 0:
                overlaps[ind, 0] = _stage2_overlap(
                    boxes[n], query_boxes[k], (iw, ih), criterion
                )
                overlaps[ind, 1] = scores_3d[n, 0]
                overlaps[ind, 2] = scores_2d[k, 0]
                overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                tensor_index[ind, 0] = k
                tensor_index[ind, 1] = n
                ind = ind + 1
                continue
            if k != k_value - 1:
                continue
            overlaps[ind, 0] = -10
            overlaps[ind, 1] = scores_3d[n, 0]
            overlaps[ind, 2] = -10
            overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
            tensor_index[ind, 0] = k
            tensor_index[ind, 1] = n
            ind = ind + 1
    return overlaps, tensor_index, ind


# pang added to build the tensor for the second stage of training, added july 13 2019
@numba.jit(nopython=True, parallel=True)
def build_stage2_training_2(boxes, query_boxes, box_2d_corner):
    n_value = boxes.shape[0]  # 3D_projected
    k_value = query_boxes.shape[0]  # 2D
    overlaps_2 = np.zeros((n_value, k_value, 24), dtype=boxes.dtype)
    for k in range(k_value):
        _ = (query_boxes[k, 2] - query_boxes[k, 0]) * (
            query_boxes[k, 5] - query_boxes[k, 1]
        )
        for n in range(n_value):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]
            )
            ih = min(boxes[n, 3], query_boxes[k, 5]) - max(
                boxes[n, 1], query_boxes[k, 1]
            )
            if iw > 0 and ih > 0:
                overlaps_2[n, k, 0:8] = query_boxes[k, :]
                overlaps_2[n, k, 8:] = box_2d_corner[n, :]
    return overlaps_2


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    n, k = boxes.shape[0], qboxes.shape[0]
    for i in range(n):
        for j in range(k):
            if rinc[i, j] <= 0:
                continue
            iw = min(boxes[i, 1], qboxes[j, 1]) - max(
                boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]
            )
            if iw <= 0:
                rinc[i, j] = 0.0
                continue
            area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
            area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
            inc = iw * rinc[i, j]
            ua = _intersection_denominator(area1, area2, inc, criterion, inc)
            rinc[i, j] = inc / ua


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def _unavailable_detection(ignored, assigned, below_threshold):
    return ignored == -1 or assigned or below_threshold


@numba.jit(nopython=True)
def _is_score_match(compute_fp, valid_match, score, current_score):
    return not compute_fp and valid_match and score > current_score


@numba.jit(nopython=True)
def _is_preferred_match(compute_fp, valid_match, preferred_match, ignored):
    return compute_fp and valid_match and preferred_match and ignored == 0


@numba.jit(nopython=True)
def _is_ignored_match(compute_fp, valid_match, no_match_yet, ignored):
    return compute_fp and valid_match and no_match_yet and ignored == 1


@numba.jit(nopython=True)
def _find_best_detection(detection_data, matching_config):
    overlaps, dt_scores, ignored_det, assigned_detection, ignored_threshold = (
        detection_data
    )
    gt_index, min_overlap, compute_fp = matching_config
    no_detection = -10000000
    det_idx = -1
    valid_detection = no_detection
    max_overlap = 0
    assigned_ignored_det = False
    for det_index in range(dt_scores.shape[0]):
        if _unavailable_detection(
            ignored_det[det_index],
            assigned_detection[det_index],
            ignored_threshold[det_index],
        ):
            continue
        overlap = overlaps[det_index, gt_index]
        dt_score = dt_scores[det_index]
        valid_match = overlap > min_overlap
        preferred_match = overlap > max_overlap or assigned_ignored_det
        if _is_score_match(compute_fp, valid_match, dt_score, valid_detection):
            det_idx = det_index
            valid_detection = dt_score
            continue
        if _is_preferred_match(
            compute_fp, valid_match, preferred_match, ignored_det[det_index]
        ):
            max_overlap = overlap
            det_idx = det_index
            valid_detection = 1
            assigned_ignored_det = False
            continue
        if _is_ignored_match(
            compute_fp,
            valid_match,
            valid_detection == no_detection,
            ignored_det[det_index],
        ):
            det_idx = det_index
            valid_detection = 1
            assigned_ignored_det = True
    return det_idx, valid_detection


@numba.jit(nopython=True)
def _count_false_positives(detection_data, overlap_config):
    dt_bboxes, dc_bboxes, assigned_detection, ignored_det, ignored_threshold = (
        detection_data
    )
    metric, min_overlap = overlap_config
    false_positives = 0
    for det_index in range(ignored_det.shape[0]):
        if not (
            assigned_detection[det_index]
            or ignored_det[det_index] == -1
            or ignored_det[det_index] == 1
            or ignored_threshold[det_index]
        ):
            false_positives += 1
    if metric != 0:
        return false_positives
    overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
    for dc_index in range(dc_bboxes.shape[0]):
        for det_index in range(ignored_det.shape[0]):
            unavailable = assigned_detection[det_index] or ignored_det[det_index] in (
                -1,
                1,
            )
            if (
                not unavailable
                and not ignored_threshold[det_index]
                and overlaps_dt_dc[det_index, dc_index] > min_overlap
            ):
                assigned_detection[det_index] = True
                false_positives -= 1
    return false_positives


@numba.jit(nopython=True)
def compute_statistics_jit(statistics_data, statistics_config):
    overlaps, gt_datas, dt_datas, ignored_gt, ignored_det, dc_bboxes = statistics_data
    metric, min_overlap, thresh, compute_fp, compute_aos = statistics_config
    gt_size, dt_scores, dt_bboxes = gt_datas.shape[0], dt_datas[:, -1], dt_datas[:, :4]
    assigned_detection, ignored_threshold = (
        [False] * dt_datas.shape[0], [False] * dt_datas.shape[0]
    )
    if compute_fp:
        for i in range(dt_datas.shape[0]):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds, thresh_idx = np.zeros((gt_size,)), 0
    delta, delta_idx = np.zeros((gt_size,)), 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx, valid_detection = _find_best_detection(
            (overlaps, dt_scores, ignored_det, assigned_detection, ignored_threshold),
            (i, min_overlap, compute_fp),
        )

        if valid_detection == -10000000:
            if ignored_gt[i] == 0:
                fn += 1
            continue
        if ignored_gt[i] == 1 or ignored_det[det_idx] == 1:
            assigned_detection[det_idx] = True
            continue
        tp += 1
        thresholds[thresh_idx] = dt_scores[det_idx]
        thresh_idx += 1
        if compute_aos:
            delta[delta_idx] = gt_datas[i, 4] - dt_datas[det_idx, 4]
            delta_idx += 1
        assigned_detection[det_idx] = True
    if compute_fp:
        fp = _count_false_positives(
            (dt_bboxes, dc_bboxes, assigned_detection, ignored_det, ignored_threshold),
            (metric, min_overlap),
        )
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp + fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(
    statistics_data, sample_counts, annotations, statistics_config
):
    overlaps, pr = statistics_data
    gt_nums, dt_nums, dc_nums = sample_counts
    gt_datas, dt_datas, dontcares, ignored_gts, ignored_dets = annotations
    metric, min_overlap, thresholds, compute_aos = statistics_config
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[
                dt_num: dt_num + dt_nums[i], gt_num: gt_num + gt_nums[i]
            ]

            gt_data = gt_datas[gt_num: gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num: dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num: gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num: dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num: dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                (overlap, gt_data, dt_data, ignored_gt, ignored_det, dontcare),
                (metric, min_overlap, thresh, True, compute_aos),
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def _annotation_overlap_boxes(annotations, metric):
    if metric == 0:
        return np.concatenate([item["bbox"] for item in annotations], 0)
    axes = [0, 2] if metric == 1 else slice(None)
    locations = np.concatenate(
        [item["location"][:, axes] for item in annotations], 0
    )
    dimensions = np.concatenate(
        [item["dimensions"][:, axes] for item in annotations], 0
    )
    rotations = np.concatenate([item["rotation_y"] for item in annotations], 0)
    return np.concatenate(
        [locations, dimensions, rotations[..., np.newaxis]], axis=1
    )


def _calculate_part_overlap(gt_annotations, dt_annotations, metric):
    gt_boxes = _annotation_overlap_boxes(gt_annotations, metric)
    dt_boxes = _annotation_overlap_boxes(dt_annotations, metric)
    if metric == 0:
        return image_box_overlap(gt_boxes, dt_boxes)
    if metric == 1:
        return bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
    if metric == 2:
        return d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
    raise ValueError("unknown metric")


def _individual_overlaps(parted_overlaps, split_parts, gt_counts, dt_counts):
    overlaps = []
    example_index = 0
    for part_index, part_size in enumerate(split_parts):
        gt_index = dt_index = 0
        for offset in range(part_size):
            gt_count = gt_counts[example_index + offset]
            dt_count = dt_counts[example_index + offset]
            overlaps.append(
                parted_overlaps[part_index][
                    gt_index: gt_index + gt_count,
                    dt_index: dt_index + dt_count,
                ]
            )
            gt_index += gt_count
            dt_index += dt_count
        example_index += part_size
    return overlaps


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """Calculate per-example and partitioned KITTI overlaps."""
    if len(gt_annos) != len(dt_annos):
        raise AssertionError
    total_dt_num = np.stack([len(annotation["name"]) for annotation in dt_annos], 0)
    total_gt_num = np.stack([len(annotation["name"]) for annotation in gt_annos], 0)
    split_parts = get_split_parts(len(gt_annos), num_parts)
    parted_overlaps = []
    example_index = 0
    for part_size in split_parts:
        end = example_index + part_size
        parted_overlaps.append(
            _calculate_part_overlap(
                gt_annos[example_index:end], dt_annos[example_index:end], metric
            )
        )
        example_index = end
    overlaps = _individual_overlaps(
        parted_overlaps, split_parts, total_gt_num, total_dt_num
    )
    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i, _ in enumerate(gt_annos):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1
        )
        dt_datas = np.concatenate(
            [
                dt_annos[i]["bbox"],
                dt_annos[i]["alpha"][..., np.newaxis],
                dt_annos[i]["score"][..., np.newaxis],
            ],
            1,
        )
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return PreparedDataResult(
        gt_datas_list,
        dt_datas_list,
        ignored_gts,
        ignored_dets,
        dontcares,
        total_dc_num,
        total_num_valid_gt,
    )


_EVAL_CLASS_PARAMETERS_SCHEMA = _ParameterSchema(
    EvalClassParameters,
    (
        "gt_annos",
        "dt_annos",
        "current_class",
        "difficulty",
        "metric",
        "min_overlap",
        "compute_aos",
        "num_parts",
    ),
    {"compute_aos": False, "num_parts": 50},
    frozenset(()),
    8,
    None,
    None,
)
_EVAL_CLASS_V3_PARAMETERS_SCHEMA = _ParameterSchema(
    EvalClassV3Parameters,
    (
        "gt_annos",
        "dt_annos",
        "current_classes",
        "difficultys",
        "metric",
        "min_overlaps",
        "compute_aos",
        "num_parts",
    ),
    {"compute_aos": False, "num_parts": 50},
    frozenset(()),
    8,
    None,
    None,
)
_DO_EVAL_V2_PARAMETERS_SCHEMA = _ParameterSchema(
    DoEvalV2Parameters,
    (
        "gt_annos",
        "dt_annos",
        "current_classes",
        "min_overlaps",
        "compute_aos",
        "difficultys",
    ),
    {"compute_aos": False, "difficultys": None},
    frozenset(()),
    6,
    None,
    None,
)


def _module_eval_class_728_part_1_chunk_1(_state):
    _state["bound_parameters"] = _bind_parameters(
        _EVAL_CLASS_PARAMETERS_SCHEMA, _state["call_args"], _state["call_kwargs"]
    )
    (
        _state["gt_annos"],
        _state["dt_annos"],
        _state["current_class"],
        _state["difficulty"],
        _state["metric"],
        _state["min_overlap"],
        _state["compute_aos"],
        _state["num_parts"],
    ) = _state["bound_parameters"]
    if not len(_state["gt_annos"]) == len(_state["dt_annos"]):
        raise AssertionError
    _state["num_examples"] = len(_state["gt_annos"])
    _state["split_parts"] = get_split_parts(_state["num_examples"], _state["num_parts"])
    _state["thresholdss"] = []
    _state["rets"] = calculate_iou_partly(
        _state["dt_annos"], _state["gt_annos"], _state["metric"], _state["num_parts"]
    )
    (
        _state["overlaps"],
        _state["parted_overlaps"],
        _state["total_dt_num"],
        _state["total_gt_num"],
    ) = _state["rets"]
    _state["rets"] = _prepare_data(
        _state["gt_annos"],
        _state["dt_annos"],
        _state["current_class"],
        _state["difficulty"],
    )
    (
        _state["gt_datas_list"],
        _state["dt_datas_list"],
        _state["ignored_gts"],
        _state["ignored_dets"],
        _state["dontcares"],
        _state["total_dc_num"],
        _state["total_num_valid_gt"],
    ) = _state["rets"]


def _module_eval_class_728_part_1_chunk_2(_state):
    for _state["i"], _state["_"] in enumerate(_state["gt_annos"]):
        _state["rets"] = compute_statistics_jit(
            (
                _state["overlaps"][_state["i"]],
                _state["gt_datas_list"][_state["i"]],
                _state["dt_datas_list"][_state["i"]],
                _state["ignored_gts"][_state["i"]],
                _state["ignored_dets"][_state["i"]],
                _state["dontcares"][_state["i"]],
            ),
            (_state["metric"], _state["min_overlap"], 0.0, False, False),
        )
        (
            _state["tp"],
            _state["fp"],
            _state["fn"],
            _state["similarity"],
            _state["thresholds"],
        ) = _state["rets"]
        _state["thresholdss"] += _state["thresholds"].tolist()
    _state["thresholdss"] = np.array(_state["thresholdss"])
    _state["thresholds"] = get_thresholds(
        _state["thresholdss"], _state["total_num_valid_gt"]
    )
    _state["thresholds"] = np.array(_state["thresholds"])
    _state["pr"] = np.zeros([len(_state["thresholds"]), 4])
    _state["idx"] = 0


def _module_eval_class_728_part_1(_state):
    _module_eval_class_728_part_1_chunk_1(_state)
    _module_eval_class_728_part_1_chunk_2(_state)


def _module_eval_class_728_part_2_chunk_1(_state):
    for _state["j"], _state["num_part"] in enumerate(_state["split_parts"]):
        _state["gt_datas_part"] = np.concatenate(
            _state["gt_datas_list"][_state["idx"]: _state["idx"] + _state["num_part"]],
            0,
        )
        _state["dt_datas_part"] = np.concatenate(
            _state["dt_datas_list"][_state["idx"]: _state["idx"] + _state["num_part"]],
            0,
        )
        _state["dc_datas_part"] = np.concatenate(
            _state["dontcares"][_state["idx"]: _state["idx"] + _state["num_part"]], 0
        )
        _state["ignored_dets_part"] = np.concatenate(
            _state["ignored_dets"][_state["idx"]: _state["idx"] + _state["num_part"]],
            0,
        )
        _state["ignored_gts_part"] = np.concatenate(
            _state["ignored_gts"][_state["idx"]: _state["idx"] + _state["num_part"]], 0
        )
        fused_compute_statistics(
            (_state["parted_overlaps"][_state["j"]], _state["pr"]),
            (
                _state["total_gt_num"][
                    _state["idx"]: _state["idx"] + _state["num_part"]
                ],
                _state["total_dt_num"][
                    _state["idx"]: _state["idx"] + _state["num_part"]
                ],
                _state["total_dc_num"][
                    _state["idx"]: _state["idx"] + _state["num_part"]
                ],
            ),
            (
                _state["gt_datas_part"],
                _state["dt_datas_part"],
                _state["dc_datas_part"],
                _state["ignored_gts_part"],
                _state["ignored_dets_part"],
            ),
            (
                _state["metric"],
                _state["min_overlap"],
                _state["thresholds"],
                _state["compute_aos"],
            ),
        )
        _state["idx"] += _state["num_part"]


def _module_eval_class_728_part_2_chunk_2(_state):
    _state["n_sample_pts"] = 41
    _state["precision"] = np.zeros([_state["n_sample_pts"]])
    _state["recall"] = np.zeros([_state["n_sample_pts"]])
    _state["aos"] = np.zeros([_state["n_sample_pts"]])
    for _state["i"], _state["_"] in enumerate(_state["thresholds"]):
        _state["recall"][_state["i"]] = _state["pr"][_state["i"], 0] / (
            _state["pr"][_state["i"], 0] + _state["pr"][_state["i"], 2]
        )
        _state["precision"][_state["i"]] = _state["pr"][_state["i"], 0] / (
            _state["pr"][_state["i"], 0] + _state["pr"][_state["i"], 1]
        )
        if _state["compute_aos"]:
            _state["aos"][_state["i"]] = _state["pr"][_state["i"], 3] / (
                _state["pr"][_state["i"], 0] + _state["pr"][_state["i"], 1]
            )


def _module_eval_class_728_part_2(_state):
    _module_eval_class_728_part_2_chunk_1(_state)
    _module_eval_class_728_part_2_chunk_2(_state)


def _module_eval_class_728_part_3(_state):
    for _state["i"], _state["_"] in enumerate(_state["thresholds"]):
        _state["precision"][_state["i"]] = np.max(_state["precision"][_state["i"]:])
        _state["recall"][_state["i"]] = np.max(_state["recall"][_state["i"]:])
        if _state["compute_aos"]:
            _state["aos"][_state["i"]] = np.max(_state["aos"][_state["i"]:])
    _state["ret_dict"] = {
        "recall": _state["recall"],
        "precision": _state["precision"],
        "orientation": _state["aos"],
    }


def eval_class(*call_args, **call_kwargs):
    """Kitti eval. Only support 2d/bev/3d/aos eval for now.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        difficulty: int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlap: float, min overlap. official:
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]]
            format: [metric, class]. choose one from matrix above.
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    _state = locals()
    _module_eval_class_728_part_1(_state)
    _module_eval_class_728_part_2(_state)
    _module_eval_class_728_part_3(_state)
    return _state["ret_dict"]


def _collect_class_thresholds(overlaps, annotation_data, match_config):
    gt_datas, dt_datas, ignored_gts, ignored_dets, dontcares = annotation_data
    metric, min_overlap, total_num_valid_gt = match_config
    threshold_values = []
    for index, overlap in enumerate(overlaps):
        statistics = compute_statistics_jit(
            (
                overlap,
                gt_datas[index],
                dt_datas[index],
                ignored_gts[index],
                ignored_dets[index],
                dontcares[index],
            ),
            (metric, min_overlap, 0.0, False, False),
        )
        threshold_values.extend(statistics[4].tolist())
    return np.array(get_thresholds(np.array(threshold_values), total_num_valid_gt))


def _accumulate_class_statistics(
    parted_overlaps, sample_counts, annotation_data, config
):
    total_gt_num, total_dt_num, total_dc_num = sample_counts
    gt_datas, dt_datas, ignored_gts, ignored_dets, dontcares = annotation_data
    split_parts, metric, min_overlap, thresholds, compute_aos = config
    precision_recall = np.zeros([len(thresholds), 4])
    start = 0
    for part_index, part_size in enumerate(split_parts):
        end = start + part_size
        part_annotations = (
            np.concatenate(gt_datas[start:end], 0),
            np.concatenate(dt_datas[start:end], 0),
            np.concatenate(dontcares[start:end], 0),
            np.concatenate(ignored_gts[start:end], 0),
            np.concatenate(ignored_dets[start:end], 0),
        )
        fused_compute_statistics(
            (parted_overlaps[part_index], precision_recall),
            (total_gt_num[start:end], total_dt_num[start:end], total_dc_num[start:end]),
            part_annotations,
            (metric, min_overlap, thresholds, compute_aos),
        )
        start = end
    return precision_recall


def _write_class_curves(outputs, location, precision_recall, compute_aos):
    recall, precision, aos = outputs
    class_index, difficulty_index, overlap_index = location
    for sample_index, row in enumerate(precision_recall):
        true_positive, false_positive, false_negative = precision_recall[
            sample_index, :3
        ]
        recall[class_index, difficulty_index, overlap_index, sample_index] = (
            true_positive / (true_positive + false_negative)
        )
        precision[class_index, difficulty_index, overlap_index, sample_index] = (
            true_positive / (true_positive + false_positive)
        )
        if compute_aos:
            aos[class_index, difficulty_index, overlap_index, sample_index] = (
                row[3] / (true_positive + false_positive)
            )
    for sample_index, _ in enumerate(precision_recall):
        curve_slice = (
            class_index,
            difficulty_index,
            overlap_index,
            slice(sample_index, None),
        )
        precision[class_index, difficulty_index, overlap_index, sample_index] = np.max(
            precision[curve_slice], axis=-1
        )
        recall[class_index, difficulty_index, overlap_index, sample_index] = np.max(
            recall[curve_slice], axis=-1
        )
        if compute_aos:
            aos[class_index, difficulty_index, overlap_index, sample_index] = np.max(
                aos[curve_slice], axis=-1
            )


def _evaluate_class_difficulty(evaluation_data, outputs, location):
    (
        gt_annos,
        dt_annos,
        overlaps,
        parted_overlaps,
        sample_counts,
        split_parts,
        metric,
        min_overlaps,
        compute_aos,
    ) = evaluation_data
    class_index, current_class, difficulty_index, difficulty = location
    prepared = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
    annotation_data = prepared[:5]
    total_dc_num, total_num_valid_gt = prepared[5:]
    total_gt_num, total_dt_num = sample_counts
    for overlap_index, min_overlap in enumerate(min_overlaps[:, metric, class_index]):
        thresholds = _collect_class_thresholds(
            overlaps, annotation_data, (metric, min_overlap, total_num_valid_gt)
        )
        precision_recall = _accumulate_class_statistics(
            parted_overlaps,
            (total_gt_num, total_dt_num, total_dc_num),
            annotation_data,
            (split_parts, metric, min_overlap, thresholds, compute_aos),
        )
        _write_class_curves(
            outputs,
            (class_index, difficulty_index, overlap_index),
            precision_recall,
            compute_aos,
        )


def eval_class_v3(*call_args, **call_kwargs):
    """Evaluate KITTI precision, recall, and orientation curves."""
    bound_parameters = _bind_parameters(
        _EVAL_CLASS_V3_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        metric,
        min_overlaps,
        compute_aos,
        num_parts,
    ) = bound_parameters
    if len(gt_annos) != len(dt_annos):
        raise ValueError("ground-truth and detection annotation counts must match")
    split_parts = get_split_parts(len(gt_annos), num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = calculate_iou_partly(
        dt_annos, gt_annos, metric, num_parts
    )
    output_shape = (len(current_classes), len(difficultys), len(min_overlaps), 41)
    outputs = tuple(np.zeros(output_shape) for _ in range(3))
    evaluation_data = (
        gt_annos,
        dt_annos,
        overlaps,
        parted_overlaps,
        (total_gt_num, total_dt_num),
        split_parts,
        metric,
        min_overlaps,
        compute_aos,
    )
    for class_index, current_class in enumerate(current_classes):
        for difficulty_index, difficulty in enumerate(difficultys):
            _evaluate_class_difficulty(
                evaluation_data,
                outputs,
                (class_index, current_class, difficulty_index, difficulty),
            )
    recall, precision, aos = outputs
    return {"recall": recall, "precision": precision, "orientation": aos}


def do_eval(gt_annos, dt_annos, current_class, min_overlaps, compute_aos=False):

    m_ap_bbox = []
    m_ap_aos = []
    for i in range(3):  # i=difficulty
        ret = eval_class(
            gt_annos, dt_annos, current_class, i, 0, min_overlaps[0], compute_aos
        )
        m_ap_bbox.append(get_m_ap(ret["precision"]))
        if compute_aos:
            m_ap_aos.append(get_m_ap(ret["orientation"]))
    m_ap_bev = []
    for i in range(3):
        ret = eval_class(gt_annos, dt_annos, current_class, i, 1, min_overlaps[1])
        m_ap_bev.append(get_m_ap(ret["precision"]))
    m_ap_3d = []
    for i in range(3):
        ret = eval_class(gt_annos, dt_annos, current_class, i, 2, min_overlaps[2])
        m_ap_3d.append(get_m_ap(ret["precision"]))
    return m_ap_bbox, m_ap_bev, m_ap_3d, m_ap_aos


# The new 40 recall points evaluation
def get_m_ap_v2(prec):
    sums = 0
    for i in range(1, prec.shape[-1], 1):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def do_eval_v2(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _DO_EVAL_V2_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, difficultys = (
        bound_parameters
    )
    if difficultys is None:
        difficultys = [0, 1, 2]
    ret = eval_class_v3(
        gt_annos, dt_annos, current_classes, difficultys, 0, min_overlaps, compute_aos
    )
    m_ap_bbox = get_m_ap_v2(ret["precision"])
    m_ap_aos = None
    if compute_aos:
        m_ap_aos = get_m_ap_v2(ret["orientation"])
    ret = eval_class_v3(
        gt_annos, dt_annos, current_classes, difficultys, 1, min_overlaps
    )
    m_ap_bev = get_m_ap_v2(ret["precision"])
    ret = eval_class_v3(
        gt_annos, dt_annos, current_classes, difficultys, 2, min_overlaps
    )
    m_ap_3d = get_m_ap_v2(ret["precision"])
    return m_ap_bbox, m_ap_bev, m_ap_3d, m_ap_aos


def do_coco_style_eval(
    gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos
):
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    m_ap_bbox, m_ap_bev, m_ap_3d, m_ap_aos = do_eval_v2(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos
    )
    m_ap_bbox = m_ap_bbox.mean(-1)
    m_ap_bev = m_ap_bev.mean(-1)
    m_ap_3d = m_ap_3d.mean(-1)
    if m_ap_aos is not None:
        m_ap_aos = m_ap_aos.mean(-1)
    return m_ap_bbox, m_ap_bev, m_ap_3d, m_ap_aos


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    sstream.write(" ".join(map(str, (value, *arg))) + "\n")
    return sstream.getvalue()


_EVAL_CLASS_NAMES = {
    0: "Car", 1: "Pedestrian", 2: "Cyclist", 3: "Van",
    4: "Person_sitting", 5: "car", 6: "tractor", 7: "trailer",
}
_V1_OVERLAPS = (
    np.array([[0.7, 0.5, 0.5, 0.7, 0.5]] * 3),
    np.array([
        [0.7, 0.5, 0.5, 0.7, 0.5],
        [0.5, 0.25, 0.25, 0.5, 0.25],
        [0.5, 0.25, 0.25, 0.5, 0.25],
    ]),
)
_OFFICIAL_OVERLAPS = np.stack([
    np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7]] * 3),
    np.array([
        [0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
        [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
    ]),
])
_COCO_RANGES = {
    0: [0.5, 0.95, 10], 1: [0.25, 0.7, 10], 2: [0.25, 0.7, 10],
    3: [0.5, 0.95, 10], 4: [0.25, 0.7, 10], 5: [0.5, 0.95, 10],
    6: [0.5, 0.95, 10], 7: [0.5, 0.95, 10],
}


def _eval_class_indices(current_classes):
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    name_to_class = {name: index for index, name in _EVAL_CLASS_NAMES.items()}
    return [name_to_class.get(value) if isinstance(value, str) else value
            for value in current_classes]


def _compute_aos(dt_annos):
    for anno in dt_annos:
        if anno["alpha"].shape[0] != 0:
            return anno["alpha"][0] != -10
    return False


def get_official_eval_result_v1(gt_annos, dt_annos, current_class):
    name_to_class = {name: index for index, name in _EVAL_CLASS_NAMES.items()}
    if isinstance(current_class, str):
        current_class = name_to_class.get(current_class)
    result = ""
    compute_aos = _compute_aos(dt_annos)
    for m_ap in _V1_OVERLAPS:
        # mAP threshold matrix: [num_minoverlap, metric, class]
        m_a_pbbox, m_a_pbev, m_ap3d, m_a_paos = do_eval(
            gt_annos, dt_annos, current_class, m_ap[:, current_class], compute_aos
        )
        result += print_str(
            (
                f"{_EVAL_CLASS_NAMES.get(current_class)} "
                "AP@{:.2f}, {:.2f}, {:.2f}:".format(*m_ap[:, current_class])
            )
        )
        result += print_str(
            (
                f"bbox AP:{m_a_pbbox[0]:.2f}, "
                f"{m_a_pbbox[1]:.2f}, "
                f"{m_a_pbbox[2]:.2f}"
            )
        )
        result += print_str(
            (f"bev  AP:{m_a_pbev[0]:.2f}, " f"{m_a_pbev[1]:.2f}, " f"{m_a_pbev[2]:.2f}")
        )
        result += print_str(
            (f"3d   AP:{m_ap3d[0]:.2f}, " f"{m_ap3d[1]:.2f}, " f"{m_ap3d[2]:.2f}")
        )
        if compute_aos:
            result += print_str(
                (
                    f"aos  AP:{m_a_paos[0]:.2f}, "
                    f"{m_a_paos[1]:.2f}, "
                    f"{m_a_paos[2]:.2f}"
                )
            )

    return result


def official_eval_metrics(gt_annos, dt_annos, current_classes, difficultys=None):
    """Return the arrays and metadata used to format official KITTI results."""
    if difficultys is None:
        difficultys = [0, 1, 2]
    current_classes = _eval_class_indices(current_classes)
    min_overlaps = _OFFICIAL_OVERLAPS[:, :, current_classes]
    compute_aos = _compute_aos(dt_annos)
    metrics = do_eval_v2(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, difficultys
    )
    return {
        "classes": current_classes,
        "overlaps": min_overlaps,
        "compute_aos": compute_aos,
        "bbox": metrics[0],
        "bev": metrics[1],
        "3d": metrics[2],
        "aos": metrics[3],
    }


def get_official_eval_result(gt_annos, dt_annos, current_classes, difficultys=None):
    evaluation = official_eval_metrics(
        gt_annos, dt_annos, current_classes, difficultys
    )
    current_classes = evaluation["classes"]
    min_overlaps = evaluation["overlaps"]
    compute_aos = evaluation["compute_aos"]
    m_a_pbbox = evaluation["bbox"]
    m_a_pbev = evaluation["bev"]
    m_ap3d = evaluation["3d"]
    m_a_paos = evaluation["aos"]
    result = ""
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (
                    f"{_EVAL_CLASS_NAMES.get(curcls)} "
                    "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])
                )
            )
            result += print_str(
                (
                    f"bbox AP:{m_a_pbbox[j, 0, i]:.2f}, "
                    f"{m_a_pbbox[j, 1, i]:.2f}, "
                    f"{m_a_pbbox[j, 2, i]:.2f}"
                )
            )
            result += print_str(
                (
                    f"bev  AP:{m_a_pbev[j, 0, i]:.2f}, "
                    f"{m_a_pbev[j, 1, i]:.2f}, "
                    f"{m_a_pbev[j, 2, i]:.2f}"
                )
            )
            result += print_str(
                (
                    f"3d   AP:{m_ap3d[j, 0, i]:.2f}, "
                    f"{m_ap3d[j, 1, i]:.2f}, "
                    f"{m_ap3d[j, 2, i]:.2f}"
                )
            )
            if compute_aos:
                result += print_str(
                    (
                        f"aos  AP:{m_a_paos[j, 0, i]:.2f}, "
                        f"{m_a_paos[j, 1, i]:.2f}, "
                        f"{m_a_paos[j, 2, i]:.2f}"
                    )
                )

    return result


def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    current_classes = _eval_class_indices(current_classes)
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(_COCO_RANGES.get(curcls))[:, np.newaxis]
    result = ""
    compute_aos = _compute_aos(dt_annos)
    m_a_pbbox, m_a_pbev, m_ap3d, m_a_paos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos
    )
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(_COCO_RANGES.get(curcls))[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str(
            (
                f"{_EVAL_CLASS_NAMES.get(curcls)} "
                "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)
            )
        )
        result += print_str(
            (
                f"bbox AP:{m_a_pbbox[j, 0]:.2f}, "
                f"{m_a_pbbox[j, 1]:.2f}, "
                f"{m_a_pbbox[j, 2]:.2f}"
            )
        )
        result += print_str(
            (
                f"bev  AP:{m_a_pbev[j, 0]:.2f}, "
                f"{m_a_pbev[j, 1]:.2f}, "
                f"{m_a_pbev[j, 2]:.2f}"
            )
        )
        result += print_str(
            (
                f"3d   AP:{m_ap3d[j, 0]:.2f}, "
                f"{m_ap3d[j, 1]:.2f}, "
                f"{m_ap3d[j, 2]:.2f}"
            )
        )
        if compute_aos:
            result += print_str(
                (
                    f"aos  AP:{m_a_paos[j, 0]:.2f}, "
                    f"{m_a_paos[j, 1]:.2f}, "
                    f"{m_a_paos[j, 2]:.2f}"
                )
            )
    return result
