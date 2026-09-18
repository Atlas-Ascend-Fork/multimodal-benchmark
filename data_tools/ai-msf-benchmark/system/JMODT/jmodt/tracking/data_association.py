# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from ortools.linear_solver import pywraplp
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch

from jmodt.ops.iou3d.iou3d_utils import boxes_iou3d_gpu
from jmodt.utils import kitti_utils


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class OrtoolsSolveParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    det_boxes: object
    pred_boxes: object
    cls_score: object
    link_score: object
    new_score: object
    end_score: object
    w_app: object
    w_iou: object
    w_dis: object


class HungarianMatchParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    det_boxes: object
    pred_boxes: object
    det_scores: object
    link_scores: object
    w_app: object
    w_iou: object
    score_threshold: object


def boxes_dist_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        distance: (M, N)
    """
    m = len(boxes_a)
    n = len(boxes_b)
    boxes_a_corners = kitti_utils.boxes3d_to_corners3d_torch(boxes_a)
    boxes_b_corners = kitti_utils.boxes3d_to_corners3d_torch(boxes_b)
    center_distance = torch.linalg.norm(
        boxes_a[:, :3].unsqueeze(1).repeat(1, n, 1)
        - boxes_b[:, :3].unsqueeze(0).repeat(m, 1, 1),
        ord=2,
        dim=-1,
    )
    corner_distance, _ = torch.max(
        torch.linalg.norm(
            boxes_a_corners.view(m, 1, 8, 1, 3).repeat(1, n, 1, 8, 1)
            - boxes_b_corners.view(1, n, 1, 8, 3).repeat(m, 1, 8, 1, 1),
            ord=2,
            dim=-1,
        ).view(m, n, 64),
        dim=-1,
    )
    return 1.0 - center_distance / corner_distance


_ORTOOLS_SOLVE_PARAMETERS_SCHEMA = _ParameterSchema(
    OrtoolsSolveParameters,
    (
        "det_boxes",
        "pred_boxes",
        "cls_score",
        "link_score",
        "new_score",
        "end_score",
        "w_app",
        "w_iou",
        "w_dis",
    ),
    {},
    frozenset(()),
    9,
    None,
    None,
)
_HUNGARIAN_MATCH_PARAMETERS_SCHEMA = _ParameterSchema(
    HungarianMatchParameters,
    (
        "det_boxes",
        "pred_boxes",
        "det_scores",
        "link_scores",
        "w_app",
        "w_iou",
        "score_threshold",
    ),
    {"score_threshold": 0},
    frozenset(()),
    7,
    None,
    None,
)


def _assignment_variables(solver, score_count, link_matrix):
    y_det = {index: solver.BoolVar("y_det[%i]" % index) for index in range(score_count)}
    y_new = {index: solver.BoolVar("y_new[%i]" % index) for index in range(score_count)}
    y_end = {index: solver.BoolVar("y_end[%i]" % index) for index in range(score_count)}
    y_link, weighted_links = {}, []
    for pred_index in range(link_matrix.shape[0]):
        y_link[pred_index] = {}
        for det_index in range(link_matrix.shape[1]):
            variable = solver.BoolVar(f"y_link[{pred_index}, {det_index}]")
            y_link[pred_index][det_index] = variable
            weighted_links.append(variable * link_matrix[pred_index, det_index])
    return y_det, y_new, y_end, y_link, weighted_links


def _assignment_constraints(solver, variables, num_pred, num_det):
    y_det, y_new, y_end, y_link = variables
    for pred_index in range(num_pred):
        terms = [y_end[pred_index], -y_det[pred_index]]
        terms += [y_link[pred_index][index] for index in range(num_det)]
        solver.Add(solver.Sum(terms) == 0)
    for det_index in range(num_det):
        score_index = num_pred + det_index
        terms = [y_new[score_index], -y_det[score_index]]
        terms += [y_link[index][det_index] for index in range(num_pred)]
        solver.Add(solver.Sum(terms) == 0)


def _assignment_result(variables, score_count, num_pred, num_det):
    y_det, y_new, y_end, y_link = variables
    assign_det = torch.tensor([y_det[i].solution_value() for i in range(score_count)])
    assign_new = torch.tensor([y_new[i].solution_value() for i in range(score_count)])
    assign_link = torch.zeros(num_pred, num_det)
    for pred_index in range(num_pred):
        for det_index in range(num_det):
            assign_link[pred_index, det_index] = y_link[pred_index][det_index].solution_value()
    matched = torch.nonzero(assign_link, as_tuple=False).tolist()
    unmatched = torch.flatten(torch.nonzero(assign_new[num_pred:], as_tuple=False)).tolist()
    tentative = torch.flatten(
        torch.nonzero(assign_det[num_pred:] == 0, as_tuple=False)
    ).tolist()
    return matched, unmatched, tentative


def ortools_solve(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _ORTOOLS_SOLVE_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        det_boxes,
        pred_boxes,
        cls_score,
        link_score,
        new_score,
        end_score,
        w_app,
        w_iou,
        w_dis,
    ) = bound_parameters
    num_det = len(det_boxes)
    num_pred = len(pred_boxes)
    iou_matrix = boxes_iou3d_gpu(pred_boxes, det_boxes)
    dis_matrix = boxes_dist_gpu(pred_boxes, det_boxes)
    link_matrix = link_score * w_app + iou_matrix * w_iou + dis_matrix * w_dis
    link_matrix = link_matrix.cpu().numpy()
    solver = pywraplp.Solver(
        "SolveAssignmentProblemMIP", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )
    variable_data = _assignment_variables(solver, len(cls_score), link_matrix)
    y_det, y_new, y_end, y_link, w_link_y = variable_data
    w_det_y = [y_det.get(i) * cls_score[i] for (i, _) in enumerate(cls_score)]
    w_new_y = [y_new.get(i) * new_score[i] for (i, _) in enumerate(cls_score)]
    w_end_y = [y_end.get(i) * end_score[i] for (i, _) in enumerate(cls_score)]

    # Objective
    solver.Maximize(solver.Sum(w_det_y + w_new_y + w_end_y + w_link_y))

    _assignment_constraints(solver, (y_det, y_new, y_end, y_link), num_pred, num_det)
    solver.Solve()
    return _assignment_result(
        (y_det, y_new, y_end, y_link), len(cls_score), num_pred, num_det
    )


def hungarian_match(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _HUNGARIAN_MATCH_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    det_boxes, pred_boxes, det_scores, link_scores, w_app, w_iou, score_threshold = (
        bound_parameters
    )
    iou_matrix = boxes_iou3d_gpu(pred_boxes, det_boxes)
    link_matrix = link_scores * w_app + iou_matrix * w_iou
    link_matrix = link_matrix.cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(-link_matrix)

    unmatched_detections = []
    tentative_detections = []
    for d, _ in enumerate(det_scores):
        if d not in row_ind:
            if det_scores[d] > score_threshold:
                unmatched_detections.append(d)
            else:
                tentative_detections.append(d)

    matched = np.vstack((row_ind, col_ind)).T.tolist()

    return matched, unmatched_detections, tentative_detections
