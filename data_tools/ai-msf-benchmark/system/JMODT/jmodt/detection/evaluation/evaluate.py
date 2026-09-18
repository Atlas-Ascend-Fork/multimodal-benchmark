# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple

import fire
import numpy as np
from shared.kitti_labels import (
    filter_annotations_by_score,
    get_label_annotation,
    get_label_annotations,
)

from jmodt.detection.evaluation.eval import (
    get_official_eval_result,
    get_coco_eval_result,
)


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class EvaluateParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    label_path: object
    result_path: object
    image_idx_list: object
    current_class: object
    coco: object
    score_thresh: object


def get_label_anno(label_path):
    return get_label_annotation(label_path, include_metadata=False)


def get_label_annos(label_folder, image_ids=None):
    return get_label_annotations(label_folder, image_ids, include_metadata=False)


def filter_annos_low_score(image_annos, thresh):
    return filter_annotations_by_score(image_annos, thresh)


_EVALUATE_PARAMETERS_SCHEMA = _ParameterSchema(
    EvaluateParameters,
    (
        "label_path",
        "result_path",
        "image_idx_list",
        "current_class",
        "coco",
        "score_thresh",
    ),
    {"current_class": 0, "coco": False, "score_thresh": -1},
    frozenset(()),
    6,
    None,
    None,
)


def evaluate(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _EVALUATE_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    label_path, result_path, image_idx_list, current_class, coco, score_thresh = (
        bound_parameters
    )
    dt_annos = get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = filter_annos_low_score(dt_annos, score_thresh)
    gt_annos = get_label_annos(label_path, image_idx_list)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class)


if __name__ == "__main__":
    fire.Fire()
