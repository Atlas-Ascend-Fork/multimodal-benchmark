# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import fire

from tools.kitti_object_eval_python.eval import (
    get_official_eval_result,
    get_coco_eval_result,
)
import tools.kitti_object_eval_python.kitti_common as kitti


from shared.parameters import bind_named as _bind_named


def _read_imageset_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(*call_args, **call_kwargs):
    bound_parameters = _bind_named(
        call_args, call_kwargs,
        ("label_path", "result_path", "label_split_file", "current_class", "coco", "score_thresh"),
        {"current_class": 0, "coco": False, "score_thresh": -1},
    )
    label_path, result_path, label_split_file, current_class, coco, score_thresh = (
        bound_parameters
    )
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class)


if __name__ == "__main__":
    fire.Fire()
