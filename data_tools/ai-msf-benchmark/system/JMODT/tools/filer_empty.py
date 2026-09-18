# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import os


LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    data_root = "/home/niangao/PycharmProjects/fusion/JMODT/data/KITTI/tracking_object"
    length_value = len(
        os.listdir(
            "/home/niangao/PycharmProjects/fusion/JMODT/data/KITTI/tracking_object/training/image_2"
        )
    )
    content = ["%06d" % n for n in list(range(0, length_value))]
    res_seq = "{}/ImageSets/{}_filter.txt".format(data_root, "val")
    content_res = content.copy()
    for fn_idx in content:
        label_fn = os.path.join(
            data_root, "training", "label_2", "{}.txt".format(fn_idx)
        )
        with open(label_fn, "r") as f2:
            lb = f2.readline()
            if len(lb) == 0:
                content_res.remove(fn_idx)
    content_res = [c + "\n" for c in content_res]
    LOGGER.info("%s %s", len(content_res), len(content))
    with open(res_seq, "w") as f3:
        f3.writelines(content_res)
