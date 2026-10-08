# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

LOGGER = logging.getLogger(__name__)
path_config = {
    "project_dir": "/home/niangao/PycharmProjects/fusion",
    "corruption_base_public_dir": "/home/niangao/disk3/kitti_corruption",
    "dataset": "kitti",
}

object_config = {
    "corruption": {
        "img": None,  # black noise_gauss
        "lidar": None,
        "calib": None,
    },
    "suffix": "object/training",
}

tracking_config = {
    "corruption": {
        "img": None,
        "lidar": None,
        "calib": None,
    },
    "suffix": "tracking/training",
}

depth_config = {
    "corruption": {
        "img": None,
        "lidar": None,
        "calib": None,
    },
    "suffix": "depth/val_selection_cropped",
    "raw_data": "depth/raw_data/",
}

task_system_map = {
    "object": ["cloc", "epnet", "fconv", "fconv_enhance"],  # "second"
    "tracking": ["jmodt", "dfmot"],
    "depth": ["twise", "mda"],
}


def get_config(name):
    import os

    if name in task_system_map["object"]:
        base_config = object_config.copy()
    elif name in task_system_map["tracking"]:
        base_config = tracking_config.copy()
    elif name in task_system_map["depth"]:
        base_config = depth_config.copy()
    else:
        raise ValueError(name)
    base_config.update(path_config)
    base_config["name"] = name
    # kitti format corruption data dir, use to soft link to systems
    base_config["work_dir"] = os.path.join(
        base_config["project_dir"], "_workdir", base_config["dataset"] + "4" + name
    )
    # corruption data
    base_config["public_dir"] = os.path.join(
        base_config["corruption_base_public_dir"], base_config["suffix"]
    )
    LOGGER.info("%s", base_config)
    return base_config
