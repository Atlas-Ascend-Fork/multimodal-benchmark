# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import os

from natsort import natsorted
from skimage import io
from tqdm import tqdm
import cv2
import numpy as np



def assert_tracking(sub_dir_path):
    sub_sub_dirs = natsorted(os.listdir(sub_dir_path))
    num = 0
    for sub_sub_dir in tqdm(sub_sub_dirs):
        d = os.path.join(sub_dir_path, sub_sub_dir)
        fns = natsorted(os.listdir(d))
        num += len(fns)
        for fn in fns:
            f = os.path.join(d, fn)
            try:
                img = cv2.imread(f)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except (OSError, TypeError, cv2.error):
                os.remove(f)
                LOGGER.info("%s", f)
    LOGGER.info("%s %s %s", sub_dir_path.split("/")[-1], "len", num)


def assert_tacking_all(dir_tracking):
    skip_arr = []
    for directory in [dir_tracking]:
        sub_dirs = natsorted(os.listdir(directory))
        LOGGER.info("%s", len(sub_dirs))
        for sub_dir in sub_dirs:
            LOGGER.info("%s", sub_dir)
            if sub_dir in skip_arr:
                continue
            if "_object" in sub_dir:
                continue
            sub_dir_path = os.path.join(directory, sub_dir)
            assert_tracking(sub_dir_path)


def assert_object_all(dir_object):
    skip_arr = []
    for directory in tqdm([dir_object]):
        sub_dirs = natsorted(os.listdir(directory))
        for sub_dir in tqdm(sub_dirs):
            if sub_dir in skip_arr:
                continue
            assert_object(directory, sub_dir)


def assert_object(directory, sub_dir):
    d = os.path.join(directory, sub_dir)
    fns = natsorted(os.listdir(d))
    LOGGER.info("%s %s %s", sub_dir, "len", len(fns))
    for fn in tqdm(fns):
        f = os.path.join(d, fn)
        try:
            img = cv2.imread(f)
            _ = io.imread(f).shape[:2]
            _ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (OSError, TypeError, ValueError, cv2.error):
            os.remove(f)
            LOGGER.info("%s", f)


def _check_depth_directory(directory):
    filenames = natsorted(os.listdir(directory))
    LOGGER.info("%s", len(filenames))
    for filename in filenames:
        path = os.path.join(directory, filename)
        try:
            image = cv2.imread(path)
            _ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except (OSError, TypeError, cv2.error):
            LOGGER.info("%s", path)


def assert_depth_all(dir_depth):
    skip_arr = []
    for directory in tqdm([dir_depth]):
        LOGGER.info("%s", directory)
        sub_dirs = natsorted(os.listdir(directory))
        for sub_dir in tqdm(sub_dirs):
            LOGGER.info("%s", sub_dir)
            if sub_dir in skip_arr:
                continue
            if "_object" in sub_dir:
                continue
            if ".zip" in sub_dir:
                continue
            _check_depth_directory(os.path.join(directory, sub_dir))


LOGGER = logging.getLogger(__name__)


def _is_nonempty_lidar(path):
    try:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    except (OSError, ValueError):
        return False
    return len(points) > 0


def _invalid_lidar_count(directory):
    count = 0
    for subdirectory in tqdm(natsorted(os.listdir(directory))):
        path = os.path.join(directory, subdirectory)
        filenames = natsorted(os.listdir(path))
        count += len(filenames)
        for filename in filenames:
            lidar_path = os.path.join(path, filename)
            if not _is_nonempty_lidar(lidar_path):
                LOGGER.info("%s", lidar_path)
    return count


def assert_tracking_lidar_all(directory):
    skip_arr = []
    sub_dirs = natsorted(os.listdir(directory))
    LOGGER.info("%s", len(sub_dirs))
    for sub_dir in sub_dirs:
        LOGGER.info("%s", sub_dir)
        if sub_dir in skip_arr:
            continue
        if "_object" in sub_dir:
            continue
        sub_dir_path = os.path.join(directory, sub_dir)
        count = _invalid_lidar_count(sub_dir_path)
        LOGGER.info("%s %s %s", sub_dir_path.split("/")[-1], "len", count)


# Check if the generated data can be read properly
if __name__ == "__main__":
    dir_fconv = "kitti_corruption/training/noise/lidar_fov/"
    assert_object(dir_fconv, "loss75")
