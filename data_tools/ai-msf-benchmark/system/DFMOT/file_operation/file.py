# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Author: wangxy

import os


def load_list_from_folder(detections_files, detections_root):
    """

    :param detections_files: Examples: File names in list form：['0005','0006']
    :param detections_root:
    :return:  filelist：Relative path of each file in the detections_files list
              num_efile：Number of files in the detections_files list
    """

    # SORT detections_files  small to large

    detections_files.sort(key=lambda x: int(x.split(".")[0]))

    filelist = list()
    for detections_file in detections_files:
        position = detections_root + "/" + detections_file
        filelist.append(position)
    num_efile = len(filelist)
    return filelist, num_efile


def mkdir_if_inexistence(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)


def fileparts(input_path, warning=True, debug=True):
    """

    :param input_path:

    :return: filename_ ：‘0005.txt’
             filename  ：'0005'
             ext：'measure'
    """
    good_path = os.path.normpath(input_path)
    if len(good_path) == 0:
        return ("", "", "")
    if good_path[-1] == "/":
        if len(good_path) > 1:
            return (good_path[:-1], "", "")  # ignore the final '/'
        else:
            return (good_path, "", "")  # ignore the final '/'
    _ = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]
    filename_ = filename + ".txt"
    ext = os.path.splitext(good_path)[1]
    return filename_, filename, ext
