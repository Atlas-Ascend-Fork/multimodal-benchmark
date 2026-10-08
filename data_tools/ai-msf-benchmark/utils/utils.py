# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""'
@Project: AI-MSF-benchmark
@Description: Please add Description
@Time:2022/9/6 15:11
@Author:NianGao

"""
import logging
import os

LOGGER = logging.getLogger(__name__)


def add_python_path(p):
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = p
    else:
        os.environ["PYTHONPATH"] += ":{}".format(p)


def remove_python_path(p):
    p = ":{}".format(p)
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"].replace(p, "")


def get_op(op):
    if op is None or op == "":
        return "clean"
    else:
        return op


# creat symlink instead of copy files
def symlink(input_path, output_path):
    if not os.path.exists(input_path):
        raise ValueError(input_path)

    if os.path.exists(output_path):
        os.remove(output_path)  # can delet soft link files
    if str(input_path).startswith("./"):
        input_path = os.path.abspath(input_path)
        # convert relateive to abslute
    LOGGER.info("%s %s %s", input_path, "-->", output_path)
    os.symlink(input_path, output_path)
