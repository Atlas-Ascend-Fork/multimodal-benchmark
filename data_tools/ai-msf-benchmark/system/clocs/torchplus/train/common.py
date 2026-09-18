# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import datetime
import os
import shutil


def create_folder(prefix, add_time=True, add_str=None, delete=False):
    additional_str = ""
    if delete:
        if os.path.exists(prefix):
            shutil.rmtree(prefix)
        os.makedirs(prefix)
    folder = prefix
    if add_time:
        # additional_str has a form such as '170903_220351'
        additional_str += datetime.datetime.now(datetime.timezone.utc).strftime(
            "%y%m%d_%H%M%S"
        )
        if add_str is not None:
            folder += "/" + additional_str + "_" + add_str
        else:
            folder += "/" + additional_str
    if delete:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    os.makedirs(folder)
    return folder
