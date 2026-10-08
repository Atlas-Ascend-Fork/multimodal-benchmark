# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""
import os

import cv2
import numpy as np


#
#


# This function takes a 4D tensor in the form NxCxWxH and save it to images according to idxs
def save_tensor_to_image(t, idxs, save_to_path):
    if not (os.path.exists(save_to_path)):
        os.mkdir(save_to_path)

    for i in range(t.size(0)):
        im = t[i, :, :, :].detach().data.cpu().numpy()
        im = np.transpose(im, (1, 2, 0)).astype(np.uint16)
        imout = im

        cv2.imwrite(
            os.path.join(
                save_to_path, str(idxs[i].data.cpu().numpy()).zfill(10) + ".png"
            ),
            imout,
            [cv2.IMWRITE_PNG_COMPRESSION, 4],
        )
