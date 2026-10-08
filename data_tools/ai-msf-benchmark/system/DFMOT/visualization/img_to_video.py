# -*-coding:utf-8-*
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# author: wangxy
import cv2

# Write your folder path here，example：/home/youname/data/img/
# Note that the last folder should have a /
IMG_ROOT = "../results/pointrcnn_Car_val/image/"
VIDEO_SAVE_PATH = "tracking_video.avi"
FRAME_RATE = 24
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video_writer = cv2.VideoWriter(VIDEO_SAVE_PATH, fourcc, FRAME_RATE, (1242, 375))

for i in range(
    297
):  # Here 297 is the number of frames in the dataset. You need to make the appropriate changes
    number = "%06d" % i
    frame = cv2.imread(IMG_ROOT + number + ".png")
    video_writer.write(frame)
video_writer.release()
