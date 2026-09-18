import numpy as np
import cv2
import random


def color_correction_error(img):
    fImg = img.astype(np.float32)
    fImg = fImg / 255.0
    #
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    s = int(random.uniform(-30, 100))
    MAX_VALUE = 100
    hlsImg[:, :, 2] = (1.0 + s / float(MAX_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    lsImg[:, :, :] = lsImg[:, :, :] * 255
    lsImg = lsImg.astype(np.uint8)
    return lsImg
