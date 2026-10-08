import cv2
import numpy as np


def white_balance_error(img, precent=0.1):
    height, width, _ = img.shape
    imgB = img[:, :, 0]
    imgG = img[:, :, 1]
    imgR = img[:, :, 2]
    bAve = cv2.mean(imgB)[0] + precent * 100
    gAve = cv2.mean(imgG)[0] + precent * 100
    rAve = cv2.mean(imgR)[0]
    aveGray = (int)(bAve + gAve + rAve) / 3
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve

    imgB = np.floor((imgB * bCoef))
    imgG = np.floor((imgG * gCoef))
    imgR = np.floor((imgR * rCoef))

    dst = np.zeros(img.shape, img.dtype)
    dst[:, :, 0] = imgB
    dst[:, :, 1] = imgG
    dst[:, :, 2] = imgR

    for i in range(0, height):
        for j in range(0, width):
            imgb = imgB[i, j]
            imgg = imgG[i, j]
            imgr = imgR[i, j]
            if imgb > 255:
                imgb = 255
            if imgg > 255:
                imgg = 255
            if imgr > 255:
                imgr = 255
            dst[i, j] = (imgb, imgg, imgr)
    return dst
