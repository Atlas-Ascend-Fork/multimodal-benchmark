import numpy as np
import cv2
import random


def blur(img):
    gauss_blur = cv2.GaussianBlur(img, (3, 3), 0, 0)
    return gauss_blur


def sp_noise(img, precent=0.01):
    image = blur(img)
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - precent
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.randint(1, 10) / 10
            if rdn < precent:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(img, mean=0, var=0.001):
    image = blur(img)
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out
