import cv2
import numpy as np


def black_level_correct_more(img, color_num=255):
    # E-BLC
    mask = np.zeros(img.shape, img.dtype)
    mask[:, :, 1] = color_num
    dst = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
    return dst


def black_level_correct_less(img, color_num=255):
    # I-BLC
    mask = np.zeros(img.shape, img.dtype)
    mask[:, :, 2] = color_num
    dst = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
    return dst
