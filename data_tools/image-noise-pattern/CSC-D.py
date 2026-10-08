import random

import cv2


def color_space_error(img, num_precent=0.6):
    y_cb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # cv2.merge([Y, Cr, Cb])
    Y, Cr, Cb = cv2.split(y_cb_img)
    height, width, _ = img.shape
    num = int(height * width * num_precent)
    pixel_position = random.sample(range(0, height * width), num)
    for pos in pixel_position:
        Y[(pos // width) - 1, pos % width] -= 16
        Cr[(pos // width) - 1, pos % width] -= 128
        Cb[(pos // width) - 1, pos % width] -= 128
    dst = cv2.merge([Y, Cr, Cb])
    output_img = cv2.cvtColor(dst, cv2.COLOR_YCrCb2BGR)
    return output_img
