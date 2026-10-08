import cv2
import random
from RGB2RAW import RGB2Bayer_RG


def ori_bad_pixel_error(img, num_precent=0.6):

    raw_img = RGB2Bayer_RG(img)
    height, width = raw_img.shape
    num = int(height * width * num_precent)
    pixel_position = random.sample(range(0, height * width), num)
    for pos in pixel_position:
        raw_img[(pos // width) - 1, pos % width] = 0
    dst = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2BGR)
    return dst
