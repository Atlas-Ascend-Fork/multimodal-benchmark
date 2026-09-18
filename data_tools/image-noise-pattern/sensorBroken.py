import logging
import cv2
import numpy as np
import random

logger = logging.getLogger(__name__)


def sensor_broken(img, spot=True, deviation=True, leak=True):
    des = img
    x = int(random.uniform(0, img.shape[0]))
    y = int(random.uniform(100, img.shape[1]) - 100)

    if spot:
        img_spot = np.zeros(img.shape, np.uint8)
        i = 10
        while i > 0:
            center = (int(random.uniform(0, img.shape[1])), int(random.uniform(0, img.shape[0])))
            cv2.circle(img_spot, center, int(random.uniform(1, 7)), (0, 0, 255), -1)
            i -= 1
        des = cv2.addWeighted(img, 1, img_spot, random.uniform(0.4, 0.7), 0)
    if deviation:
        img_deviation = np.zeros(img.shape, np.uint8)
        i = 50
        axis = int(random.uniform(10, 20))
        while i >= 0:
            img_deviation[:, y - i + axis, 0] = 255 - i * 5 - 5
            img_deviation[:, y - i + axis, 2] = 255 - i * 5 - 5
            i -= 1
        j = 50
        while j > 0:
            img_deviation[:, y + j + axis, 0] = 255 - j * 5 - 5
            img_deviation[:, y + j + axis, 2] = 255 - j * 5 - 5
            j -= 1

        des = cv2.addWeighted(des, 1, img_deviation, 0.6, 0)
    if leak:

        axis = int(random.uniform(10, 30))
        des[x - 1:x + 1, :, :] = 0
        i = 0
        while i < 40:
            des[x:int(x + 16 - (i / 10) ** 2) + int(random.uniform(0, 20 - i // 2)),
                y - i + axis, :] = 0
            i += 1
        j = 0
        while j < 40:
            des[x:int(x + 16 - (j / 10) ** 2) + int(random.uniform(0, 20 - j // 2)),
                y + j + axis, :] = 0
            j += 1
    else:
        logger.info('----')

        r = int(random.uniform(0, 255))
        g = int(random.uniform(0, 255))
        b = int(random.uniform(0, 255))
        # 给十字线填充颜色
        des[:, y:y + 1, 0] = b
        des[:, y:y + 1, 1] = g
        des[:, y:y + 1, 2] = r
    return des
