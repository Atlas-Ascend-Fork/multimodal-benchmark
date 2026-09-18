import logging
import numpy as np
import random
import least_object

logger = logging.getLogger(__name__)


def timing_error(img, img_name):
    des = np.zeros(img.shape, np.uint8)
    row, col, chs = img.shape
    least_object_height, _ = least_object.get_least_object_height_width(img_name)
    logger.info(least_object_height)
    errorNum = int(random.randint(0, least_object_height // 2))
    error = []
    gapCount = 0
    for i in range(errorNum):
        error.append(int(random.uniform(0, row)))
    error.sort()

    des[0:error[0], 0:col, 0:chs] = img[0:error[0], 0:col, 0:chs]
    for i in range(errorNum - 1):

        gap = int(random.uniform(10, 20))
        gapCount = gapCount + gap

        if error[i] + gapCount > row or error[i + 1] + gapCount > row:
            des[error[i]:row - gapCount, 0:col, 0:chs] = img[error[i] + gapCount:row, 0:col, 0:chs]
            break

        else:
            des[error[i]:error[i + 1], 0:col, 0:chs] = \
                img[error[i] + gapCount:error[i + 1] + gapCount, 0:col, 0:chs]
    des[error[errorNum - 1]:row - gapCount, 0:col, 0:chs] = \
        img[error[errorNum - 1] + gapCount:row, 0:col, 0:chs]
    return des


def sensor_timing_error(img, img_name, row_precent=1):
    row, col, chs = img.shape
    least_object_height = least_object.get_least_object_height(img_name)
    errorNum = int((least_object_height // 2) * row_precent)
    error_start = random.randint(0, row - errorNum - 1)
    img[error_start:error_start + errorNum, :] = (0, 0, 0)
    return img
