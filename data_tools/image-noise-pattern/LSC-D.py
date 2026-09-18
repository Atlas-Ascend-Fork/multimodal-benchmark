import logging
import cv2
import numpy as np
import math
from RGB2RAW import RGB2Bayer_RG

logger = logging.getLogger(__name__)


def lens_shading_error(img, alpha=0.5):

    img = RGB2Bayer_RG(img)
    row, col = img.shape
    shading = np.ones(img.shape)
    r = ((max(row, col) - min(row, col)) // 2 * alpha)
    logger.info(r)
    size = math.sqrt((row // 2) ** 2 + (col // 2) ** 2)
    center = (row // 2, col // 2)
    a = 1 / (r - size)
    b = 0 - size * a
    for i in range(row):
        for j in range(col):
            d = math.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if d <= r:
                continue
            else:
                scale = a * d + b
                shading[i][j] = shading[i][j] * scale
            img[i][j] = round(img[i][j] * shading[i][j])
    dst = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)

    return dst
