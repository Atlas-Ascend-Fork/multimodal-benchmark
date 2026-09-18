import numpy as np
import cv2
import random
import least_object


def memory_corruption(img):
    b, g, r = cv2.split(img)
    row, col, chs = img.shape
    damageB = np.random.randint(0, 256, img.shape[0:2], dtype=np.uint8)
    damageG = np.random.randint(0, 256, img.shape[0:2], dtype=np.uint8)
    damageR = np.random.randint(0, 256, img.shape[0:2], dtype=np.uint8)

    tB, damageB = cv2.threshold(damageB, 32, 255, cv2.THRESH_BINARY)
    tG, damageG = cv2.threshold(damageG, 32, 255, cv2.THRESH_BINARY)
    tR, damageR = cv2.threshold(damageR, 32, 255, cv2.THRESH_BINARY)
    b = cv2.bitwise_and(b, damageB)
    g = cv2.bitwise_and(g, damageG)
    r = cv2.bitwise_and(r, damageR)

    dst = cv2.merge([b, g, r])
    return dst


def memory_damage(img, img_name, num_precent=1):
    B, G, R = cv2.split(img)
    row, col, chs = img.shape
    least_object_height, least_object_width = least_object.get_least_object_height_width(img_name)
    damage_row = int((least_object_height // 2) * num_precent)
    damage_col = int((least_object_width // 2) * num_precent)
    start_row = random.randint(0, row - damage_row - 1)
    start_col = random.randint(0, col - damage_col - 1)
    img[start_row:start_row + damage_row, start_col:start_col + damage_col] = [0, 0, 0]
    return img
