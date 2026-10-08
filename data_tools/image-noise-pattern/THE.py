import math

import numpy as np
import cv2
import random


def line_damage_error(img, num_precent=0.1):
    YUVImg = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, Cr, Cb = cv2.split(YUVImg)
    row, col, chs = YUVImg.shape
    damage = np.ones(img.shape[0:2], dtype=np.uint8)
    dataChs = random.randint(0, int(math.pow(2, 7)))  # 224#random.randint(0, 255)
    for i in range(int(row * num_precent)):
        for j in range(int(row * num_precent)):
            damage[i][j] = dataChs
    y = cv2.bitwise_and(y, damage)
    Cr = cv2.bitwise_and(Cr, damage)
    Cb = cv2.bitwise_and(Cb, damage)
    dst = cv2.merge([y, Cr, Cb])

    return dst


def replace_str(int_str, num):
    position = random.sample(range(0, 8), num)
    new_int_ls = ['0'] * 8
    int_str = list(int_str)
    for i in range(len(int_str)):
        if (i in position) or (int_str[i] == 0):
            continue
        else:
            new_int_ls[i] = int_str[i]
    return ''.join(new_int_ls)


def line_damage_error2(img, num_precent=0.2):
    YUVImg = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(YUVImg)
    row, col, chs = YUVImg.shape
    for i in random.sample(range(row), int(row * num_precent)):
        for j in random.sample(range(col), int(col * num_precent)):
            Y[i][j] = int(replace_str(bin(Y[i][j])[2:], 1), 2)
            Cr[i][j] = int(replace_str(bin(Cr[i][j])[2:], 1), 2)
            Cb[i][j] = int(replace_str(bin(Cb[i][j])[2:], 1), 2)
    dst = cv2.merge([Y, Cr, Cb])
    return cv2.cvtColor(dst, cv2.COLOR_YCrCb2BGR)
