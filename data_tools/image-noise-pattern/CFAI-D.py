import random


def cfa_interpolation_error(img, num_precent=0.6):
    height, width, _ = img.shape
    num = int(height * width * num_precent)
    pixel_position = random.sample(range(0, height * width), num)
    for pos in pixel_position:
        B, G, R = img[pos // width, pos % width]
        img[pos // width, pos % width] = int(0.299 * R + 0.587 * G + 0.114 * B)
    return img
