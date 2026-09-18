import cv2
import random


def sensor_damage(img, h, w, num=25, sensor='ccd'):
    max_l = 4
    damage_grade = 3
    axes_l = random.randint(max_l // 2 + 1, max_l)
    axes_s = axes_l // 2
    axesSize = (axes_l, axes_s)
    ptCenter = (random.randint(axesSize[1] // 2, w - axesSize[1] // 2),
                random.randint(axesSize[0] // 2, h - axesSize[0] // 2))
    while num > 0:
        rotateAngle = 90
        startAngle = 0
        endAngle = 360
        point_color = (255, 255, 255)  # BGR
        thickness = -1
        lineType = 4
        cv2.ellipse(img, ptCenter, axesSize, rotateAngle, startAngle,
                    endAngle, point_color, thickness, lineType)
        if damage_grade > 2:
            if sensor == 'ccd':
                cv2.line(img, (ptCenter[0], 0), (ptCenter[0], h),
                         (255, 255, 255), damage_grade - 2, 8)
            else:
                cv2.line(img, (0, ptCenter[1]), (w, ptCenter[1]),
                         (0, 0, 0), damage_grade - 2, 8)
                cv2.line(img, (ptCenter[0], 0), (ptCenter[0], h),
                         (0, 0, 0), damage_grade - 2, 8)

        num -= 1
        new_axes_l = random.randint(max_l // 2 + 1, max_l)
        new_axes_s = new_axes_l // 2
        axesSize = (new_axes_l, new_axes_s)
        while True:
            new_x = random.randint(axesSize[1] // 2, w - axesSize[1] // 2)
            new_y = random.randint(axesSize[0] // 2, h - axesSize[0] // 2)
            if (new_x - new_axes_l // 2 > ptCenter[0] + axes_l // 2
                    or new_x + new_axes_l // 2 < ptCenter[0] - axes_l // 2
                    or new_y - new_axes_s // 2 > ptCenter[1] + axes_s // 2
                    or new_y + new_axes_s // 2 < ptCenter[1] - axes_s // 2):
                break
        ptCenter = (new_x, new_y)

    return img


if __name__ == '__main__':
    path = '../VOCdevkit/VOC2007/JPEGImages/000000.jpg'
    img = cv2.imread(path)
    cv2.imshow('ori', img)
    h, w, _ = img.shape
    new_img = sensor_damage(img, h, w, 50, 'cmos')
    cv2.imshow('test2', new_img)
    cv2.waitKey(0)
