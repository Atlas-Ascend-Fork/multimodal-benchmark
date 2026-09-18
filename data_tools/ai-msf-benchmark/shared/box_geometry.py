# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Polygon and 3D bounding-box geometry helpers."""

from scipy.spatial import ConvexHull
import numpy as np


def camera_box_corners(center, dimensions, yaw, bottom_center=True):
    """Return KITTI camera-space corners for ``(length, width, height)``."""
    length_value, width, height = dimensions
    x_signs = np.array((1, 1, -1, -1, 1, 1, -1, -1))
    z_signs = np.array((1, -1, -1, 1, 1, -1, -1, 1))
    x_values = x_signs * length_value / 2
    z_values = z_signs * width / 2
    if bottom_center:
        y_values = np.array((0, 0, 0, 0, -height, -height, -height, -height))
    else:
        y_values = np.array((1, 1, 1, 1, -1, -1, -1, -1)) * height / 2
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))
    corners = rotation.dot(np.vstack((x_values, y_values, z_values)))
    return (corners + np.asarray(center).reshape(3, 1)).T


def draw_projected_box(image, vertices, cv2_module, color, thickness):
    """Draw the twelve edges of a projected cuboid."""
    points = vertices.astype(np.int32)
    edge_pairs = []
    for index in range(4):
        edge_pairs.extend(((index, (index + 1) % 4),
                           (index + 4, (index + 1) % 4 + 4),
                           (index, index + 4)))
    line_type = getattr(cv2_module, "LINE_AA", getattr(cv2_module, "CV_AA", 16))
    for start, end in edge_pairs:
        cv2_module.line(
            image, tuple(points[start, :2]), tuple(points[end, :2]),
            color, thickness, line_type,
        )
    return image


def boxes_iou3d_gpu(boxes_a, boxes_b, cuda_module, box_utils, cast_bev=False):
    """Compute pairwise 3D IoU using a caller-provided CUDA overlap kernel."""
    import torch

    boxes_a_bev = box_utils.boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = box_utils.boxes3d_to_bev_torch(boxes_b)
    if cast_bev:
        boxes_a_bev, boxes_b_bev = boxes_a_bev.float(), boxes_b_bev.float()
    overlaps_bev = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0]))
    ).zero_()
    cuda_module.boxes_overlap_bev_gpu(
        boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev
    )
    lower = torch.max(
        (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1),
        (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1),
    )
    upper = torch.min(boxes_a[:, 1].view(-1, 1), boxes_b[:, 1].view(1, -1))
    overlap_volume = overlaps_bev * torch.clamp(upper - lower, min=0)
    volume_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    volume_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    return overlap_volume / torch.clamp(volume_a + volume_b - overlap_volume, min=1e-7)


def gpu_nms(boxes, scores, threshold, cuda_kernel, torch_module):
    """Run a CUDA NMS kernel and restore indices in descending score order."""
    order = scores.sort(0, descending=True)[1]
    sorted_boxes = boxes[order].contiguous()
    keep = torch_module.LongTensor(sorted_boxes.size(0))
    output_count = cuda_kernel(sorted_boxes, keep, threshold)
    selected = keep[:output_count].to(order.device)
    return order[selected].contiguous()


def polygon_clip(subject_polygon, clip_polygon):
    """Clip a polygon by a counter-clockwise convex polygon."""

    def inside(point):
        return (
            (clip_end[0] - clip_start[0]) * (point[1] - clip_start[1])
            > (clip_end[1] - clip_start[1]) * (point[0] - clip_start[0])
        )

    def intersection():
        clip_delta = [
            clip_start[0] - clip_end[0],
            clip_start[1] - clip_end[1],
        ]
        subject_delta = [start[0] - end[0], start[1] - end[1]]
        clip_cross = clip_start[0] * clip_end[1] - clip_start[1] * clip_end[0]
        subject_cross = start[0] * end[1] - start[1] * end[0]
        scale = 1.0 / (
            clip_delta[0] * subject_delta[1]
            - clip_delta[1] * subject_delta[0]
        )
        return [
            (clip_cross * subject_delta[0] - subject_cross * clip_delta[0]) * scale,
            (clip_cross * subject_delta[1] - subject_cross * clip_delta[1]) * scale,
        ]

    output = subject_polygon
    clip_start = clip_polygon[-1]
    for clip_end in clip_polygon:
        input_points = output
        output = []
        start = input_points[-1]
        for end in input_points:
            end_inside = inside(end)
            start_inside = inside(start)
            if end_inside and not start_inside:
                output.append(intersection())
            if end_inside:
                output.append(end)
            if not end_inside and start_inside:
                output.append(intersection())
            start = end
        clip_start = clip_end
        if not output:
            return None
    return output


def poly_area(x_coordinates, y_coordinates):
    """Return the area enclosed by polygon coordinates."""
    return 0.5 * np.abs(
        np.dot(x_coordinates, np.roll(y_coordinates, 1))
        - np.dot(y_coordinates, np.roll(x_coordinates, 1))
    )


def convex_hull_intersection(first, second):
    """Return the intersection polygon and its area."""
    intersection = polygon_clip(first, second)
    if intersection is None:
        return None, 0.0
    hull = ConvexHull(intersection)
    return intersection, hull.volume


def box3d_volume(corners):
    """Return the volume of an eight-corner 3D box."""
    first_edge = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    second_edge = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    height = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return first_edge * second_edge * height


def box3d_iou(corners1, corners2):
    """Return 3D and bird's-eye-view IoU for two ground-parallel boxes."""
    rectangle1 = [(corners1[index, 0], corners1[index, 2]) for index in range(3, -1, -1)]
    rectangle2 = [(corners2[index, 0], corners2[index, 2]) for index in range(3, -1, -1)]
    area1 = poly_area(np.array(rectangle1)[:, 0], np.array(rectangle1)[:, 1])
    area2 = poly_area(np.array(rectangle2)[:, 0], np.array(rectangle2)[:, 1])
    _, intersection_area = convex_hull_intersection(rectangle1, rectangle2)
    iou_2d = intersection_area / (area1 + area2 - intersection_area)
    maximum_y = min(corners1[0, 1], corners2[0, 1])
    minimum_y = max(corners1[4, 1], corners2[4, 1])
    intersection_volume = intersection_area * max(0.0, maximum_y - minimum_y)
    volume1 = box3d_volume(corners1)
    volume2 = box3d_volume(corners2)
    iou_3d = intersection_volume / (volume1 + volume2 - intersection_volume)
    return iou_3d, iou_2d
