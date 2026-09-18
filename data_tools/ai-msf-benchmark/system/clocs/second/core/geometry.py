# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numba
import numpy as np


@numba.njit
def is_line_segment_intersection_jit(lines1, lines2):
    """check if line segments1 and line segments2 have cross point

    Args:
        lines1 (float, [N, 2, 2]): [description]
        lines2 (float, [M, 2, 2]): [description]

    Returns:
        [type]: [description]
    """

    # Return true if line segments AB and CD intersect
    n = lines1.shape[0]
    m = lines2.shape[0]
    ret = np.zeros((n, m), dtype=np.bool_)
    for i in range(n):
        for j in range(m):
            a = lines1[i, 0]
            b = lines1[i, 1]
            c = lines2[j, 0]
            d = lines2[j, 1]
            acd = (d[1] - a[1]) * (c[0] - a[0]) > (c[1] - a[1]) * (d[0] - a[0])
            bcd = (d[1] - b[1]) * (c[0] - b[0]) > (c[1] - b[1]) * (d[0] - b[0])
            abc = (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
            abd = (d[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (d[0] - a[0])
            if acd != bcd and abc != abd:
                ret[i, j] = True
    return ret


@numba.njit
def _counter_clockwise(origin, first, second):
    first_x = first[0] - origin[0]
    first_y = first[1] - origin[1]
    second_x = second[0] - origin[0]
    second_y = second[1] - origin[1]
    return second_y * first_x > first_y * second_x


@numba.njit
def _determinant(point1, point2):
    return point1[0] * point2[1] - point2[0] * point1[1]


@numba.njit
def line_segment_intersection(line1, line2, intersection):
    a = line1[0]
    b = line1[1]
    c = line2[0]
    d = line2[1]
    segments_cross = _counter_clockwise(a, c, d) != _counter_clockwise(b, c, d)
    segments_cross &= _counter_clockwise(a, b, c) != _counter_clockwise(a, b, d)
    if segments_cross:
        direction1 = b - a
        direction2 = d - c
        denominator = direction1[1] * direction2[0] - direction1[0] * direction2[1]
        line1_det = _determinant(a, b)
        line2_det = _determinant(c, d)
        intersection[0] = (
            line1_det * direction2[0] - direction1[0] * line2_det
        ) / denominator
        intersection[1] = (
            line1_det * direction2[1] - direction1[1] * line2_det
        ) / denominator
        return True
    return False


def _ccw(a, b, c):
    return (c[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0]) > (
        b[..., 1] - a[..., 1]
    ) * (c[..., 0] - a[..., 0])


def is_line_segment_cross(lines1, lines2):
    # 10x slower than jit version with 1000-1000 random lines input.
    a = lines1[:, 0, :][:, np.newaxis, :]
    b = lines1[:, 1, :][:, np.newaxis, :]
    c = lines2[:, 0, :][np.newaxis, :, :]
    d = lines2[:, 1, :][np.newaxis, :, :]
    return np.logical_and(
        _ccw(a, c, d) != _ccw(b, c, d), _ccw(a, b, c) != _ccw(a, b, d)
    )


def surface_equ_3d(polygon_surfaces):
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    d = np.einsum("aij, aij->ai", normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


def points_in_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    _ = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    return _points_in_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


@numba.njit
def _point_in_convex_polygon_3d(point, normal_vec, d, surface_count):
    for surface_index in range(normal_vec.shape[0]):
        if surface_index > surface_count:
            break
        sign = (
            point[0] * normal_vec[surface_index, 0]
            + point[1] * normal_vec[surface_index, 1]
            + point[2] * normal_vec[surface_index, 2]
            + d[surface_index]
        )
        if sign >= 0:
            return False
    return True


@numba.njit
def _points_in_convex_polygon_3d_jit(
    points, polygon_surfaces, normal_vec, d, num_surfaces
):
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    for i in range(num_points):
        for j in range(num_polygons):
            ret[i, j] = _point_in_convex_polygon_3d(
                points[i], normal_vec[j], d[j], num_surfaces[j]
            )
    return ret


@numba.njit
def _point_in_convex_polygon_2d(point, polygon, edge_vectors):
    for k in range(polygon.shape[0]):
        cross = edge_vectors[k, 1] * (polygon[k, 0] - point[0])
        cross -= edge_vectors[k, 0] * (polygon[k, 1] - point[1])
        if cross >= 0:
            return False
    return True


@numba.jit
def points_in_convex_polygon_jit(points, polygon, clockwise=True):
    """check points is in 2d convex polygons. True when point in polygon
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = (
            polygon
            - polygon[
                :,
                [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)),
                :,
            ]
        )
    else:
        vec1 = (
            polygon[
                :,
                [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)),
                :,
            ]
            - polygon
        )
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    for i in range(num_points):
        for j in range(num_polygons):
            ret[i, j] = _point_in_convex_polygon_2d(
                points[i], polygon[j], vec1[j]
            )
    return ret


def points_in_convex_polygon(points, polygon, clockwise=True):
    """check points is in convex polygons. may run 2x faster when write in
    cython(don't need to calculate all cross-product between edge and point)
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_lines = polygon.shape[1]
    polygon_next = polygon[:, [num_lines - 1] + list(range(num_lines - 1)), :]
    if clockwise:
        vec1 = (polygon - polygon_next)[np.newaxis, ...]
    else:
        vec1 = (polygon_next - polygon)[np.newaxis, ...]
    vec2 = polygon[np.newaxis, ...] - points[:, np.newaxis, np.newaxis, :]
    cross = np.cross(vec1, vec2)
    return np.all(cross > 0, axis=2)
