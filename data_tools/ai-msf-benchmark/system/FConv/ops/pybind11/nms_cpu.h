#ifndef NMS_CPU_H
#define NMS_CPU_H
#include <pybind11/pybind11.h>
// must include pybind11/stl.h if using containers in STL in arguments.
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <boost/geometry.hpp>
#include "array_utils.h"

namespace py = pybind11;
using namespace pybind11::literals;


template <typename DType, typename CornerReader>
boost::geometry::model::polygon<boost::geometry::model::point<DType, 2, boost::geometry::cs::cartesian>>
make_nms_polygon(const CornerReader &corners, int row, int a, int b, int c, int d, int axis) {
    namespace bg = boost::geometry;
    typedef bg::model::point<DType, 2, bg::cs::cartesian> Point;
    bg::model::polygon<Point> polygon;
    const int indices[] = {a, b, c, d, a};
    for (int index : indices) bg::append(polygon, Point(corners(row, index, 0), corners(row, index, axis)));
    return polygon;
}

template <typename DType, typename CornerReader>
DType rotated_overlap_2d(const CornerReader &corners, int first, int second) {
    namespace bg = boost::geometry;
    auto poly = make_nms_polygon<DType>(corners, first, 0, 1, 2, 3, 1);
    auto qpoly = make_nms_polygon<DType>(corners, second, 0, 1, 2, 3, 1);
    std::vector<decltype(poly)> intersection, polygon_union;
    bg::intersection(poly, qpoly, intersection);
    if (intersection.empty()) return 0;
    bg::union_(poly, qpoly, polygon_union);
    if (polygon_union.empty()) return 0;
    DType union_area = bg::area(polygon_union.front());
    return union_area > 0 ? bg::area(intersection.front()) / union_area : 0;
}

template <typename DType, typename CornerReader>
DType rotated_overlap_3d(const CornerReader &corners, int first, int second) {
    namespace bg = boost::geometry;
    auto poly = make_nms_polygon<DType>(corners, first, 6, 7, 4, 5, 2);
    auto qpoly = make_nms_polygon<DType>(corners, second, 6, 7, 4, 5, 2);
    std::vector<decltype(poly)> intersection;
    bg::intersection(poly, qpoly, intersection);
    if (intersection.empty()) return 0;
    DType top = std::min(corners(first, 0, 1), corners(second, 0, 1));
    DType bottom = std::max(corners(first, 4, 1), corners(second, 4, 1));
    DType intersection_volume = bg::area(intersection.front()) * std::max(DType(0), top - bottom);
    DType first_volume = bg::area(poly) * (corners(first, 0, 1) - corners(first, 4, 1));
    DType second_volume = bg::area(qpoly) * (corners(second, 0, 1) - corners(second, 4, 1));
    DType union_volume = first_volume + second_volume - intersection_volume;
    return union_volume > 0 ? intersection_volume / union_volume : 0;
}

template <typename DType>
std::vector<int> rotated_non_max_suppression_impl(
    py::array_t<DType> box_corners,
    py::array_t<int> order,
    py::array_t<DType> standup_iou,
    DType thresh,
    bool use_3d_overlap)
{
    auto ndets = box_corners.shape(0);
    auto corners = box_corners.template unchecked<3>();
    auto ordered = order.template unchecked<1>();
    auto standup = standup_iou.template unchecked<2>();
    auto suppressed = zeros<int>({ndets});
    auto suppressed_rw = suppressed.template mutable_unchecked<1>();
    std::vector<int> keep;
    for (int order_index = 0; order_index < ndets; ++order_index) {
        int first = ordered(order_index);
        if (suppressed_rw(first) == 1) continue;
        keep.push_back(first);
        for (int candidate_index = order_index + 1; candidate_index < ndets;
             ++candidate_index) {
            int second = ordered(candidate_index);
            if (suppressed_rw(second) == 1 || standup(first, second) <= 0.0) continue;
            DType overlap = use_3d_overlap
                ? rotated_overlap_3d<DType>(corners, first, second)
                : rotated_overlap_2d<DType>(corners, first, second);
            if (overlap >= thresh) suppressed_rw(second) = 1;
        }
    }
    return keep;
}

template <typename DType>
std::vector<int> non_max_suppression_cpu(
    py::array_t<DType> boxes,
    py::array_t<int> order,
    DType thresh,
    DType eps=0)
{
    auto ndets = boxes.shape(0);
    auto boxes_r = boxes.template unchecked<2>();
    auto order_r = order.template unchecked<1>();
    auto suppressed = zeros<int>({ndets});
    auto suppressed_rw = suppressed.template mutable_unchecked<1>();
    auto area = zeros<DType>({ndets});
    auto area_rw = area.template mutable_unchecked<1>();
    // get areas
    for(int i = 0; i < ndets; ++i){
        area_rw(i) = (boxes_r(i, 2) - boxes_r(i, 0) + eps) * (boxes_r(i, 3) - boxes_r(i, 1) + eps);
    }
    std::vector<int> keep;
    int i, j;
    DType xx1, xx2, w, h, inter, ovr;
    for(int _i = 0; _i < ndets; ++_i){
        i = order_r(_i);
        if(suppressed_rw(i) == 1)
            continue;
        keep.push_back(i);
        for(int _j = _i + 1; _j < ndets; ++_j){
            j = order_r(_j);
            if(suppressed_rw(j) == 1)
                continue;
            xx2 = std::min(boxes_r(i, 2), boxes_r(j, 2));
            xx1 = std::max(boxes_r(i, 0), boxes_r(j, 0));
            w = xx2 - xx1 + eps;
            if (w <= 0) continue;
            xx2 = std::min(boxes_r(i, 3), boxes_r(j, 3));
            xx1 = std::max(boxes_r(i, 1), boxes_r(j, 1));
            h = xx2 - xx1 + eps;
            if (h <= 0) continue;
            inter = w * h;
            DType union_area = area_rw(i) + area_rw(j) - inter;
            ovr = union_area > 0 ? inter / union_area : 0;
            if(ovr >= thresh)
                suppressed_rw(j) = 1;
        }
    }
    return keep;
}

template <typename DType>
std::vector<int> rotate_non_max_suppression_cpu(
    py::array_t<DType> box_corners,
    py::array_t<int> order,
    py::array_t<DType> standup_iou,
    DType thresh)
{
    return rotated_non_max_suppression_impl(
        box_corners, order, standup_iou, thresh, false);
}

template <typename DType>
std::vector<int> rotate_non_max_suppression_3d_cpu(
    py::array_t<DType> box_corners,
    py::array_t<int> order,
    py::array_t<DType> standup_iou,
    DType thresh)
 /*
  camera coordinate
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
  */
{
    return rotated_non_max_suppression_impl(
        box_corners, order, standup_iou, thresh, true);
}


#endif
