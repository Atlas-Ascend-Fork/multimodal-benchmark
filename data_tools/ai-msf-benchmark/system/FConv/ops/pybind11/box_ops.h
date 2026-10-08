#ifndef BOX_OPS_H
#define BOX_OPS_H

#include <math.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <boost/geometry.hpp>
#include <pybind11/numpy.h>
#include <utility>
#include "array_utils.h"
namespace py = pybind11;
using namespace pybind11::literals;

template <typename DType, typename CornerReader>
boost::geometry::model::polygon<boost::geometry::model::point<DType, 2, boost::geometry::cs::cartesian>>
make_corner_polygon(const CornerReader &corners, int row, int a, int b, int c, int d, int axis) {
  namespace bg = boost::geometry;
  typedef bg::model::point<DType, 2, bg::cs::cartesian> Point;
  bg::model::polygon<Point> polygon;
  const int indices[] = {a, b, c, d, a};
  for (int index : indices) bg::append(polygon, Point(corners(row, index, 0), corners(row, index, axis)));
  return polygon;
}

template <typename DType, typename CornerReader>
std::pair<DType, DType> calculate_3d_overlaps(
    const boost::geometry::model::polygon<
        boost::geometry::model::point<DType, 2, boost::geometry::cs::cartesian>> &poly,
    const boost::geometry::model::polygon<
        boost::geometry::model::point<DType, 2, boost::geometry::cs::cartesian>> &qpoly,
    const CornerReader &boxes, int box_index, const CornerReader &query_boxes, int query_index,
    DType intersection_area) {
  namespace bg = boost::geometry;
  DType ymax = std::min(boxes(box_index, 0, 1), query_boxes(query_index, 0, 1));
  DType ymin = std::max(boxes(box_index, 4, 1), query_boxes(query_index, 4, 1));
  DType height = boxes(box_index, 0, 1) - boxes(box_index, 4, 1);
  DType query_height = query_boxes(query_index, 0, 1) - query_boxes(query_index, 4, 1);
  DType intersection_volume = intersection_area * std::max(DType(0), ymax - ymin);
  DType volume = std::max(DType(0), bg::area(poly) * height);
  DType query_volume = std::max(DType(0), bg::area(qpoly) * query_height);
  DType union_area = bg::area(poly) + bg::area(qpoly) - intersection_area;
  DType union_volume = volume + query_volume - intersection_volume;
  return {union_area > 0 ? intersection_area / union_area : 0,
          union_volume > 0 ? intersection_volume / union_volume : 0};
}
template <typename DType>
py::array_t<DType> rbbox_iou_impl(
    py::array_t<DType> box_corners, py::array_t<DType> qbox_corners,
    py::array_t<DType> standup_iou, DType standup_thresh, bool use_3d) {
  namespace bg = boost::geometry;
  typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
  typedef bg::model::polygon<point_t> polygon_t;
  polygon_t poly, qpoly; std::vector<polygon_t> poly_inter, poly_union;
  auto box_corners_r = box_corners.template unchecked<3>();
  auto qbox_corners_r = qbox_corners.template unchecked<3>();
  auto standup_iou_r = standup_iou.template unchecked<2>();
  py::array_t<DType> overlaps = zeros<DType>(
      {box_corners_r.shape(0), qbox_corners_r.shape(0)});
  auto overlaps_rw = overlaps.template mutable_unchecked<2>();
  for (int k = 0; k < qbox_corners_r.shape(0); ++k) {
    for (int n = 0; n < box_corners_r.shape(0); ++n) {
      if (standup_iou_r(n, k) <= standup_thresh)
        continue;
      if (use_3d) {
        poly = make_corner_polygon<DType>(box_corners_r, n, 6, 7, 4, 5, 2);
        qpoly = make_corner_polygon<DType>(qbox_corners_r, k, 6, 7, 4, 5, 2);
      } else {
        poly = make_corner_polygon<DType>(box_corners_r, n, 0, 1, 2, 3, 1);
        qpoly = make_corner_polygon<DType>(qbox_corners_r, k, 0, 1, 2, 3, 1);
      }

      bg::intersection(poly, qpoly, poly_inter);
      if (poly_inter.empty()) {
        poly.clear();
        qpoly.clear();
        continue;
      }
      DType inter_area = bg::area(poly_inter.front());
      bg::union_(poly, qpoly, poly_union);
      if (!poly_union.empty()) {
        if (use_3d) {
          overlaps_rw(n, k) = calculate_3d_overlaps<DType>(
              poly, qpoly, box_corners_r, n, qbox_corners_r, k,
              inter_area).second;
        } else {
          DType union_area = bg::area(poly_union.front());
          overlaps_rw(n, k) = union_area > 0 ? inter_area / union_area : 0;
        }
      }
      poly_union.clear();
      poly.clear();
      qpoly.clear();
      poly_inter.clear();
    }
  }
  return overlaps;
}

template <typename DType>
py::array_t<DType> rbbox_iou(
    py::array_t<DType> box_corners, py::array_t<DType> qbox_corners,
    py::array_t<DType> standup_iou, DType standup_thresh) {
  return rbbox_iou_impl(
      box_corners, qbox_corners, standup_iou, standup_thresh, false);
}

template <typename DType>
py::array_t<DType> rbbox_iou_3d(
    py::array_t<DType> box_corners, py::array_t<DType> qbox_corners,
    py::array_t<DType> standup_iou, DType standup_thresh) {
  return rbbox_iou_impl(
      box_corners, qbox_corners, standup_iou, standup_thresh, true);
}

template <typename DType>
py::array_t<DType>
rbbox_iou_3d_pair(py::array_t<DType> box_corners, py::array_t<DType> qbox_corners) {
  namespace bg = boost::geometry;
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

  typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
  typedef bg::model::polygon<point_t> polygon_t;
  polygon_t poly, qpoly;
  std::vector<polygon_t> poly_inter, poly_union;

  DType inter_area;
  auto box_corners_r = box_corners.template unchecked<3>();
  auto qbox_corners_r = qbox_corners.template unchecked<3>();
  auto N = box_corners_r.shape(0);
  auto K = qbox_corners_r.shape(0);

  py::array_t<DType> overlaps = zeros<DType>({N, 2});
  auto overlaps_rw = overlaps.template mutable_unchecked<2>();
  if (N == 0 || K == 0 || N != K) {
    return overlaps;
  }
  for (int n = 0; n < N; ++n) {

    poly = make_corner_polygon<DType>(box_corners_r, n, 6, 7, 4, 5, 2);
    qpoly = make_corner_polygon<DType>(qbox_corners_r, n, 6, 7, 4, 5, 2);

    bg::intersection(poly, qpoly, poly_inter);

    if (!poly_inter.empty()) {
      inter_area = bg::area(poly_inter.front());
      bg::union_(poly, qpoly, poly_union);
      if (!poly_union.empty()) {
        auto values = calculate_3d_overlaps<DType>(
            poly, qpoly, box_corners_r, n, qbox_corners_r, n, inter_area);
        overlaps_rw(n, 0) = values.first;
        overlaps_rw(n, 1) = values.second;

      }
      poly_union.clear();
    }
    poly.clear();
    qpoly.clear();
    poly_inter.clear();
  }

  return overlaps;
}

#endif
