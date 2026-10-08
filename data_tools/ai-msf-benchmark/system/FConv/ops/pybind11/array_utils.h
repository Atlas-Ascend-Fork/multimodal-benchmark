// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.

#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

#include <algorithm>
#include <vector>

#include <pybind11/numpy.h>

template <typename DType, typename ShapeContainer>
inline pybind11::array_t<DType> constant(ShapeContainer shape, DType value) {
    pybind11::array_t<DType> array(shape);
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), value);
    return array;
}

template <typename DType>
inline pybind11::array_t<DType> zeros(std::vector<long int> shape) {
    return constant<DType, std::vector<long int>>(shape, 0);
}

#endif
