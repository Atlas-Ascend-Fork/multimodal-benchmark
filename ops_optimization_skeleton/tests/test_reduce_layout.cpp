// Copyright (c) 2026 multimodal-bench contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "common/inc/op_api/aclnn_check.h"

int main()
{
    float data[16] = {};
    int64_t shape[] = {2, 1, 3};
    int64_t storage[] = {2, 1, 8};
    int64_t strides[] = {8, 8, 2};
    auto tensor = aclCreateTensor(shape, 3, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, storage, 3, data);
    if (tensor == nullptr) {
        return 1;
    }
    for (int64_t dim : {0, 1, -2}) {
        auto dims = aclCreateIntArray(&dim, 1);
        if (dims == nullptr) {
            aclDestroyTensor(tensor);
            return 1;
        }
        bool moving = dim != 0;
        bool valid = op::IsReduceTensorMoving(tensor, dims) == moving;
        valid = (op::IsReduceNonContiguousSupport(tensor, dims) == !moving) && valid;
        aclDestroyIntArray(dims);
        if (!valid) {
            aclDestroyTensor(tensor);
            return 1;
        }
    }
    auto allDims = aclCreateIntArray(nullptr, 0);
    if (allDims == nullptr) {
        aclDestroyTensor(tensor);
        return 1;
    }
    bool valid = !op::IsReduceTensorMoving(tensor, allDims);
    valid = op::IsReduceNonContiguousSupport(tensor, allDims) && valid;
    aclDestroyIntArray(allDims);
    aclDestroyTensor(tensor);
    return valid ? 0 : 1;
}
