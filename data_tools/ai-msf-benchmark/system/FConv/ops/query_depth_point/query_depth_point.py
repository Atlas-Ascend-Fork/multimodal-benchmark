# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from torch import nn
from torch.autograd import Function

from . import query_depth_point_cuda


class _QueryDepthPoint(Function):

    @staticmethod
    def forward(ctx, dis_z, nsample, xyz1, xyz2):
        """
        Input:
            dis_z: float32, depth distance search distance
            nsample: int32, number of points selected in each ball region
            xyz1: (batch_size, 3, ndataset) float32 array, input points
            xyz2: (batch_size, 3, npoint) float32 array, query points
        Output:
            idx: (batch_size, npoint, nsample) int32 array, indices to input points
            pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
        """
        if not (xyz1.is_cuda and xyz1.size(1) == 3):
            raise AssertionError
        if not (xyz2.is_cuda and xyz2.size(1) == 3):
            raise AssertionError
        if not (xyz1.size(0) == xyz2.size(0)):
            raise AssertionError
        if not (xyz1.is_contiguous()):
            raise AssertionError
        if not (xyz2.is_contiguous()):
            raise AssertionError

        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()

        b = xyz1.size(0)
        n = xyz1.size(1)
        m = xyz2.size(1)

        idx = xyz1.new(b, m, nsample).long().zero_()
        pts_cnt = xyz1.new(b, m).int().zero_()

        query_depth_point_cuda.forward(
            b, n, m, dis_z, nsample, xyz1, xyz2, idx, pts_cnt
        )
        return idx, pts_cnt

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 6


class QueryDepthPoint(nn.Module):
    def __init__(self, dis_z, nsample):
        super(QueryDepthPoint, self).__init__()
        self.dis_z = dis_z
        self.nsample = nsample

    def forward(self, xyz1, xyz2):
        return _QueryDepthPoint.apply(self.dis_z, self.nsample, xyz1, xyz2)
