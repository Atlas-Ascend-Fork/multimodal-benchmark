# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import numpy as np
import torch


def huber_loss(error, delta, weight=None):
    delta = torch.ones_like(error) * delta
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = abs_error - quadratic
    losses = 0.5 * quadratic**2 + delta * linear
    # Note condisder batch mean
    if weight is not None:
        losses *= weight

    return losses.mean()


def smooth_l1_loss(input_tensor, target, sigma=1.0, size_average=True):
    """
    input: B, *
    target: B, *

    """
    # smooth_l1_loss with sigma
    if not (input_tensor.shape == target.shape):
        raise AssertionError

    diff = torch.abs(input_tensor - target)

    mask = (diff < (1.0 / sigma**2)).detach().type_as(diff)

    output = mask * torch.pow(sigma * diff, 2) / 2.0 + (1 - mask) * (
        diff - 1.0 / (2.0 * sigma**2.0)
    )
    loss = output.sum()
    if size_average:
        loss = loss / input_tensor.shape[0]

    return loss


def get_box3d_corners_helper(centers, headings, sizes):

    n = centers.shape[0]
    length_value = sizes[:, 0]  # (N)
    w = sizes[:, 1]  # (N)
    h = sizes[:, 2]  # (N)
    x_corners = torch.stack(
        [
            length_value / 2,
            length_value / 2,
            -length_value / 2,
            -length_value / 2,
            length_value / 2,
            length_value / 2,
            -length_value / 2,
            -length_value / 2,
        ],
        1,
    )  # (N,8)
    y_corners = torch.stack(
        [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], 1
    )  # (N,8)
    z_corners = torch.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], 1
    )  # (N,8)
    corners = torch.stack([x_corners, y_corners, z_corners], 1)  # (N,3,8)

    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = headings.new_ones(n)
    zeros = headings.new_zeros(n)
    row1 = torch.stack([c, zeros, s], 1)  # (N,3)
    row2 = torch.stack([zeros, ones, zeros], 1)
    row3 = torch.stack([-s, zeros, c], 1)
    r = torch.stack([row1, row2, row3], 1)  # (N,3,3)

    corners_3d = torch.bmm(r, corners)  # (N,3,8)
    corners_3d = corners_3d + centers.unsqueeze(2)  # (N,3,8)
    corners_3d = torch.transpose(corners_3d, 1, 2).contiguous()  # (N,8,3)
    return corners_3d


def point_cloud_masking(
    point_cloud, logits, xyz_only=True, num_object_point=1024, return_idx=False
):
    """Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.

    Input:
        point_cloud: (B, 3, N)
        logits: shape (B, N, 2)
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: (B, 3, M)
            M = num_object_point as a hyper-parameter
        mask_xyz_mean: (B, 3)
        mask B, N
    """

    _ = point_cloud.shape[0]
    _ = point_cloud.shape[-1]
    # b, n, 3
    point_cloud = point_cloud.transpose(2, 1).contiguous()
    mask = logits[:, :, 0] < logits[:, :, 1]  # B, N
    mask = mask.float()

    mask_count = torch.sum(mask, 1, keepdim=True)  # B, 1
    point_cloud_xyz = point_cloud[:, :, :3]
    # b, n, 1 - b, n, 3 -> b, n, 3
    fg_pts = mask.unsqueeze(-1) * point_cloud_xyz
    # avoid divide zero
    mask_count = torch.max(mask_count, torch.ones_like(mask_count))
    # b, 3 - b, 1 -> b, 3
    mask_xyz_mean = torch.sum(fg_pts, 1) / mask_count

    # Translate to masked points' centroid
    # b, n, 3 - b, 1, 3
    point_cloud_xyz_stage1 = point_cloud_xyz - mask_xyz_mean.unsqueeze(1)

    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = point_cloud[:, :, 3:]
        point_cloud_stage1 = torch.cat(
            [point_cloud_xyz_stage1, point_cloud_features], axis=-1
        )

    object_point_cloud, idx = gather_object_pc(
        point_cloud_stage1, mask, num_object_point
    )
    # b, m, 3  -> b, 3, m
    object_point_cloud = object_point_cloud.transpose(2, 1).contiguous()
    if return_idx:
        return object_point_cloud, mask_xyz_mean, mask, idx
    else:
        return object_point_cloud, mask_xyz_mean, mask


def gather_object_pc(point_cloud, mask, npoints=512):
    """Gather object point clouds according to predicted masks.
    Input:
        point_cloud: (B,N,C)
        mask: shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc:(B, npoint, C)
        indices: (B, npoint)
    """

    def mask_to_indices(mask):
        # input b, n
        # output b, npoints

        indices = mask.new_zeros(mask.shape[0], npoints, dtype=torch.long)

        for i in range(mask.shape[0]):
            pos_indices = (mask[i, :] > 0.5).nonzero()
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                pos_indices = pos_indices[:, 0]
                if len(pos_indices) > npoints:
                    choice = np.random.choice(len(pos_indices), npoints, replace=False)
                else:
                    choice = np.random.choice(
                        len(pos_indices), npoints - len(pos_indices), replace=True
                    )
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                choice = choice.astype(np.int64)
                indices[i, :] = pos_indices[choice]

        return indices

    indices = mask_to_indices(mask.detach())

    batch_size = point_cloud.shape[0]
    feat_num = point_cloud.shape[2]
    object_pc = torch.gather(
        point_cloud,
        1,
        indices.view(batch_size, npoints, 1).expand(batch_size, npoints, feat_num),
    )

    return object_pc, indices
