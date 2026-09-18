# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import math

import numpy as np
import torch


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class ResultUpdateParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    irmse: object
    imae: object
    mse: object
    rmse: object
    mae: object
    absrel: object
    squared_rel: object
    lg10: object
    delta1: object
    delta2: object
    delta3: object
    gpu_time: object
    data_time: object
    silog: object
    photometric: object
    tmae: object
    trmse: object


lg_e_10 = math.log(10)


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x."""
    return torch.log(x) / lg_e_10


def accum_errorstat_linearscale(label_map, error):

    discrete_bins = np.arange(1, 8000, 100)

    label_mapfin = label_map[np.logical_not(np.isnan(label_map))]
    error = error[np.logical_not(np.isnan(label_map))]

    label_mapfin = np.reshape(label_mapfin, (-1,))
    error = np.reshape(error, (-1,))

    error_bins = np.zeros((80,), np.float32)
    pix_bins = np.zeros((80,), np.float32)

    for ii in range(len(discrete_bins) - 1):
        pixmask = np.logical_and(
            label_mapfin >= discrete_bins[ii], label_mapfin < discrete_bins[ii + 1]
        )
        error_masked = np.multiply(error, np.float32(pixmask))
        error_bins[ii] = np.nansum(error_masked)
        pix_bins[ii] = np.sum(pixmask)

    return error_bins, pix_bins


def accum_errorstat(label_map, pred_map, maxdep=90.0, step=10.0, method="ae"):

    discrete_bins = torch.arange(0, maxdep, step)
    nbins = int(maxdep / step) - 1
    mask = torch.isclose(label_map, torch.tensor(0.0, device=label_map.device))
    pix_valid = ~mask
    label_mapfin = label_map[pix_valid]
    pred_mapfin = pred_map[pix_valid]
    if method == "ae":
        error = torch.abs(pred_mapfin - label_mapfin)
    elif method == "se":
        error = (pred_mapfin - label_mapfin) ** 2
    error_bins = torch.zeros(
        nbins,
    )
    pix_bins = torch.zeros(
        nbins,
    )

    for ii in range(len(discrete_bins) - 1):
        pixmask = (label_mapfin >= discrete_bins[ii]) & (
            label_mapfin < discrete_bins[ii + 1]
        )
        error_masked = error * pixmask.double()
        error_bins[ii] = torch.sum(error_masked)
        pix_bins[ii] = torch.sum(pixmask)

    return error_bins, pix_bins


def trmse_error(label, pred, depth_maxrange, threshold=1000, oormask=None):
    if oormask is None:
        oormask = []
    pred[pred > depth_maxrange] = depth_maxrange
    pred[pred < 0] = 0
    mask = (label == 0) | (label > depth_maxrange)
    pix_valid = ~mask
    result = torch.sqrt(
        torch.mean(
            torch.min(
                (label[pix_valid] - pred[pix_valid]) ** 2,
                torch.ones((torch.sum(pix_valid),)).to(pred.device) * threshold**2,
            )
        )
    )
    result = float(result)
    return result


def tmae_error(label, pred, depth_maxrange, threshold=1000):
    pred[pred > depth_maxrange] = depth_maxrange
    pred[pred < 0] = 0
    mask = (label == 0) | (label > depth_maxrange)
    pix_valid = ~mask
    result = torch.mean(
        torch.min(
            torch.abs(pred[pix_valid] - label[pix_valid]),
            torch.ones((torch.sum(pix_valid),)).to(pred.device) * threshold,
        )
    )
    result = float(result)
    return result


_RESULT_UPDATE_PARAMETERS_SCHEMA = _ParameterSchema(
    ResultUpdateParameters,
    (
        "irmse",
        "imae",
        "mse",
        "rmse",
        "mae",
        "absrel",
        "squared_rel",
        "lg10",
        "delta1",
        "delta2",
        "delta3",
        "gpu_time",
        "data_time",
        "silog",
        "photometric",
        "tmae",
        "trmse",
    ),
    {"photometric": 0, "tmae": 0, "trmse": 0},
    frozenset(()),
    17,
    None,
    None,
)


class Result(object):
    def __init__(self, threshold=1000, dep_maxrange=80000):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.tmae = 0
        self.trmse = 0
        self.threshold = threshold
        self.dep_maxrange = dep_maxrange
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.silog = 0  # Scale invariant logarithmic error [log(m)*100]
        self.photometric = 0

    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.tmae = np.inf
        self.trmse = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0

    def update(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _RESULT_UPDATE_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            irmse,
            imae,
            mse,
            rmse,
            mae,
            absrel,
            squared_rel,
            lg10,
            delta1,
            delta2,
            delta3,
            gpu_time,
            data_time,
            silog,
            photometric,
            tmae,
            trmse,
        ) = bound_parameters
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.lg10 = lg10
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.data_time = data_time
        self.gpu_time = gpu_time
        self.silog = silog
        self.photometric = photometric
        self.tmae = tmae
        self.trmse = trmse

    def evaluate(self, output, target, photometric=0):
        valid_mask = target > 0.1

        # convert from meters to mm
        output_mm = 1e3 * output[valid_mask]
        target_mm = 1e3 * target[valid_mask]

        abs_diff = (output_mm - target_mm).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        self.squared_rel = float(((abs_diff / target_mm) ** 2).mean())
        self.tmae = tmae_error(
            target_mm, output_mm, self.dep_maxrange, threshold=self.threshold
        )
        self.trmse = trmse_error(
            target_mm, output_mm, self.dep_maxrange, threshold=self.threshold
        )
        max_ratio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((max_ratio < 1.25).float().mean())
        self.delta2 = float((max_ratio < 1.25**2).float().mean())
        self.delta3 = float((max_ratio < 1.25**3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        # silog uses meters
        err_log = torch.log(target[valid_mask]) - torch.log(output[valid_mask])
        normalized_squared_log = (err_log**2).mean()
        log_mean = err_log.mean()
        self.silog = math.sqrt(normalized_squared_log - log_mean * log_mean) * 100

        # convert from meters to km
        inv_output_km = (1e-3 * output[valid_mask]) ** (-1)
        inv_target_km = (1e-3 * target[valid_mask]) ** (-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        self.photometric = float(photometric)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_tmae = 0
        self.sum_trmse = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.sum_photometric = 0
        self.sum_silog = 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_tmae += n * result.tmae
        self.sum_trmse += n * result.trmse
        self.sum_squared_rel += n * result.squared_rel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time
        self.sum_silog += n * result.silog
        self.sum_photometric += n * result.photometric

    def average(self):
        avg = Result()
        if self.count > 0:
            avg.update(
                self.sum_irmse / self.count,
                self.sum_imae / self.count,
                self.sum_mse / self.count,
                self.sum_rmse / self.count,
                self.sum_mae / self.count,
                self.sum_absrel / self.count,
                self.sum_squared_rel / self.count,
                self.sum_lg10 / self.count,
                self.sum_delta1 / self.count,
                self.sum_delta2 / self.count,
                self.sum_delta3 / self.count,
                self.sum_gpu_time / self.count,
                self.sum_data_time / self.count,
                self.sum_silog / self.count,
                self.sum_photometric / self.count,
                self.sum_tmae / self.count,
                self.sum_trmse / self.count,
            )
        return avg
