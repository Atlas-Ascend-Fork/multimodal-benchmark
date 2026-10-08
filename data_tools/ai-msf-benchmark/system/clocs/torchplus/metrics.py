# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class PrecisionrecallInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    dim: object
    ignore_idx: object
    thresholds: object
    use_sigmoid_score: object
    encode_background_as_zeros: object


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.FloatTensor([0.0]))
        self.register_buffer("count", torch.FloatTensor([0.0]))

    @property
    def value(self):
        return self.total / self.count

    def forward(self, scalar):
        if not scalar.eq(0.0):
            self.count += 1
            self.total += scalar.data.float()
        return self.value.cpu()

    def clear(self):
        self.total.zero_()
        self.count.zero_()


class Accuracy(nn.Module):
    def __init__(
        self, dim=1, ignore_idx=-1, threshold=0.5, encode_background_as_zeros=True
    ):
        super().__init__()
        self.register_buffer("total", torch.FloatTensor([0.0]))
        self.register_buffer("count", torch.FloatTensor([0.0]))
        self.ignore_index = ignore_idx
        self.dimension = dim
        self.threshold = threshold
        self._encode_background_as_zeros = encode_background_as_zeros

    @property
    def value(self):
        return self.total / self.count

    def forward(self, labels, preds, weights=None):
        if self._encode_background_as_zeros:
            scores = torch.sigmoid(preds)
            labels_pred = torch.max(preds, dim=self.dimension)[1] + 1
            pred_labels = torch.where(
                (scores > self.threshold).any(self.dimension),
                labels_pred,
                torch.tensor(0).type_as(labels_pred),
            )
        else:
            pred_labels = torch.max(preds, dim=self.dimension)[1]
        n, *ds = labels.shape
        labels = labels.view(n, int(np.prod(ds)))
        pred_labels = pred_labels.view(n, int(np.prod(ds)))
        if weights is None:
            weights = (labels != self.ignore_index).float()
        else:
            weights = weights.float()

        num_examples = torch.sum(weights)
        num_examples = torch.clamp(num_examples, min=1.0).float()
        total = torch.sum((pred_labels == labels.long()).float())
        self.count += num_examples
        self.total += total
        return self.value.cpu()

    def clear(self):
        self.total.zero_()
        self.count.zero_()


class _BinaryMetric(nn.Module):
    def __init__(self, dim=1, ignore_idx=-1, threshold=0.5):
        super().__init__()
        self.register_buffer("total", torch.FloatTensor([0.0]))
        self.register_buffer("count", torch.FloatTensor([0.0]))
        self.ignore_index = ignore_idx
        self.dimension = dim
        self.threshold = threshold

    @property
    def value(self):
        return self.total / self.count

    def clear(self):
        self.total.zero_()
        self.count.zero_()


class Precision(_BinaryMetric):

    def forward(self, labels, preds, weights=None):
        labels, pred_labels, weights = _binary_metric_inputs(
            self, labels, preds, weights
        )
        pred_trues = pred_labels > 0
        trues = labels > 0
        falses = labels == 0
        true_positives = (weights * (trues & pred_trues).float()).sum()
        false_positives = (weights * (falses & pred_trues).float()).sum()
        return _record_metric_ratio(self, true_positives, false_positives)


class Recall(_BinaryMetric):
    def forward(self, labels, preds, weights=None):
        labels, pred_labels, weights = _binary_metric_inputs(
            self, labels, preds, weights
        )
        pred_trues = pred_labels == 1
        pred_falses = pred_labels == 0
        trues = labels == 1
        true_positives = (weights * (trues & pred_trues).float()).sum()
        false_negatives = (weights * (trues & pred_falses).float()).sum()
        return _record_metric_ratio(self, true_positives, false_negatives)


def _binary_metric_inputs(metric, labels, preds, weights):
    """Normalize binary labels, predictions, and optional sample weights."""
    if preds.shape[metric.dimension] == 1:  # BCE
        pred_labels = (
            (torch.sigmoid(preds) > metric.threshold).long().squeeze(metric.dimension)
        )
    else:
        if preds.shape[metric.dimension] != 2:
            raise AssertionError("binary metrics only support one or two channels")
        pred_labels = torch.max(preds, dim=metric.dimension)[1]
    batch_size, *dimensions = labels.shape
    labels = labels.view(batch_size, int(np.prod(dimensions)))
    pred_labels = pred_labels.view(batch_size, int(np.prod(dimensions)))
    if weights is None:
        weights = (labels != metric.ignore_index).float()
    return labels, pred_labels, weights.float()


def _record_metric_ratio(metric, numerator, other_count):
    count = numerator + other_count
    if count > 0:
        metric.count += count
        metric.total += numerator
    return metric.value.cpu()


def _calc_binary_metrics(labels, scores, weights=None, ignore_idx=-1, threshold=0.5):

    pred_labels = (scores > threshold).long()
    n, *ds = labels.shape
    labels = labels.view(n, int(np.prod(ds)))
    pred_labels = pred_labels.view(n, int(np.prod(ds)))
    pred_trues = pred_labels > 0
    pred_falses = pred_labels == 0
    trues = labels > 0
    falses = labels == 0
    true_positives = (weights * (trues & pred_trues).float()).sum()
    true_negatives = (weights * (falses & pred_falses).float()).sum()
    false_positives = (weights * (falses & pred_trues).float()).sum()
    false_negatives = (weights * (trues & pred_falses).float()).sum()
    return true_positives, true_negatives, false_positives, false_negatives


_PRECISIONRECALL_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    PrecisionrecallInitParameters,
    (
        "dim",
        "ignore_idx",
        "thresholds",
        "use_sigmoid_score",
        "encode_background_as_zeros",
    ),
    {
        "dim": 1,
        "ignore_idx": -1,
        "thresholds": 0.5,
        "use_sigmoid_score": False,
        "encode_background_as_zeros": True,
    },
    frozenset(()),
    5,
    None,
    None,
)


class PrecisionRecall(nn.Module):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _PRECISIONRECALL_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        dim, ignore_idx, thresholds, use_sigmoid_score, encode_background_as_zeros = (
            bound_parameters
        )
        super().__init__()
        if not isinstance(thresholds, (list, tuple)):
            thresholds = [thresholds]

        self.register_buffer("prec_total", torch.FloatTensor(len(thresholds)).zero_())
        self.register_buffer("prec_count", torch.FloatTensor(len(thresholds)).zero_())
        self.register_buffer("rec_total", torch.FloatTensor(len(thresholds)).zero_())
        self.register_buffer("rec_count", torch.FloatTensor(len(thresholds)).zero_())

        self.ignore_index = ignore_idx
        self.dimension = dim
        self._thresholds = thresholds
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros

    @property
    def value(self):
        prec_count = torch.clamp(self.prec_count, min=1.0)
        rec_count = torch.clamp(self.rec_count, min=1.0)
        return (
            (self.prec_total / prec_count).cpu(),
            (self.rec_total / rec_count).cpu(),
        )

    @property
    def thresholds(self):
        return self._thresholds

    def forward(self, labels, preds, weights=None):
        if self._encode_background_as_zeros:
            # this don't support softmax
            if not (self._use_sigmoid_score):
                raise AssertionError
            total_scores = torch.sigmoid(preds)
        else:
            if self._use_sigmoid_score:
                total_scores = torch.sigmoid(preds)[..., 1:]
            else:
                total_scores = F.softmax(preds, dim=-1)[..., 1:]
        scores = torch.max(total_scores, dim=-1)[0]
        if weights is None:
            weights = (labels != self.ignore_index).float()
        else:
            weights = weights.float()
        for i, thresh in enumerate(self._thresholds):
            tp, tn, fp, fn = _calc_binary_metrics(
                labels, scores, weights, self.ignore_index, thresh
            )
            rec_count = tp + fn
            prec_count = tp + fp
            if rec_count > 0:
                self.rec_count[i] += rec_count
                self.rec_total[i] += tp
            if prec_count > 0:
                self.prec_count[i] += prec_count
                self.prec_total[i] += tp

        return self.value

    def clear(self):
        self.rec_count.zero_()
        self.prec_count.zero_()
        self.prec_total.zero_()
        self.rec_total.zero_()
