# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Compatibility exports for the canonical refined-car sample provider."""

__all__ = (
    "ProviderDataset",
    "collate_fn",
    "compute_alpha",
    "from_prediction_to_label_format",
)

from datasets import provider_sample_refine as _provider

ProviderDataset = _provider.ProviderDataset
collate_fn = _provider.collate_fn
compute_alpha = _provider.compute_alpha
from_prediction_to_label_format = _provider.from_prediction_to_label_format
