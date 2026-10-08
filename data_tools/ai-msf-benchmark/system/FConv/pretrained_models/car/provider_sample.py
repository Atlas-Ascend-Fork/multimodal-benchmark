# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Compatibility exports for the canonical car sample provider."""

__all__ = [
    "ProviderDataset",
    "collate_fn",
    "compute_alpha",
    "from_prediction_to_label_format",
]

from datasets.provider_sample import (
    ProviderDataset,
    collate_fn,
    compute_alpha,
    from_prediction_to_label_format,
)
