# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Checkpoint persistence shared by compatible trainers."""

import os

import torch
from shared.parameters import bind_named as _bind_named


def checkpoint_state(model=None, optimizer=None, epoch=None, iteration=None):
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict() if model is not None else None
    return {
        "epoch": epoch,
        "it": iteration,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
    }


def save_checkpoint(state, filename="checkpoint"):
    torch.save(state, "{}.pth".format(filename))


def load_checkpoint(*call_args, **call_kwargs):
    (
        model,
        optimizer,
        filename,
        logger,
        strict,
        report_keys,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('model', 'optimizer', 'filename', 'logger', 'strict', 'report_keys'),
        {'strict': True, 'report_keys': False},
    )
    if not os.path.isfile(filename):
        raise FileNotFoundError
    logger.info("==> Loading from checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    epoch = checkpoint.get("epoch", -1)
    iteration = checkpoint.get("it", 0.0)
    if model is not None and checkpoint["model_state"] is not None:
        keys = model.load_state_dict(checkpoint["model_state"], strict=strict)
        if report_keys:
            logger.info("missing keys: %s\n", keys.missing_keys)
            logger.info("unexpected keys: %s\n", keys.unexpected_keys)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    logger.info("==> Done")
    return iteration, epoch


def load_part_checkpoint(model, filename, logger, total_keys=-1):
    if not os.path.isfile(filename):
        raise FileNotFoundError
    logger.info("==> Loading part model from checkpoint '{}'".format(filename))
    model_state = torch.load(filename)["model_state"]
    compatible = {
        key: value for key, value in model_state.items() if key in model.state_dict()
    }
    state_dict = model.state_dict()
    state_dict.update(compatible)
    model.load_state_dict(state_dict)
    loaded_count = len(compatible)
    if loaded_count == 0:
        raise RuntimeError
    logger.info("==> Done (loaded %d/%d)", loaded_count, total_keys)
