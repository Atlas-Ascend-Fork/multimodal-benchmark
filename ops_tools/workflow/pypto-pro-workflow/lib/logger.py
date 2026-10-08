#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import sys
from pathlib import Path


_LOG_FMT = "[%(asctime)s][%(levelname)s][%(funcName)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_SIMPLE_FMT = "[%(asctime)s] %(message)s"
_NODE_FMT = "[%(asctime)s][%(levelname)s] %(message)s"


def _get_level(name: str, default: int) -> int:
    return getattr(logging, name.upper(), default)


def setup_logger(
    work_dir: str = None,
    level: str = "DEBUG",
    console_level: str = "INFO",
) -> logging.Logger:
    """Set up global + state + node loggers."""
    log = logging.getLogger("pypto-pro-workflow")
    if log.handlers:
        return log
    log.setLevel(_get_level(level, logging.DEBUG))

    fmt = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(_get_level(console_level, logging.INFO))
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if work_dir:
        log_dir = Path(work_dir) / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(
            log_dir / "workflow.log", encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


def setup_state_logger(work_dir: str) -> logging.Logger:
    """State transition logger."""
    log = logging.getLogger("pypto-pro-state")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)

    fmt = logging.Formatter(_SIMPLE_FMT, datefmt=_LOG_DATEFMT)

    log_dir = Path(work_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(
        log_dir / "state_transitions.log", encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


def setup_history_logger(work_dir: str) -> logging.Logger:
    """History change logger with full JSON snapshots."""
    log = logging.getLogger("pypto-pro-history")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)

    fmt = logging.Formatter(_SIMPLE_FMT, datefmt=_LOG_DATEFMT)

    log_dir = Path(work_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(
        log_dir / "history.log", encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log


def setup_node_logger(work_dir: str, node: str) -> logging.Logger:
    """Per-node logger (N1/N2/N3/N4)."""
    logger_name = f"pypto-{node}"
    log = logging.getLogger(logger_name)
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(_NODE_FMT, datefmt=_LOG_DATEFMT)

    log_dir = Path(work_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(
        log_dir / f"{node}.log", encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log
