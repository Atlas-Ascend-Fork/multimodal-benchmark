# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib
import logging
import os
import shutil
import sys

from configs.config import cfg


def import_from_file(def_file, cover=True):
    folder = os.path.dirname(def_file)
    file = os.path.basename(def_file)
    # save setting file
    save_file = os.path.join(cfg.OUTPUT_DIR, file)
    if not os.path.exists(save_file) or cover:
        shutil.copy(def_file, os.path.join(cfg.OUTPUT_DIR, file))

    path = os.path.abspath(folder)
    sys.path.append(path)
    model_file = importlib.import_module(file[:-3])
    sys.path.remove(path)
    return model_file


def get_accuracy(output, target, topk=(1,), ignore=None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    if not (output.shape[0] == target.shape[0]):
        raise AssertionError
    if ignore is not None:
        if not (isinstance(ignore, int)):
            raise AssertionError
        keep = (target != ignore).nonzero().view(-1)
        output = output[keep]
        target = target[keep]

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 1e-14

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(log_file):
    log_format = "%(message)s"
    logging.root.handlers = []
    logging.basicConfig(filename=log_file, format=log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    return logger
