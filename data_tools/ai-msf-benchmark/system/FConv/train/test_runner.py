# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import logging
import os
import pprint
import random
import time

import numpy as np
import torch

from configs.config import assert_and_infer_cfg, cfg
from configs.config import merge_cfg_from_file, merge_cfg_from_list
from datasets.dataset_info import DATASET_INFO
from utils.utils import get_logger, import_from_file


def _parse_args():
    parser = argparse.ArgumentParser(description="Test a detection network")
    parser.add_argument("--cfg", dest="cfg_file", default=None, type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cfg_file is None:
        parser.error("--cfg is required")
    return args


def _configure(args):
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()


def _build_test_objects():
    model_class = import_from_file(cfg.MODEL.FILE).PointNetDet
    dataset_module = import_from_file(cfg.DATA.FILE)
    dataset = dataset_module.ProviderDataset(
        cfg.DATA.NUM_SAMPLES,
        split=cfg.TEST.DATASET,
        random_flip=False,
        random_shift=False,
        one_hot=True,
        from_rgb_detection=cfg.FROM_RGB_DET,
        overwritten_data_path=cfg.OVER_WRITE_TEST_FILE,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset_module.collate_fn,
    )
    input_channels = 3 if not cfg.DATA.WITH_EXTRA_FEAT else cfg.DATA.EXTRA_FEAT_DIM
    category_info = DATASET_INFO[cfg.DATA.DATASET_NAME]
    model = model_class(
        input_channels, num_vec=len(category_info.CLASSES),
        num_classes=cfg.MODEL.NUM_CLASSES,
    ).cuda()
    return model, dataset, loader


def _load_weights(model):
    if not os.path.isfile(cfg.TEST.WEIGHTS):
        raise FileNotFoundError(cfg.TEST.WEIGHTS)
    checkpoint = torch.load(cfg.TEST.WEIGHTS)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    logging.info("=> loaded checkpoint '%s'", cfg.TEST.WEIGHTS)
    if cfg.NUM_GPUS > 1:
        model = torch.nn.DataParallel(model)
    return model


def run_detection_test(test_function):
    args = _parse_args()
    _configure(args)
    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.SAVE_SUB_DIR)
    os.makedirs(save_dir, exist_ok=True)
    cfg_name = os.path.basename(args.cfg_file).split(".")[0]
    log_name = "{}_{}_val.log".format(cfg_name, time.strftime("%Y-%m-%d-%H-%M"))
    logger = get_logger(os.path.join(save_dir, log_name))
    logger.info("config:\n %s", pprint.pformat(cfg))
    model, dataset, loader = _build_test_objects()
    model = _load_weights(model)
    result_dir = os.path.join(save_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(save_dir, "detection.pkl")
    test_function(model, dataset, loader, output_file, result_dir)
