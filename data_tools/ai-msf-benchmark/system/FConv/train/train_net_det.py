# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import NamedTuple

import argparse
import logging
import os
import pprint
import random as pyrandom
import sys
import time

from configs.config import assert_and_infer_cfg
from configs.config import cfg
from configs.config import merge_cfg_from_file
from configs.config import merge_cfg_from_list
import numpy as np
import torch
import torch.optim as optim

from datasets.dataset_info import DATASET_INFO
from utils.training_states import TrainingStates
from utils.utils import AverageMeter, import_from_file, get_logger


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class TrainParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    data_loader: object
    model: object
    optimizer: object
    lr_scheduler: object
    epoch: object
    logger: object


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a network with Detectron")
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Config file for training (and optionally testing)",
        default=None,
        type=str,
    )
    parser.add_argument("--stage", default=None, type=int)
    parser.add_argument(
        "opts",
        help="See configs/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        raise ValueError("command-line arguments are required")

    return parser.parse_args()


def set_random_seed(seed=3):
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_module_bn_momentum(model, momentum=0.1):
    def set_bn_momentum(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.momentum = momentum

    model.apply(set_bn_momentum)


def get_bn_decay(epoch):
    # 0.5 - 0.01
    bn_init_decay = 0.1
    bn_decay_rate = 0.5
    bn_decay_step = cfg.TRAIN.LR_STEPS
    bn_decay_clip = 0.01
    bn_momentum = max(
        bn_init_decay * bn_decay_rate ** (epoch // bn_decay_step), bn_decay_clip
    )

    return bn_momentum


_TRAIN_PARAMETERS_SCHEMA = _ParameterSchema(
    TrainParameters,
    ("data_loader", "model", "optimizer", "lr_scheduler", "epoch", "logger"),
    {"logger": None},
    frozenset(()),
    6,
    None,
    None,
)


def _train_batch(model, optimizer, data_dicts):
    batch_size = data_dicts["point_cloud"].shape[0]
    data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}
    optimizer.zero_grad()
    losses, metrics = model(data_dicts_var)
    losses["total_loss"].mean().backward()
    optimizer.step()
    losses_reduce = {
        key: value.detach().mean().item() for key, value in losses.items()
    }
    metrics_reduce = {
        key: value.detach().mean().item() for key, value in metrics.items()
    }
    return batch_size, dict(**losses_reduce, **metrics_reduce)


def _log_train_progress(index, progress, meters, training_states):
    loader_size, epoch, cur_lr = progress
    if (index + 1) % cfg.disp != 0 and (index + 1) != loader_size:
        return
    data_time_meter, batch_time_meter = meters
    states = training_states.get_states(avg=False)
    output = "Train Epoch: {:03d} [{:04d}/{}] lr:{:.6f} Time:{:.3f}/{:.3f} ".format(
        epoch + 1, index + 1, loader_size, cur_lr,
        data_time_meter.val, batch_time_meter.val,
    )
    logging.info(output + training_states.format_states(states))
    if (index + 1) == loader_size:
        states = training_states.get_states(avg=True)
        output = "Train Epoch(AVG): {:03d} [{:04d}/{}] lr:{:.6f} Time:{:.3f}/{:.3f} ".format(
            epoch + 1, index + 1, loader_size, cur_lr,
            data_time_meter.val, batch_time_meter.val,
        )
        logging.info(output + training_states.format_states(states))


def train(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _TRAIN_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    data_loader, model, optimizer, lr_scheduler, epoch, logger = bound_parameters
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    model.train()

    min_lr = cfg.TRAIN.MIN_LR
    lr_scheduler.step(epoch)
    if min_lr > 0:
        if lr_scheduler.get_lr()[0] < min_lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = min_lr

    cur_lr = optimizer.param_groups[0]["lr"]

    tic = time.time()
    loader_size = len(data_loader)

    training_states = TrainingStates()

    for i, data_dicts in enumerate(data_loader):
        data_time_meter.update(time.time() - tic)
        batch_size, states = _train_batch(model, optimizer, data_dicts)
        training_states.update_states(states, batch_size)
        batch_time_meter.update(time.time() - tic)
        tic = time.time()
        meters = (data_time_meter, batch_time_meter)
        _log_train_progress(
            i, (loader_size, epoch, cur_lr), meters, training_states
        )
        if logger is not None:
            states = training_states.get_states(avg=True)
            for tag, value in states.items():
                logger.scalar_summary(tag, value, int(epoch))


def validate(data_loader, model, epoch, logger=None):
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    model.eval()

    tic = time.time()
    training_states = TrainingStates()

    for _, (data_dicts) in enumerate(data_loader):
        data_time_meter.update(time.time() - tic)

        batch_size = data_dicts["point_cloud"].shape[0]

        with torch.no_grad():
            data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}

            losses, metrics = model(data_dicts_var)
            # mean for multi-gpu setting
            losses_reduce = {
                key: value.detach().mean().item() for key, value in losses.items()
            }
            metrics_reduce = {
                key: value.detach().mean().item() for key, value in metrics.items()
            }

        training_states.update_states(
            dict(**losses_reduce, **metrics_reduce), batch_size
        )

        batch_time_meter.update(time.time() - tic)
        tic = time.time()

    states = training_states.get_states(avg=True)

    states_str = training_states.format_states(states)
    output_str = "Validation Epoch: {:03d} Time:{:.3f}/{:.3f} ".format(
        epoch + 1, data_time_meter.val, batch_time_meter.val
    )

    logging.info(output_str + states_str)

    if logger is not None:
        for tag, value in states.items():
            logger.scalar_summary(tag, value, int(epoch))

    return states["IoU_" + str(cfg.IOU_THRESH)]


def _configure_training(args):
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg_name = os.path.basename(args.cfg_file).split(".")[0]
    log_file = "{}_{}_train.log".format(cfg_name, time.strftime("%Y-%m-%d-%H-%M"))
    logger = get_logger(os.path.join(cfg.OUTPUT_DIR, log_file))
    logger.info(pprint.pformat(args))
    logger.info("config:\n {}".format(pprint.pformat(cfg)))
    return logger


def _visualization_loggers():
    logger_train = None
    logger_val = None
    if cfg.USE_TFBOARD:
        from utils.logger import Logger

        train_dir = os.path.join(cfg.OUTPUT_DIR, "tb_logger", "train")
        val_dir = os.path.join(cfg.OUTPUT_DIR, "tb_logger", "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        logger_train, logger_val = Logger(train_dir), Logger(val_dir)
    return logger_train, logger_val


def _training_data_loaders():
    logging.info(cfg.DATA.FILE)
    dataset_def = import_from_file(cfg.DATA.FILE)
    collate_fn = dataset_def.collate_fn
    dataset_def = dataset_def.ProviderDataset
    train_dataset = dataset_def(
        cfg.DATA.NUM_SAMPLES,
        split=cfg.TRAIN.DATASET,
        one_hot=True,
        random_flip=True,
        random_shift=True,
        extend_from_det=cfg.DATA.EXTEND_FROM_DET,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataset = dataset_def(
        cfg.DATA.NUM_SAMPLES,
        split=cfg.TEST.DATASET,
        one_hot=True,
        random_flip=False,
        random_shift=False,
        extend_from_det=cfg.DATA.EXTEND_FROM_DET,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    logging.info(
        "training: sample {} / batch {} ".format(len(train_dataset), len(train_loader))
    )
    logging.info(
        "validation: sample {} / batch {} ".format(len(val_dataset), len(val_loader))
    )
    return train_loader, val_loader


def _build_training_model():
    logging.info(cfg.MODEL.FILE)
    model_def = import_from_file(cfg.MODEL.FILE)
    model_def = model_def.PointNetDet
    input_channels = 3 if not cfg.DATA.WITH_EXTRA_FEAT else cfg.DATA.EXTRA_FEAT_DIM
    dataset_name = cfg.DATA.DATASET_NAME
    if dataset_name not in DATASET_INFO:
        raise AssertionError
    datset_category_info = DATASET_INFO[dataset_name]
    model = model_def(
        input_channels,
        num_vec=len(datset_category_info.CLASSES),
        num_classes=cfg.MODEL.NUM_CLASSES,
    )
    logging.info(pprint.pformat(model))
    if cfg.NUM_GPUS > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    logging.info("parameters: %d" % sum(p.numel() for p in model.parameters()))
    return model


def _training_optimizer(model):
    logging.info("using optimizer method {}".format(cfg.TRAIN.OPTIMIZER))
    if cfg.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
    elif cfg.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.BASE_LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
    else:
        raise AssertionError("Not support now.")
    if len(cfg.TRAIN.LR_STEPS) > 1:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.TRAIN.LR_STEPS, gamma=cfg.TRAIN.GAMMA
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.TRAIN.LR_STEPS[0], gamma=cfg.TRAIN.GAMMA
        )
    return optimizer, scheduler


def _resume_training(args, model, optimizer, logger):
    state = {"start_epoch": 0, "best_precision": 0, "best_epoch": 0}
    if not cfg.RESUME:
        return state
    pretrained_root = (
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/"
        "system/FConv/pretrained_models"
    )
    weight_paths = {
        1: os.path.join(pretrained_root, "car", "model_0050.pth"),
        2: os.path.join(pretrained_root, "car_refine", "model_0050.pth"),
    }
    if args.stage not in weight_paths:
        raise ValueError()
    cfg.TRAIN.WEIGHTS = weight_paths[args.stage]
    if not os.path.isfile(cfg.TRAIN.WEIGHTS):
        raise AssertionError
    checkpoint = torch.load(cfg.TRAIN.WEIGHTS)
    state = {
        "start_epoch": checkpoint["epoch"],
        "best_precision": checkpoint["best_prec1"],
        "best_epoch": checkpoint["best_epoch"],
    }
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    logger.info(
        "=> loaded checkpoint '%s' (epoch %s)", cfg.TRAIN.WEIGHTS, checkpoint["epoch"]
    )
    if state["start_epoch"] == cfg.TRAIN.MAX_EPOCH:
        return {"start_epoch": 0, "best_precision": 0, "best_epoch": 0}
    return state


def _run_training_epochs(context, state):
    for epoch in range(state["start_epoch"], cfg.TRAIN.MAX_EPOCH):
        train(
            context["train_loader"],
            context["model"],
            context["optimizer"],
            context["scheduler"],
            epoch,
            context["train_logger"],
        )
        precision = validate(
            context["val_loader"], context["model"], epoch, context["val_logger"]
        )
        is_best = precision > state["best_precision"]
        if is_best:
            state["best_precision"] = precision
            state["best_epoch"] = epoch + 1
            is_best = True
            logging.info(
                "Best model {:04d}, Validation Accuracy {:.6f}".format(
                    state["best_epoch"], state["best_precision"]
                )
            )
        save_data = {
            "epoch": epoch + 1,
            "state_dict": (
                context["model"].state_dict()
                if cfg.NUM_GPUS == 1
                else context["model"].module.state_dict()
            ),
            "optimizer": context["optimizer"].state_dict(),
            "best_prec1": state["best_precision"],
            "best_epoch": state["best_epoch"],
        }
        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.TRAIN.MAX_EPOCH:
            torch.save(
                save_data,
                os.path.join(cfg.OUTPUT_DIR, "model_%04d.pth" % (epoch + 1)),
            )
        if is_best:
            torch.save(save_data, os.path.join(cfg.OUTPUT_DIR, "model_best.pth"))
        if (epoch + 1) == cfg.TRAIN.MAX_EPOCH:
            torch.save(save_data, os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
    logging.info(
        "Best model {:04d}, Validation Accuracy {:.6f}".format(
            state["best_epoch"], state["best_precision"]
        )
    )


def main():
    args = parse_args()
    logger = _configure_training(args)
    train_logger, val_logger = _visualization_loggers()
    set_random_seed()
    train_loader, val_loader = _training_data_loaders()
    model = _build_training_model()
    optimizer, scheduler = _training_optimizer(model)
    state = _resume_training(args, model, optimizer, logger)
    if cfg.EVAL_MODE:
        validate(val_loader, model, state["start_epoch"], val_logger)
        return
    context = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_logger": train_logger,
        "val_logger": val_logger,
    }
    _run_training_epochs(context, state)


if __name__ == "__main__":
    main()
