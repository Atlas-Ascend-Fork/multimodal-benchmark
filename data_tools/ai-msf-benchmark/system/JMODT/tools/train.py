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

from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch

from jmodt.config import cfg, print_config_to_log, cfg_from_list
from jmodt.detection.datasets.kitti_dataset import KittiDataset
from jmodt.detection.modeling import train_functions
from jmodt.detection.modeling.point_rcnn import PointRCNN
from jmodt.utils import train_utils

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument(
    "--data_root", type=str, default="data/KITTI", help="specify the data root"
)
parser.add_argument(
    "--challenge", type=str, default="tracking", help="specify the KITTI system"
)
parser.add_argument(
    "--finetune", action="store_false", help="whether to finetune the pretrained model"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=12,
    required=True,
    help="the batch size for training",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="specify an output directory if needed",
)
parser.add_argument(
    "--ckpt", type=str, default=None, help="continue training from this checkpoint"
)
parser.add_argument("--mgpus", action="store_true", help="whether to use multiple gpu")
parser.add_argument(
    "--train_with_eval", action="store_true", help="whether to train with evaluation"
)
parser.add_argument(
    "--set",
    dest="set_cfgs",
    default=None,
    nargs=argparse.REMAINDER,
    help="set extra config keys if needed",
)
args = parser.parse_args()


def create_logger(log_file):
    log_format = "%(asctime)s  %(levelname)5s  %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def create_dataloader(logger, split):
    data_set = KittiDataset(
        root_dir=args.data_root,
        npoints=cfg.RPN.NUM_POINTS,
        split=split,
        mode="TRAIN",
        logger=logger,
        classes=cfg.CLASSES,
        challenge=args.challenge,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
        collate_fn=data_set.collate_batch,
        drop_last=True,
    )
    return data_set, data_loader


def _module_main_72_part_1(_state):
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if args.finetune:
        cfg.RPN.FIXED = True
        cfg.TRAIN.FINETUNE = True
    else:
        cfg.RPN.FIXED = False
        cfg.TRAIN.FINETUNE = False
    _state["root_result_dir"] = args.output_dir
    os.makedirs(_state["root_result_dir"], exist_ok=True)
    _state["log_file"] = os.path.join(_state["root_result_dir"], "log_train.txt")
    _state["logger"] = create_logger(_state["log_file"])
    _state["logger"].info("**********************Start logging**********************")
    _state["gpu_list"] = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys()
        else "ALL"
    )
    _state["logger"].info("CUDA_VISIBLE_DEVICES=%s" % _state["gpu_list"])
    for _state["key"], _state["val"] in vars(args).items():
        _state["logger"].info("{:16} {}".format(_state["key"], _state["val"]))
    print_config_to_log(cfg, logger=_state["logger"])
    _state["tb_log"] = SummaryWriter(
        logdir=os.path.join(_state["root_result_dir"], "tensorboard")
    )
    _state["train_set"], _state["train_loader"] = create_dataloader(
        _state["logger"], split=cfg.TRAIN.SPLIT
    )
    _state["val_set"], _state["val_loader"] = (
        (
            create_dataloader(_state["logger"], split=cfg.TRAIN.VAL_SPLIT)
            if args.train_with_eval
            else None
        ),
        None,
    )
    _state["fn_decorator"] = train_functions.model_joint_fn_decorator()
    _state["model"] = PointRCNN(
        num_classes=_state["train_set"].num_class, use_xyz=True, mode="TRAIN"
    )
    if args.mgpus:
        _state["model"] = nn.DataParallel(_state["model"])
    _state["model"].cuda()
    _state["params_to_update"] = _state["model"].parameters()
    _state["start_epoch"] = _state["it"] = 0
    _state["last_epoch"] = -1


def _module_main_72_part_2_chunk_1(_state):
    if args.ckpt is not None:
        _state["pure_model"] = (
            _state["model"].module
            if isinstance(_state["model"], torch.nn.DataParallel)
            else _state["model"]
        )
        if cfg.TRAIN.FINETUNE:
            for _state["param"] in _state["pure_model"].parameters():
                _state["param"].requires_grad = False
            _state["params_to_update"] = list(
                _state["pure_model"].rcnn_net.link_layer.parameters()
            ) + list(_state["pure_model"].rcnn_net.se_layer.parameters())
            for _state["param"] in _state["params_to_update"]:
                _state["param"].requires_grad = True
            _state["optimizer"] = optim.AdamW(
                [
                    {"params": _state["pure_model"].rcnn_net.link_layer.parameters()},
                    {"params": _state["pure_model"].rcnn_net.se_layer.parameters()},
                ],
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            )
        else:
            _state["optimizer"] = optim.AdamW(
                _state["params_to_update"],
                lr=cfg.TRAIN.LR,
                weight_decay=cfg.TRAIN.WEIGHT_DECAY,
            )
        if cfg.TRAIN.RELOAD_OPTIMIZER:
            _state["it"], _state["start_epoch"] = train_utils.load_checkpoint(
                _state["pure_model"],
                _state["optimizer"],
                filename=args.ckpt,
                logger=_state["logger"],
            )
            _state["last_epoch"] = _state["start_epoch"] + 1
        else:
            train_utils.load_checkpoint(
                _state["pure_model"],
                optimizer=None,
                filename=args.ckpt,
                logger=_state["logger"],
            )
    else:
        _state["optimizer"] = optim.AdamW(
            _state["params_to_update"],
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )


def _module_main_72_part_2_chunk_2(_state):
    _state["lr_scheduler"] = optim.lr_scheduler.CosineAnnealingLR(
        _state["optimizer"],
        T_max=cfg.TRAIN.TMAX,
        eta_min=cfg.TRAIN.ETA_MIN,
        last_epoch=_state["last_epoch"],
    )


def _module_main_72_part_2(_state):
    _module_main_72_part_2_chunk_1(_state)
    _module_main_72_part_2_chunk_2(_state)


def _module_main_72_part_3(_state):
    _state["logger"].info("**********************Start training**********************")
    _state["ckpt_dir"] = os.path.join(_state["root_result_dir"], "ckpt")
    os.makedirs(_state["ckpt_dir"], exist_ok=True)
    _state["trainer"] = train_utils.Trainer(
        model=_state["model"],
        params_to_update=_state["params_to_update"],
        model_fn_train=_state["fn_decorator"],
        optimizer=_state["optimizer"],
        ckpt_dir=_state["ckpt_dir"],
        lr_scheduler=_state["lr_scheduler"],
        model_fn_val=_state["fn_decorator"],
        tb_log=_state["tb_log"],
        eval_frequency=1,
        grad_norm_clip=cfg.TRAIN.GRAD_NORM_CLIP,
    )
    _state["trainer"].train(
        _state["it"],
        _state["start_epoch"],
        cfg.TRAIN.EPOCHS,
        _state["train_loader"],
        _state["val_loader"],
    )
    _state["logger"].info("**********************End training**********************")


def main():
    _state = locals()
    _module_main_72_part_1(_state)
    _module_main_72_part_2(_state)
    _module_main_72_part_3(_state)


if __name__ == "__main__":
    main()
