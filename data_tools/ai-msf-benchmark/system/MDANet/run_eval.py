# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""
import argparse
import json
import os
import sys

from dataloader.data_loaders import kitti_data_loader
from mdanet_runtime import build_runtime, build_trainer, mode_and_sets
import torch
import torch.backends.cudnn as cudnn


#

#


# Fix CUDNN error for non-contiguous inputs

cudnn.enabled = True
cudnn.benchmark = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


torch.manual_seed(1)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)

parser.add_argument(
    "--mode", action="store", dest="mode", default="eval", help='"eval" or "train" mode'
)
parser.add_argument("--model", default=None)
parser.add_argument("--loss1", default=None)
parser.add_argument("--loss2", default=None)
parser.add_argument(
    "--exp",
    action="store",
    dest="exp",
    default="exp_msg_chn",
    help="Experiment name as in workspace directory",
)

parser.add_argument(
    "--ckpt",
    action="store",
    dest="chkpt",
    default="results/quickstart/checkpoints/net.pth.tar",
    help="Checkpoint number to load",
)

parser.add_argument(
    "--set",
    action="store",
    dest="set",
    default="train",
    type=str,
    nargs="?",
    help='Which set to evaluate on "val", "selval" or "test"',
)
parser.add_argument("--save_output", action="store_true", help="Whether to save output")
args = parser.parse_args()

# Path to the workspace directory
TRAINING_WS_PATH = "workspace/"
exp = args.exp
exp_dir = os.path.join(TRAINING_WS_PATH, exp)

# Add the experiment's folder to python path
sys.path.append(exp_dir)

# Read parameters file
with open(os.path.join("config", args.config + ".json"), "r") as fp:
    params = json.load(fp)

# Use GPU or not

dataloaders, dataset_sizes, model, t, optimizer, lr_decay = build_runtime(
    params, args, kitti_data_loader
)

mode, sets = mode_and_sets(args)
if args.mode == "eval":
    save_output = args.save_output

# Objective function
loss1 = args.loss1 if args.loss1 is not None else params["loss1"]
loss2 = args.loss2 if args.loss2 is not None else params["loss2"]
objective1 = locals()[loss1]()
objective2 = locals()[loss2]()

mytrainer = build_trainer(
    runtime=(dataloaders, dataset_sizes, model, t, optimizer, lr_decay),
    parameters=params,
    objectives=(objective1, objective2),
    experiment_dir=exp_dir,
    sets=sets,
    checkpoint=args.chkpt,
)

if mode == "train":
    # train the network
    net = mytrainer.train(params["num_epochs"])  #
elif os.path.isfile(args.chkpt):
    net = mytrainer.evaluate(save_output)
elif os.path.isdir(args.chkpt):
    net = mytrainer.evaluate_folder()
else:
    raise NotImplementedError
