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
import time
import warnings

from dataloader.data_loaders import kitti_data_loader
from mdanet_runtime import build_runtime, build_trainer, mode_and_sets
import torch
import torch.backends.cudnn as cudnn


#

#


# Fix CUDNN error for non-contiguous inputs

warnings.filterwarnings("ignore")
cudnn.enabled = True
cudnn.benchmark = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.cuda.manual_seed(1)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--model", default=None)
parser.add_argument(
    "--mode",
    action="store",
    dest="mode",
    default="train",
    help='"eval" or "train" mode',
)
parser.add_argument(
    "--exp",
    action="store",
    dest="exp",
    default="exp_msg_chn",
    help="Experiment name as in workspace directory",
)
parser.add_argument(
    "--chkpt",
    action="store",
    dest="chkpt",
    default=None,
    nargs="?",  # None or number
    help="Checkpoint number to load",
)

parser.add_argument(
    "--set",
    action="store",
    dest="set",
    default="selval",
    type=str,
    nargs="?",
    help='Which set to evaluate on "val", "selval" or "test"',
)
args = parser.parse_args()

# Path to the workspace directory
exp_dir = os.path.join(
    "results", time.strftime("%m%d%H%M%S", time.localtime(time.time()))
)
os.makedirs(exp_dir)

# # Add the experiment's folder to python path

# Read parameters file
with open(os.path.join("config", args.params + ".json"), "r") as fp:
    params = json.load(fp)

# Use GPU or not

dataloaders, dataset_sizes, model, t, optimizer, lr_decay = build_runtime(
    params, args, kitti_data_loader
)

mode, sets = mode_and_sets(args)

# Objective function
objective1 = locals()[params["loss1"]]()
objective2 = locals()[params["loss2"]]()

mytrainer = build_trainer(
    (dataloaders, dataset_sizes, model, t, optimizer, lr_decay),
    params,
    (objective1, objective2),
    exp_dir,
    sets,
    args.chkpt,
)

if mode == "train":
    # train the network
    net = mytrainer.train(params["num_epochs"])  #
else:
    net = mytrainer.evaluate()
