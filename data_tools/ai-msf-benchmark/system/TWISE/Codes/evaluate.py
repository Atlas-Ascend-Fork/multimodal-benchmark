# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import argparse
import logging
import os
import sys
import time

from dataloaders.kitti_loader import owidth, input_options, KittiDepth
from metrics import AverageMeter, Result
from model import MultiResNetworkAvgpoolDiffspatialsizes
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
import helper
import imageio
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from utils import smooth2chandep


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class IterateParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    mode: object
    runtime_args: object
    loader1: object
    model: object
    optimizer: object
    logger: object
    epoch: object
    curr_step: object



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


LOGGER = logging.getLogger(__name__)

params = {
    "depth_maxrange": 80.0,
    "threshold": 100,
}

parser = argparse.ArgumentParser(description="Sparse-to-Dense")
parser.add_argument(
    "-w",
    "--workers",
    default=1,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)

parser.add_argument(
    "--data-folder",
    default="/home/imransai/Documents/Depth_SuperResolution/TWISE_Public/TWISE",
    type=str,
    metavar="PATH",
    help="data folder (default: none)",
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="rgbd",
    choices=input_options,
    help="input: | ".join(input_options),
)
parser.add_argument(
    "--rank-metric",
    type=str,
    default="rmse",
    choices=[m for m in dir(Result()) if not m.startswith("_")],
    help="metrics for which best result is sbatch_datacted",
)
parser.add_argument(
    "-m",
    "--train-mode",
    type=str,
    default="dense",
    choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
    help="dense | sparse | photo | sparse+photo | dense+photo",
)
parser.add_argument(
    "--save_imageflag", default="F", type=str, help="save images for test"
)
parser.add_argument("--cpu", action="store_true", help="run on cpu")

args = parser.parse_args()

MODEL_NAME = "TWISE_gamma2.5"

modelcheckpt_dir = os.path.join("../pretrained_models", MODEL_NAME)
modellog_dir = os.path.join("./log_dir", MODEL_NAME)

args.result = modelcheckpt_dir
args.use_rgb = True
args.use_d = "d" in args.input
args.use_g = False
LOGGER.info("%s", args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
LOGGER.info("%s", "=> using '{}' for computation.".format(device))


_ITERATE_PARAMETERS_SCHEMA = _ParameterSchema(
    IterateParameters,
    (
        "mode",
        "runtime_args",
        "loader1",
        "model",
        "optimizer",
        "logger",
        "epoch",
        "curr_step",
    ),
    {"curr_step": None},
    frozenset(()),
    8,
    None,
    None,
)


def _predict_depth(batch_data, model):
    color_image = batch_data["rgb"]
    sparse_depth = batch_data["d"] / params["depth_maxrange"]
    prediction, _, _ = model(sparse_depth, color_image)
    return smooth2chandep(prediction, params=params, device=device)


def _save_completion(batch_data, model, runtime_args, step):
    save_path = os.path.join(runtime_args.data_folder, "Eval_DepthModel", MODEL_NAME)
    os.makedirs(save_path, exist_ok=True)
    prediction = _predict_depth(batch_data, model).detach().cpu().numpy().squeeze()
    height, _ = np.shape(prediction)
    padding = np.zeros((352 - height, owidth))
    prediction = np.concatenate((padding, prediction))
    filename = "%010d.png" % step
    imageio.imwrite(os.path.join(save_path, filename), np.uint16(prediction * 256))
    LOGGER.info("Finished writing sample %s", filename)


def _evaluate_batch(batch_data, model, runtime_args, filename):
    start = time.time()
    prediction = _predict_depth(batch_data, model)
    gpu_time = time.time() - start
    if runtime_args.save_imageflag == "T":
        save_path = os.path.join("Save_ImageResults", MODEL_NAME)
        os.makedirs(save_path, exist_ok=True)
        image = Image.fromarray(prediction.detach().cpu().numpy().squeeze())
        image.convert("L").save(os.path.join(save_path, filename))
    result = Result()
    result.evaluate(prediction.data, batch_data["gt"].data, 0)
    return result, gpu_time


def _evaluate_iteration(context, batch_data, filename):
    with torch.no_grad():
        result, gpu_time = _evaluate_batch(
            batch_data, context["model"], context["runtime_args"], filename
        )
    batch_size = next(iter(batch_data.values())).size(0)
    for meter in context["meters"]:
        meter.update(result, gpu_time, context["data_time"], batch_size)
    context["logger"].conditional_print(
        context["mode"], context["index"], context["epoch"], 0,
        context["loader_length"], *context["meters"],
    )
    average = context["logger"].conditional_save_info(
        context["mode"], context["meters"][1], context["epoch"]
    )
    is_best = context["logger"].rank_conditional_save_best(
        context["mode"], average, context["epoch"]
    )
    return average, is_best


def iterate(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _ITERATE_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    mode, runtime_args, loader1, model, optimizer, logger, epoch, curr_step = (
        bound_parameters
    )
    block_average_meter, average_meter = AverageMeter(), AverageMeter()
    meters = [block_average_meter, average_meter]
    if curr_step is None:
        curr_step = 0

    # switch to appropriate mode
    if not (mode in ["train", "val", "test", "val_selection", "test_completion"]):
        raise AssertionError("unsupported mode: {}".format(mode))

    model.eval()
    learning_rate = 0
    average = is_best = None

    fns = natsorted(
        os.listdir("../Data/depth_selection/val_selection_cropped/velodyne_raw")
    )
    for i, batch_data in tqdm(enumerate(loader1)):
        batch_data = {
            key: val.to(device) for key, val in batch_data.items() if val is not None
        }
        start = time.time()
        data_time = time.time() - start
        if mode == "test_completion":
            _save_completion(batch_data, model, runtime_args, curr_step)
            curr_step += 1
            continue
        if mode in ["test", "val_selection"]:
            average, is_best = _evaluate_iteration(
                {
                    "model": model,
                    "runtime_args": runtime_args,
                    "meters": meters,
                    "data_time": data_time,
                    "logger": logger,
                    "mode": mode,
                    "index": i,
                    "epoch": epoch,
                    "loader_length": len(loader1),
                },
                batch_data,
                fns[i],
            )
            curr_step += 1

    return average, is_best


def main():
    global args
    checkpoint = None
    is_eval = False
    args.curr_step = 0
    args_new = args

    complete_modelname = os.path.join(modelcheckpt_dir, "model_best.pth.tar")
    if os.path.isfile(complete_modelname):
        LOGGER.info("=> loading checkpoint '%s' ...", complete_modelname)
        checkpoint = torch.load(complete_modelname, map_location=device)
        args = checkpoint["args"]
        args.data_folder = args_new.data_folder
        args.save_imageflag = args_new.save_imageflag
        is_eval = True
        LOGGER.info("%s", "Completed.")
    else:
        LOGGER.info("%s", "No model found at '{}'".format(modelcheckpt_dir))
        return

    LOGGER.info("%s", "=> creating model and optimizer ...")

    model = MultiResNetworkAvgpoolDiffspatialsizes().to(device)

    model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        model_named_params, lr=args.lr, weight_decay=args.weight_decay
    )
    LOGGER.info("%s", "completed.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        LOGGER.info("%s", "=> checkpoint state loaded.")

    model = torch.nn.DataParallel(model)

    # Data loading code
    LOGGER.info("%s", "=> creating data loaders ... ")

    val_dataset = KittiDepth("val_selection", args, trainvalsplit=0.95)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )  # set batch size to be 1 for validation
    LOGGER.info("%s", "\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = helper.logger(args, output_directory=modelcheckpt_dir)
    if checkpoint is not None:
        logger.best_result = checkpoint["best_result"]
    LOGGER.info("%s", "=> logger created.")

    if is_eval:
        LOGGER.info("%s", "=> starting model evaluation ...")
        result, is_best = iterate(
            "val_selection", args, val_loader, model, None, logger, checkpoint["epoch"]
        )
        return
    # main loop


if __name__ == "__main__":
    main()
