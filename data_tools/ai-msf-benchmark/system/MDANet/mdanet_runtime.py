# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Model and optimizer setup for MDANet entry points."""

from collections import namedtuple
import importlib

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from shared.parameters import bind_named as _bind_named


RuntimeComponents = namedtuple(
    "RuntimeComponents",
    "dataloaders dataset_sizes model trainer_module optimizer scheduler",
)


def build_runtime(parameters, arguments, kitti_data_loader):
    """Build loaders, network, trainer module, optimizer, and scheduler."""
    loader_name = parameters.get("data_loader", "KittiDataLoader")
    loaders = {
        "KittiDataLoader": kitti_data_loader,
        "kitti_data_loader": kitti_data_loader,
    }
    dataloaders, dataset_sizes = loaders.get(
        loader_name, kitti_data_loader
    )(parameters)
    model_name = arguments.model or parameters["model"]
    model_module = importlib.import_module("modules." + model_name)
    model = nn.DataParallel(model_module.network().cuda())
    trainer_module = importlib.import_module("trainers." + parameters["trainer"])
    trainable = filter(lambda value: value.requires_grad, model.parameters())
    optimizer = getattr(optim, parameters["optimizer"])(
        trainable,
        lr=parameters["lr"],
        weight_decay=parameters["weight_decay"],
    )
    decay = lr_scheduler.StepLR(
        optimizer,
        step_size=parameters["lr_decay_step"],
        gamma=parameters["lr_decay"],
    )
    return RuntimeComponents(
        dataloaders, dataset_sizes, model, trainer_module, optimizer, decay
    )


def mode_and_sets(arguments):
    if arguments.mode == "train":
        return "train", ["train"]
    if arguments.mode == "eval":
        return "eval", [arguments.set]
    raise ValueError("unsupported mode: {}".format(arguments.mode))


def build_trainer(*call_args, **call_kwargs):
    (
        runtime,
        parameters,
        objectives,
        experiment_dir,
        sets,
        checkpoint,
    ) = _bind_named(
        call_args,
        call_kwargs,
        ('runtime', 'parameters', 'objectives', 'experiment_dir', 'sets', 'checkpoint'),
    )
    dataloaders, dataset_sizes, model, trainer_module, optimizer, decay = runtime
    objective1, objective2 = objectives
    return trainer_module.KittiDepthTrainer(
        model,
        parameters,
        optimizer,
        objective1,
        objective2,
        decay,
        dataloaders,
        dataset_sizes,
        workspace_dir=experiment_dir,
        sets=sets,
        use_load_checkpoint=checkpoint,
        exp_dir=experiment_dir,
    )
