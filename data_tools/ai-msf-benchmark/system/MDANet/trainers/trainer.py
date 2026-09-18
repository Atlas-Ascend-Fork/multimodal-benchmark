# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import glob
import logging
import os

import torch


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class TrainerInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    net: object
    optimizer: object
    lr_scheduler: object
    objective1: object
    objective2: object
    use_gpu: object
    workspace_dir: object
    gpu_id: object


LOGGER = logging.getLogger(__name__)


_TRAINER_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    TrainerInitParameters,
    (
        "net",
        "optimizer",
        "lr_scheduler",
        "objective1",
        "objective2",
        "use_gpu",
        "workspace_dir",
        "gpu_id",
    ),
    {"use_gpu": True, "workspace_dir": None, "gpu_id": 0},
    frozenset(()),
    8,
    None,
    None,
)


class Trainer(object):
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function.
    """

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRAINER_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            net,
            optimizer,
            lr_scheduler,
            objective1,
            objective2,
            use_gpu,
            workspace_dir,
            gpu_id,
        ) = bound_parameters
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.objective1 = objective1
        self.objective2 = objective2
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.workspace_dir = None
        self.use_save_checkpoint = workspace_dir is not None

        if workspace_dir is not None:
            self.workspace_dir = os.path.expanduser(workspace_dir)
            if not os.path.exists(self.workspace_dir):
                os.makedirs(self.workspace_dir)

        self.epoch = 1
        self.training_time = 0.0
        self.stats = {}

        if self.use_gpu:
            self.net.cuda()

    def train(self, max_epochs):
        """Do training for the given number of epochs."""

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            self.train_epoch()
            if self.use_save_checkpoint:
                self.save_checkpoint()

        LOGGER.info("%s", "Finished training!")

    def train_epoch(self, logger):
        raise NotImplementedError

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net_type = type(self.net).__name__
        state = {
            "epoch": self.epoch,
            "net_type": net_type,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "stats": self.stats,
            "use_gpu": self.use_gpu,
            "training_time": self.training_time,
        }

        chkpt_path = os.path.join(self.workspace_dir, "checkpoints")
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)

        file_path = "{}/{}_ep{:04d}.pth.tar".format(chkpt_path, net_type, self.epoch)
        torch.save(state, file_path)

    def load_checkpoint(self, checkpoint=None):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        net_type = type(self.net).__name__

        chkpt_path = os.path.join(self.workspace_dir, "checkpoints")

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(
                glob.glob("{}/{}_ep*.pth.tar".format(chkpt_path, net_type))
            )
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                LOGGER.info("%s", "No matching checkpoint file found!\n")
                return False
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = "{}/{}_ep{:04d}.pth.tar".format(
                chkpt_path, net_type, checkpoint
            )
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        device = torch.device(
            "cuda:" + str(self.gpu_id) if torch.cuda.is_available() else "cpu"
        )
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)

        self.net.load_state_dict(checkpoint_dict["net"])

        if isinstance(checkpoint, int) or (checkpoint is None):
            self.epoch = checkpoint_dict["epoch"] + 1
            self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
            if "lr_scheduler" in checkpoint_dict:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.lr_scheduler.last_epoch = checkpoint_dict["epoch"] - 1
            self.stats = checkpoint_dict["stats"]
            self.use_gpu = checkpoint_dict["use_gpu"]

        return True
