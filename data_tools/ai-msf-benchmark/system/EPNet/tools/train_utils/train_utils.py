# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import logging
import math
import os

from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import tqdm


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters
from shared.checkpoint import checkpoint_state as _checkpoint_state
from shared.checkpoint import load_checkpoint as _load_checkpoint
from shared.checkpoint import load_part_checkpoint, save_checkpoint


class TrainerInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    model: object
    model_fn: object
    optimizer: object
    ckpt_dir: object
    lr_scheduler: object
    bnm_scheduler: object
    model_fn_eval: object
    tb_log: object
    eval_frequency: object
    lr_warmup_scheduler: object
    warmup_epoch: object
    grad_norm_clip: object


class TrainerTrainParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    start_it: object
    start_epoch: object
    n_epochs: object
    train_loader: object
    test_loader: object
    ckpt_save_interval: object
    lr_scheduler_each_iter: object


LOGGER = logging.getLogger(__name__)

logging.getLogger(__name__).addHandler(logging.StreamHandler())
cur_logger = logging.getLogger(__name__)


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model).__name__)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched.LRScheduler):
    def __init__(self, optimizer, t_max, eta_min=0, last_epoch=-1):
        self.t_max = t_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 - math.cos(math.pi * self.last_epoch / self.t_max))
            / 2
            for base_lr in self.base_lrs
        ]


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    return _checkpoint_state(model, optimizer, epoch, it)


def load_checkpoint(
    model=None, optimizer=None, filename="checkpoint", logger=cur_logger
):
    return _load_checkpoint(model, optimizer, filename, logger)


def load_part_ckpt(model, filename, logger=cur_logger, total_keys=-1):
    load_part_checkpoint(model, filename, logger, total_keys)


_TRAINER_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    TrainerInitParameters,
    (
        "model",
        "model_fn",
        "optimizer",
        "ckpt_dir",
        "lr_scheduler",
        "bnm_scheduler",
        "model_fn_eval",
        "tb_log",
        "eval_frequency",
        "lr_warmup_scheduler",
        "warmup_epoch",
        "grad_norm_clip",
    ),
    {
        "eval_frequency": 1,
        "lr_warmup_scheduler": None,
        "warmup_epoch": -1,
        "grad_norm_clip": 1.0,
    },
    frozenset(()),
    12,
    None,
    None,
)
_TRAINER_TRAIN_PARAMETERS_SCHEMA = _ParameterSchema(
    TrainerTrainParameters,
    (
        "start_it",
        "start_epoch",
        "n_epochs",
        "train_loader",
        "test_loader",
        "ckpt_save_interval",
        "lr_scheduler_each_iter",
    ),
    {"test_loader": None, "ckpt_save_interval": 5, "lr_scheduler_each_iter": False},
    frozenset(()),
    7,
    None,
    None,
)


class Trainer(object):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRAINER_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            model,
            model_fn,
            optimizer,
            ckpt_dir,
            lr_scheduler,
            bnm_scheduler,
            model_fn_eval,
            tb_log,
            eval_frequency,
            lr_warmup_scheduler,
            warmup_epoch,
            grad_norm_clip,
        ) = bound_parameters
        (
            self.model,
            self.model_fn,
            self.optimizer,
            self.lr_scheduler,
            self.bnm_scheduler,
            self.model_fn_eval,
        ) = (
            model,
            model_fn,
            optimizer,
            lr_scheduler,
            bnm_scheduler,
            model_fn_eval,
        )

        self.ckpt_dir = ckpt_dir
        self.eval_frequency = eval_frequency
        self.tb_log = tb_log
        self.lr_warmup_scheduler = lr_warmup_scheduler
        self.warmup_epoch = warmup_epoch
        self.grad_norm_clip = grad_norm_clip

    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = count = 0.0

        # eval one epoch
        for _, data in tqdm.tqdm(
            enumerate(d_loader, 0), total=len(d_loader), leave=False, desc="val"
        ):
            self.optimizer.zero_grad()

            loss, tb_dict, disp_dict = self.model_fn_eval(self.model, data)

            total_loss += loss.item()
            count += 1
            for k, v in tb_dict.items():
                eval_dict[k] = eval_dict.get(k, 0) + v

        # statistics this epoch
        for k, _ in eval_dict.items():
            eval_dict[k] = eval_dict.get(k) / max(count, 1)

        cur_performance = 0
        if "recalled_cnt" in eval_dict:
            eval_dict["recall"] = eval_dict.get("recalled_cnt") / max(
                eval_dict.get("gt_cnt"), 1
            )
            cur_performance = eval_dict.get("recall")
        elif "iou" in eval_dict:
            cur_performance = eval_dict.get("iou")

        return total_loss / count, eval_dict, cur_performance

    def train(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRAINER_TRAIN_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            start_it,
            start_epoch,
            n_epochs,
            train_loader,
            test_loader,
            ckpt_save_interval,
            lr_scheduler_each_iter,
        ) = bound_parameters
        eval_frequency = self.eval_frequency if self.eval_frequency > 0 else 1
        for name, value in self.model.named_parameters():
            LOGGER.info("%s %s %s", "***", name, value.requires_grad)
        it = start_it
        with tqdm.trange(start_epoch, n_epochs, desc="epochs") as tbar, tqdm.tqdm(
            total=len(train_loader), leave=False, desc="train"
        ) as pbar:
            for epoch in tbar:
                self._step_epoch_schedulers(epoch, it, lr_scheduler_each_iter)
                bars = (tbar, pbar)
                it = self._train_epoch(
                    train_loader, epoch, it, lr_scheduler_each_iter, bars
                )
                self._save_epoch_checkpoint(epoch, it, ckpt_save_interval)
                pbar.close()
                self._evaluate_epoch(epoch, it, test_loader, eval_frequency)
                pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc="train")
                pbar.set_postfix(dict(total_it=it))

        return None

    def _log_training_metrics(self, loss, learning_rate, metrics, iteration):
        if self.tb_log is None:
            return
        self.tb_log.add_scalar("train_loss", loss, iteration)
        self.tb_log.add_scalar("learning_rate", learning_rate, iteration)
        for key, value in metrics.items():
            self.tb_log.add_scalar("train_" + key, value, iteration)

    def _log_evaluation_metrics(self, loss, metrics, iteration):
        if self.tb_log is None:
            return
        self.tb_log.add_scalar("val_loss", loss, iteration)
        for key, value in metrics.items():
            self.tb_log.add_scalar("val_" + key, value, iteration)

    def _step_epoch_schedulers(self, epoch, iteration, scheduler_each_iter):
        if (
            self.lr_scheduler is not None
            and self.warmup_epoch <= epoch
            and not scheduler_each_iter
        ):
            self.lr_scheduler.step(epoch)
        if self.bnm_scheduler is None:
            return
        self.bnm_scheduler.step(iteration)
        if self.tb_log is not None:
            momentum = self.bnm_scheduler.lmbd(epoch)
            self.tb_log.add_scalar("bn_momentum", momentum, iteration)

    def _learning_rate(self, epoch, iteration, scheduler_each_iter):
        if scheduler_each_iter:
            self.lr_scheduler.step(iteration)
            learning_rate = float(self.optimizer.lr)
            if self.tb_log is not None:
                self.tb_log.add_scalar("learning_rate", learning_rate, iteration)
            return learning_rate
        if self.lr_warmup_scheduler is not None and epoch < self.warmup_epoch:
            self.lr_warmup_scheduler.step(iteration)
            return self.lr_warmup_scheduler.get_lr()[0]
        return self.lr_scheduler.get_lr()[0]

    def _train_epoch(self, train_loader, epoch, iteration, scheduler_each_iter, bars):
        epoch_bar, progress_bar = bars
        for batch in train_loader:
            LOGGER.info("%s", batch["sample_id"][0])
            learning_rate = self._learning_rate(epoch, iteration, scheduler_each_iter)
            loss, metrics, display = self._train_it(batch)
            iteration += 1
            display.update({"loss": loss, "lr": learning_rate})
            progress_bar.update()
            progress_bar.set_postfix(dict(total_it=iteration))
            epoch_bar.set_postfix(display)
            epoch_bar.refresh()
            self._log_training_metrics(loss, learning_rate, metrics, iteration)
        return iteration

    def _save_epoch_checkpoint(self, epoch, iteration, interval):
        trained_epoch = epoch + 1
        if trained_epoch % interval != 0:
            return
        name = os.path.join(self.ckpt_dir, "checkpoint_epoch_%d" % trained_epoch)
        state = checkpoint_state(self.model, self.optimizer, trained_epoch, iteration)
        save_checkpoint(state, filename=name)

    def _evaluate_epoch(self, epoch, iteration, loader, frequency):
        if epoch % frequency != 0 or loader is None:
            return
        with torch.set_grad_enabled(False):
            loss, metrics, _ = self.eval_epoch(loader)
        self._log_evaluation_metrics(loss, metrics, iteration)

    def _train_it(self, batch):
        self.model.train()

        self.optimizer.zero_grad()
        loss, tb_dict, disp_dict = self.model_fn(self.model, batch)
        LOGGER.info("%s %s", "===================", loss)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return loss.item(), tb_dict, disp_dict
