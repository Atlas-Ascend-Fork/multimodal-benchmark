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
import tqdm


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters
from shared.checkpoint import checkpoint_state as _checkpoint_state
from shared.checkpoint import load_checkpoint as _load_checkpoint
from shared.checkpoint import load_part_checkpoint, save_checkpoint


class TrainerInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    model: object
    params_to_update: object
    model_fn_train: object
    optimizer: object
    ckpt_dir: object
    lr_scheduler: object
    model_fn_val: object
    tb_log: object
    eval_frequency: object
    grad_norm_clip: object


class TrainerTrainParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    start_it: object
    start_epoch: object
    n_epochs: object
    train_loader: object
    val_loader: object
    stop_thres: object


logging.getLogger(__name__).addHandler(logging.StreamHandler())
cur_logger = logging.getLogger(__name__)


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    return _checkpoint_state(model, optimizer, epoch, it)


def load_checkpoint(
    model=None, optimizer=None, filename="checkpoint", logger=cur_logger
):
    return _load_checkpoint(
        model, optimizer, filename, logger, strict=False, report_keys=True
    )


def load_part_ckpt(model, filename, logger=cur_logger, total_keys=-1):
    load_part_checkpoint(model, filename, logger, total_keys)


_TRAINER_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    TrainerInitParameters,
    (
        "model",
        "params_to_update",
        "model_fn_train",
        "optimizer",
        "ckpt_dir",
        "lr_scheduler",
        "model_fn_val",
        "tb_log",
        "eval_frequency",
        "grad_norm_clip",
    ),
    {"eval_frequency": 1, "grad_norm_clip": 1.0},
    frozenset(()),
    10,
    None,
    None,
)
_TRAINER_TRAIN_PARAMETERS_SCHEMA = _ParameterSchema(
    TrainerTrainParameters,
    ("start_it", "start_epoch", "n_epochs", "train_loader", "val_loader", "stop_thres"),
    {"val_loader": None, "stop_thres": 5},
    frozenset(()),
    6,
    None,
    None,
)


class Trainer:
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRAINER_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            model,
            params_to_update,
            model_fn_train,
            optimizer,
            ckpt_dir,
            lr_scheduler,
            model_fn_val,
            tb_log,
            eval_frequency,
            grad_norm_clip,
        ) = bound_parameters
        self.model = model
        self.params_to_update = params_to_update
        self.model_fn = model_fn_train
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_fn_eval = model_fn_val
        self.ckpt_dir = ckpt_dir
        self.eval_frequency = eval_frequency
        self.tb_log = tb_log
        self.grad_norm_clip = grad_norm_clip

    @staticmethod
    def _update_early_stopping(train_loss, val_loss, previous, counter, threshold):
        previous_train, previous_val = previous
        if previous_train == -1 or previous_val == -1:
            return (train_loss, val_loss), counter, False
        if train_loss >= previous_train or val_loss <= previous_val:
            return previous, 0, False
        counter += 1
        cur_logger.info("Bad train")
        should_stop = counter > threshold
        if should_stop:
            cur_logger.info("Early stopping")
        return previous, counter, should_stop

    def eval_epoch(self, val_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = []
        nan_dict = {}

        # eval one epoch
        with tqdm.tqdm(val_loader, leave=False, desc="val") as tbar:
            for data in tbar:
                loss, tb_dict, disp_dict = self.model_fn_eval(self.model, data)
                if loss > 0:
                    total_loss.append(loss.item())
                tbar.set_postfix(disp_dict)
                self._accumulate_eval_metrics(eval_dict, nan_dict, tb_dict)

        # statistics this epoch
        for k, _ in eval_dict.items():
            eval_dict[k] = eval_dict.get(k) / (len(val_loader) - nan_dict.get(k, 0))

        cur_performance = 0
        if "recalled_cnt" in eval_dict:
            eval_dict["recall"] = eval_dict.get("recalled_cnt") / max(
                eval_dict.get("gt_cnt"), 1
            )
            cur_performance = eval_dict.get("recall")
        elif "iou" in eval_dict:
            cur_performance = eval_dict.get("iou")

        return (
            sum(total_loss) / len(total_loss) if len(total_loss) > 0 else 0,
            eval_dict,
            cur_performance,
        )

    def train(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRAINER_TRAIN_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        start_it, start_epoch, n_epochs, train_loader, val_loader, stop_thres = (
            bound_parameters
        )
        eval_frequency = self.eval_frequency if self.eval_frequency > 0 else 1

        it = start_it
        min_val_loss = float("inf")
        prev_train_loss = -1
        prev_val_loss = -1
        counter = 0
        scaler = torch.cuda.amp.GradScaler()

        with tqdm.trange(start_epoch, n_epochs, desc="epochs") as tbar, tqdm.tqdm(
            total=len(train_loader), leave=False, desc="train"
        ) as pbar:

            for epoch in tbar:
                bars = (tbar, pbar)
                it, train_loss_epoch = self._train_epoch(train_loader, scaler, it, bars)
                trained_epoch = epoch + 1
                self._step_learning_rate(trained_epoch)
                if self.tb_log is not None:
                    self.tb_log.add_scalar(
                        "train_loss_epoch", train_loss_epoch, trained_epoch
                    )
                pbar.close()
                self._save_epoch_checkpoint(trained_epoch, it)
                val_loss_epoch = self._evaluate_epoch(
                    epoch, trained_epoch, val_loader, eval_frequency
                )
                if val_loss_epoch is None:
                    pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc="train")
                    pbar.set_postfix(dict(total_it=it))
                    continue
                previous = (prev_train_loss, prev_val_loss)
                previous, counter, should_stop = self._update_early_stopping(
                    train_loss_epoch, val_loss_epoch, previous, counter, stop_thres
                )
                prev_train_loss, prev_val_loss = previous
                min_val_loss = self._save_best_model(val_loss_epoch, min_val_loss)
                if should_stop:
                    break
                pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc="train")
                pbar.set_postfix(dict(total_it=it))

    @staticmethod
    def _accumulate_eval_metrics(eval_dict, nan_dict, metrics):
        for key, value in metrics.items():
            if math.isnan(value):
                nan_dict[key] = nan_dict.get(key, 0) + 1
            else:
                eval_dict[key] = eval_dict.get(key, 0) + value

    def _log_training_metrics(self, loss, metrics, iteration):
        if self.tb_log is None:
            return
        self.tb_log.add_scalar("train_loss", loss, iteration)
        for key, value in metrics.items():
            self.tb_log.add_scalar("train_" + key, value, iteration)

    def _log_evaluation_metrics(self, loss, metrics, epoch):
        if self.tb_log is None:
            return
        self.tb_log.add_scalar("val_loss_epoch", loss, epoch)
        for key, value in metrics.items():
            self.tb_log.add_scalar("val_" + key, value, epoch)

    def _train_epoch(self, loader, scaler, iteration, bars):
        epoch_bar, progress_bar = bars
        self.model.train()
        losses = []
        for batch in loader:
            iteration += 1
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, metrics, display = self.model_fn(self.model, batch)
            if loss > 0:
                scaler.scale(loss).backward()
                clip_grad_norm_(self.params_to_update, self.grad_norm_clip)
                losses.append(loss.item())
                scaler.step(self.optimizer)
                scaler.update()
                self._log_training_metrics(loss.item(), metrics, iteration)
            progress_bar.update()
            progress_bar.set_postfix(dict(total_it=iteration))
            epoch_bar.set_postfix(display)
            epoch_bar.refresh()
        average_loss = sum(losses) / len(losses) if losses else 0
        return iteration, average_loss

    def _step_learning_rate(self, trained_epoch):
        if self.lr_scheduler is None:
            return
        self.lr_scheduler.step()
        if self.tb_log is None:
            return
        learning_rate = self.lr_scheduler.get_last_lr()
        if isinstance(learning_rate, list):
            self.tb_log.add_scalar("learning_rate_1", learning_rate[0], trained_epoch)
            self.tb_log.add_scalar("learning_rate_2", learning_rate[-1], trained_epoch)
            return
        self.tb_log.add_scalar("learning_rate", learning_rate, trained_epoch)

    def _save_epoch_checkpoint(self, trained_epoch, iteration):
        name = os.path.join(self.ckpt_dir, "checkpoint_epoch_%d" % trained_epoch)
        state = checkpoint_state(self.model, self.optimizer, trained_epoch, iteration)
        save_checkpoint(state, filename=name)

    def _evaluate_epoch(self, epoch, trained_epoch, loader, frequency):
        if epoch % frequency != 0 or loader is None:
            return None
        with torch.set_grad_enabled(False):
            loss, metrics, _ = self.eval_epoch(loader)
        self._log_evaluation_metrics(loss, metrics, trained_epoch)
        return loss

    def _save_best_model(self, val_loss, min_val_loss):
        if val_loss >= min_val_loss:
            return min_val_loss
        name = os.path.join(self.ckpt_dir, "best_model")
        model = (
            self.model.module
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model
        )
        save_checkpoint({"model_state": model.state_dict()}, filename=name)
        return val_loss
