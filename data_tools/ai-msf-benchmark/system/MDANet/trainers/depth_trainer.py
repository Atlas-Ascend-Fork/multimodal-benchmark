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
from typing import NamedTuple
import os.path
import time

from tqdm import tqdm
from trainers.trainer import Trainer  # from CVLPyDL repo
import torch

from utils.average_meter import AverageMeter
from utils.save_tensor_to_image import save_tensor_to_image


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class KittidepthtrainerInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    net: object
    params: object
    optimizer: object
    objective1: object
    objective2: object
    lr_scheduler: object
    dataloaders: object
    dataset_sizes: object
    workspace_dir: object
    sets: object
    use_load_checkpoint: object
    exp_dir: object


#

#


err_metrics = ["MAE()", "RMSE()", "IMae()", "IRmse()"]


_KITTIDEPTHTRAINER_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    KittidepthtrainerInitParameters,
    (
        "net",
        "params",
        "optimizer",
        "objective1",
        "objective2",
        "lr_scheduler",
        "dataloaders",
        "dataset_sizes",
        "workspace_dir",
        "sets",
        "use_load_checkpoint",
        "exp_dir",
    ),
    {"sets": None, "use_load_checkpoint": None, "exp_dir": None},
    frozenset(()),
    12,
    None,
    None,
)


class KittiDepthTrainer(Trainer):
    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _KITTIDEPTHTRAINER_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            net,
            params,
            optimizer,
            objective1,
            objective2,
            lr_scheduler,
            dataloaders,
            dataset_sizes,
            workspace_dir,
            sets,
            use_load_checkpoint,
            exp_dir,
        ) = bound_parameters
        if sets is None:
            sets = ["train", "val"]
        super(KittiDepthTrainer, self).__init__(
            net,
            optimizer,
            lr_scheduler,
            objective1,
            objective2,
            use_gpu=params["use_gpu"],
            workspace_dir=workspace_dir,
        )

        self.lr_scheduler = lr_scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.use_load_checkpoint = use_load_checkpoint

        self.params = params
        self.save_chkpt_each = params["save_chkpt_each"]
        self.sets = sets
        self.save_images = params["save_out_imgs"]
        self.load_rgb = params["load_rgb"] if "load_rgb" in params else False
        self.model_name = params["model"]
        self.use_bin_loss = True if params["loss1"] == "BINLoss" else False

        self.exp_dir = exp_dir

        for s in self.sets:
            self.stats[s + "_loss"] = []

    # Training Function #######

    @staticmethod
    def _update_error_metrics(outputs, target, error_meters, batch_size):
        for metric_name in err_metrics:
            metric = globals()["Deltas" if "Delta" in metric_name else metric_name]()
            error = metric(outputs, target)
            if "Delta" in metric_name:
                for index, name in enumerate(("Delta1", "Delta2", "Delta3")):
                    error_meters[name].update(error[index], batch_size)
                break
            error_meters[metric_name].update(error.item(), batch_size)

    @staticmethod
    def _update_min_errors(error_meters, minimum_errors, epoch, minimum_epoch):
        if error_meters["RMSE()"].avg >= minimum_errors["RMSE()"]:
            return minimum_epoch
        for metric_name in err_metrics:
            minimum_errors[metric_name] = error_meters[metric_name].avg
        return epoch

    @staticmethod
    def _write_evaluation_header(logger, model_name):
        logger.write(f"[Eval] - {model_name}\n")
        logger.write("=" * 15 * (4 + len(err_metrics)) + "\n")
        logger.write(f'{"Epoch":^15}{"Val Loss":^15}')
        for metric in err_metrics:
            logger.write(f"{metric:^15}")
        logger.write(f'{"Time":^15}{"Time_av":^15}\n')

    def train(self, max_epochs):
        logger = open(os.path.join(self.exp_dir, "train.log"), mode="w")

        logger.write(f"[Train] - {self.model_name}\n\n")

        logger.write("[Experiment Parameters]\n")
        logger.write(
            "Number of model parameters: {}\n".format(
                sum([p.data.nelement() for p in self.net.parameters()])
            )
        )

        for k, v in self.params.items():
            logger.write("{0:<22s} : {1:}\n".format(k, v))

        self._load_training_checkpoint(logger)

        logger.write("\n")
        logger.write("=" * 15 * 4)
        logger.write("\n")
        logger.write(f'{"Epoch":^15}{"lr":^15}{"Loss":^15}{"Time(h)":^15}\n')

        for epoch in range(
            self.epoch, max_epochs + 1
        ):  # range function returns max_epochs-1

            logger.write("-" * 15 * 4)
            logger.write("\n")
            logger.flush()

            start_epoch_time = time.time()

            self.epoch = epoch

            # Decay Learning Rate
            self.lr_scheduler.step(epoch)  # LR decay
            logger.write(f'{epoch:^15}{self.optimizer.param_groups[0]["lr"]:^15.4g}')

            # Train the epoch
            loss_meter = self.train_epoch(logger)

            # Add the average loss for this epoch to stats
            for s in self.sets:
                self.stats[s + "_loss"].append(loss_meter[s].avg)

            # Save checkpoint
            if self.use_save_checkpoint and self.epoch % self.save_chkpt_each == 0:
                self.save_checkpoint()

            end_epoch_time = time.time()
            epoch_duration = end_epoch_time - start_epoch_time
            logger.write(f"{epoch_duration / 3600:^15.2f}\n")

        logger.write("=" * 15 * 4)
        logger.write("\n")

        # Save the final model
        torch.save(self.net, self.workspace_dir + "/final_model.pth")
        logger.close()
        return self.net

    def train_epoch(self, logger):

        loss_meter = {}
        for s in self.sets:
            loss_meter[s] = AverageMeter()

        for s in self.sets:
            # Iterate over data.
            for data in tqdm(self.dataloaders[s]):
                inputs_d, c, item_idxs, inputs_rgb, gt_depth = data

                inputs_d = inputs_d.cuda()
                gt_depth = gt_depth.cuda()
                inputs_rgb = inputs_rgb.cuda()

                outputs = self.net(inputs_d, inputs_rgb)

                # Calculate loss for valid pixel in the ground truth
                mse_loss11 = self.objective1(outputs[0], gt_depth)
                mse_loss12 = self.objective1(outputs[1], gt_depth)
                mse_loss14 = self.objective1(outputs[2], gt_depth)

                mae_loss11 = self.objective2(outputs[0], gt_depth)
                mae_loss12 = self.objective2(outputs[1], gt_depth)
                mae_loss14 = self.objective2(outputs[2], gt_depth)

                if self.epoch < 41:
                    gamma = 1
                else:
                    gamma = 0

                if self.epoch < 6:
                    delta = 1
                elif self.epoch < 11:
                    delta = 0.1
                else:
                    delta = 0

                mse_loss = delta * mse_loss14 + delta * mse_loss12 + mse_loss11
                mae_loss = delta * mae_loss14 + delta * mae_loss12 + mae_loss11

                loss_value = mse_loss + gamma * mae_loss

                # backward + optimize only if in training phase
                loss_value.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # statistics
                loss_meter[s].update(mse_loss11.item(), inputs_d.size(0))

            logger.write(f"{loss_meter[s].avg:^15.8f}")
            torch.cuda.empty_cache()

        return loss_meter

    def evaluate(self, save_output):
        log_path = os.path.join(
            os.path.split(self.use_load_checkpoint)[0], "..", "eval.log"
        )
        epoch = self._load_evaluation_checkpoint()
        self.net.train(False)
        meters = (
            {set_name: AverageMeter() for set_name in self.sets},
            {metric: AverageMeter() for metric in err_metrics},
            AverageMeter(),
        )
        with open(log_path, mode="w") as logger:
            self._write_evaluation_header(logger, self.model_name)
            with torch.no_grad():
                for set_name in self.sets:
                    self._evaluate_set(set_name, meters, (logger, epoch, save_output))

    def evaluate_folder(self):
        logger = open(
            os.path.join(self.use_load_checkpoint, "..", "eval_folder.log"), mode="w"
        )

        min_err = {}
        min_epoch = 0
        for m in err_metrics:
            min_err[m] = float("Inf")

        # Load last save checkpoint
        if self.use_load_checkpoint is not None and isinstance(
            self.use_load_checkpoint, str
        ):

            logger.write(f"[Eval] - {self.model_name}\n\n")
            logger.write("=" * 15 * (4 + len(err_metrics)))
            logger.write("\n")
            logger.write(f'{"Epoch":^15}{"Val Loss":^15}')
            for m in err_metrics:
                logger.write(f"{m:^15}")
            logger.write(f'{"Time":^15}{"Time_av":^15}\n')
            logger.flush()

            for ckpt in sorted(os.listdir(self.use_load_checkpoint))[-20:]:
                min_epoch = self._evaluate_checkpoint(ckpt, min_err, min_epoch, logger)
            logger.write("=" * 15 * (4 + len(err_metrics)))
            logger.write("\n")

        logger.write("\nMin Error show:\n")
        logger.write(f"Min Epoch: {min_epoch}\n")
        for m in err_metrics:
            logger.write("Min Error: [{}]: {:.8f}\n".format(m, min_err.get(m)))
        logger.close()

    def _load_training_checkpoint(self, logger):
        checkpoint = self.use_load_checkpoint
        if checkpoint is None:
            return
        if isinstance(checkpoint, str):
            logger.write("loading checkpoint from: \n" + checkpoint)
            loaded = self.load_checkpoint(checkpoint)
            if loaded:
                self.epoch = int(checkpoint[-10:-8]) + 1
            logger.write(
                "Checkpoint was loaded successfully!\n"
                if loaded
                else "Evaluating using initial parameters\n"
            )
            return
        if not isinstance(checkpoint, int) or checkpoint == 0:
            return
        label = str(checkpoint) if checkpoint > 0 else "last"
        logger.write("=> Loading checkpoint {} ...\n".format(label))
        loaded = (
            self.load_checkpoint(checkpoint)
            if checkpoint > 0
            else self.load_checkpoint()
        )
        logger.write(
            "Checkpoint was loaded successfully!\n"
            if loaded
            else "Evaluating using initial parameters\n"
        )

    def _normalize_depth(self, outputs, target):
        outputs[outputs == 0] = -1
        if not self.load_rgb:
            outputs[outputs == outputs[0, 0, 0, 0]] = -1
        target[target == 0] = -1
        if self.params["invert_depth"]:
            outputs = 1 / outputs
            target = 1 / target
        outputs[outputs == -1] = 0
        target[target == -1] = 0
        scale = self.params["data_normalize_factor"] / 256
        return outputs * scale, target * scale

    def _save_folder_output(self, set_name, outputs, item_indices):
        if set_name != "test":
            return
        output_path = os.path.join(
            self.workspace_dir, set_name + "_output_epoch_" + str(self.epoch)
        )
        save_tensor_to_image(outputs.data * 256, item_indices, output_path)

    def _evaluate_folder_batch(self, data, set_name, meters):
        loss_meter, error_meters, times = meters
        start_time = time.time()
        inputs_d, _, item_indices, inputs_rgb, target = data
        inputs_d = inputs_d.cuda()
        target = target.cuda()
        inputs_rgb = inputs_rgb.cuda()
        outputs = self.net(inputs_d, inputs_rgb)
        if self.use_bin_loss:
            outputs = outputs[-1]
        elif len(outputs) > 1:
            outputs = outputs[0]
        duration = time.time() - start_time
        times.update(duration / inputs_d.size(0), inputs_d.size(0))
        if set_name in {"selval", "val", "test"}:
            loss_value = self.objective1(outputs, target, self.epoch)
            loss_meter[set_name].update(loss_value.item(), inputs_d.size(0))
            outputs, target = self._normalize_depth(outputs, target)
            self._update_error_metrics(outputs, target, error_meters, inputs_d.size(0))
        return outputs, item_indices

    # Evaluation Function #######
    def _load_evaluation_checkpoint(self):
        if self.use_load_checkpoint is None:
            return 0
        epoch = 0
        if isinstance(self.use_load_checkpoint, str):
            if self.use_load_checkpoint[-12:-8].isdigit():
                epoch = int(self.use_load_checkpoint[-12:-8])
            self.load_checkpoint(self.use_load_checkpoint)
        elif self.use_load_checkpoint > 0:
            self.load_checkpoint(self.use_load_checkpoint)
        elif self.use_load_checkpoint == -1:
            self.load_checkpoint()
        return epoch

    # Evaluation Function #######
    def _evaluate_set(self, set_name, meters, context):
        logger, epoch, save_output = context
        loss_meter, errors, times = meters
        started = time.time()
        logger.write("-" * 15 * (4 + len(err_metrics)) + "\n")
        for data in tqdm(self.dataloaders[set_name]):
            outputs, item_indices = self._evaluate_folder_batch(data, set_name, meters)
            if save_output:
                output_path = os.path.join(
                    os.path.split(self.use_load_checkpoint)[0],
                    "..",
                    set_name + "_output_epoch_" + str(epoch),
                )
                save_tensor_to_image(outputs.data * 256, item_indices, output_path)
        average_time = (time.time() - started) / len(self.dataloaders[set_name].dataset)
        logger.write(f"{epoch:^15}{loss_meter[set_name].avg:^15.8f}")
        for metric in err_metrics:
            logger.write(f"{errors[metric].avg:^15.8f}")
        logger.write(f"{times.avg:^15.4f}{average_time:^15.4f}\n")
        logger.write("=" * 15 * (4 + len(err_metrics)) + "\n")
        torch.cuda.empty_cache()

    def _evaluate_checkpoint_set(self, set_name, ckpt, meters, logger):
        loss_meter, err, times = meters
        epoch = int(ckpt[15:19])
        start_time_value = time.time()
        logger.write("-" * 15 * (4 + len(err_metrics)) + "\n")
        logger.flush()
        for data in tqdm(self.dataloaders[set_name]):
            outputs, item_idxs = self._evaluate_folder_batch(data, set_name, meters)
            self._save_folder_output(set_name, outputs, item_idxs)
        average_time = (time.time() - start_time_value) / len(
            self.dataloaders[set_name].dataset
        )
        logger.write(f"{epoch:^15}{loss_meter[set_name].avg:^15.8f}")
        for metric in err_metrics:
            logger.write(f"{err.get(metric).avg:^15.8f}")
        logger.write(f"{times.avg:^15.4f}{average_time:^15.4f}\n")
        torch.cuda.empty_cache()
        return epoch

    def _evaluate_checkpoint(self, ckpt, min_err, min_epoch, logger):
        self.load_checkpoint(os.path.join(self.use_load_checkpoint, ckpt))
        self.net.train(False)
        meters = (
            {set_name: AverageMeter() for set_name in self.sets},
            {metric: AverageMeter() for metric in err_metrics},
            AverageMeter(),
        )
        with torch.no_grad():
            for set_name in self.sets:
                epoch = self._evaluate_checkpoint_set(set_name, ckpt, meters, logger)
                min_epoch = self._update_min_errors(
                    meters[1], min_err, epoch, min_epoch
                )
        return min_epoch
