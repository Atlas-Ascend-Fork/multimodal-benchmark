# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import NamedTuple
import csv
import logging
import os
import shutil
import time

from metrics import Result
import torch
import vis_utils
from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class LoggerConditionalSaveImgComparisonParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    mode: object
    i: object
    ele: object
    pred: object
    epoch: object


class LoggerConditionalPrintParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    split: object
    i: object
    epoch: object
    lr: object
    n_set: object
    blk_avg_meter: object
    avg_meter: object


class LoggerLossSummarizeV2Parameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    mode: object
    trainloss: object
    trainlossmaince: object
    valloss: object
    vallossmaince: object
    duration: object
    curr_step: object


class LoggerLossSummarizeV3Parameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    mode: object
    trainloss: object
    trainlossmaince: object
    trainlossfusion: object
    valloss: object
    vallossmaince: object
    vallossfusion: object
    duration: object
    curr_step: object


LOGGER = logging.getLogger(__name__)

fieldnames = [
    "epoch",
    "rmse",
    "photo",
    "mae",
    "irmse",
    "imae",
    "mse",
    "tmae",
    "trmse",
    "absrel",
    "lg10",
    "silog",
    "squared_rel",
    "delta1",
    "delta2",
    "delta3",
    "data_time",
    "gpu_time",
]


_LOGGER_LOSS_SUMMARIZE_V2_PARAMETERS_SCHEMA = _ParameterSchema(
    LoggerLossSummarizeV2Parameters,
    (
        "mode",
        "trainloss",
        "trainlossmaince",
        "valloss",
        "vallossmaince",
        "duration",
        "curr_step",
    ),
    {"curr_step": 0},
    frozenset(()),
    7,
    None,
    None,
)
_LOGGER_LOSS_SUMMARIZE_V3_PARAMETERS_SCHEMA = _ParameterSchema(
    LoggerLossSummarizeV3Parameters,
    (
        "mode",
        "trainloss",
        "trainlossmaince",
        "trainlossfusion",
        "valloss",
        "vallossmaince",
        "vallossfusion",
        "duration",
        "curr_step",
    ),
    {"curr_step": 0},
    frozenset(()),
    9,
    None,
    None,
)
_LOGGER_CONDITIONAL_PRINT_PARAMETERS_SCHEMA = _ParameterSchema(
    LoggerConditionalPrintParameters,
    ("split", "i", "epoch", "lr", "n_set", "blk_avg_meter", "avg_meter"),
    {},
    frozenset(()),
    7,
    None,
    None,
)
_LOGGER_CONDITIONAL_SAVE_IMG_COMPARISON_PARAMETERS_SCHEMA = _ParameterSchema(
    LoggerConditionalSaveImgComparisonParameters,
    ("mode", "i", "ele", "pred", "epoch"),
    {},
    frozenset(()),
    5,
    None,
    None,
)


class Logger:
    def __init__(self, args, prepare=True, output_directory=None):
        self.args = args
        if output_directory is None:
            output_directory = get_folder_name(args)
        elif not os.path.exists(output_directory):
            os.makedirs(output_directory)

        self.output_directory = output_directory
        self.best_result = Result()
        self.best_result.set_to_worst()
        self.img_merge = None
        self.old_best_result = None

        if not prepare:
            return
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.train_csv = os.path.join(output_directory, "train.csv")
        self.val_csv = os.path.join(output_directory, "val.csv")
        self.best_txt = os.path.join(output_directory, "best.txt")

        # backup the source code
        if args.resume == "":
            LOGGER.info("%s", "=> creating source code backup ...")
            backup_directory = os.path.join(output_directory, "code_backup")
            self.backup_directory = backup_directory
            # create new csv files with only header
            with open(self.train_csv, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            with open(self.val_csv, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            LOGGER.info("%s", "=> finished creating source code backup.")

    @staticmethod
    def loss_summarize(mode, trainloss, valloss, duration, curr_step=0):
        LOGGER.info(
            "%s",
            "curr_step {:d},trainloss_tot = {:.7f}, valloss_tot = {:.7f}, ({:.7f} sec/step)".format(
                curr_step, trainloss, valloss, duration
            ),
        )

    @staticmethod
    def loss_summarize_v2(*call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _LOGGER_LOSS_SUMMARIZE_V2_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            mode,
            trainloss,
            trainlossmaince,
            valloss,
            vallossmaince,
            duration,
            curr_step,
        ) = bound_parameters
        LOGGER.info(
            "%s",
            (
                "curr_step {:d},trainloss_tot = {:.7f}, train_maince = {:.7f}, "
                "valloss_tot = {:.7f}, valmaince = {:.7f}, ({:.7f} sec/step)"
            ).format(
                curr_step, trainloss, trainlossmaince, valloss, vallossmaince, duration
            ),
        )

    @staticmethod
    def loss_summarize_v3(*call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _LOGGER_LOSS_SUMMARIZE_V3_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            mode,
            trainloss,
            trainlossmaince,
            trainlossfusion,
            valloss,
            vallossmaince,
            vallossfusion,
            duration,
            curr_step,
        ) = bound_parameters
        LOGGER.info(
            "%s",
            (
                "curr_step {:d},trainloss_tot = {:.7f}, train_maince = {:.7f}, train_fusion = {:7f}, "
                "valloss_tot = {:.7f}, valmaince = {:.7f}, valloss_fusion = {:7f}, "
                "({:.7f} sec/step)"
            ).format(
                curr_step,
                trainloss,
                trainlossmaince,
                trainlossfusion,
                valloss,
                vallossmaince,
                vallossfusion,
                duration,
            ),
        )

    def conditional_print(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _LOGGER_CONDITIONAL_PRINT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        split, i, epoch, lr, n_set, blk_avg_meter, avg_meter = bound_parameters
        if (i + 1) % self.args.print_freq == 0:
            _ = avg_meter.average()
            _ = blk_avg_meter.average()
            blk_avg_meter.reset()

    def conditional_save_info(self, split, average_meter, epoch):
        avg = average_meter.average()
        if split == "train":
            csvfile_name = self.train_csv
        elif split == "val":
            csvfile_name = self.val_csv
        elif split == "eval":
            eval_filename = os.path.join(self.output_directory, "eval.txt")
            self.save_single_txt(eval_filename, avg, epoch)
            return avg
        elif split in ["test", "val_selection", "full"]:
            eval_filename = os.path.join(self.output_directory, "eval.txt")
            self.save_single_txt(eval_filename, avg, epoch)
            return avg
        else:
            raise ValueError("wrong split provided to logger")
        with open(csvfile_name, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(
                {
                    "epoch": epoch,
                    "rmse": avg.rmse,
                    "photo": avg.photometric,
                    "mae": avg.mae,
                    "irmse": avg.irmse,
                    "imae": avg.imae,
                    "mse": avg.mse,
                    "tmae": avg.tmae,
                    "trmse": avg.trmse,
                    "silog": avg.silog,
                    "squared_rel": avg.squared_rel,
                    "absrel": avg.absrel,
                    "lg10": avg.lg10,
                    "delta1": avg.delta1,
                    "delta2": avg.delta2,
                    "delta3": avg.delta3,
                    "gpu_time": avg.gpu_time,
                    "data_time": avg.data_time,
                }
            )
        return avg

    def save_single_txt(self, filename, result, epoch):
        with open(filename, "w") as txtfile:
            txtfile.write(
                (
                    "rank_metric={}\n"
                    + "epoch={}\n"
                    + "rmse={:.3f}\n"
                    + "mae={:.3f}\n"
                    + "silog={:.3f}\n"
                    + "squared_rel={:.3f}\n"
                    + "irmse={:.3f}\n"
                    + "imae={:.3f}\n"
                    + "mse={:.3f}\n"
                    + "absrel={:.3f}\n"
                    + "lg10={:.3f}\n"
                    + "delta1={:.3f}\n"
                    + "tmae={:.3f}\n"
                    + "trmse={:.3f}\n"
                    "t_gpu={:.4f}"
                ).format(
                    self.args.rank_metric,
                    epoch,
                    result.rmse,
                    result.mae,
                    result.silog,
                    result.squared_rel,
                    result.irmse,
                    result.imae,
                    result.mse,
                    result.absrel,
                    result.lg10,
                    result.delta1,
                    result.tmae,
                    result.trmse,
                    result.gpu_time,
                )
            )

    def save_best_txt(self, result, epoch):
        self.save_single_txt(self.best_txt, result, epoch)

    def conditional_save_img_comparison(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _LOGGER_CONDITIONAL_SAVE_IMG_COMPARISON_PARAMETERS_SCHEMA,
            call_args,
            call_kwargs,
        )
        mode, i, ele, pred, epoch = bound_parameters
        if mode == "val" or mode == "eval":
            skip = 100
            if i == 0:
                self.img_merge = vis_utils.merge_into_row(ele, pred)
            elif i % skip == 0 and i < 8 * skip:
                row = vis_utils.merge_into_row(ele, pred)
                self.img_merge = vis_utils.add_row(self.img_merge, row)
            elif i == 8 * skip:
                filename = self._get_img_comparison_name(mode, epoch)
                vis_utils.save_image(self.img_merge, filename)

    def save_img_comparison_as_best(self, mode, epoch):
        if mode == "val":
            filename = self._get_img_comparison_name(mode, epoch, is_best=True)
            vis_utils.save_image(self.img_merge, filename)

    def get_ranking_error(self, result):
        return getattr(result, self.args.rank_metric)

    def rank_conditional_save_best(self, mode, result, epoch):
        error = self.get_ranking_error(result)
        best_error = self.get_ranking_error(self.best_result)
        is_best = error < best_error
        if is_best and mode == "val":
            self.old_best_result = self.best_result
            self.best_result = result
            self.save_best_txt(result, epoch)
        return is_best

    def conditional_save_pred(self, mode, i, pred, epoch):
        if ("test" in mode or mode == "eval") and self.args.save_pred:

            # save images for visualization/ testing
            image_folder = os.path.join(self.output_directory, mode + "_output")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(pred.data.cpu()).numpy()
            filename = os.path.join(image_folder, "{0:010d}.png".format(i))
            vis_utils.save_depth_as_uint16png(img, filename)

    def conditional_summarize(self, mode, avg, is_best, curr_step=0):
        LOGGER.info("%s %s %s", "\n*\nSummary of ", mode, "round")
        LOGGER.info(
            "%s",
            (
                "RMSE={average.rmse:.3f}\nMAE={average.mae:.3f}\nTMAE={average.tmae: .3f}\n"
                "TRMSE={average.trmse: .3f}\nPhoto={average.photometric:.3f}\n"
                "iRMSE={average.irmse:.3f}\niMAE={average.imae:.3f}\n"
                "squared_rel={average.squared_rel}\nsilog={average.silog}\n"
                "Delta1={average.delta1:.3f}\nREL={average.absrel:.3f}\n"
                "Lg10={average.lg10:.3f}\nt_GPU={time:.3f}\ncurr_step = {curr_step: }"
            ).format(average=avg, time=avg.gpu_time, curr_step=curr_step),
        )
        if is_best and mode == "val":
            LOGGER.info(
                "%s",
                "New best model by %s (was %.3f)"
                % (self.args.rank_metric, self.get_ranking_error(self.old_best_result)),
            )
        elif mode == "val":
            LOGGER.info(
                "%s",
                "(best %s is %.3f)"
                % (self.args.rank_metric, self.get_ranking_error(self.best_result)),
            )
        LOGGER.info("%s", "*\n")

    def _get_img_comparison_name(self, mode, epoch, is_best=False):
        if mode == "eval":
            return self.output_directory + "/comparison_eval.png"
        if mode == "val":
            if is_best:
                return self.output_directory + "/comparison_best.png"
            else:
                return self.output_directory + "/comparison_" + str(epoch) + ".png"
        raise ValueError("unsupported comparison mode: {}".format(mode))


ignore_hidden = shutil.ignore_patterns(
    ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*"
)


def backup_source_code(backup_directory):
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)
    shutil.copytree(".", backup_directory, ignore=ignore_hidden)


def adjust_learning_rate(lr_init, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.5 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(
        output_directory, "checkpoint-" + str(epoch) + ".pth.tar"
    )
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, "model_best.pth.tar")
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(
            output_directory, "checkpoint-" + str(epoch - 1) + ".pth.tar"
        )
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def save_checkpoint_modified(
    state, epoch, output_directory, is_best=True, curr_step=None
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    checkpoint_filename = os.path.join(
        output_directory, "checkpoint-" + str(epoch) + "_" + str(curr_step) + ".pth.tar"
    )
    torch.save(state, checkpoint_filename)
    LOGGER.info("%s", "Checkpoint Saved Successfully\n")
    if is_best:
        best_filename = os.path.join(output_directory, "model_best.pth.tar")
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(
            output_directory, "checkpoint-" + str(epoch - 1) + "*.pth.tar"
        )
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def get_folder_name(args):
    current_time = time.strftime("%Y-%m-%d@%H-%M")
    if args.use_pose:
        prefix = "mode={}.w1={}.w2={}.".format(args.train_mode, args.w1, args.w2)
    else:
        prefix = "mode={}.".format(args.train_mode)
    return os.path.join(
        args.result,
        prefix
        + "input={}.resnet{}.criterion={}.lr={}.bs={}.wd={}.pretrained={}.jitter={}.time={}".format(
            args.input,
            args.layers,
            args.criterion,
            args.lr,
            args.batch_size,
            args.weight_decay,
            args.pretrained,
            args.jitter,
            current_time,
        ),
    )


avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2).cuda()


def multiscale(img):
    img1 = avgpool(img)
    img2 = avgpool(img1)
    img3 = avgpool(img2)
    img4 = avgpool(img3)
    img5 = avgpool(img4)
    return img5, img4, img3, img2, img1
