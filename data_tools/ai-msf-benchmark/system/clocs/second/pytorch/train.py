# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections import namedtuple
from typing import NamedTuple
import json
import logging
import pathlib
import pickle
import time
import warnings

from google.protobuf import text_format
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaPerformanceWarning,
    NumbaWarning,
)
from tensorboardX import SummaryWriter
import fire
import numpy as np
import torch
import torch.nn.functional as F

from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (
    box_coder_builder,
    input_reader_builder,
    lr_scheduler_builder,
    optimizer_builder,
    second_builder,
)
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import SigmoidFocalClassificationLoss
from second.pytorch.models import fusion
from second.utils.eval import get_official_eval_result, d3_box_overlap
from second.utils.progress_bar import ProgressBar
import second.data.kitti_common as kitti


TrainingLoadersResult = namedtuple(
    "TrainingLoadersResult",
    "input_config evaluation_input_config dataset evaluation_dataset loader evaluation_loader",
)
import torchplus


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


class BuildInferenceNetParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    config_path: object
    model_dir: object
    result_path: object
    predict_test: object
    ckpt_path: object
    ref_detfile: object
    pickle_result: object
    measure_time: object
    batch_size: object


class TrainParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    config_path: object
    model_dir: object
    result_path: object
    create_folder: object
    display_step: object
    summary_step: object
    pickle_result: object
    patchs: object
    pretrain_model_dir: object


class PredictKittiToFileParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    net: object
    detection_2d_path: object
    fusion_layer: object
    example: object
    result_save_path: object
    class_names: object
    center_limit_range: object
    lidar_input: object


class PredictKittiToAnnoParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    net: object
    detection_2d_path: object
    fusion_layer: object
    example: object
    class_names: object
    center_limit_range: object
    lidar_input: object
    global_set: object


class EvaluateParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    config_path: object
    model_dir: object
    eval_map: object
    result_path: object
    predict_test: object
    ckpt_path: object
    ref_detfile: object
    pickle_result: object
    measure_time: object
    batch_size: object


LOGGER = logging.getLogger(__name__)

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.simplefilter("ignore", category=NumbaWarning)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


def example_convert_to_torch(example, dtype=torch.float32, device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels",
        "anchors",
        "reg_targets",
        "reg_weights",
        "bev_map",
        "rect",
        "Trv2c",
        "P2",
        "d3_gt_boxes",
        "gt_2d_boxes",
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.tensor(v, dtype=torch.float32, device=device).to(
                dtype
            )
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch


_BUILD_INFERENCE_NET_PARAMETERS_SCHEMA = _ParameterSchema(
    BuildInferenceNetParameters,
    (
        "config_path",
        "model_dir",
        "result_path",
        "predict_test",
        "ckpt_path",
        "ref_detfile",
        "pickle_result",
        "measure_time",
        "batch_size",
    ),
    {
        "result_path": None,
        "predict_test": False,
        "ckpt_path": None,
        "ref_detfile": None,
        "pickle_result": True,
        "measure_time": False,
        "batch_size": 1,
    },
    frozenset(()),
    9,
    None,
    None,
)
_TRAIN_PARAMETERS_SCHEMA = _ParameterSchema(
    TrainParameters,
    (
        "config_path",
        "model_dir",
        "result_path",
        "create_folder",
        "display_step",
        "summary_step",
        "pickle_result",
        "patchs",
        "pretrain_model_dir",
    ),
    {
        "result_path": None,
        "create_folder": False,
        "display_step": 50,
        "summary_step": 5,
        "pickle_result": True,
        "patchs": None,
        "pretrain_model_dir": None,
    },
    frozenset(()),
    9,
    None,
    None,
)
_PREDICT_KITTI_TO_FILE_PARAMETERS_SCHEMA = _ParameterSchema(
    PredictKittiToFileParameters,
    (
        "net",
        "detection_2d_path",
        "fusion_layer",
        "example",
        "result_save_path",
        "class_names",
        "center_limit_range",
        "lidar_input",
    ),
    {"center_limit_range": None, "lidar_input": False},
    frozenset(()),
    8,
    None,
    None,
)
_PREDICT_KITTI_TO_ANNO_PARAMETERS_SCHEMA = _ParameterSchema(
    PredictKittiToAnnoParameters,
    (
        "net",
        "detection_2d_path",
        "fusion_layer",
        "example",
        "class_names",
        "center_limit_range",
        "lidar_input",
        "global_set",
    ),
    {"center_limit_range": None, "lidar_input": False, "global_set": None},
    frozenset(()),
    8,
    None,
    None,
)
_EVALUATE_PARAMETERS_SCHEMA = _ParameterSchema(
    EvaluateParameters,
    (
        "config_path",
        "model_dir",
        "eval_map",
        "result_path",
        "predict_test",
        "ckpt_path",
        "ref_detfile",
        "pickle_result",
        "measure_time",
        "batch_size",
    ),
    {
        "eval_map": True,
        "result_path": None,
        "predict_test": False,
        "ckpt_path": None,
        "ref_detfile": None,
        "pickle_result": True,
        "measure_time": False,
        "batch_size": None,
    },
    frozenset(()),
    10,
    None,
    None,
)


def _module_build_inference_net_325_part_1(_state):
    _state["bound_parameters"] = _bind_parameters(
        _BUILD_INFERENCE_NET_PARAMETERS_SCHEMA,
        _state["call_args"],
        _state["call_kwargs"],
    )
    (
        _state["config_path"],
        _state["model_dir"],
        _state["result_path"],
        _state["predict_test"],
        _state["ckpt_path"],
        _state["ref_detfile"],
        _state["pickle_result"],
        _state["measure_time"],
        _state["batch_size"],
    ) = _state["bound_parameters"]
    _state["model_dir"] = pathlib.Path(_state["model_dir"])
    if _state["predict_test"]:
        _state["result_name"] = "predict_test"
    else:
        _state["result_name"] = "eval_results"
    if _state["result_path"] is None:
        _state["result_path"] = _state["model_dir"] / _state["result_name"]
    else:
        _state["result_path"] = pathlib.Path(_state["result_path"])
    _state["config"] = pipeline_pb2.TrainEvalPipelineConfig()
    with open(_state["config_path"], "r") as _state["f"]:
        _state["proto_str"] = _state["f"].read()
        text_format.Merge(_state["proto_str"], _state["config"])
    _state["model_cfg"] = _state["config"].model.second
    _state["input_cfg"] = _state["config"].eval_input_reader
    _state["detection_2d_path"] = _state["config"].train_config.detection_2d_path


def _module_build_inference_net_325_part_2(_state):
    _state["center_limit_range"] = _state["model_cfg"].post_center_limit_range
    _state["voxel_generator"] = voxel_builder.build(_state["model_cfg"].voxel_generator)
    _state["bv_range"] = _state["voxel_generator"].point_cloud_range[[0, 1, 3, 4]]
    _state["box_coder"] = box_coder_builder.build(_state["model_cfg"].box_coder)
    _state["target_assigner_cfg"] = _state["model_cfg"].target_assigner
    _state["target_assigner"] = target_assigner_builder.build(
        _state["target_assigner_cfg"], _state["bv_range"], _state["box_coder"]
    )
    _state["class_names"] = _state["target_assigner"].classes
    _state["net"] = second_builder.build(
        _state["model_cfg"],
        _state["voxel_generator"],
        _state["target_assigner"],
        measure_time=_state["measure_time"],
    )
    _state["net"].cuda()
    if _state["ckpt_path"] is None:
        LOGGER.info("%s", "load existing model")
        torchplus.train.try_restore_latest_checkpoints(
            _state["model_dir"], [_state["net"]]
        )
    else:
        torchplus.train.restore(_state["ckpt_path"], _state["net"])
    _state["batch_size"] = _state["batch_size"] or _state["input_cfg"].batch_size
    _state["net"].eval()


def build_inference_net(*call_args, **call_kwargs):
    _state = locals()
    _module_build_inference_net_325_part_1(_state)
    _module_build_inference_net_325_part_2(_state)
    return _state["net"]


def _next_training_batch(state):
    try:
        return next(state["data_iter"])
    except StopIteration:
        LOGGER.info("%s", "end epoch")
        if state["clear_metrics_every_epoch"]:
            state["net"].clear_metrics()
        state["data_iter"] = iter(state["dataloader"])
        return next(state["data_iter"])


def _report_fusion_step(state):
    global_step = state["net"].get_global_step()
    state["global_step"] = global_step
    if global_step % state["display_step"] != 0:
        return
    average_loss = state["cls_loss_sum"] / state["display_step"]
    values = (
        "now it is",
        global_step,
        "steps",
        " and the cls_loss is :",
        average_loss,
        "learning_rate: ",
        float(state["optimizer"].lr),
    )
    state["logf"].write(" ".join(str(value) for value in values) + "\n")
    LOGGER.info("%s %s %s %s %s %s %s", *values)
    state["cls_loss_sum"] = 0


def _save_elapsed_checkpoint(state):
    elapsed = time.time() - state["ckpt_start_time"]
    if elapsed <= state["train_cfg"].save_checkpoints_secs:
        return
    torchplus.train.save_models(
        state["model_dir"],
        [state["fusion_layer"], state["optimizer"]],
        state["net"].get_global_step(),
    )
    state["ckpt_start_time"] = time.time()


def _train_fusion_step(state):
    net = state["net"]
    state["lr_scheduler"].step(net.get_global_step())
    example = _next_training_batch(state)
    example = example_convert_to_torch(example, state["float_dtype"])
    output_camera, _, _, fusion_input, tensor_index = net(
        example, state["detection_2d_path"]
    )
    targets, positives, negatives = _get_fusion_targets(example, output_camera)
    predictions, valid = state["fusion_layer"](fusion_input.cuda(), tensor_index.cuda())
    target_tensor = torch.from_numpy(targets).type(torch.float32).cuda()
    weights = negatives.type(torch.float32) + positives.type(torch.float32)
    normalizer = positives.sum(1, keepdim=True).type(torch.float32)
    weights /= torch.clamp(normalizer, min=1.0)
    if valid == 1:
        losses = state["focal_loss"](predictions, target_tensor, weights=weights.cuda())
        loss = losses.sum() / example["labels"].shape[0]
        state["cls_loss_sum"] += loss
        if state["train_cfg"].enable_mixed_precision:
            loss *= state["loss_scale"]
        loss.backward()
        state["mixed_optimizer"].step()
        state["mixed_optimizer"].zero_grad()
    net.update_global_step()
    state["t"] = time.time()
    _report_fusion_step(state)
    _save_elapsed_checkpoint(state)


def _collect_validation_predictions(state, result_path):
    annotations = []
    validation_loss = 0
    progress = ProgressBar()
    progress.start(
        (len(state["eval_dataset"]) + state["eval_input_cfg"].batch_size - 1)
        // state["eval_input_cfg"].batch_size
    )
    for example in state["eval_dataloader"]:
        example = example_convert_to_torch(example, state["float_dtype"])
        if state["pickle_result"]:
            batch_annotations, batch_loss = predict_kitti_to_anno(
                state["net"],
                state["detection_2d_path"],
                state["fusion_layer"],
                example,
                state["class_names"],
                state["center_limit_range"],
                state["model_cfg"].lidar_input,
            )
            annotations.extend(batch_annotations)
            validation_loss += batch_loss
        else:
            _predict_kitti_to_file(
                state["net"],
                state["detection_2d_path"],
                state["fusion_layer"],
                example,
                result_path,
                state["class_names"],
                state["center_limit_range"],
                state["model_cfg"].lidar_input,
            )
        progress.print_bar()
    return annotations, validation_loss


def _evaluate_fusion_model(state):
    fusion_layer, net = state["fusion_layer"], state["net"]
    fusion_layer.eval()
    net.eval()
    result_path = state["result_path"] / f"step_{net.get_global_step()}"
    result_path.mkdir(parents=True, exist_ok=True)
    for message in (
        "#################################",
        "# EVAL",
        "#################################",
        "Generate output labels...",
    ):
        LOGGER.info("%s", message)
        state["logf"].write(message + "\n")
    start = time.time()
    net.clear_timer()
    annotations, validation_loss = _collect_validation_predictions(state, result_path)
    average_loss = validation_loss / len(state["eval_dataloader"])
    LOGGER.info("validation_loss: %s", average_loss)
    state["logf"].write(f"validation_loss: {average_loss}\n")
    speed = len(state["eval_dataset"]) / (time.time() - start)
    message = f"generate label finished({speed:.2f}/s). start eval:"
    LOGGER.info("%s", message)
    state["logf"].write(message + "\n")
    ground_truth = [info["annos"] for info in state["eval_dataset"].dataset.kitti_infos]
    if not state["pickle_result"]:
        annotations = kitti.get_label_annos(result_path)
    result = get_official_eval_result(ground_truth, annotations, state["class_names"])
    state["logf"].write(str(result) + "\n")
    LOGGER.info("%s", result)
    state["writer"].add_text(
        "eval_result", json.dumps(result, indent=2), state["global_step"]
    )
    fusion_layer.train()


def _run_training_cycles(state):
    for loop_index in range(state["total_loop"]):
        LOGGER.info("************** cur loop %s", loop_index)
        remaining = state["train_cfg"].steps - state["total_step_elapsed"]
        steps = min(state["train_cfg"].steps_per_eval, remaining)
        for _ in range(steps):
            _train_fusion_step(state)
        state["total_step_elapsed"] += steps
        torchplus.train.save_models(
            state["model_dir"],
            [state["fusion_layer"], state["optimizer"]],
            state["net"].get_global_step(),
        )
        _evaluate_fusion_model(state)


def _load_train_configuration(config_path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as file_handle:
        proto_str = file_handle.read()
    text_format.Merge(proto_str, config)
    model_cfg = config.model.second
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner = target_assigner_builder.build(
        model_cfg.target_assigner,
        voxel_generator.point_cloud_range[[0, 1, 3, 4]],
        box_coder,
    )
    return config, proto_str, model_cfg, voxel_generator, target_assigner


def _build_training_models(config_path, model_dir, pretrain_model_dir, config):
    train_cfg = config.train_config
    net = build_inference_net(
        config_path,
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/system/CLOCs/model_dir",
    )
    fusion_layer = fusion.fusion()
    torchplus.train.try_restore_latest_checkpoints(pretrain_model_dir, [fusion_layer])
    fusion_layer.cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    optimizer = optimizer_builder.build(
        train_cfg.optimizer, fusion_layer,
        mixed=train_cfg.enable_mixed_precision,
        loss_scale=train_cfg.loss_scale_factor,
    )
    torchplus.train.try_restore_latest_checkpoints(model_dir, [optimizer])
    scheduler = lr_scheduler_builder.build(
        train_cfg.optimizer, optimizer, train_cfg.steps
    )
    return net, fusion_layer, optimizer, scheduler


def _training_worker_init(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    LOGGER.info("%s %s", f"WORKER {worker_id} seed:", np.random.get_state()[1][0])


def _build_training_loaders(config, model_cfg, voxel_generator, target_assigner):
    input_cfg, eval_input_cfg = config.train_input_reader, config.eval_input_reader
    dataset = input_reader_builder.build(
        input_cfg, model_cfg, training=True,
        voxel_generator=voxel_generator, target_assigner=target_assigner,
    )
    eval_dataset = input_reader_builder.build(
        eval_input_cfg, model_cfg, training=True,
        voxel_generator=voxel_generator, target_assigner=target_assigner,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=input_cfg.batch_size, shuffle=True,
        num_workers=input_cfg.num_workers, pin_memory=False,
        collate_fn=merge_second_batch, worker_init_fn=_training_worker_init,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=eval_input_cfg.batch_size, shuffle=False,
        num_workers=eval_input_cfg.num_workers, pin_memory=False,
        collate_fn=merge_second_batch,
    )
    return TrainingLoadersResult(
        input_cfg, eval_input_cfg, dataset, eval_dataset, dataloader, eval_dataloader
    )


def _training_outputs(model_dir, proto_str):
    detail_path = model_dir / "log.json"
    if detail_path.exists():
        with open(detail_path, "r") as file_handle:
            json.load(file_handle)
    log_path = model_dir / "log.txt"
    summary_dir = model_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return log_path, SummaryWriter(str(summary_dir))


def _initial_training_state(context):
    parameters = context["parameters"]
    train_cfg = context["train_cfg"]
    return {
        "net": context["net"], "fusion_layer": context["fusion_layer"],
        "optimizer": context["optimizer"], "mixed_optimizer": context["optimizer"],
        "lr_scheduler": context["lr_scheduler"], "train_cfg": train_cfg,
        "model_cfg": context["model_cfg"], "model_dir": context["model_dir"],
        "result_path": context["result_path"], "pickle_result": parameters.pickle_result,
        "detection_2d_path": train_cfg.detection_2d_path,
        "center_limit_range": context["model_cfg"].post_center_limit_range,
        "class_names": context["target_assigner"].classes,
        "input_cfg": context["input_cfg"], "eval_input_cfg": context["eval_input_cfg"],
        "dataset": context["dataset"], "eval_dataset": context["eval_dataset"],
        "dataloader": context["dataloader"], "eval_dataloader": context["eval_dataloader"],
        "data_iter": iter(context["dataloader"]),
        "focal_loss": SigmoidFocalClassificationLoss(),
        "float_dtype": torch.float16 if train_cfg.enable_mixed_precision else torch.float32,
        "loss_scale": train_cfg.loss_scale_factor, "display_step": parameters.display_step,
        "clear_metrics_every_epoch": train_cfg.clear_metrics_every_epoch,
        "cls_loss_sum": 0, "total_step_elapsed": 0, "total_loop": 1,
        "ckpt_start_time": time.time(), "logf": context["logf"],
        "writer": context["writer"],
    }


def train(*call_args, **call_kwargs):
    parameters = _bind_parameters(_TRAIN_PARAMETERS_SCHEMA, call_args, call_kwargs)
    torch.manual_seed(3)
    np.random.seed(3)
    model_dir = parameters.model_dir
    if parameters.create_folder and pathlib.Path(model_dir).exists():
        model_dir = torchplus.train.create_folder(model_dir)
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = parameters.result_path or model_dir / "results"
    config, proto_str, model_cfg, voxel_generator, target_assigner = (
        _load_train_configuration(parameters.config_path)
    )
    train_cfg = config.train_config
    LOGGER.info("%s %s", "2d detection path:", train_cfg.detection_2d_path)
    net, fusion_layer, optimizer, lr_scheduler = _build_training_models(
        parameters.config_path, model_dir, parameters.pretrain_model_dir, config
    )
    loaders = _build_training_loaders(
        config, model_cfg, voxel_generator, target_assigner
    )
    input_cfg, eval_input_cfg, dataset, eval_dataset, dataloader, eval_dataloader = loaders
    log_path, writer = _training_outputs(model_dir, proto_str)
    with open(log_path, "a") as logf:
        logf.write(proto_str + "\n")
        net.set_global_step(torch.tensor([0]))
        optimizer.zero_grad()
        state = _initial_training_state(locals())
        LOGGER.info("%s", "############################# train")
        try:
            _run_training_cycles(state)
        except Exception:
            torchplus.train.save_models(
                model_dir, [fusion_layer, optimizer, net], net.get_global_step()
            )
            raise
    LOGGER.info("%s", f"==================,save to {model_dir}")
    torchplus.train.save_models(
        model_dir, [fusion_layer, optimizer], net.get_global_step()
    )


def _outside_image(bbox, image_shape):
    return (
        bbox[0] > image_shape[1]
        or bbox[1] > image_shape[0]
        or bbox[2] < 0
        or bbox[3] < 0
    )


def _outside_center_limit(box_lidar, center_limit_range):
    if center_limit_range is None:
        return False
    limits = np.asarray(center_limit_range)
    return np.any(box_lidar[:3] < limits[:3]) or np.any(box_lidar[:3] > limits[3:])


def _kitti_prediction_lines(prediction, image_shape, class_names, options):
    center_limit_range, lidar_input = options
    if prediction["bbox"] is None or prediction["bbox"].numel() == 0:
        return []
    boxes_2d = prediction["bbox"].data.cpu().numpy()
    boxes_camera = prediction["box3d_camera"].data.cpu().numpy()
    boxes_lidar = prediction["box3d_lidar"].data.cpu().numpy()
    scores = prediction["scores"].data.cpu().numpy()
    labels = prediction["label_preds"].data.cpu().numpy()
    boxes_camera = boxes_camera[:, [0, 1, 2, 4, 5, 3, 6]]
    lines = []
    for box, box_lidar, bbox, score, label in zip(
        boxes_camera, boxes_lidar, boxes_2d, scores, labels
    ):
        if not lidar_input and _outside_image(bbox, image_shape):
            continue
        if _outside_center_limit(box_lidar, center_limit_range):
            continue
        bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
        bbox[:2] = np.maximum(bbox[:2], [0, 0])
        result = {
            "name": class_names[int(label)],
            "alpha": -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
            "bbox": bbox,
            "location": box[:3],
            "dimensions": box[3:6],
            "rotation_y": box[6],
            "score": score,
        }
        lines.append(kitti.kitti_result_line(result))
    return lines


def _write_kitti_predictions(
    predictions, image_shapes, result_path, class_names, options
):
    for index, prediction in enumerate(predictions):
        lines = _kitti_prediction_lines(
            prediction, image_shapes[index], class_names, options
        )
        image_index = kitti.get_image_index_str(prediction["image_idx"])
        with open(f"{result_path}/{image_index}.txt", "w") as result_file:
            result_file.write("\n".join(lines))


def _predict_kitti_to_file(*call_args, **call_kwargs):
    bound_parameters = _bind_parameters(
        _PREDICT_KITTI_TO_FILE_PARAMETERS_SCHEMA, call_args, call_kwargs
    )
    (
        net,
        detection_2d_path,
        fusion_layer,
        example,
        result_save_path,
        class_names,
        center_limit_range,
        lidar_input,
    ) = bound_parameters
    _, output_3d, _, fusion_input, indices = net(example, detection_2d_path)
    fusion_scores, _ = fusion_layer(fusion_input.cuda(), indices.cuda())
    output_3d["cls_preds"] = fusion_scores.reshape(1, 200, 176, 2)
    predictions = predict_v2(net, example, output_3d)
    options = (center_limit_range, lidar_input)
    _write_kitti_predictions(
        predictions, example["image_shape"], result_save_path, class_names, options
    )


def _get_fusion_targets(example, output_camera):
    gt_boxes = example["d3_gt_boxes"][0]
    if gt_boxes.shape[0] == 0:
        targets = np.zeros((1, 70400, 1))
        positives = torch.zeros(1, 70400, dtype=torch.float32, device="cuda")
        negatives = torch.ones(1, 70400, dtype=torch.float32, device="cuda")
        return targets, positives, negatives
    camera_boxes = box_torch_ops.box_lidar_to_camera(
        gt_boxes, example["rect"][0], example["Trv2c"][0]
    )
    predicted_boxes = output_camera[0]["box3d_camera"]
    overlaps = d3_box_overlap(
        camera_boxes.detach().cpu().numpy(),
        predicted_boxes.squeeze().detach().cpu().numpy(),
        criterion=-1,
    )
    maximum_overlaps = np.amax(overlaps, axis=0)
    targets = (maximum_overlaps >= 0.7).reshape(1, -1, 1)
    positives = (
        torch.from_numpy((maximum_overlaps >= 0.7).reshape(1, -1))
        .type(torch.float32)
        .cuda()
    )
    negatives = (
        torch.from_numpy((maximum_overlaps <= 0.5).reshape(1, -1))
        .type(torch.float32)
        .cuda()
    )
    return targets, positives, negatives


def _unique_prediction_score(score, used_scores):
    if used_scores is None:
        return score
    for _ in range(100000):
        if score not in used_scores:
            used_scores.add(score)
            return score
        score -= 1 / 100000
    return score


def _prediction_to_anno(prediction, image_shape, conversion_config):
    class_names, center_limit_range, lidar_input, used_scores = conversion_config
    if prediction["bbox"] is None or prediction["bbox"].size.numel() == 0:
        return kitti.empty_result_anno()
    arrays = [
        prediction[name].detach().cpu().numpy()
        for name in ("box3d_camera", "box3d_lidar", "bbox", "scores", "label_preds")
    ]
    anno = kitti.get_start_result_anno()
    for box, box_lidar, bbox, score, label in zip(*arrays):
        outside_image = (
            bbox[0] > image_shape[1]
            or bbox[1] > image_shape[0]
            or bbox[2] < 0
            or bbox[3] < 0
        )
        if not lidar_input and outside_image:
            continue
        if center_limit_range is not None:
            limit_range = np.array(center_limit_range)
            outside_range = np.any(box_lidar[:3] < limit_range[:3]) or np.any(
                box_lidar[:3] > limit_range[3:]
            )
            if outside_range:
                continue
        bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
        bbox[:2] = np.maximum(bbox[:2], [0, 0])
        anno["name"].append(class_names[int(label)])
        anno["truncated"].append(0.0)
        anno["occluded"].append(0)
        anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
        anno["bbox"].append(bbox)
        anno["dimensions"].append(box[3:6])
        anno["location"].append(box[:3])
        anno["rotation_y"].append(box[6])
        anno["score"].append(_unique_prediction_score(score, used_scores))
    if not anno["name"]:
        return kitti.empty_result_anno()
    return {name: np.stack(values) for name, values in anno.items()}


def _module_predict_kitti_to_anno_915_part_1(_state):
    _state["bound_parameters"] = _bind_parameters(
        _PREDICT_KITTI_TO_ANNO_PARAMETERS_SCHEMA,
        _state["call_args"],
        _state["call_kwargs"],
    )
    (
        _state["net"],
        _state["detection_2d_path"],
        _state["fusion_layer"],
        _state["example"],
        _state["class_names"],
        _state["center_limit_range"],
        _state["lidar_input"],
        _state["global_set"],
    ) = _state["bound_parameters"]
    _state["focal_loss_val"] = SigmoidFocalClassificationLoss()
    _state["batch_image_shape"] = _state["example"]["image_shape"]
    _state["batch_imgidx"] = _state["example"]["image_idx"]
    (
        _state["all_3d_output_camera_dict"],
        _state["all_3d_output"],
        _state["top_predictions"],
        _state["fusion_input"],
        _state["torch_index"],
    ) = _state["net"](_state["example"], _state["detection_2d_path"])
    _state["t_start"] = time.time()
    _state["fusion_cls_preds"], _state["flag"] = _state["fusion_layer"](
        _state["fusion_input"].cuda(), _state["torch_index"].cuda()
    )
    _state["t_end"] = time.time()
    _state["t_fusion"] = _state["t_end"] - _state["t_start"]
    _state["fusion_cls_preds_reshape"] = _state["fusion_cls_preds"].reshape(
        1, 200, 176, 2
    )
    _state["all_3d_output"].update({"cls_preds": _state["fusion_cls_preds_reshape"]})
    _state["predictions_dicts"] = predict_v2(
        _state["net"], _state["example"], _state["all_3d_output"]
    )


def _module_predict_kitti_to_anno_915_part_2(_state):
    _state["target_for_fusion"], _state["positives"], _state["negatives"] = (
        _get_fusion_targets(_state["example"], _state["all_3d_output_camera_dict"])
    )
    _state["one_hot_targets"] = (
        torch.from_numpy(_state["target_for_fusion"]).type(torch.float32).cuda()
    )
    _state["cls_weights"] = _state["negatives"].type(torch.float32) + _state[
        "positives"
    ].type(torch.float32)
    _state["cls_weights"] /= torch.clamp(
        _state["positives"].sum(1, keepdim=True).type(torch.float32), min=1.0
    )
    _state["cls_losses"] = _state["focal_loss_val"](
        _state["fusion_cls_preds"],
        _state["one_hot_targets"],
        weights=_state["cls_weights"].cuda(),
    )
    _state["cls_losses_reduced"] = (
        (_state["cls_losses"].sum() / _state["example"]["labels"].shape[0])
        .detach()
        .cpu()
        .numpy()
    )
    _state["annos"] = []
    for _state["batch_index"], _state["preds_dict"] in enumerate(
        _state["predictions_dicts"]
    ):
        _state["image_shape"] = _state["batch_image_shape"][_state["batch_index"]]
        _state["img_idx"] = _state["preds_dict"]["image_idx"]
        _state["conversion_config"] = (
            _state["class_names"],
            _state["center_limit_range"],
            _state["lidar_input"],
            _state["global_set"],
        )
        _state["annos"].append(
            _prediction_to_anno(
                _state["preds_dict"], _state["image_shape"], _state["conversion_config"]
            )
        )
        _state["num_example"] = _state["annos"][-1]["name"].shape[0]
        _state["annos"][-1]["image_idx"] = np.array(
            [_state["img_idx"]] * _state["num_example"], dtype=np.int64
        )


def predict_kitti_to_anno(*call_args, **call_kwargs):
    _state = locals()
    _module_predict_kitti_to_anno_915_part_1(_state)
    _module_predict_kitti_to_anno_915_part_2(_state)
    return (_state["annos"], _state["cls_losses_reduced"])


def _load_evaluation_config(config_path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    return config


def _build_evaluation_components(config_path, config):
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner = target_assigner_builder.build(
        model_cfg.target_assigner, bv_range, box_coder
    )
    net = build_inference_net(
        config_path,
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/system/CLOCs/model_dir",
    )
    fusion_layer = fusion.fusion()
    fusion_layer.cuda()
    net.cuda()
    return {
        "input_cfg": input_cfg,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "voxel_generator": voxel_generator,
        "target_assigner": target_assigner,
        "class_names": target_assigner.classes,
        "net": net,
        "fusion_layer": fusion_layer,
    }


def _prepare_evaluation_models(components, model_dir, ckpt_path):
    net, fusion_layer = components["net"], components["fusion_layer"]
    if ckpt_path is None:
        LOGGER.info("%s", "load existing model for fusion layer")
        torchplus.train.try_restore_latest_checkpoints(model_dir, [fusion_layer])
    else:
        torchplus.train.restore(ckpt_path, fusion_layer)
    if components["train_cfg"].enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    net.eval()
    fusion_layer.eval()
    if net.get_global_step() == 0:
        raise ValueError("****************", net.get_global_step(), "****************")


def _build_evaluation_loader(components, predict_test, batch_size):
    input_cfg = components["input_cfg"]
    model_cfg = components["model_cfg"]
    batch_size = batch_size or input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=not predict_test,
        voxel_generator=components["voxel_generator"],
        target_assigner=components["target_assigner"],
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
    )
    return eval_dataset, eval_dataloader, batch_size


def _evaluation_progress_bar(context):
    bar = ProgressBar()
    bar.start(
        (len(context["dataset"]) + context["batch_size"] - 1)
        // context["batch_size"]
    )
    return bar


def _run_evaluation_batches(context):
    net = context["components"]["net"]
    model_cfg = context["components"]["model_cfg"]
    fusion_layer = context["components"]["fusion_layer"]
    train_cfg = context["components"]["train_cfg"]
    float_dtype = torch.float16 if train_cfg.enable_mixed_precision else torch.float32
    dt_annos, prep_example_times, prep_times = [], [], []
    global_set = None
    LOGGER.info("%s", "Generate output labels...")
    bar = _evaluation_progress_bar(context)
    t2 = time.time()
    val_loss_final = 0
    for example in iter(context["dataloader"]):
        if context["measure_time"]:
            prep_times.append(time.time() - t2)
            t1 = time.time()
            torch.cuda.synchronize()
        example = example_convert_to_torch(example, float_dtype)
        if context["measure_time"]:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)
        if context["pickle_result"]:
            dt_annos_i, val_losses = predict_kitti_to_anno(
                net,
                train_cfg.detection_2d_path,
                fusion_layer,
                example,
                context["components"]["class_names"],
                model_cfg.post_center_limit_range,
                model_cfg.lidar_input,
                global_set,
            )
            dt_annos += dt_annos_i
            val_loss_final += val_losses
        else:
            _predict_kitti_to_file(
                net,
                train_cfg.detection_2d_path,
                fusion_layer,
                example,
                context["result_path"],
                context["components"]["class_names"],
                model_cfg.post_center_limit_range,
                model_cfg.lidar_input,
            )
        bar.print_bar()
        if context["measure_time"]:
            t2 = time.time()
    return dt_annos, val_loss_final, prep_example_times, prep_times


def _finish_evaluation(context, outputs, start_time):
    dt_annos, val_loss, prep_example_times, prep_times = outputs
    dataset, dataloader = context["dataset"], context["dataloader"]
    sec_per_example = len(dataset) / (time.time() - start_time)
    LOGGER.info("%s", f"generate label finished({sec_per_example:.2f}/s). start eval:")
    LOGGER.info("%s %s", "validation_loss:", val_loss / len(dataloader))
    if context["measure_time"]:
        LOGGER.info(
            "%s",
            f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms",
        )
        LOGGER.info("%s", f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in context["components"]["net"].get_avg_time_dict().items():
        LOGGER.info("%s", f"avg {name} time = {val * 1000:.3f} ms")
    result_path = context["result_path"]
    if not context["predict_test"]:
        gt_annos = [info["annos"] for info in dataset.dataset.kitti_infos]
        if not context["pickle_result"]:
            dt_annos = kitti.get_label_annos(result_path)
        result = get_official_eval_result(
            gt_annos, dt_annos, context["components"]["class_names"]
        )
        LOGGER.info("%s", result)
        with open("/home/niangao/PycharmProjects/fusion/temp_clocs", mode="a") as ff:
            ff.write(result)
            ff.write("\n")
    if context["pickle_result"]:
        with open(result_path / "result.pkl", "wb") as result_file:
            pickle.dump(dt_annos, result_file)


def evaluate(*call_args, **call_kwargs):
    parameters = _bind_parameters(_EVALUATE_PARAMETERS_SCHEMA, call_args, call_kwargs)
    config_path, model_dir, eval_map, result_path, predict_test, ckpt_path = parameters[:6]
    pickle_result, measure_time, batch_size = parameters[7:]
    model_dir = pathlib.Path(model_dir)
    LOGGER.info("%s %s", "Predict_test: ", predict_test)
    result_name = "predict_test" if predict_test else "eval_results"
    result_path = pathlib.Path(result_path) if result_path else model_dir / result_name
    config = _load_evaluation_config(config_path)
    components = _build_evaluation_components(config_path, config)
    _prepare_evaluation_models(components, model_dir, ckpt_path)
    dataset, dataloader, batch_size = _build_evaluation_loader(
        components, predict_test, batch_size
    )
    result_path = result_path / f"step_{components['net'].get_global_step()}"
    result_path.mkdir(parents=True, exist_ok=True)
    context = {
        "components": components,
        "dataset": dataset,
        "dataloader": dataloader,
        "batch_size": batch_size,
        "result_path": result_path,
        "predict_test": predict_test,
        "pickle_result": pickle_result,
        "measure_time": measure_time,
    }
    start_time = time.time()
    outputs = _run_evaluation_batches(context)
    if eval_map:
        _finish_evaluation(context, outputs, start_time)


def save_config(config_path, save_path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    ret = text_format.MessageToString(config, indent=2)
    with open(save_path, "w") as f:
        f.write(ret)


def _decoded_prediction_batches(net, example, predictions):
    batch_size = example["anchors"].shape[0]
    batch_anchors = example["anchors"].view(batch_size, -1, 7)
    boxes = predictions["box_preds"].view(batch_size, -1, net.box_coder.code_size)
    num_class_with_bg = net.num_class
    if not net.encode_background_as_zeros:
        num_class_with_bg = net.num_class + 1
    classes = predictions["cls_preds"].view(batch_size, -1, num_class_with_bg)
    boxes = net.box_coder.decode_torch(boxes, batch_anchors)
    if net.use_direction_classifier:
        directions = predictions["dir_cls_preds"].view(batch_size, -1, 2)
    else:
        directions = [None] * batch_size
    masks = [None] * batch_size
    if "anchors_mask" in example:
        masks = example["anchors_mask"].view(batch_size, -1)
    transforms = zip(
        example["rect"], example["Trv2c"], example["P2"], example["image_idx"]
    )
    return zip(boxes, classes, directions, masks, transforms), boxes


def _class_scores(net, class_predictions):
    if net.encode_background_as_zeros:
        if not net.use_sigmoid_score:
            raise ValueError("sigmoid scores are required with zero-encoded background")
        return torch.sigmoid(class_predictions)
    if net.use_sigmoid_score:
        return torch.sigmoid(class_predictions)[..., 1:]
    return F.softmax(class_predictions, dim=-1)[..., 1:]


def _boxes_for_nms(net, boxes):
    boxes_for_nms = boxes[:, [0, 1, 3, 4, 6]]
    if net.use_rotate_nms:
        return boxes_for_nms
    corners = box_torch_ops.center_to_corner_box2d(
        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4]
    )
    return box_torch_ops.corner_to_standup_nd(corners)


def _multiclass_selection(net, boxes, scores, direction_labels):
    nms_function = box_torch_ops.rotate_nms if net.use_rotate_nms else box_torch_ops.nms
    selected_by_class = box_torch_ops.multiclass_nms(
        nms_func=nms_function,
        boxes=_boxes_for_nms(net, boxes).unsqueeze(1),
        scores=scores,
        num_class=net.num_class,
        pre_max_size=net.nms_pre_max_size,
        post_max_size=net.nms_post_max_size,
        iou_threshold=net.nms_iou_threshold,
        score_thresh=net.nms_score_threshold,
    )
    selected_boxes, selected_labels, selected_scores, selected_directions = (
        [],
        [],
        [],
        [],
    )
    for class_index, selected in enumerate(selected_by_class):
        if selected is None:
            continue
        count = selected.shape[0]
        selected_boxes.append(boxes[selected])
        selected_labels.append(
            torch.full([count], class_index, dtype=torch.int64, device=boxes.device)
        )
        selected_scores.append(scores[selected, class_index])
        if net.use_direction_classifier:
            selected_directions.append(direction_labels[selected])
    directions = None
    if net.use_direction_classifier:
        directions = torch.cat(selected_directions, dim=0)
    return (
        torch.cat(selected_boxes, dim=0),
        torch.cat(selected_labels, dim=0),
        torch.cat(selected_scores, dim=0),
        directions,
    )


def _singleclass_selection(net, boxes, scores, direction_labels):
    if scores.shape[-1] == 1:
        top_scores = scores.squeeze(-1)
        top_labels = torch.zeros(
            scores.shape[0], device=scores.device, dtype=torch.long
        )
    else:
        top_scores, top_labels = torch.max(scores, dim=-1)
    if net.nms_score_threshold > 0.0:
        keep = top_scores >= top_scores.new_tensor(net.nms_score_threshold)
        top_scores = top_scores[keep]
        boxes = boxes[keep]
        top_labels = top_labels[keep]
        if net.use_direction_classifier:
            direction_labels = direction_labels[keep]
    selected = []
    if top_scores.shape[0] != 0:
        nms_function = (
            box_torch_ops.rotate_nms if net.use_rotate_nms else box_torch_ops.nms
        )
        selected = nms_function(
            _boxes_for_nms(net, boxes),
            top_scores,
            pre_max_size=net.nms_pre_max_size,
            post_max_size=net.nms_post_max_size,
            iou_threshold=net.nms_iou_threshold,
        )
    directions = direction_labels[selected] if net.use_direction_classifier else None
    return boxes[selected], top_labels[selected], top_scores[selected], directions


def _select_predictions(net, boxes, classes, directions, anchor_mask):
    if anchor_mask is not None:
        boxes = boxes[anchor_mask]
        classes = classes[anchor_mask]
        if net.use_direction_classifier:
            directions = directions[anchor_mask]
    boxes, classes = boxes.float(), classes.float()
    direction_labels = None
    if net.use_direction_classifier:
        direction_labels = torch.max(directions, dim=-1)[1]
    scores = _class_scores(net, classes)
    if net.multiclass_nms:
        return _multiclass_selection(net, boxes, scores, direction_labels)
    return _singleclass_selection(net, boxes, scores, direction_labels)


def _empty_prediction(box_batch, labels, image_index):
    dtype, device = box_batch.dtype, box_batch.device
    return {
        "bbox": torch.zeros([0, 4], dtype=dtype, device=device),
        "box3d_camera": torch.zeros([0, 7], dtype=dtype, device=device),
        "box3d_lidar": torch.zeros([0, 7], dtype=dtype, device=device),
        "scores": torch.zeros([0], dtype=dtype, device=device),
        "label_preds": torch.zeros([0], dtype=labels.dtype, device=device),
        "image_idx": image_index,
    }


def _format_prediction(net, selected, transform, box_batch):
    boxes, labels, scores, directions = selected
    rect, trv2c, projection, image_index = transform
    if boxes.shape[0] == 0:
        return _empty_prediction(box_batch, labels, image_index)
    if net.use_direction_classifier:
        opposite = (boxes[..., -1] > 0) ^ directions.to(torch.bool)
        boxes[..., -1] += torch.where(
            opposite,
            torch.tensor(np.pi).type_as(boxes),
            torch.tensor(0.0).type_as(boxes),
        )
    camera_boxes = box_torch_ops.box_lidar_to_camera(boxes, rect.float(), trv2c.float())
    corners = box_torch_ops.center_to_corner_box3d(
        camera_boxes[:, :3],
        camera_boxes[:, 3:6],
        camera_boxes[:, 6],
        [0.5, 1.0, 0.5],
        axis=1,
    )
    image_corners = box_torch_ops.project_to_image(corners, projection.float())
    bbox = torch.cat(
        [torch.min(image_corners, dim=1)[0], torch.max(image_corners, dim=1)[0]], dim=1
    )
    return {
        "bbox": bbox,
        "box3d_camera": camera_boxes,
        "box3d_lidar": boxes,
        "scores": scores,
        "label_preds": labels,
        "image_idx": image_index,
    }


def predict_v2(net, example, preds_dict):
    batches, box_batch = _decoded_prediction_batches(net, example, preds_dict)
    predictions = []
    for boxes, classes, directions, mask, transform in batches:
        selected = _select_predictions(net, boxes, classes, directions, mask)
        predictions.append(_format_prediction(net, selected, transform, box_batch))
    return predictions


if __name__ == "__main__":
    fire.Fire()
