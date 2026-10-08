# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import abc
import contextlib

from google.protobuf import text_format
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from second.data.preprocess import merge_second_batch, prep_pointcloud
from second.protos import pipeline_pb2


def _inferencecontext_get_inference_input_dict_31_part_1(_state):
    if not _state["self"].anchor_cache is not None:
        raise AssertionError
    if not _state["self"].target_assigner is not None:
        raise AssertionError
    if not _state["self"].voxel_generator is not None:
        raise AssertionError
    if not _state["self"].config is not None:
        raise AssertionError
    if not _state["self"].built:
        raise AssertionError
    _state["rect"] = _state["info"]["calib/R0_rect"]
    _state["p2"] = _state["info"]["calib/P2"]
    _state["trv2c"] = _state["info"]["calib/Tr_velo_to_cam"]
    _state["input_cfg"] = _state["self"].config.eval_input_reader
    _state["model_cfg"] = _state["self"].config.model.second
    _state["pil2tensor"] = transforms.ToTensor()
    _state["pil_image"] = Image.open(str(_state["i_path"])).convert("RGB")
    _state["image_pang_bgr"] = np.array(_state["pil_image"])[:, :, [2, 1, 0]]
    _state["image_pang"] = _state["image_pang_bgr"]
    _state["input_dict"] = {
        "points": _state["points"],
        "rect": _state["rect"],
        "Trv2c": _state["trv2c"],
        "P2": _state["p2"],
        "image_shape": np.array(_state["info"]["img_shape"], dtype=np.int32),
        "image_idx": _state["info"]["image_idx"],
        "image_path": _state["info"]["img_path"],
        "images": _state["image_pang"],
    }
    _state["out_size_factor"] = (
        _state["model_cfg"].rpn.layer_strides[0]
        // _state["model_cfg"].rpn.upsample_strides[0]
    )


def _inferencecontext_get_inference_input_dict_31_part_2(_state):
    _state["example"] = prep_pointcloud(
        input_dict=_state["input_dict"],
        root_path=str(_state["self"].root_path),
        voxel_generator=_state["self"].voxel_generator,
        target_assigner=_state["self"].target_assigner,
        max_voxels=_state["input_cfg"].max_number_of_voxels,
        class_names=_state["self"].target_assigner.classes,
        training=False,
        create_targets=False,
        shuffle_points=_state["input_cfg"].shuffle_points,
        generate_bev=False,
        without_reflectivity=_state["model_cfg"].without_reflectivity,
        num_point_features=_state["model_cfg"].num_point_features,
        anchor_area_threshold=_state["input_cfg"].anchor_area_threshold,
        anchor_cache=_state["self"].anchor_cache,
        out_size_factor=_state["out_size_factor"],
        out_dtype=np.float32,
    )
    _state["example"]["image_idx"] = _state["info"]["image_idx"]
    _state["example"]["image_shape"] = _state["input_dict"]["image_shape"]
    _state["example"]["points"] = _state["points"]
    if "anchors_mask" in _state["example"]:
        _state["example"]["anchors_mask"] = _state["example"]["anchors_mask"].astype(
            np.uint8
        )
    _state["example"] = merge_second_batch([_state["example"]])


class InferenceContext:
    def __init__(self):
        self.config = None
        self.root_path = None
        self.target_assigner = None
        self.voxel_generator = None
        self.anchor_cache = None
        self.built = False

    @staticmethod
    def get_config(path):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        return config

    def get_inference_input_dict(self, info, points, i_path):
        _state = {"self": self, "info": info, "points": points, "i_path": i_path}
        _inferencecontext_get_inference_input_dict_31_part_1(_state)
        _inferencecontext_get_inference_input_dict_31_part_2(_state)
        return _state["example"]

    @abc.abstractmethod
    def _build(self):
        raise NotImplementedError(type(self).__name__)

    def build(self, config_path):
        self.config = self.get_config(config_path)
        ret = self._build()
        self.built = True
        return ret

    @abc.abstractmethod
    def _inference(self, example):
        raise NotImplementedError(type(self).__name__)

    def inference(self, example):
        return self._inference(example)

    @abc.abstractmethod
    def _restore(self, ckpt_path):
        raise NotImplementedError(type(self).__name__)

    def restore(self, ckpt_path):
        return self._restore(ckpt_path)

    @abc.abstractmethod
    def _ctx(self):
        raise NotImplementedError(type(self).__name__)

    @contextlib.contextmanager
    def ctx(self):
        yield self._ctx()
