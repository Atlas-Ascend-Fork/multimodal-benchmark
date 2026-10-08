# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import base64
import logging
import pickle
import time

from flask import Flask, jsonify, request
from flask_cors import CORS
import fire
import numpy as np

from second.data import kitti_common as kitti
from second.pytorch.inference import TorchInferenceContext
import second.core.box_np_ops as box_np_ops

LOGGER = logging.getLogger(__name__)

app = Flask("second")
CORS(app)


class SecondBackend:
    def __init__(self):
        self.root_path = None
        self.info_path = None
        self.kitti_infos = None
        self.image_idxes = None
        self.dt_annos = None
        self.inference_ctx = None


BACKEND = SecondBackend()


def error_response(msg):
    response = {}
    response["status"] = "error"
    response["message"] = "[ERROR]" + msg
    LOGGER.info("%s", "[ERROR]" + msg)
    return response


@app.route("/api/readinfo", methods=["POST"])
def readinfo():
    global BACKEND
    instance = request.json
    root_path = Path(instance["root_path"])
    LOGGER.info("%s %s", "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", root_path)

    response = {"status": "normal"}
    if not (root_path / "training").exists():
        response["status"] = "error"
        response["message"] = "ERROR: your root path is incorrect."
        LOGGER.info("%s", "ERROR: your root path is incorrect.7777777")
        return response
    BACKEND.root_path = root_path
    info_path = Path(instance["info_path"])

    if not info_path.exists():
        response["status"] = "error"
        response["message"] = "ERROR: info file not exist."
        LOGGER.info("%s", "ERROR: your root path is incorrect.8888888")
        return response
    BACKEND.info_path = info_path
    with open(info_path, "rb") as f:
        kitti_infos = pickle.load(f)
    BACKEND.kitti_infos = kitti_infos
    BACKEND.image_idxes = [info["image_idx"] for info in kitti_infos]
    response["image_indexes"] = BACKEND.image_idxes

    response = jsonify(results=[response])
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


@app.route("/api/read_detection", methods=["POST"])
def read_detection():
    global BACKEND
    instance = request.json
    det_path = Path(instance["det_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")

    if Path(det_path).is_file():
        with open(det_path, "rb") as f:
            dt_annos = pickle.load(f)
    else:
        dt_annos = kitti.get_label_annos(det_path)
    LOGGER.info("%s %s", "da_annos shape:", len(dt_annos))
    BACKEND.dt_annos = dt_annos
    response = jsonify(results=[response])
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def _annotation_response(annos, rect, trv2c, whwh, prefix=""):
    dims = annos["dimensions"]
    num_obj = dims.shape[0]
    loc = annos["location"]
    bbox = annos["bbox"] / whwh
    boxes_camera = np.concatenate(
        [loc, dims, annos["rotation_y"][..., np.newaxis]], axis=1
    )
    boxes = box_np_ops.box_camera_to_lidar(boxes_camera, rect, trv2c)
    box_np_ops.change_box3d_center_(
        boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5]
    )
    rotations = np.concatenate(
        [np.zeros([num_obj, 2], dtype=np.float32), -boxes[:, 6:7]], axis=1
    )
    result = {
        prefix + "locs": boxes[:, :3].tolist(),
        prefix + "dims": boxes[:, 3:6].tolist(),
        prefix + "rots": rotations.tolist(),
        prefix + "bbox": bbox.tolist(),
        prefix + "labels": annos["name"][:num_obj].tolist(),
    }
    if "score" in annos:
        result[prefix + "scores"] = annos["score"].tolist()
    return result


def _requested_sample():
    if BACKEND.root_path is None:
        return None, error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return None, error_response("kitti info is not loaded")
    instance = request.json
    image_idx = instance["image_idx"]
    index = BACKEND.image_idxes.index(image_idx)
    return (instance, index, BACKEND.kitti_infos[index]), None


def _json_response(response):
    result = jsonify(results=[response])
    result.headers["Access-Control-Allow-Headers"] = "*"
    return result


@app.route("/api/get_pointcloud", methods=["POST"])
def get_pointcloud():
    global BACKEND
    sample, error = _requested_sample()
    if error is not None:
        return error
    instance, idx, kitti_info = sample
    response = {"status": "normal"}
    rect = kitti_info["calib/R0_rect"]
    _ = kitti_info["calib/P2"]
    trv2c = kitti_info["calib/Tr_velo_to_cam"]
    img_shape = kitti_info["img_shape"]  # hw
    wh = np.array(img_shape[::-1])
    whwh = np.tile(wh, 2)
    if "annos" in kitti_info:
        annos = kitti_info["annos"]
        num_obj = len([n for n in annos["name"] if n != "DontCare"])
        gt_fields = ("dimensions", "location", "rotation_y", "bbox", "name")
        gt_annos = {key: annos[key][:num_obj] for key in gt_fields}
        response.update(_annotation_response(gt_annos, rect, trv2c, whwh))

    v_path = str(Path(BACKEND.root_path) / kitti_info["velodyne_path"])
    LOGGER.info("%s %s", "*******************", v_path)
    with open(v_path, "rb") as f:
        pc_str = base64.encodebytes(f.read())
    response["pointcloud"] = pc_str.decode("utf-8")
    if "with_det" in instance and instance["with_det"]:
        if BACKEND.dt_annos is None:
            return error_response("det anno is not loaded")
        dt_annos = BACKEND.dt_annos[idx]
        response.update(_annotation_response(dt_annos, rect, trv2c, whwh, "dt_"))

    LOGGER.info("%s", "send response!")
    return _json_response(response)


@app.route("/api/get_image", methods=["POST"])
def get_image():
    global BACKEND
    sample, error = _requested_sample()
    if error is not None:
        return error
    _, _, kitti_info = sample
    response = {"status": "normal"}
    _ = kitti_info["calib/R0_rect"]
    _ = kitti_info["calib/P2"]
    _ = kitti_info["calib/Tr_velo_to_cam"]
    if "img_path" in kitti_info:
        img_path = kitti_info["img_path"]
        if img_path != "":
            image_path = BACKEND.root_path / img_path
            LOGGER.info("%s", image_path)
            with open(str(image_path), "rb") as f:
                image_str = f.read()
            response["image_b64"] = base64.b64encode(image_str).decode("utf-8")
            response["image_b64"] = (
                "data:image/{};base64,".format(image_path.suffix[1:])
                + response["image_b64"]
            )
            LOGGER.info(
                "%s", "send an image with size {}!".format(len(response["image_b64"]))
            )
    return _json_response(response)


@app.route("/api/build_network", methods=["POST"])
def build_network():
    global BACKEND
    instance = request.json
    cfg_path = Path(instance["config_path"])
    ckpt_path = Path(instance["checkpoint_path"])
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")
    if not cfg_path.exists():
        return error_response("config file not exist.")
    if not ckpt_path.exists():
        return error_response("ckpt file not exist.")
    BACKEND.inference_ctx = TorchInferenceContext()
    BACKEND.inference_ctx.build(str(cfg_path))
    BACKEND.inference_ctx.restore(str(ckpt_path))
    response = jsonify(results=[response])
    response.headers["Access-Control-Allow-Headers"] = "*"
    LOGGER.info("%s", "build_network successful!")
    return response


@app.route("/api/inference_by_idx", methods=["POST"])
def inference_by_idx():
    global BACKEND
    instance = request.json
    response = {"status": "normal"}
    if BACKEND.root_path is None:
        return error_response("root path is not set")
    if BACKEND.kitti_infos is None:
        return error_response("kitti info is not loaded")
    if BACKEND.inference_ctx is None:
        return error_response("inference_ctx is not loaded")
    image_idx = instance["image_idx"]
    idx = BACKEND.image_idxes.index(image_idx)
    kitti_info = BACKEND.kitti_infos[idx]

    v_path = str(Path(BACKEND.root_path) / kitti_info["velodyne_path"])
    i_path = str(Path(BACKEND.root_path) / kitti_info["img_path"])
    num_features = 4
    points = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape(
        [-1, num_features]
    )
    rect = kitti_info["calib/R0_rect"]
    p2 = kitti_info["calib/P2"]
    trv2c = kitti_info["calib/Tr_velo_to_cam"]
    if "img_shape" in kitti_info:
        image_shape = kitti_info["img_shape"]
        points = box_np_ops.remove_outside_points(points, rect, trv2c, p2, image_shape)
        LOGGER.info("%s", points.shape[0])
    img_shape = kitti_info["img_shape"]  # hw
    wh = np.array(img_shape[::-1])
    whwh = np.tile(wh, 2)

    t = time.time()
    inputs = BACKEND.inference_ctx.get_inference_input_dict(kitti_info, points, i_path)
    LOGGER.info("%s %s", "input preparation time:", time.time() - t)
    t = time.time()
    with BACKEND.inference_ctx.ctx():
        dt_annos = BACKEND.inference_ctx.inference(inputs)[0]
    LOGGER.info("%s %s", "detection time:", time.time() - t)
    response.update(_annotation_response(dt_annos, rect, trv2c, whwh, "dt_"))

    response = jsonify(results=[response])
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def main(port=16666):
    app.run(host="127.0.0.1", threaded=True, port=port)


if __name__ == "__main__":
    fire.Fire()
