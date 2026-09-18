# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

from easydict import EasyDict as edict
import numpy as np
from shared.configuration import apply_config_list, merge_config

LOGGER = logging.getLogger(__name__)

__C = edict()
cfg = __C

# 0. basic config
__C.TAG = "default"
__C.CLASSES = "Car"

__C.INCLUDE_SIMILAR_TYPE = False

# config of augmentation
__C.AUG_DATA = True
__C.AUG_METHOD_LIST = ["rotation", "scaling", "flip"]
__C.AUG_METHOD_PROB = [0.5, 0.5, 0.5]
__C.AUG_ROT_RANGE = 18

__C.GT_AUG_ENABLED = False
__C.GT_EXTRA_NUM = 15
__C.GT_AUG_RAND_NUM = False
__C.GT_AUG_APPLY_PROB = 0.75
__C.GT_AUG_HARD_RATIO = 0.6

__C.PC_REDUCE_BY_RANGE = True
__C.PC_AREA_SCOPE = np.array(
    [[-40, 40], [-1, 3], [0, 70.4]]
)  # x, y, z scope in rect camera coords

__C.CLS_MEAN_SIZE = np.array([[1.52, 1.63, 3.88]], dtype=np.float32)

# 0.1 config of use img
__C.USE_IOU_BRANCH = False

# config of LI-Fusion
__C.LI_FUSION = edict()
__C.LI_FUSION.ENABLED = False
__C.LI_FUSION.IMG_FEATURES_CHANNEL = 128
__C.LI_FUSION.ADD_Image_Attention = False
__C.LI_FUSION.IMG_CHANNELS = [3, 64, 128, 256, 512]
__C.LI_FUSION.POINT_CHANNELS = [96, 256, 512, 1024]

__C.LI_FUSION.DeConv_Reduce = [16, 16, 16, 16]
__C.LI_FUSION.DeConv_Kernels = [2, 4, 8, 16]
__C.LI_FUSION.DeConv_Strides = [2, 4, 8, 16]


# 1. config of rpn network
__C.RPN = edict()
__C.RPN.ENABLED = True
__C.RPN.FIXED = False

__C.RPN.USE_INTENSITY = True

# 1.1 config of use img_rgb input (x)
__C.RPN.USE_RGB = False

# config of bin-based loss
__C.RPN.LOC_XZ_FINE = False
__C.RPN.LOC_SCOPE = 3.0
__C.RPN.LOC_BIN_SIZE = 0.5
__C.RPN.NUM_HEAD_BIN = 12

# config of network structure
__C.RPN.BACKBONE = "pointnet2_msg"

__C.RPN.USE_BN = True
__C.RPN.NUM_POINTS = 16384

__C.RPN.SA_CONFIG = edict()
__C.RPN.SA_CONFIG.NPOINTS = [4096, 1024, 256, 64]
__C.RPN.SA_CONFIG.RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
__C.RPN.SA_CONFIG.NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
__C.RPN.SA_CONFIG.MLPS = [
    [[16, 16, 32], [32, 32, 64]],
    [[64, 64, 128], [64, 96, 128]],
    [[128, 196, 256], [128, 196, 256]],
    [[256, 256, 512], [256, 384, 512]],
]
__C.RPN.FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
__C.RPN.CLS_FC = [128]
__C.RPN.REG_FC = [128]
__C.RPN.DP_RATIO = 0.5

# config of training
__C.RPN.LOSS_CLS = "DiceLoss"
__C.RPN.FG_WEIGHT = 15
__C.RPN.FOCAL_ALPHA = [0.25, 0.75]
__C.RPN.FOCAL_GAMMA = 2.0
__C.RPN.REG_LOSS_WEIGHT = [1.0, 1.0, 1.0, 1.0]
__C.RPN.LOSS_WEIGHT = [1.0, 1.0]
__C.RPN.NMS_TYPE = "normal"  # normal, rotate

# config of testing
__C.RPN.SCORE_THRESH = 0.3

# 2. config of rcnn network
__C.RCNN = edict()
__C.RCNN.ENABLED = False

# config of input
__C.RCNN.USE_RPN_FEATURES = True
__C.RCNN.USE_MASK = True
__C.RCNN.MASK_TYPE = "seg"
__C.RCNN.USE_INTENSITY = False
__C.RCNN.USE_DEPTH = True
__C.RCNN.USE_SEG_SCORE = False
__C.RCNN.ROI_SAMPLE_JIT = False
__C.RCNN.ROI_FG_AUG_TIMES = 10

__C.RCNN.REG_AUG_METHOD = "multiple"  # multiple, single, normal
__C.RCNN.POOL_EXTRA_WIDTH = 1.0

__C.RCNN.USE_RGB = False
# config of bin-based loss
__C.RCNN.LOC_SCOPE = 1.5
__C.RCNN.LOC_BIN_SIZE = 0.5
__C.RCNN.NUM_HEAD_BIN = 9
__C.RCNN.LOC_Y_BY_BIN = False
__C.RCNN.LOC_Y_SCOPE = 0.5
__C.RCNN.LOC_Y_BIN_SIZE = 0.25
__C.RCNN.SIZE_RES_ON_ROI = False

# config of network structure
__C.RCNN.USE_BN = False
__C.RCNN.DP_RATIO = 0.0

__C.RCNN.BACKBONE = "pointnet"  # pointnet, pointsift
__C.RCNN.XYZ_UP_LAYER = [128, 128]

__C.RCNN.NUM_POINTS = 512
__C.RCNN.SA_CONFIG = edict()
__C.RCNN.SA_CONFIG.NPOINTS = [128, 32, -1]
__C.RCNN.SA_CONFIG.RADIUS = [0.2, 0.4, 100]
__C.RCNN.SA_CONFIG.NSAMPLE = [64, 64, 64]
__C.RCNN.SA_CONFIG.MLPS = [[128, 128, 128], [128, 128, 256], [256, 256, 512]]
__C.RCNN.CLS_FC = [256, 256]
__C.RCNN.REG_FC = [256, 256]

# config of training
__C.RCNN.LOSS_CLS = "BinaryCrossEntropy"
__C.RCNN.FOCAL_ALPHA = [0.25, 0.75]
__C.RCNN.FOCAL_GAMMA = 2.0
__C.RCNN.CLS_WEIGHT = np.array([1.0, 1.0, 1.0], dtype=np.float32)
__C.RCNN.CLS_FG_THRESH = 0.6
__C.RCNN.CLS_BG_THRESH = 0.45
__C.RCNN.CLS_BG_THRESH_LO = 0.05
__C.RCNN.REG_FG_THRESH = 0.55
__C.RCNN.FG_RATIO = 0.5
__C.RCNN.ROI_PER_IMAGE = 64
__C.RCNN.HARD_BG_RATIO = 0.6
#
__C.RCNN.IOU_LOSS_TYPE = "raw"
__C.RCNN.IOU_ANGLE_POWER = 1

# config of testing
__C.RCNN.SCORE_THRESH = 0.3
__C.RCNN.NMS_THRESH = 0.1

# general training config
__C.TRAIN = edict()
__C.TRAIN.SPLIT = "train"
__C.TRAIN.VAL_SPLIT = "smallval"

__C.TRAIN.LR = 0.002
__C.TRAIN.LR_CLIP = 0.00001
__C.TRAIN.LR_DECAY = 0.5
__C.TRAIN.DECAY_STEP_LIST = [50, 100, 150, 200, 250, 300]
__C.TRAIN.LR_WARMUP = False
__C.TRAIN.WARMUP_MIN = 0.0002
__C.TRAIN.WARMUP_EPOCH = 5

__C.TRAIN.BN_MOMENTUM = 0.9
__C.TRAIN.BN_DECAY = 0.5
__C.TRAIN.BNM_CLIP = 0.01
__C.TRAIN.BN_DECAY_STEP_LIST = [50, 100, 150, 200, 250, 300]

__C.TRAIN.OPTIMIZER = "adam"
__C.TRAIN.WEIGHT_DECAY = 0.0  # "L2 regularization coeff [default: 0.0]"
__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.MOMS = [0.95, 0.85]
__C.TRAIN.DIV_FACTOR = 10.0
__C.TRAIN.PCT_START = 0.4

__C.TRAIN.GRAD_NORM_CLIP = 1.0

__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
__C.TRAIN.RPN_POST_NMS_TOP_N = 2048
__C.TRAIN.RPN_NMS_THRESH = 0.85
__C.TRAIN.RPN_DISTANCE_BASED_PROPOSE = True

#
__C.TRAIN.RPN_TRAIN_WEIGHT = 1.0
__C.TRAIN.RCNN_TRAIN_WEIGHT = 1.0
__C.TRAIN.CE_WEIGHT = 5.0
__C.TRAIN.IOU_LOSS_TYPE = "cls_mask_with_bin"
__C.TRAIN.BBOX_AVG_BY_BIN = True
__C.TRAIN.RY_WITH_BIN = False

__C.TEST = edict()
__C.TEST.SPLIT = "val"
__C.TEST.RPN_PRE_NMS_TOP_N = 9000
__C.TEST.RPN_POST_NMS_TOP_N = 300
__C.TEST.RPN_NMS_THRESH = 0.7
__C.TEST.RPN_DISTANCE_BASED_PROPOSE = True
#
__C.TEST.BBOX_AVG_BY_BIN = True
__C.TEST.RY_WITH_BIN = False


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    merge_config(a, b, edict, LOGGER)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    apply_config_list(__C, cfg_list)


def save_config_to_file(config, pre="config", logger=None):
    for key, val in config.items():
        if isinstance(config[key], edict):
            if logger is not None:
                logger.info("\n%s.%s = edict()" % (pre, key))
            else:
                LOGGER.info("%s", "\n%s.%s = edict()" % (pre, key))
            save_config_to_file(config[key], pre=pre + "." + key, logger=logger)
            continue

        if logger is not None:
            logger.info("%s.%s: %s" % (pre, key, val))
        else:
            LOGGER.info("%s", "%s.%s: %s" % (pre, key, val))
