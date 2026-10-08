# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FConv compatibility layer for the repository-wide KITTI utilities."""

from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(_REPOSITORY_ROOT))

from shared.compatibility import load_from_path


_shared = load_from_path(
    "ai_msf_shared_kitti_util", _REPOSITORY_ROOT / "utils" / "kitti_util.py"
)


class Object3d(_shared.Object3d):
    """Retain the score and serialization API used by the FConv data tools."""

    def __init__(self, label_file_line):
        super(Object3d, self).__init__(label_file_line)
        fields = label_file_line.split()
        self.score = float(fields[15]) if len(fields) > 15 else 1

    def get_string_ann(self):
        values = (
            self.truncation, self.occlusion, self.alpha,
            self.xmin, self.ymin, self.xmax, self.ymax,
            self.h, self.w, self.length_value,
            self.t[0], self.t[1], self.t[2], self.ry, self.score,
        )
        return self.type + " %d %d" % values[:2] + " " + " ".join(
            "%.6f" % value for value in values[2:]
        )


Calibration = _shared.Calibration
rotx = _shared.rotx
roty = _shared.roty
rotz = _shared.rotz
transform_from_rot_trans = _shared.transform_from_rot_trans
inverse_rigid_trans = _shared.inverse_rigid_trans
load_image = _shared.load_image
load_velo_scan = _shared.load_velo_scan
project_to_image = _shared.project_to_image
compute_box_3d = _shared.compute_box_3d
compute_orientation_3d = _shared.compute_orientation_3d


def read_label(label_filename):
    with open(label_filename) as label_file:
        return [Object3d(line.rstrip()) for line in label_file]
