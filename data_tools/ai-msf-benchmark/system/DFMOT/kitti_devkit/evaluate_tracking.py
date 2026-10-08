#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
    function that does the evaluation
    input:
      - result_sha (sha key where the results are located
      - mail (messenger object for output messages sent via email and to cout)
    output:
      - True if at least one of the sub-benchmarks could be processed successfully
      - False otherwise
    data:
      - at this point the submitted files are located in results/<result_sha>/data
      - the results shall be saved as follows
        -> summary statistics of the method: results/<result_sha>/stats_task.txt
           here task refers to the sub-system (e.g., um_lane, uu_road etc.)
           file contents: numbers for main table, format: %.6f (single space separated)
           note: only files with successful sub-system evaluation must be created
        -> detailed results/graphics/plots: results/<result_sha>/subdir
           with appropriate subdir and file names (all subdir's need to be created)
"""
from collections import defaultdict
from typing import NamedTuple
import copy
import logging
import math
import os
import sys

from munkres import Munkres
import kitti_devkit.mailpy as mailpy


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters
from shared.parameters import bind_named as _bind_named


class TdataInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    frame: object
    obj_type: object
    truncation: object
    occlusion: object
    obs_angle: object
    x1: object
    y1: object
    x2: object
    y2: object
    w: object
    h: object
    length_value: object
    x: object
    y: object
    z: object
    yaw: object
    score: object
    track_id: object


class TrackingevaluationInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    t_sha: object
    root: object
    part: object
    gt_path: object
    min_overlap: object
    max_truncation: object
    min_height: object
    max_occlusion: object
    mail: object
    cls: object


class TrackingEvaluationResult(NamedTuple):
    """Named result returned without changing tuple unpacking behavior."""

    value_0: object
    value_1: object
    value_2: object
    value_3: object
    value_4: object
    value_5: object
    value_6: object
    value_7: object


class TrackingStatsResult(NamedTuple):
    """Named result returned without changing tuple unpacking behavior."""

    value_0: object
    value_1: object
    value_2: object
    value_3: object
    value_4: object
    value_5: object
    value_6: object
    value_7: object


class TrackingFileStats(NamedTuple):
    """Values written to the KITTI tracking statistics file."""

    mota: object
    motp: object
    motal: object
    moda: object
    modp: object
    recall: object
    precision: object
    f1: object
    far: object
    mt: object
    pt: object
    ml: object
    tp: object
    fp: object
    fn: object
    id_switches: object
    fragments: object
    n_gt: object
    n_gt_trajectories: object
    n_tr: object
    n_tr_trajectories: object


LOGGER = logging.getLogger(__name__)


_TDATA_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    TdataInitParameters,
    (
        "frame",
        "obj_type",
        "truncation",
        "occlusion",
        "obs_angle",
        "x1",
        "y1",
        "x2",
        "y2",
        "w",
        "h",
        "length_value",
        "x",
        "y",
        "z",
        "yaw",
        "score",
        "track_id",
    ),
    {
        "frame": -1,
        "obj_type": "unset",
        "truncation": -1,
        "occlusion": -1,
        "obs_angle": -10,
        "x1": -1,
        "y1": -1,
        "x2": -1,
        "y2": -1,
        "w": -1,
        "h": -1,
        "length_value": -1,
        "x": -1000,
        "y": -1000,
        "z": -1000,
        "yaw": -10,
        "score": -1000,
        "track_id": -1,
    },
    frozenset(()),
    18,
    None,
    None,
)
_TRACKINGEVALUATION_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    TrackingevaluationInitParameters,
    (
        "t_sha",
        "root",
        "part",
        "gt_path",
        "min_overlap",
        "max_truncation",
        "min_height",
        "max_occlusion",
        "mail",
        "cls",
    ),
    {
        "part": "all",
        "gt_path": "./data/tracking",
        "min_overlap": 0.5,
        "max_truncation": 0,
        "min_height": 25,
        "max_occlusion": 2,
        "mail": None,
        "cls": "car",
    },
    frozenset(()),
    10,
    None,
    None,
)


class TData:
    """
    Utility class to load data.
    """

    def __init__(self, *call_args, **call_kwargs):
        """
        Constructor, initializes the object given the parameters.
        """
        bound_parameters = _bind_parameters(
            _TDATA_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            frame,
            obj_type,
            truncation,
            occlusion,
            obs_angle,
            x1,
            y1,
            x2,
            y2,
            w,
            h,
            length_value,
            x,
            y,
            z,
            yaw,
            score,
            track_id,
        ) = bound_parameters

        # init object data
        self.frame = frame
        self.track_id = track_id
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.length_value = length_value
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.score = score
        self.ignored = False
        self.valid = False
        self.tracker = -1

    def __str__(self):
        """
        Print read data.
        """

        attrs = vars(self)
        return "\n".join("%s: %s" % item for item in attrs.items())


TRAIN_SEQ_ID = [
    "0003",
    "0001",
    "0013",
    "0009",
    "0004",
    "0020",
    "0006",
    "0015",
    "0008",
    "0012",
]
VALID_SEQ_ID = [
    "0005",
    "0007",
    "0017",
    "0011",
    "0002",
    "0014",
    "0000",
    "0010",
    "0016",
    "0019",
    "0018",
]
# Valid sequence 0017 has no cars in detection,
# so it should not be included if val with GT detection
# '0000', '0010', '0016', '0019', '0018']
TRAINVAL_SEQ_ID = [f"{i:04d}" for i in range(21)]


def initialize_tracking_statistics(evaluation):
    scalar_fields = (
        "n_gt", "n_igt", "n_gt_trajectories", "n_tr", "n_itr",
        "n_igttr", "n_tr_trajectories", "mota", "motp", "motal", "moda",
        "modp", "recall", "precision", "f1", "far", "total_cost", "itp",
        "tp", "fn", "ifn", "fp", "mme", "fragments", "id_switches",
        "mt", "pt", "ml",
    )
    list_fields = (
        "n_gts", "n_igts", "n_gt_seq", "n_trs", "n_itrs", "n_tr_seq",
        "modp_t", "itps", "tps", "fns", "ifns", "fps",
    )
    for field in scalar_fields:
        setattr(evaluation, field, 0)
    for field in list_fields:
        setattr(evaluation, field, [])


class TrackingEvaluation(object):
    """tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
    MOTA	- Multi-object tracking accuracy in [0,100]
    MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
    MOTAL	- Multi-object tracking accuracy in [0,100] with log10(id-switches)

    id-switches - number of id switches
    fragments   - number of fragmentations

    MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories

    recall	        - recall = percentage of detected targets
    precision	    - precision = percentage of correctly detected targets
    FAR		    - number of false alarms per frame
    falsepositives - number of false positives (FP)
    missed         - number of missed targets (FN)
    """

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRACKINGEVALUATION_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            t_sha,
            root,
            part,
            gt_path,
            min_overlap,
            max_truncation,
            min_height,
            max_occlusion,
            mail,
            cls,
        ) = bound_parameters
        filename_test_mapping = "./data/tracking/evaluate_tracking.seqmap"
        self.n_frames = []
        self.sequence_name = []
        self.sequence_id = []
        if part == "val":
            self.sequence_id = VALID_SEQ_ID
        elif part == "train":
            self.sequence_id = TRAIN_SEQ_ID
        elif part == "all":
            self.sequence_id = TRAINVAL_SEQ_ID
        with open(filename_test_mapping, "r") as fh:
            for _, length_value in enumerate(fh):
                fields = length_value.split(" ")
                seq_id = "%04d" % int(fields[0])
                if seq_id in self.sequence_id:
                    self.sequence_name.append(seq_id)
                    self.n_frames.append(int(fields[3]) - int(fields[2]) + 1)
        fh.close()
        self.n_sequences = len(self.sequence_name)

        # mail object
        self.mail = mail

        # class to evaluate, i.e. pedestrian or car
        self.cls = cls

        # data and parameter
        self.gt_path = os.path.join(gt_path, "label_02")
        self.t_sha = t_sha
        self.t_path = os.path.join(root, t_sha, part)

        initialize_tracking_statistics(self)

        self.min_overlap = min_overlap
        self.max_truncation = max_truncation
        self.max_occlusion = max_occlusion
        self.min_height = min_height  # minimum height of an object for evaluation
        self.n_sample_points = 500

        # this should be enough to hold all groundtruth trajectories
        # is expanded if necessary and reduced in any case
        self.gt_trajectories = [[] for x in range(self.n_sequences)]
        self.ign_trajectories = [[] for x in range(self.n_sequences)]

    @staticmethod
    def _new_sequence_stats():
        return {
            key: 0
            for key in ("tp", "itp", "fn", "ifn", "fp", "igt", "itr", "n_gts", "n_trs")
        }

    @staticmethod
    def _validate_frame_counts(
        frame_info, frame_stats, ignored_stats, association_matrix, logger
    ):
        sequence_index, frame_index, ground_truth, tracker = frame_info
        ignored_tracker, ignored_fn, ignored_tp, ignored_pairs = ignored_stats
        true_positive, false_positive, false_negative = (
            frame_stats["tp"],
            frame_stats["fp"],
            frame_stats["fn"],
        )
        if min(true_positive, false_positive, false_negative) < 0:
            logger.info(
                "invalid tracking counts at sequence %s frame %s: tp=%s fp=%s fn=%s",
                sequence_index,
                frame_index,
                true_positive,
                false_positive,
                false_negative,
            )
            raise ValueError("tracking counts must not be negative")
        if (
            true_positive + false_negative
            != len(ground_truth) - ignored_fn - ignored_tp
        ):
            raise ValueError("ground-truth count is not TP+FN")
        tracker_count = (
            true_positive
            + false_positive
            + ignored_tp
            + ignored_tracker
            - ignored_pairs
        )
        if tracker_count != len(tracker):
            logger.info(
                "invalid tracker count at sequence %s frame %s: %s != %s; associations=%s",
                sequence_index,
                frame_index,
                tracker_count,
                len(tracker),
                association_matrix,
            )
            raise ValueError("tracker count is not TP+FP")

    @staticmethod
    def _is_id_switch(previous_id, current_id, previous_frame_id):
        return (
            previous_id != current_id
            and previous_id != -1
            and current_id != -1
            and previous_frame_id != -1
        )

    @staticmethod
    def _is_fragment(trajectory, frame_index, previous_id, current_id):
        return (
            frame_index < len(trajectory) - 1
            and trajectory[frame_index - 1] != current_id
            and previous_id != -1
            and current_id != -1
            and trajectory[frame_index + 1] != -1
        )

    @staticmethod
    def _stats_descriptions():
        return (
            ("MOTA", "MOTP", "MOTAL", "MODA", "MODP", "recall", "precision", "F1", "FAR"),
            ("MT", "PT", "ML", "tp", "fp", "fn", "id_switches", "fragments"),
            ("n_gt", "n_gt_trajectories", "n_tr", "n_tr_trajectories"),
        )

    @staticmethod
    def _tracking_classes(cls):
        if "car" in cls.lower():
            return ["car", "van", "dontcare"]
        if "pedestrian" in cls.lower():
            return ["pedestrian", "person_sitting", "dontcare"]
        return [cls.lower(), "dontcare"]

    @staticmethod
    def _parse_tracking_values(fields):
        detection = TData()
        detection.frame = int(float(fields[0]))
        detection.track_id = int(float(fields[1]))
        detection.obj_type = fields[2].lower()
        detection.truncation = int(float(fields[3]))
        detection.occlusion = int(float(fields[4]))
        detection.obs_angle = float(fields[5])
        detection.x1, detection.y1 = float(fields[6]), float(fields[7])
        detection.x2, detection.y2 = float(fields[8]), float(fields[9])
        detection.h, detection.w = float(fields[10]), float(fields[11])
        detection.length_value = float(fields[12])
        detection.x, detection.y, detection.z = map(float, fields[13:16])
        detection.yaw = float(fields[16])
        return detection

    @staticmethod
    def _record_sequence_core(
        detection, state, loading_groundtruth, extension_logger, duplicate_logger
    ):
        frame_data, trajectory_ids, id_frame_cache, eval_2d, eval_3d = state
        while detection.frame >= len(frame_data):
            extension_logger(detection.frame, len(frame_data))
            extension = max(500, detection.frame - len(frame_data) + 1)
            frame_data.extend([[] for _ in range(extension)])
        id_frame = (detection.frame, detection.track_id)
        if id_frame in id_frame_cache and not loading_groundtruth:
            duplicate_logger(detection)
            return None
        id_frame_cache.add(id_frame)
        frame_data[detection.frame].append(copy.copy(detection))
        if detection.track_id not in trajectory_ids and detection.obj_type != "dontcare":
            trajectory_ids.append(detection.track_id)
        if not loading_groundtruth:
            eval_2d = eval_2d and -1 not in (
                detection.x1, detection.x2, detection.y1, detection.y2,
            )
            eval_3d = eval_3d and -1000 not in (
                detection.x, detection.y, detection.z,
            )
        return frame_data, trajectory_ids, id_frame_cache, eval_2d, eval_3d

    def create_eval_dir(self):
        """
        Creates directory to store evaluation results and data for visualization.
        """

        self.eval_dir = os.path.join(self.t_path, "eval", self.cls)
        if not os.path.exists(self.eval_dir):
            LOGGER.info("%s %s", "create directory:", self.eval_dir)
            os.makedirs(self.eval_dir)
            LOGGER.info("%s", "done")

    def load_groundtruth(self):
        """
        Helper function to load ground truth.
        """

        try:
            self._load_data(self.gt_path, cls=self.cls, loading_groundtruth=True)
        except IOError:
            return False
        return True

    def load_tracker(self):
        """
        Helper function to load tracker data.
        """

        try:
            if not self._load_data(
                self.t_path, cls=self.cls, loading_groundtruth=False
            ):
                return False
        except IOError:
            return False
        return True

    def boxoverlap(self, a, b, criterion="union"):
        """
        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
        """

        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        w = x2 - x1
        h = y2 - y1

        if w <= 0.0 or h <= 0.0:
            return 0.0
        inter = w * h
        aarea = (a.x2 - a.x1) * (a.y2 - a.y1)
        barea = (b.x2 - b.x1) * (b.y2 - b.y1)
        # intersection over union overlap
        if criterion.lower() == "union":
            o = inter / float(aarea + barea - inter)
        elif criterion.lower() == "a":
            o = float(inter) / float(aarea)
        else:
            raise TypeError("Unkown type for criterion")
        return o

    def _build_frame_association(self, ground_truth, tracker, trajectories, ignored):
        max_cost = 1e9
        cost_matrix = []
        current_ids = [[], []]
        for gt_item in ground_truth:
            current_ids[0].append(gt_item.track_id)
            current_ids[1].append(-1)
            gt_item.tracker = -1
            gt_item.id_switch = 0
            gt_item.fragmentation = 0
            cost_row = []
            for tracker_item in tracker:
                cost = 1 - self.boxoverlap(gt_item, tracker_item)
                cost_row.append(cost if cost <= self.min_overlap else max_cost)
            cost_matrix.append(cost_row)
            trajectories[gt_item.track_id].append(-1)
            ignored[gt_item.track_id].append(False)
        return ([[]] if not ground_truth else cost_matrix), current_ids

    def _apply_association(self, ground_truth, tracker, association_data, trajectories):
        cost_matrix, association_matrix, current_ids = association_data
        stats = {"tp": 0, "fp": 0, "fn": 0, "cost": 0, "costs": [0] * len(ground_truth)}
        for row, column in association_matrix:
            cost = cost_matrix[row][column]
            if cost >= 1e9:
                ground_truth[row].tracker = -1
                self.fn += 1
                stats["fn"] += 1
                continue
            gt_item, tracker_item = ground_truth[row], tracker[column]
            gt_item.tracker = tracker_item.track_id
            current_ids[1][row] = tracker_item.track_id
            tracker_item.valid = True
            gt_item.distance = cost
            overlap = 1 - cost
            self.total_cost += overlap
            stats["cost"] += overlap
            stats["costs"][row] = overlap
            trajectories[gt_item.track_id][-1] = tracker_item.track_id
            self.tp += 1
            stats["tp"] += 1
        return stats

    def _ignored_trackers(self, tracker, dontcare):
        ignored_count = 0
        ignored_by_id = {}
        for tracker_item in tracker:
            ignored_by_id[tracker_item.track_id] = -1
            neighboring = (self.cls == "car" and tracker_item.obj_type == "van") or (
                self.cls == "pedestrian" and tracker_item.obj_type == "person_sitting"
            )
            if not tracker_item.valid and (
                neighboring or abs(tracker_item.y1 - tracker_item.y2) <= self.min_height
            ):
                tracker_item.ignored = True
            if not tracker_item.valid and not tracker_item.ignored:
                tracker_item.ignored = any(
                    self.boxoverlap(tracker_item, area, "a") > 0.5 for area in dontcare
                )
            if tracker_item.ignored:
                ignored_count += 1
                ignored_by_id[tracker_item.track_id] = 1
        return ignored_count, ignored_by_id

    def _ground_truth_is_ignored(self, item):
        neighboring = (self.cls == "car" and item.obj_type == "van") or (
            self.cls == "pedestrian" and item.obj_type == "person_sitting"
        )
        return (
            item.occlusion > self.max_occlusion
            or item.truncation > self.max_truncation
            or neighboring
        )

    def _ignored_ground_truth(
        self, ground_truth, ignored_trajectories, ignored_by_id, frame_stats
    ):
        ignored_fn = 0
        ignored_tp = 0
        ignored_pairs = 0
        for index, gt_item in enumerate(ground_truth):
            if not self._ground_truth_is_ignored(gt_item):
                continue
            ignored_trajectories[gt_item.track_id][-1] = True
            gt_item.ignored = True
            if gt_item.tracker < 0:
                ignored_fn += 1
                continue
            ignored_tp += 1
            if ignored_by_id[gt_item.tracker] > 0:
                ignored_pairs += 1
            frame_stats["cost"] -= frame_stats["costs"][index]
        return ignored_fn, ignored_tp, ignored_pairs

    def _update_frame_counts(self, *call_args, **call_kwargs):
        (
            ground_truth,
            tracker,
            association_matrix,
            frame_stats,
            ignored_stats,
            sequence_stats,
        ) = _bind_named(
            call_args,
            call_kwargs,
            ('ground_truth', 'tracker', 'association_matrix', 'frame_stats', 'ignored_stats', 'sequence_stats'),
        )
        ignored_tracker, ignored_fn, ignored_tp, ignored_pairs = ignored_stats
        frame_stats["tp"] -= ignored_tp
        self.itp += ignored_tp
        self.n_gt -= ignored_fn + ignored_tp
        self.n_igt += ignored_fn + ignored_tp
        self.n_itr += ignored_tracker
        self.n_igttr += ignored_pairs
        unassociated = len(ground_truth) - len(association_matrix) - ignored_fn
        frame_stats["fn"] += unassociated
        self.fn += unassociated
        self.ifn += ignored_fn
        frame_stats["fp"] = (
            len(tracker)
            - frame_stats["tp"]
            - ignored_tracker
            - ignored_tp
            + ignored_pairs
        )
        self.fp += frame_stats["fp"]
        for key, value in (
            ("tp", frame_stats["tp"]),
            ("itp", ignored_tp),
            ("fp", frame_stats["fp"]),
            ("fn", frame_stats["fn"]),
            ("ifn", ignored_fn),
            ("igt", ignored_fn + ignored_tp),
            ("itr", ignored_tracker),
        ):
            sequence_stats[key] += value

    def _mark_identity_changes(self, ground_truth, current_ids, previous_ids):
        for index, ground_truth_id in enumerate(current_ids[0]):
            if ground_truth_id not in previous_ids[0]:
                continue
            previous_index = previous_ids[0].index(ground_truth_id)
            tracker_id = current_ids[1][index]
            previous_tracker_id = previous_ids[1][previous_index]
            changed = tracker_id != previous_tracker_id and previous_tracker_id != -1
            if (
                changed
                and tracker_id != -1
                and ground_truth[index].truncation < self.max_truncation
            ):
                ground_truth[index].id_switch = 1
            if changed and ground_truth[index].truncation < self.max_truncation:
                ground_truth[index].fragmentation = 1

    def _evaluate_frame(self, frame_info, sequence_context, munkres, logger):
        sequence_index, frame_index, ground_truth, dontcare, tracker = frame_info
        trajectories, ignored_trajectories, sequence_stats, previous_ids = (
            sequence_context
        )
        self.n_gt += len(ground_truth)
        self.n_tr += len(tracker)
        sequence_stats["n_gts"] += len(ground_truth)
        sequence_stats["n_trs"] += len(tracker)
        cost_matrix, current_ids = self._build_frame_association(
            ground_truth, tracker, trajectories, ignored_trajectories
        )
        association_matrix = munkres.compute(cost_matrix)
        frame_stats = self._apply_association(
            ground_truth,
            tracker,
            (cost_matrix, association_matrix, current_ids),
            trajectories,
        )
        ignored_tracker, ignored_by_id = self._ignored_trackers(tracker, dontcare)
        ignored_fn, ignored_tp, ignored_pairs = self._ignored_ground_truth(
            ground_truth, ignored_trajectories, ignored_by_id, frame_stats
        )
        ignored_stats = (ignored_tracker, ignored_fn, ignored_tp, ignored_pairs)
        self._update_frame_counts(
            ground_truth,
            tracker,
            association_matrix,
            frame_stats,
            ignored_stats,
            sequence_stats,
        )
        self._validate_frame_counts(
            (sequence_index, frame_index, ground_truth, tracker),
            frame_stats,
            ignored_stats,
            association_matrix,
            logger,
        )
        self._mark_identity_changes(ground_truth, current_ids, previous_ids)
        modp_t = (
            frame_stats["cost"] / float(frame_stats["tp"]) if frame_stats["tp"] else 1
        )
        self.modp_t.append(modp_t)
        return current_ids

    def _evaluate_sequence(self, sequence_index, munkres, logger):
        sequence_stats = self._new_sequence_stats()
        trajectories, ignored_trajectories = defaultdict(list), defaultdict(list)
        previous_ids = [[], []]
        sequence_context = (
            trajectories,
            ignored_trajectories,
            sequence_stats,
            previous_ids,
        )
        for frame_index, ground_truth in enumerate(self.groundtruth[sequence_index]):
            frame_info = (
                sequence_index,
                frame_index,
                ground_truth,
                self.dcareas[sequence_index][frame_index],
                self.tracker[sequence_index][frame_index],
            )
            previous_ids = self._evaluate_frame(
                frame_info, sequence_context, munkres, logger
            )
            sequence_context = (
                trajectories,
                ignored_trajectories,
                sequence_stats,
                previous_ids,
            )
        self.gt_trajectories[sequence_index] = trajectories
        self.ign_trajectories[sequence_index] = ignored_trajectories
        for key, target in (
            ("n_gts", self.n_gts),
            ("n_trs", self.n_trs),
            ("tp", self.tps),
            ("itp", self.itps),
            ("fp", self.fps),
            ("fn", self.fns),
            ("ifn", self.ifns),
            ("igt", self.n_igts),
            ("itr", self.n_itrs),
        ):
            target.append(sequence_stats[key])

    def _trajectory_events(self, trajectory, ignored):
        last_id = trajectory[0]
        tracked = 1 if trajectory[0] >= 0 else 0
        for frame_index, current_id in enumerate(trajectory[1:], start=1):
            if ignored[frame_index]:
                last_id = -1
                continue
            if self._is_id_switch(
                last_id, current_id, trajectory[frame_index - 1]
            ):
                self.id_switches += 1
            if self._is_fragment(trajectory, frame_index, last_id, current_id):
                self.fragments += 1
            if current_id != -1:
                tracked += 1
                last_id = current_id
        last_changed = len(trajectory) > 1 and trajectory[-2] != trajectory[-1]
        valid_last_ids = last_id != -1 and trajectory[-1] != -1
        if last_changed and valid_last_ids and not ignored[-1]:
            self.fragments += 1
        return tracked

    def _evaluate_trajectories(self):
        ignored_total = 0
        for trajectories, ignored_trajectories in zip(
            self.gt_trajectories, self.ign_trajectories
        ):
            for trajectory, ignored in zip(
                trajectories.values(), ignored_trajectories.values()
            ):
                if all(ignored):
                    ignored_total += 1
                    continue
                if all(item == -1 for item in trajectory):
                    self.ml += 1
                    continue
                tracked = self._trajectory_events(trajectory, ignored)
                tracking_ratio = tracked / float(len(trajectory) - sum(ignored))
                if tracking_ratio > 0.8:
                    self.mt += 1
                elif tracking_ratio < 0.2:
                    self.ml += 1
                else:
                    self.pt += 1
        return ignored_total

    def _finalize_tracking_ratios(self, ignored_trajectories):
        valid_trajectories = self.n_gt_trajectories - ignored_trajectories
        if valid_trajectories == 0:
            self.mt = self.pt = self.ml = 0.0
        else:
            self.mt /= float(valid_trajectories)
            self.pt /= float(valid_trajectories)
            self.ml /= float(valid_trajectories)
        if self.fp + self.tp == 0 or self.tp + self.fn == 0:
            self.recall = self.precision = 0.0
        else:
            self.recall = self.tp / float(self.tp + self.fn)
            self.precision = self.tp / float(self.fp + self.tp)
        self.f1 = (
            0.0
            if self.recall + self.precision == 0
            else (2.0 * self.precision * self.recall / (self.precision + self.recall))
        )
        self.far = (
            "n/a" if sum(self.n_frames) == 0 else self.fp / float(sum(self.n_frames))
        )

    def _finalize_clear_mot(self):
        if self.n_gt == 0:
            self.mota = self.moda = -float("inf")
        else:
            self.mota = 1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)
            self.moda = 1 - (self.fn + self.fp) / float(self.n_gt)
        self.motp = float("inf") if self.tp == 0 else self.total_cost / float(self.tp)
        if self.n_gt == 0:
            self.motal = -float("inf")
        elif self.id_switches == 0:
            self.motal = 1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)
        else:
            self.motal = 1 - (self.fn + self.fp + math.log10(self.id_switches)) / float(
                self.n_gt
            )
        self.modp = (
            "n/a"
            if sum(self.n_frames) == 0
            else sum(self.modp_t) / float(sum(self.n_frames))
        )

    def compute3rd_party_metrics(self):
        logger = LOGGER
        """Compute CLEAR MOT and trajectory tracking metrics."""
        munkres = Munkres()
        for sequence_index, _ in enumerate(self.groundtruth):
            self._evaluate_sequence(sequence_index, munkres, logger)
        ignored_trajectories = self._evaluate_trajectories()
        self._finalize_tracking_ratios(ignored_trajectories)
        self._finalize_clear_mot()
        return True

    def create_summary(self):
        sections = (
            (
                ("Multiple Object Tracking Accuracy (MOTA)", self.mota),
                ("Multiple Object Tracking Precision (MOTP)", self.motp),
                ("Multiple Object Tracking Accuracy (MOTAL)", self.motal),
                ("Multiple Object Detection Accuracy (MODA)", self.moda),
                ("Multiple Object Detection Precision (MODP)", self.modp),
            ),
            (
                ("Recall", self.recall),
                ("Precision", self.precision),
                ("F1", self.f1),
                ("False Alarm Rate", self.far),
            ),
            (
                ("Mostly Tracked", self.mt),
                ("Partly Tracked", self.pt),
                ("Mostly Lost", self.ml),
            ),
            (
                ("True Positives", self.tp),
                ("Ignored True Positives", self.itp),
                ("False Positives", self.fp),
                ("False Negatives", self.fn),
                ("Ignored False Negatives", self.ifn),
                ("Missed Targets", self.fn),
                ("ID-switches", self.id_switches),
                ("Fragmentations", self.fragments),
            ),
            (
                ("Ground Truth Objects (Total)", self.n_gt + self.n_igt),
                ("Ignored Ground Truth Objects", self.n_igt),
                ("Ground Truth Trajectories", self.n_gt_trajectories),
            ),
            (
                ("Tracker Objects (Total)", self.n_tr),
                ("Ignored Tracker Objects", self.n_itr),
                ("Tracker Trajectories", self.n_tr_trajectories),
            ),
        )
        lines = ["tracking evaluation summary".center(80, "=")]
        for section in sections:
            lines.extend(self.print_entry(name, value) for name, value in section)
            lines.append("")
        lines[-1] = "=" * 80
        return "\n".join(lines)

    def print_entry(self, key, val, width=(70, 10)):
        """
        Pretty print an entry in a table fashion.
        """

        s_out = key.ljust(width[0])
        if isinstance(val, int):
            s = "%%%dd" % width[1]
            s_out += s % val
        elif isinstance(val, float):
            s = "%%%df" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s" % val).rjust(width[1])
        return s_out

    def _stats_values(self):
        return TrackingFileStats(
            self.mota, self.motp, self.motal, self.moda, self.modp,
            self.recall, self.precision, self.f1, self.far,
            self.mt, self.pt, self.ml, self.tp, self.fp, self.fn,
            self.id_switches, self.fragments, self.n_gt,
            self.n_gt_trajectories, self.n_tr, self.n_tr_trajectories,
        )

    def _write_stats_files(self, summary):
        summary_path = os.path.join(self.eval_dir, "summary_%s.txt" % self.cls)
        with open(summary_path, "w+") as summary_file:
            summary_file.write(str(summary) + "\n")
        stats_path = os.path.join(self.eval_dir, "stats_%s.txt" % self.cls)
        with open(stats_path, "w+") as stats_file:
            stats_file.write(str("%.6f " * 21 % self._stats_values()) + "\n")
        description_path = os.path.join(self.eval_dir, "description.txt")
        with open(description_path, "w+") as description_file:
            description_file.writelines(
                " ".join(description) + "\n"
                for description in self._stats_descriptions()
            )

    def save_to_stats(self):
        """Save tracking statistics and their human-readable summary."""
        summary = self.create_summary()
        self.mail.msg(summary)
        self._write_stats_files(summary)
        return self._tracking_stats_result()

    def _tracking_stats_result(self):
        return TrackingStatsResult(
            self.mota,
            self.motp,
            self.recall,
            self.precision,
            self.f1,
            self.fp,
            self.fn,
            self.id_switches,
        )

    def _parse_tracking_fields(self, fields, loading_groundtruth):
        detection = self._parse_tracking_values(fields)
        if loading_groundtruth:
            return detection
        if len(fields) not in (17, 18):
            self.mail.msg("file is not in KITTI format")
            return None
        detection.score = -1 if len(fields) == 17 else float(fields[17])
        return detection

    def _record_sequence_detection(self, detection, state, loading_groundtruth):
        def log_extension(frame, length):
            LOGGER.info("%s %s %s", "extend f_data", frame, length)

        def log_duplicate(item):
            self.mail.msg(
                "track ids are not unique for sequence %d: frame %d"
                % (item.frame, item.track_id)
            )
            self.mail.msg(
                "track id %d occured at least twice for this frame"
                % item.track_id
            )
            self.mail.msg("Exiting...")

        return self._record_sequence_core(
            detection, state, loading_groundtruth, log_extension, log_duplicate
        )

    def _load_sequence_data_core(self, *call_args, **call_kwargs):
        (
            root_dir,
            sequence_info,
            classes,
            loading_groundtruth,
            parse_detection,
            record_detection,
        ) = _bind_named(
            call_args,
            call_kwargs,
            ('root_dir', 'sequence_info', 'classes', 'loading_groundtruth', 'parse_detection', 'record_detection'),
        )
        sequence_index, sequence_name = sequence_info
        frame_data = [[] for _ in range(self.n_frames[sequence_index])]
        trajectory_ids = []
        id_frame_cache = set()
        eval_2d, eval_3d = True, True
        filename = os.path.join(root_dir, "%s.txt" % sequence_name)
        with open(filename, "r") as file_handle:
            for line in file_handle:
                fields = line.strip().split(" ")
                if not any(class_name in fields[2].lower() for class_name in classes):
                    continue
                detection = parse_detection(fields)
                if detection is None:
                    return None
                if detection.track_id == -1 and detection.obj_type != "dontcare":
                    continue
                state = (frame_data, trajectory_ids, id_frame_cache, eval_2d, eval_3d)
                state = record_detection(detection, state)
                if state is None:
                    return None
                frame_data, trajectory_ids, id_frame_cache, eval_2d, eval_3d = state
        return frame_data, len(trajectory_ids), eval_2d, eval_3d

    def _load_sequence_data(
        self, root_dir, sequence_info, classes, loading_groundtruth
    ):
        return self._load_sequence_data_core(
            root_dir,
            sequence_info,
            classes,
            loading_groundtruth,
            lambda fields: self._parse_tracking_fields(fields, loading_groundtruth),
            lambda detection, state: self._record_sequence_detection(
                detection, state, loading_groundtruth
            ),
        )

    def _load_data_core(self, root_dir, cls, loading_groundtruth, load_sequence):
        classes = self._tracking_classes(cls)
        sequence_data, trajectory_counts = [], []
        eval_2d, eval_3d = True, True
        for sequence_info in enumerate(self.sequence_name):
            loaded = load_sequence(root_dir, sequence_info, classes)
            if loaded is None:
                return False
            frame_data, count, sequence_eval_2d, sequence_eval_3d = loaded
            sequence_data.append(frame_data)
            trajectory_counts.append(count)
            eval_2d = eval_2d and sequence_eval_2d
            eval_3d = eval_3d and sequence_eval_3d
        if loading_groundtruth:
            self._store_ground_truth(sequence_data, trajectory_counts)
            return True
        self.tracker = sequence_data
        self.n_tr_trajectories = sum(trajectory_counts)
        self.eval_2d, self.eval_3d = eval_2d, eval_3d
        self.n_tr_seq = trajectory_counts
        return self.n_tr_trajectories != 0

    def _store_ground_truth(self, sequence_data, trajectory_counts):
        self.dcareas, self.groundtruth = [], []
        for sequence in sequence_data:
            sequence_gt, sequence_dc = [], []
            for frame in sequence:
                sequence_gt.append(
                    [item for item in frame if item.obj_type != "dontcare"]
                )
                sequence_dc.append(
                    [item for item in frame if item.obj_type == "dontcare"]
                )
            self.groundtruth.append(sequence_gt)
            self.dcareas.append(sequence_dc)
        self.n_gt_seq = trajectory_counts
        self.n_gt_trajectories = sum(trajectory_counts)

    def _load_data(self, root_dir, cls, min_score=-1000, loading_groundtruth=False):
        """
        Generic loader for ground truth and tracking data.
        Use loadGroundtruth() or loadTracker() to load this data.
        Loads detections in KITTI format from textfiles.
        """
        return self._load_data_core(
            root_dir,
            cls,
            loading_groundtruth,
            lambda directory, sequence, classes: self._load_sequence_data(
                directory, sequence, classes, loading_groundtruth
            ),
        )


def evaluate(result_sha, root, part="all", mail=None):
    """Evaluate available car and pedestrian tracking results."""

    if mail is None:
        mail = mailpy.Mail("")

    # start evaluation and instanciated eval object
    mail.msg("Processing Result for KITTI Tracking Benchmark")
    classes = []
    for c in ("car", "pedestrian"):
        e = TrackingEvaluation(t_sha=result_sha, root=root, part=part, mail=mail, cls=c)
        # load tracker data and check provided classes
        try:
            if not e.load_tracker():
                continue
            mail.msg("Loading Results - Success")
            mail.msg("Evaluate Object Class: %s" % c.upper())
            classes.append(c)
        except (OSError, TypeError, ValueError, IndexError):
            mail.msg(
                "Feel free to contact us (lenz@kit.edu), if you receive this error message:"
            )
            mail.msg("   Caught exception while loading result data.")
            break
        # load groundtruth data for this class
        if not e.load_groundtruth():
            raise ValueError("Ground truth not found.")
        mail.msg("Loading Groundtruth - Success")
        # sanity checks
        if len(e.groundtruth) is not len(e.tracker):
            mail.msg("The uploaded data does not provide results for every sequence.")
            return False
        mail.msg("Loaded %d Sequences." % len(e.groundtruth))
        mail.msg("Start Evaluation...")
        # create needed directories, evaluate and save stats
        try:
            e.create_eval_dir()
        except OSError:
            mail.msg(
                "Feel free to contact us (lenz@kit.edu), if you receive this error message:"
            )
            mail.msg("   Caught exception while creating results.")
        if e.compute3rd_party_metrics():
            mota, motp, recall, prec, f1, fp, fn, id_switches = e.save_to_stats()
        else:
            mail.msg(
                "There seem to be no true positives or false positives at all in the submitted data."
            )

    # finish
    if len(classes) == 0:
        mail.msg(
            "The uploaded results could not be evaluated. Check for format errors."
        )
        return False
    mail.msg("Thank you for participating in our system!")
    return TrackingEvaluationResult(mota, motp, recall, prec, f1, fp, fn, id_switches)


#
# entry point of evaluation script
# input:
# - result_sha (unique key of results)
# - user_sha (key of user who submitted the results, optional)
# - user_sha (email of user who submitted the results, optional)
if __name__ == "__main__":

    # check for correct number of arguments. if user_sha and email are not supplied,
    # no notification email is sent (this option is used for auto-updates)
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        LOGGER.info(
            "%s", "Usage: python eval_tracking.py root result_sha [user_sha email]"
        )
        sys.exit(1)

    # get unique sha key of submitted results
    result_sha = sys.argv[2]
    root = sys.argv[1]

    # create mail messenger and debug output object
    if len(sys.argv) == 5:
        mail = mailpy.Mail(sys.argv[3])
    else:
        mail = mailpy.Mail("")

    # evaluate results and send notification email to user
    success = evaluate(result_sha, root, mail)
    if len(sys.argv) == 4:
        mail.finalize(success, "tracking", result_sha, sys.argv[2])
    else:
        mail.finalize(success, "tracking", result_sha, "")
