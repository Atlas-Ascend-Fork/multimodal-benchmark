# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""JMODT adapter for the shared KITTI tracking evaluator."""

from pathlib import Path
from typing import NamedTuple
import os

from munkres import Munkres

from shared.compatibility import import_from_root
from jmodt.config import TRAIN_SEQ_ID, VALID_SEQ_ID, TEST_SEQ_ID


from shared.parameters import ParameterSchema as _ParameterSchema, bind_parameters as _bind_parameters


_TRACKING_BASE = import_from_root(
    "kitti_devkit.evaluate_tracking",
    Path(__file__).parents[3] / "DFMOT",
)


class TrackingevaluationInitParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    t_sha: object
    result_root: object
    part: object
    gt_path: object
    min_overlap: object
    max_truncation: object
    min_height: object
    max_occlusion: object
    cls: object


class TrackingevaluationLoadDataParameters(NamedTuple):
    """Named parameters for a previously wide function signature."""

    root_dir: object
    cls: object
    logger: object
    min_score: object
    loading_groundtruth: object


TData = _TRACKING_BASE.TData
TrackingStatsResult = _TRACKING_BASE.TrackingStatsResult
TrackingFileStats = _TRACKING_BASE.TrackingFileStats


seqmap = {
    "0000": "000154",
    "0001": "000447",
    "0002": "000233",
    "0003": "000144",
    "0004": "000314",
    "0005": "000297",
    "0006": "000270",
    "0007": "000800",
    "0008": "000390",
    "0009": "000803",
    "0010": "000294",
    "0011": "000373",
    "0012": "000078",
    "0013": "000340",
    "0014": "000106",
    "0015": "000376",
    "0016": "000209",
    "0017": "000145",
    "0018": "000339",
    "0019": "001059",
    "0020": "000837",
}


_TRACKINGEVALUATION_INIT_PARAMETERS_SCHEMA = _ParameterSchema(
    TrackingevaluationInitParameters,
    (
        "t_sha",
        "result_root",
        "part",
        "gt_path",
        "min_overlap",
        "max_truncation",
        "min_height",
        "max_occlusion",
        "cls",
    ),
    {
        "min_overlap": 0.5,
        "max_truncation": 0,
        "min_height": 25,
        "max_occlusion": 2,
        "cls": "car",
    },
    frozenset(()),
    9,
    None,
    None,
)
_TRACKINGEVALUATION_LOAD_DATA_PARAMETERS_SCHEMA = _ParameterSchema(
    TrackingevaluationLoadDataParameters,
    ("root_dir", "cls", "logger", "min_score", "loading_groundtruth"),
    {"min_score": -1000, "loading_groundtruth": False},
    frozenset(()),
    5,
    None,
    None,
)


initialize_tracking_statistics = _TRACKING_BASE.initialize_tracking_statistics


class TrackingEvaluation(_TRACKING_BASE.TrackingEvaluation):
    """JMODT-specific paths and logging around CLEAR MOT statistics."""

    def __init__(self, *call_args, **call_kwargs):
        bound_parameters = _bind_parameters(
            _TRACKINGEVALUATION_INIT_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        (
            t_sha,
            result_root,
            part,
            gt_path,
            min_overlap,
            max_truncation,
            min_height,
            max_occlusion,
            cls,
        ) = bound_parameters
        self.n_frames = []
        self.sequence_name = []
        self.sequence_id = []
        if part == "val":
            self.sequence_id = VALID_SEQ_ID
        elif part == "train":
            self.sequence_id = TRAIN_SEQ_ID
        elif part == "test":
            self.sequence_id = TEST_SEQ_ID

        for seq_id, n_frames in seqmap.items():
            if seq_id in self.sequence_id:
                self.sequence_name.append(seq_id)
                self.n_frames.append(int(n_frames))
        self.n_sequences = len(seqmap)

        # class to evaluate, i.e. pedestrian or car
        self.cls = cls

        # data and parameter
        self.gt_path = os.path.join(gt_path, "label_02")
        self.t_sha = t_sha
        self.t_path = os.path.join(result_root, t_sha, part)

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
    def _parse_tracking_fields(fields):
        return TrackingEvaluation._parse_tracking_values(fields)

    @staticmethod
    def _record_sequence_detection(detection, state, options):
        loading_groundtruth, logger = options

        def log_extension(frame, length):
            logger.info("extend f_data %s %s", frame, length)

        def log_duplicate(item):
            logger.info(
                "track ids are not unique for sequence %d: frame %d",
                item.frame,
                item.track_id,
            )
            logger.info(
                "track id %d occured at least twice for this frame", item.track_id
            )
            logger.info("Exiting...")

        return TrackingEvaluation._record_sequence_core(
            detection, state, loading_groundtruth, log_extension, log_duplicate
        )

    def create_eval_dir(self, logger):
        """
        Creates directory to store evaluation results and data for visualization.
        """

        self.eval_dir = os.path.join(self.t_path, "eval", self.cls)
        if not os.path.exists(self.eval_dir):
            logger.info(f"create directory: {self.eval_dir}")
            os.makedirs(self.eval_dir)
            logger.info("done")

    def load_groundtruth(self, logger):
        """
        Helper function to load ground truth.
        """

        try:
            self._load_data(
                self.gt_path, cls=self.cls, loading_groundtruth=True, logger=logger
            )
        except IOError:
            return False
        return True

    def load_tracker(self, logger):
        """
        Helper function to load tracker data.
        """

        try:
            if not self._load_data(
                self.t_path, cls=self.cls, loading_groundtruth=False, logger=logger
            ):
                return False
        except IOError:
            return False
        return True

    def compute3rd_party_metrics(self, logger):
        """Compute tracking metrics using the shared evaluator core."""
        munkres = Munkres()
        for sequence_index, _ in enumerate(self.groundtruth):
            self._evaluate_sequence(sequence_index, munkres, logger)
        ignored = self._evaluate_trajectories()
        self._finalize_tracking_ratios(ignored)
        self._finalize_clear_mot()
        return True

    def save_to_stats(self, logger):
        """Save tracking statistics and their human-readable summary."""
        summary = self.create_summary()
        logger.info(summary)
        self._write_stats_files(summary)
        return self._tracking_stats_result()

    def _load_sequence_data(self, root_dir, sequence_info, classes, options):
        loading_groundtruth, logger = options
        return self._load_sequence_data_core(
            root_dir,
            sequence_info,
            classes,
            loading_groundtruth,
            self._parse_tracking_fields,
            lambda detection, state: self._record_sequence_detection(
                detection, state, options
            ),
        )

    def _load_data(self, *call_args, **call_kwargs):
        """
        Generic loader for ground truth and tracking data.
        Use loadGroundtruth() or loadTracker() to load this data.
        Loads detections in KITTI format from textfiles.
        """
        bound_parameters = _bind_parameters(
            _TRACKINGEVALUATION_LOAD_DATA_PARAMETERS_SCHEMA, call_args, call_kwargs
        )
        root_dir, cls, logger, min_score, loading_groundtruth = bound_parameters
        return self._load_data_core(
            root_dir,
            cls,
            loading_groundtruth,
            lambda directory, sequence, classes: self._load_sequence_data(
                directory, sequence, classes, (loading_groundtruth, logger)
            ),
        )


def evaluate(result_sha, result_root, part, gt_path, logger):
    """Evaluate available car and pedestrian tracking results."""

    # start evaluation and instanciated eval object
    logger.info("Processing Result for KITTI Tracking Benchmark")
    classes = []
    for c in ("car", "pedestrian"):
        e = TrackingEvaluation(t_sha=result_sha, result_root=result_root, part=part,
                               cls=c, gt_path=gt_path)
        # load tracker data and check provided classes
        try:
            if not e.load_tracker(logger):
                continue
            logger.info("Loading Results - Success")
            logger.info("Evaluate Object Class: %s" % c.upper())
            classes.append(c)
        except (OSError, TypeError, ValueError, IndexError):
            logger.info(
                "Feel free to contact us (lenz@kit.edu), if you receive this error message:"
            )
            logger.info("   Caught exception while loading result data.")
            break
        # load groundtruth data for this class
        if not e.load_groundtruth(logger):
            raise ValueError("Ground truth not found.")
        logger.info("Loading Groundtruth - Success")
        # sanity checks
        if len(e.groundtruth) is not len(e.tracker):
            logger.info(
                "The uploaded data does not provide results for every sequence."
            )
            return False
        logger.info("Loaded %d Sequences." % len(e.groundtruth))
        logger.info("Start Evaluation...")
        # create needed directories, evaluate and save stats
        try:
            e.create_eval_dir(logger=logger)
        except OSError:
            logger.info(
                "Feel free to contact us (lenz@kit.edu), if you receive this error message:"
            )
            logger.info("   Caught exception while creating results.")
        if e.compute3rd_party_metrics(logger):
            mota, motp, recall, prec, f1, fp, fn, id_switches = e.save_to_stats(logger)
            return mota
        else:
            logger.info(
                "There seem to be no true positives or false positives at all in the submitted data."
            )

    # finish
    if len(classes) == 0:
        logger.info(
            "The uploaded results could not be evaluated. Check for format errors."
        )
        return False
    logger.info("Thank you for participating in our system!")
    return True
