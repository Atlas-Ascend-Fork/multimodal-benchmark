# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import csv
import logging
import os

from scipy.optimize import linear_sum_assignment
import numpy as np

from .. import _timing
from .. import utils
from ..utils import TrackEvalException
from ._base_dataset import _BaseDataset

LOGGER = logging.getLogger(__name__)


class Kitti2DBox(_BaseDataset):
    """Dataset class for KITTI 2D bounding box tracking"""

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )
        self.gt_fol = self.config["GT_FOLDER"]
        self.tracker_fol = self.config["TRACKERS_FOLDER"]
        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config["INPUT_AS_ZIP"]

        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        self.max_occlusion = 2
        self.max_truncation = 0
        self.min_height = 25

        # Get classes to eval
        self.valid_classes = ["car", "pedestrian"]
        self.class_list = [
            cls.lower() if cls.lower() in self.valid_classes else None
            for cls in self.config["CLASSES_TO_EVAL"]
        ]
        if not all(self.class_list):
            raise TrackEvalException(
                "Attempted to evaluate an invalid class. Only classes [car, pedestrian] are valid."
            )
        self.class_name_to_class_id = {
            "car": 1,
            "van": 2,
            "truck": 3,
            "pedestrian": 4,
            "person": 5,  # person sitting
            "cyclist": 6,
            "tram": 7,
            "misc": 8,
            "dontcare": 9,
            "car_2": 1,
        }

        self._load_sequences()
        self._configure_trackers()
        self._validate_tracker_files()

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            "GT_FOLDER": os.path.join(
                code_path, "/home/niangao/disk1/DeepFusionMOT/datasets/kitti/train"
            ),  # Location of GT data
            "TRACKERS_FOLDER": os.path.join(
                code_path, "/home/niangao/disk1/DeepFusionMOT/results/train/data"
            ),  # Trackers location
            "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            "TRACKERS_TO_EVAL": None,  # Filenames of trackers to eval (if None, all in folder)
            "CLASSES_TO_EVAL": ["car", "pedestrian"],  # Valid: ['car', 'pedestrian']
            "SPLIT_TO_EVAL": "training",  # Valid: 'training', 'val', 'training_minus_val', 'test'
            "INPUT_AS_ZIP": False,  # Whether tracker input files are zipped
            "PRINT_CONFIG": True,  # Whether to print current config
            "TRACKER_SUB_FOLDER": "data",  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            "OUTPUT_SUB_FOLDER": "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        }
        return default_config

    @staticmethod
    def _require_file(path, label):
        if not os.path.isfile(path):
            raise TrackEvalException(
                f"{label} file not found: " + os.path.basename(path)
            )

    @staticmethod
    def _selected_timestep_data(raw_data, timestep, cls_id, distractor_classes):
        gt_mask = np.sum(
            [raw_data["gt_classes"][timestep] == class_id
             for class_id in [cls_id] + distractor_classes],
            axis=0,
        ).astype(np.bool)
        tracker_mask = np.atleast_1d(
            raw_data["tracker_classes"][timestep] == cls_id
        ).astype(np.bool)
        gt_extras = raw_data["gt_extras"][timestep]
        return {
            "gt_ids": raw_data["gt_ids"][timestep][gt_mask],
            "gt_dets": raw_data["gt_dets"][timestep][gt_mask],
            "gt_classes": raw_data["gt_classes"][timestep][gt_mask],
            "gt_occlusion": gt_extras["occlusion"][gt_mask],
            "gt_truncation": gt_extras["truncation"][gt_mask],
            "tracker_ids": raw_data["tracker_ids"][timestep][tracker_mask],
            "tracker_dets": raw_data["tracker_dets"][timestep][tracker_mask],
            "tracker_confidences": raw_data["tracker_confidences"][timestep][tracker_mask],
            "similarity_scores": raw_data["similarity_scores"][timestep][gt_mask, :][:, tracker_mask],
        }

    @staticmethod
    def _relabel_ids(id_values):
        unique_ids = np.unique(
            [item for timestep_ids in id_values for item in np.unique(timestep_ids)]
        )
        if len(unique_ids) == 0:
            return unique_ids
        id_map = np.nan * np.ones((np.max(unique_ids) + 1))
        id_map[unique_ids] = np.arange(len(unique_ids))
        for timestep, timestep_ids in enumerate(id_values):
            if len(timestep_ids) > 0:
                id_values[timestep] = id_map[timestep_ids].astype(np.int)
        return unique_ids

    @staticmethod
    def _validate_time_keys(read_data, num_timesteps, seq, is_gt):
        valid_keys = {str(timestep) for timestep in range(num_timesteps)}
        extra_keys = [key for key in read_data if key not in valid_keys]
        if extra_keys:
            source = "Ground-truth" if is_gt else "Tracking"
            raise TrackEvalException(
                source
                + " data contains the following invalid timesteps in seq %s: " % seq
                + ", ".join(str(key) + ", " for key in extra_keys)
            )

    @staticmethod
    def _set_timestep_data(raw_data, read_data, timestep, is_gt):
        time_key = str(timestep)
        if time_key not in read_data:
            raw_data["dets"][timestep] = np.empty((0, 4))
            raw_data["ids"][timestep] = np.empty(0).astype(int)
            raw_data["classes"][timestep] = np.empty(0).astype(int)
            empty_key = "gt_extras" if is_gt else "tracker_confidences"
            raw_data[empty_key][timestep] = (
                {"truncation": np.empty(0), "occlusion": np.empty(0)}
                if is_gt
                else np.empty(0)
            )
            return
        time_data = np.asarray(read_data[time_key], dtype=np.float)
        raw_data["dets"][timestep] = np.atleast_2d(time_data[:, 6:10])
        raw_data["ids"][timestep] = np.atleast_1d(time_data[:, 1]).astype(int)
        raw_data["classes"][timestep] = np.atleast_1d(time_data[:, 2]).astype(int)
        if is_gt:
            raw_data["gt_extras"][timestep] = {
                "truncation": np.atleast_1d(time_data[:, 3].astype(int)),
                "occlusion": np.atleast_1d(time_data[:, 4].astype(int)),
            }
        else:
            raw_data["tracker_confidences"][timestep] = (
                np.atleast_1d(time_data[:, 17])
                if time_data.shape[1] > 17
                else np.ones(time_data.shape[0])
            )

    @staticmethod
    def _set_ignore_regions(raw_data, ignore_data, timestep):
        time_key = str(timestep)
        if time_key in ignore_data:
            time_ignore = np.asarray(ignore_data[time_key], dtype=np.float)
            raw_data["gt_crowd_ignore_regions"][timestep] = np.atleast_2d(
                time_ignore[:, 6:10]
            )
        else:
            raw_data["gt_crowd_ignore_regions"][timestep] = np.empty((0, 4))

    def _load_sequences(self):
        self.seq_list = []
        self.seq_lengths = {}
        seqmap_name = "evaluate_tracking.seqmap." + self.config["SPLIT_TO_EVAL"]
        seqmap_file = os.path.join(self.gt_fol, seqmap_name)
        if not os.path.isfile(seqmap_file):
            LOGGER.info("%s", seqmap_file)
            raise TrackEvalException(
                "no seqmap found: " + os.path.basename(seqmap_file)
            )
        with open(seqmap_file) as file_handle:
            dialect = csv.Sniffer().sniff(file_handle.read(1024))
            file_handle.seek(0)
            for row in csv.reader(file_handle, dialect):
                if len(row) < 4:
                    continue
                sequence = row[0]
                self.seq_list.append(sequence)
                self.seq_lengths[sequence] = int(row[3])
                if not self.data_is_zipped:
                    self._require_file(
                        os.path.join(self.gt_fol, "label_02", sequence + ".txt"), "GT"
                    )
        if self.data_is_zipped:
            self._require_file(os.path.join(self.gt_fol, "data.zip"), "GT")

    def _configure_trackers(self):
        configured = self.config["TRACKERS_TO_EVAL"]
        self.tracker_list = (
            os.listdir(self.tracker_fol) if configured is None else configured
        )
        display_names = self.config["TRACKER_DISPLAY_NAMES"]
        if display_names is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
            return
        if configured is None or len(display_names) != len(self.tracker_list):
            raise TrackEvalException(
                "List of tracker files and tracker display names do not match."
            )
        self.tracker_to_disp = dict(zip(self.tracker_list, display_names))

    def _validate_tracker_files(self):
        for tracker in self.tracker_list:
            if self.data_is_zipped:
                path = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol + ".zip"
                )
                self._require_file(path, "Tracker")
                continue
            for sequence in self.seq_list:
                path = os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol, sequence + ".txt"
                )
                if not os.path.isfile(path):
                    relative = os.path.join(
                        tracker, self.tracker_sub_fol, os.path.basename(path)
                    )
                    raise TrackEvalException("Tracker file not found: " + relative)

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _evaluation_class_ids(self, cls):
        if cls == "pedestrian":
            distractor_classes = [self.class_name_to_class_id["person"]]
        elif cls == "car":
            distractor_classes = [self.class_name_to_class_id["van"]]
        else:
            raise TrackEvalException("Class %s is not evaluatable" % cls)
        return self.class_name_to_class_id[cls], distractor_classes

    def _matched_tracker_removals(self, selected, distractor_classes):
        tracker_count = selected["tracker_ids"].shape[0]
        unmatched_indices = np.arange(tracker_count)
        if selected["gt_ids"].shape[0] == 0 or tracker_count == 0:
            return np.array([], np.int), unmatched_indices
        matching_scores = selected["similarity_scores"].copy()
        epsilon = np.finfo("float").eps
        matching_scores[matching_scores < 0.5 - epsilon] = 0
        match_rows, match_cols = linear_sum_assignment(-matching_scores)
        actually_matched = matching_scores[match_rows, match_cols] > epsilon
        match_rows = match_rows[actually_matched]
        match_cols = match_cols[actually_matched]
        is_distractor = np.isin(selected["gt_classes"][match_rows], distractor_classes)
        is_invalid = np.logical_or(
            selected["gt_occlusion"][match_rows] > self.max_occlusion + epsilon,
            selected["gt_truncation"][match_rows] > self.max_truncation + epsilon,
        )
        to_remove = match_cols[np.logical_or(is_distractor, is_invalid)]
        return to_remove, np.delete(unmatched_indices, match_cols, axis=0)

    def _unmatched_tracker_removals(self, raw_data, timestep, tracker_dets, unmatched_indices):
        unmatched_dets = tracker_dets[unmatched_indices, :]
        heights = unmatched_dets[:, 3] - unmatched_dets[:, 1]
        too_small = heights <= self.min_height + np.finfo("float").eps
        ignore_overlap = self._calculate_box_ious(
            unmatched_dets,
            raw_data["gt_crowd_ignore_regions"][timestep],
            box_format="x0y0x1y1",
            do_ioa=True,
        )
        within_ignore = np.any(
            ignore_overlap > 0.5 + np.finfo("float").eps, axis=1
        )
        return unmatched_indices[np.logical_or(too_small, within_ignore)]

    def _preprocess_timestep(self, raw_data, timestep, cls_id, distractor_classes):
        selected = self._selected_timestep_data(
            raw_data, timestep, cls_id, distractor_classes
        )
        matched, unmatched_indices = self._matched_tracker_removals(
            selected, distractor_classes
        )
        unmatched = self._unmatched_tracker_removals(
            raw_data, timestep, selected["tracker_dets"], unmatched_indices
        )
        tracker_remove = np.concatenate((matched, unmatched), axis=0)
        similarity_scores = np.delete(
            selected["similarity_scores"], tracker_remove, axis=1
        )
        gt_keep = (
            (np.less_equal(selected["gt_occlusion"], self.max_occlusion))
            & (np.less_equal(selected["gt_truncation"], self.max_truncation))
            & (np.equal(selected["gt_classes"], cls_id))
        )
        return {
            "gt_ids": selected["gt_ids"][gt_keep],
            "gt_dets": selected["gt_dets"][gt_keep, :],
            "tracker_ids": np.delete(selected["tracker_ids"], tracker_remove, axis=0),
            "tracker_dets": np.delete(selected["tracker_dets"], tracker_remove, axis=0),
            "tracker_confidences": np.delete(
                selected["tracker_confidences"], tracker_remove, axis=0
            ),
            "similarity_scores": similarity_scores[gt_keep],
        }

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """Preprocess one KITTI sequence and class for metric evaluation."""
        cls_id, distractor_classes = self._evaluation_class_ids(cls)
        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data["num_timesteps"]):
            timestep_data = self._preprocess_timestep(
                raw_data, t, cls_id, distractor_classes
            )
            for key in data_keys:
                data[key][t] = timestep_data[key]
            num_tracker_dets += len(data["tracker_ids"][t])
            num_gt_dets += len(data["gt_ids"][t])
        unique_gt_ids = self._relabel_ids(data["gt_ids"])
        unique_tracker_ids = self._relabel_ids(data["tracker_ids"])
        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _raw_file_location(self, tracker, seq, is_gt):
        if self.data_is_zipped:
            zip_file = (
                os.path.join(self.gt_fol, "data.zip")
                if is_gt
                else os.path.join(
                    self.tracker_fol, tracker, self.tracker_sub_fol + ".zip"
                )
            )
            return seq + ".txt", zip_file
        if is_gt:
            return os.path.join(self.gt_fol, "label_02", seq + ".txt"), None
        return (
            os.path.join(
                self.tracker_fol, tracker, self.tracker_sub_fol, seq + ".txt"
            ),
            None,
        )

    def _raw_data_filters(self, is_gt):
        valid_filter = {2: list(self.class_list)}
        if is_gt and "car" in self.class_list:
            valid_filter[2].append("van")
        if is_gt and "pedestrian" in self.class_list:
            valid_filter[2].append("person")
        return valid_filter, {2: ["dontcare"]} if is_gt else None

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the kitti 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        file, zip_file = self._raw_file_location(tracker, seq, is_gt)
        valid_filter, crowd_ignore_filter = self._raw_data_filters(is_gt)

        # Convert kitti class strings to class ids
        convert_filter = {2: self.class_name_to_class_id}

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(
            file,
            time_col=0,
            id_col=1,
            remove_negative_ids=True,
            valid_filter=valid_filter,
            crowd_ignore_filter=crowd_ignore_filter,
            convert_filter=convert_filter,
            is_zipped=self.data_is_zipped,
            zip_file=zip_file,
        )
        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_crowd_ignore_regions", "gt_extras"]
        else:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        self._validate_time_keys(read_data, num_timesteps, seq, is_gt)

        for t in range(num_timesteps):
            self._set_timestep_data(raw_data, read_data, t, is_gt)
            if is_gt:
                self._set_ignore_regions(raw_data, ignore_data, t)

        if is_gt:
            key_map = {"ids": "gt_ids", "classes": "gt_classes", "dets": "gt_dets"}
        else:
            key_map = {
                "ids": "tracker_ids",
                "classes": "tracker_classes",
                "dets": "tracker_dets",
            }
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data["num_timesteps"] = num_timesteps
        raw_data["seq"] = seq
        return raw_data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(
            gt_dets_t, tracker_dets_t, box_format="x0y0x1y1"
        )
        return similarity_scores
