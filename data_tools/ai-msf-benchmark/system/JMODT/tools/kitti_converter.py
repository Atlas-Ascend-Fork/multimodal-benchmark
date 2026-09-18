# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import logging
import os
import shutil

from jmodt.config import TRAIN_SEQ_ID, VALID_SEQ_ID, SMALL_VAL_SEQ_ID, TEST_SEQ_ID

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--data_root", type=str, default="data/KITTI")
args = parser.parse_args()


def init_or_clear_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for f in os.listdir(path):
            f_path = os.path.join(path, f)
            os.remove(f_path)


def _sequence_paths(training_root, sequence):
    return {
        "image": os.path.join(training_root, "image_02", sequence),
        "lidar": os.path.join(training_root, "velodyne", sequence),
        "calib": os.path.join(training_root, "calib", f"{sequence}.txt"),
        "label": os.path.join(training_root, "label_02", f"{sequence}.txt"),
    }


def _sequence_frames(lidar_path):
    return sorted(os.path.splitext(filename)[0] for filename in os.listdir(lidar_path))


def _mapped_track_id(sequence, track_id, track_ids, used_track_id):
    if track_id == -1:
        return track_id, used_track_id
    key = f"{sequence}_{track_id}"
    if key not in track_ids:
        used_track_id += 1
        track_ids[key] = used_track_id
    return track_ids[key], used_track_id


def _load_sequence_labels(label_path, sequence, frames, track_ids, used_track_id):
    labels = {frame: [] for frame in frames}
    with open(label_path, "r") as label_file:
        for line in label_file:
            fields = line.strip().split()
            frame = fields[0].zfill(6)
            if frame not in labels:
                continue
            track_id, used_track_id = _mapped_track_id(
                sequence, int(fields[1]), track_ids, used_track_id
            )
            labels[frame].append(" ".join(fields[2:]) + f" {track_id}\n")
    return labels, used_track_id


def _require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _copy_training_sample(paths, output_dirs, frame, sample):
    image = _require_file(os.path.join(paths["image"], f"{frame}.png"))
    lidar = _require_file(os.path.join(paths["lidar"], f"{frame}.bin"))
    calibration = _require_file(paths["calib"])
    shutil.copyfile(image, os.path.join(output_dirs["image"], f"{sample}.png"))
    shutil.copyfile(lidar, os.path.join(output_dirs["lidar"], f"{sample}.bin"))
    shutil.copyfile(calibration, os.path.join(output_dirs["calib"], f"{sample}.txt"))


def _write_sample_labels(label_dir, sample, labels):
    regular = [label for label in labels if "DontCare" not in label]
    ignored = [label for label in labels if "DontCare" in label]
    with open(os.path.join(label_dir, f"{sample}.txt"), "w+") as label_file:
        label_file.writelines(regular + ignored)


def _write_sample_mappings(output_dir, sample_to_frame, sequence_to_samples):
    sample_path = os.path.join(output_dir, "sample2frame.txt")
    with open(sample_path, "w+") as mapping_file:
        for sample, (sequence, frame) in sample_to_frame.items():
            mapping_file.write(f"{sample} {sequence} {frame}\n")
    sequence_path = os.path.join(output_dir, "seq2sample.txt")
    with open(sequence_path, "w+") as mapping_file:
        for sequence, samples in sequence_to_samples.items():
            mapping_file.write(f"{sequence} {' '.join(samples)} \n")


def _split_lines(samples, paired):
    if paired:
        return (f"{left} {right}\n" for left, right in zip(samples, samples[1:]))
    return (f"{sample}\n" for sample in samples)


def _write_train_splits(output_root, sequence_to_samples):
    split_dir = os.path.join(output_root, "ImageSets")
    os.makedirs(split_dir, exist_ok=True)
    split_sequences = {
        "train.txt": (TRAIN_SEQ_ID, True),
        "val.txt": (VALID_SEQ_ID, False),
        "small_val.txt": (SMALL_VAL_SEQ_ID, True),
    }
    for filename, (sequences, paired) in split_sequences.items():
        with open(os.path.join(split_dir, filename), "w+") as split_file:
            for sequence in sequences:
                split_file.writelines(
                    _split_lines(sequence_to_samples[sequence], paired)
                )


def _write_test_mappings(output_dir, sample_to_frame, sequence_to_samples, count):
    with open(os.path.join(output_dir, "sample2frame.txt"), "w+") as mapping_file:
        for sample_id in range(count):
            sample = str(sample_id).zfill(6)
            sequence, frame = sample_to_frame[sample]
            mapping_file.write(f"{sample} {sequence} {frame}\n")
    with open(os.path.join(output_dir, "seq2sample.txt"), "w+") as mapping_file:
        for sequence_id in range(29):
            sequence = "%04d" % sequence_id
            mapping_file.write(f"{sequence} {' '.join(sequence_to_samples[sequence])} \n")


def _write_test_split(output_root, sequence_to_samples):
    split_dir = os.path.join(output_root, "ImageSets")
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "test.txt"), "w+") as split_file:
        for sequence in TEST_SEQ_ID:
            split_file.writelines(
                f"{sample}\n" for sample in sequence_to_samples[sequence]
            )


def create_train_sample_data(
    input_root, output_root, init_or_clear_dirs=True, only_labels=False
):
    res_training = os.path.join(output_root, "training")

    output_dirs = {
        "calib": os.path.join(res_training, "calib"),
        "image": os.path.join(res_training, "image_2"),
        "label": os.path.join(res_training, "label_2"),
        "lidar": os.path.join(res_training, "velodyne"),
    }

    if init_or_clear_dirs:
        for directory in output_dirs.values():
            init_or_clear_dir(directory)

    in_training = os.path.join(input_root, "training")

    sample_id = 0
    used_tid = 0  # start from tid = 1
    tid_dict = {}  # change sparse tid to compact tid for all sequences

    sample_to_real_frame = {}
    seq_to_sample = {}

    for seq in range(21):
        seq = "%04d" % seq
        paths = _sequence_paths(in_training, seq)
        frames = _sequence_frames(paths["lidar"])
        LOGGER.info("%s %s %s %s", "processing sequence", seq, ", length:", len(frames))
        label_dict, used_tid = _load_sequence_labels(
            paths["label"], seq, frames, tid_dict, used_tid
        )

        for frame in frames:
            sample_str = str(sample_id).zfill(6)
            if not only_labels:
                _copy_training_sample(paths, output_dirs, frame, sample_str)
            _write_sample_labels(output_dirs["label"], sample_str, label_dict[frame])
            sample_to_real_frame[sample_str] = (seq, frame)
            seq_to_sample.setdefault(seq, []).append(sample_str)
            sample_id += 1

    _write_sample_mappings(res_training, sample_to_real_frame, seq_to_sample)
    LOGGER.info("%s %s", sample_id, used_tid)
    _write_train_splits(output_root, seq_to_sample)


def create_test_sample_data(input_root, output_root, init_or_clear_dirs=True):
    out_test_dir = os.path.join(output_root, "testing")
    output_dirs = {
        "calib": os.path.join(out_test_dir, "calib"),
        "image": os.path.join(out_test_dir, "image_2"),
        "lidar": os.path.join(out_test_dir, "velodyne"),
    }

    if init_or_clear_dirs:
        for directory in output_dirs.values():
            init_or_clear_dir(directory)

    in_test_dir = os.path.join(input_root, "testing")
    sample_id = 0
    sample_to_real_frame = {}
    seq_to_sample = {}

    for seq in TEST_SEQ_ID:
        paths = _sequence_paths(in_test_dir, seq)
        frames = _sequence_frames(paths["lidar"])
        LOGGER.info("%s %s %s %s", "processing sequence", seq, ", length:", len(frames))

        for frame in frames:
            sample_str = str(sample_id).zfill(6)
            _copy_training_sample(paths, output_dirs, frame, sample_str)
            sample_to_real_frame[sample_str] = (seq, frame)
            seq_to_sample.setdefault(seq, []).append(sample_str)
            sample_id += 1

    _write_test_mappings(out_test_dir, sample_to_real_frame, seq_to_sample, sample_id)
    _write_test_split(output_root, seq_to_sample)


if __name__ == "__main__":
    in_dir = os.path.join(args.data_root, "tracking")
    out_dir = os.path.join(args.data_root, "tracking_object")
    create_train_sample_data(
        input_root=in_dir,
        output_root=out_dir,
        init_or_clear_dirs=True,
        only_labels=False,
    )
    create_test_sample_data(
        input_root=in_dir, output_root=out_dir, init_or_clear_dirs=True
    )
