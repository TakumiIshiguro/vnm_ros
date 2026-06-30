#!/usr/bin/env python3
import argparse
import csv
import os
import pickle
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def numeric_image_name(directory: str, index: int) -> str:
    stem = str(index)
    for extension in IMAGE_EXTENSIONS:
        name = stem + extension
        if os.path.exists(os.path.join(directory, name)):
            return name
    return ""


def load_pickle(path: str) -> Dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a dict")
    return data


def sample_count(data: Dict) -> int:
    lengths = []
    for value in data.values():
        array = np.asarray(value)
        if array.ndim > 0:
            lengths.append(array.shape[0])
    if not lengths:
        return 0
    return max(lengths)


def value_columns(name: str, value, count: int) -> List[str]:
    array = np.asarray(value)
    if array.ndim == 0:
        return [name]
    if array.shape[0] != count:
        return []
    if array.ndim == 1:
        return [name]
    width = int(np.prod(array.shape[1:]))
    return [f"{name}_{i}" for i in range(width)]


def row_values(value, index: int, count: int) -> List:
    array = np.asarray(value)
    if array.ndim == 0:
        return [array.item()]
    if array.shape[0] != count:
        return []
    item = array[index]
    if np.asarray(item).ndim == 0:
        return [np.asarray(item).item()]
    return np.asarray(item).reshape(-1).tolist()


def csv_schema(data: Dict, count: int) -> Tuple[List[str], List[str]]:
    keys = sorted(data.keys())
    header = ["index", "image"]
    included_keys = []
    for key in keys:
        columns = value_columns(key, data[key], count)
        if columns:
            header.extend(columns)
            included_keys.append(key)
    return header, included_keys


def convert_pickle(pkl_path: str, output_path: str = None) -> str:
    pkl_path = os.path.abspath(pkl_path)
    directory = os.path.dirname(pkl_path)
    if output_path is None:
        output_path = os.path.join(directory, "traj_data.csv")
    data = load_pickle(pkl_path)
    count = sample_count(data)
    header, keys = csv_schema(data, count)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index in range(count):
            row = [index, numeric_image_name(directory, index)]
            for key in keys:
                row.extend(row_values(data[key], index, count))
            writer.writerow(row)
    return output_path


def iter_pickle_paths(path: str, recursive: bool) -> Iterable[str]:
    path = os.path.abspath(path)
    if os.path.isfile(path):
        yield path
        return
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    direct = os.path.join(path, "traj_data.pkl")
    if os.path.isfile(direct):
        yield direct
        return
    if recursive:
        for root, _, files in os.walk(path):
            if "traj_data.pkl" in files:
                yield os.path.join(root, "traj_data.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert vnm_ros traj_data.pkl files to CSV."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="traj_data.pkl files, trajectory directories, or dataset directories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path. Only valid with one input traj_data.pkl or trajectory directory.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively convert all traj_data.pkl files under directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pkl_paths = []
    for path in args.paths:
        pkl_paths.extend(iter_pickle_paths(path, args.recursive))
    if not pkl_paths:
        raise FileNotFoundError("No traj_data.pkl files found")
    if args.output and len(pkl_paths) != 1:
        raise ValueError("--output can only be used when converting one pickle")

    for pkl_path in pkl_paths:
        output = convert_pickle(pkl_path, args.output)
        print(f"{pkl_path} -> {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
