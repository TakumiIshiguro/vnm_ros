import os
from typing import Any, Dict

import yaml


def package_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def set_if_missing(config: Dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if key not in config or config[key] in (None, ""):
        config[key] = value


def merge_paths(
    runtime_cfg: Dict[str, Any], training_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    paths = {}
    for source in (runtime_cfg.get("paths", {}), training_cfg.get("paths", {})):
        for section, values in source.items():
            paths.setdefault(section, {}).update(values or {})
    return paths


def apply_shared_paths(config: Dict[str, Dict[str, Any]]) -> None:
    paths = config.get("paths", {})
    dataset_paths = paths.get("dataset", {})
    rosbag_paths = paths.get("rosbag", {})
    topomap_paths = paths.get("topomap", {})

    train_cfg = config["train"]
    dataset_cfg = train_cfg.setdefault("dataset", {})
    collection_cfg = train_cfg.setdefault("collection", {})
    topomap_cfg = config["topomap"]

    set_if_missing(dataset_cfg, "train_data_dir", dataset_paths.get("train_data_dir"))
    set_if_missing(dataset_cfg, "test_data_dir", dataset_paths.get("test_data_dir"))
    set_if_missing(collection_cfg, "bag_path", rosbag_paths.get("path"))
    set_if_missing(topomap_cfg, "bag_path", rosbag_paths.get("path"))
    set_if_missing(topomap_cfg, "topomap_dir", topomap_paths.get("topomap_dir"))


def resolve_path(path: str, base_dir: str = None) -> str:
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = package_root()
    return os.path.abspath(os.path.join(base_dir, path))


def load_runtime_config(config_dir: str = None) -> Dict[str, Dict[str, Any]]:
    if config_dir is None:
        config_dir = os.path.join(package_root(), "config")
    runtime_cfg = load_yaml(os.path.join(config_dir, "runtime.yaml"))
    training_cfg = load_yaml(os.path.join(config_dir, "training.yaml"))
    config = {
        "paths": merge_paths(runtime_cfg, training_cfg),
        "topics": load_yaml(os.path.join(config_dir, "topics.yaml")),
        "model": load_yaml(os.path.join(config_dir, "model.yaml")),
        "robot": runtime_cfg["robot"],
        "topomap": runtime_cfg["topomap"],
        "visualization": runtime_cfg["visualization"],
        "train": training_cfg,
    }
    apply_shared_paths(config)
    return config
