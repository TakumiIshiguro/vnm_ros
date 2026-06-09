import os
from typing import Any, Dict

import yaml


def package_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(path: str, base_dir: str = None) -> str:
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = package_root()
    return os.path.abspath(os.path.join(base_dir, path))


def load_runtime_config(config_dir: str = None) -> Dict[str, Dict[str, Any]]:
    if config_dir is None:
        config_dir = os.path.join(package_root(), "config")
    config = {
        "topics": load_yaml(os.path.join(config_dir, "topics.yaml")),
        "robot": load_yaml(os.path.join(config_dir, "robot.yaml")),
        "model": load_yaml(os.path.join(config_dir, "model.yaml")),
        "topomap": load_yaml(os.path.join(config_dir, "topomap.yaml")),
    }
    train_path = os.path.join(config_dir, "train.yaml")
    if os.path.isfile(train_path):
        config["train"] = load_yaml(train_path)
    return config
