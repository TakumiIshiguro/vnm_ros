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


def expand_model_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_type = model_cfg["model_type"]
    if "checkpoint_path" in model_cfg and "common" not in model_cfg:
        return dict(model_cfg)
    if "checkpoint_path" in model_cfg:
        raise ValueError(
            "Top-level checkpoint_path is no longer supported with nested model config. "
            "Set checkpoint_path only under the selected model section, "
            f"for example '{model_type}: checkpoint_path: ...'."
        )
    common_cfg = model_cfg.get("common", {})
    type_cfg = model_cfg.get(model_type)
    if type_cfg is None:
        raise ValueError(f"model config is missing a '{model_type}' section")
    if not type_cfg.get("checkpoint_path"):
        raise ValueError(
            f"model config section '{model_type}' must define checkpoint_path"
        )

    expanded = {
        key: value
        for key, value in model_cfg.items()
        if key not in ("common", "vint", "nomad")
    }
    expanded.update(common_cfg)
    expanded.update(type_cfg)
    return expanded


def selected_model_type(runtime_cfg: Dict[str, Any]) -> str:
    model_type = runtime_cfg.get("model_type")
    if model_type is None:
        model_type = runtime_cfg.get("model", {}).get("type")
    if not model_type:
        raise ValueError("runtime.yaml must define model_type")
    return str(model_type)


def resolve_path(path: str, base_dir: str = None) -> str:
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = package_root()
    return os.path.abspath(os.path.join(base_dir, path))


def model_dataset_dir(dataset_cfg: Dict[str, Any], dataset_type: str, model_type: str) -> str:
    data_dir_key = "train_data_dir" if dataset_type == "train" else "test_data_dir"
    base_dir = resolve_path(dataset_cfg[data_dir_key], package_root())
    return os.path.join(base_dir, str(model_type))


def load_runtime_config(config_dir: str = None) -> Dict[str, Dict[str, Any]]:
    if config_dir is None:
        config_dir = os.path.join(package_root(), "config")
    runtime_cfg = load_yaml(os.path.join(config_dir, "runtime.yaml"))
    model_type = selected_model_type(runtime_cfg)
    model_file_cfg = load_yaml(os.path.join(config_dir, f"{model_type}.yaml"))
    model_cfg = expand_model_config(model_file_cfg["model"])
    if model_cfg["model_type"] != model_type:
        raise ValueError(
            f"runtime.yaml model_type={model_type} does not match "
            f"{model_type}.yaml model.model_type={model_cfg['model_type']}"
        )
    training_cfg = {key: value for key, value in model_file_cfg.items() if key != "model"}
    if "device" in training_cfg and "device" not in model_cfg:
        model_cfg["device"] = training_cfg["device"]
    config = {
        "paths": merge_paths(runtime_cfg, training_cfg),
        "topics": load_yaml(os.path.join(config_dir, "topics.yaml")),
        "model": model_cfg,
        "robot": runtime_cfg["robot"],
        "topomap": runtime_cfg["topomap"],
        "visualization": runtime_cfg["visualization"],
        "train": training_cfg,
    }
    apply_shared_paths(config)
    return config


def load_care_config(config_dir: str = None) -> Dict[str, Dict[str, Any]]:
    if config_dir is None:
        config_dir = os.path.join(package_root(), "config")
    config_dir = os.path.abspath(os.path.expanduser(config_dir))
    care = load_yaml(os.path.join(config_dir, "care.yaml"))
    runtime = dict(care.get("runtime", {}))
    map_config = dict(care.get("map", {}))
    avoidance = dict(care.get("avoidance", {}))
    target_direction = dict(care.get("target_direction", {}))
    topics = load_yaml(os.path.join(config_dir, "topics.yaml"))

    runtime["rate"] = float(runtime.get("rate", 0.0))
    runtime["stale_timeout_seconds"] = float(
        runtime.get("stale_timeout_seconds", 0.0)
    )
    runtime["robot_frame"] = str(runtime.get("robot_frame", "")).strip()
    if runtime["rate"] <= 0.0:
        raise ValueError("care runtime.rate must be positive")
    if runtime["stale_timeout_seconds"] <= 0.0:
        raise ValueError("care runtime.stale_timeout_seconds must be positive")
    if not runtime["robot_frame"]:
        raise ValueError("care runtime.robot_frame must not be empty")

    map_config["width_m"] = float(map_config.get("width_m", 0.0))
    map_config["height_m"] = float(map_config.get("height_m", 0.0))
    map_config["grid_spacing_m"] = float(
        map_config.get("grid_spacing_m", 0.0)
    )
    map_config["obstacle_radius_pixels"] = int(
        map_config.get("obstacle_radius_pixels", -1)
    )
    image_size = map_config.get("image_size", [])
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(int(value) < 256 for value in image_size)
    ):
        raise ValueError("care map.image_size must contain two values >= 256")
    map_config["image_size"] = [int(value) for value in image_size]
    if map_config["width_m"] <= 0.0 or map_config["height_m"] <= 0.0:
        raise ValueError("care map dimensions must be positive")
    if map_config["grid_spacing_m"] <= 0.0:
        raise ValueError("care map.grid_spacing_m must be positive")
    if map_config["obstacle_radius_pixels"] < 0:
        raise ValueError("care map.obstacle_radius_pixels must be non-negative")

    avoidance["require_obstacle_data"] = bool(
        avoidance.get("require_obstacle_data", True)
    )
    avoidance["num_action_samples"] = int(
        avoidance.get("num_action_samples", 8)
    )
    if avoidance["num_action_samples"] <= 0:
        raise ValueError(
            "care avoidance.num_action_samples must be positive"
        )
    avoidance["waypoint_index"] = int(
        avoidance.get("waypoint_index", 1)
    )
    if avoidance["waypoint_index"] < 0:
        raise ValueError(
            "care avoidance.waypoint_index must be non-negative"
        )
    avoidance["base_action_strategy"] = str(
        avoidance.get("base_action_strategy", "mean")
    ).strip().lower()
    if avoidance["base_action_strategy"] not in ("mean", "first"):
        raise ValueError(
            "care avoidance.base_action_strategy must be mean or first"
        )
    for key, default in (
        ("maximum_forward_range_m", 1.5),
        ("path_influence_radius_m", 0.40),
        ("depth_offset_m", 0.05),
        ("minimum_force_distance_m", 0.02),
        ("force_balance_ratio_threshold", 0.15),
        ("theta_clip_degrees", 45.0),
        ("safe_fov_threshold_degrees", 30.0),
    ):
        avoidance[key] = float(avoidance.get(key, default))
    if avoidance["maximum_forward_range_m"] <= 0.0:
        raise ValueError(
            "care avoidance.maximum_forward_range_m must be positive"
        )
    if avoidance["path_influence_radius_m"] <= 0.0:
        raise ValueError(
            "care avoidance.path_influence_radius_m must be positive"
        )
    if avoidance["depth_offset_m"] < 0.0:
        raise ValueError(
            "care avoidance.depth_offset_m must be non-negative"
        )
    if avoidance["minimum_force_distance_m"] <= 0.0:
        raise ValueError(
            "care avoidance.minimum_force_distance_m must be positive"
        )
    if not 0.0 <= avoidance["force_balance_ratio_threshold"] < 1.0:
        raise ValueError(
            "care avoidance.force_balance_ratio_threshold must be in [0, 1)"
        )
    if not 0.0 < avoidance["theta_clip_degrees"] <= 180.0:
        raise ValueError(
            "care avoidance.theta_clip_degrees must be in (0, 180]"
        )
    if not (
        0.0
        < avoidance["safe_fov_threshold_degrees"]
        <= avoidance["theta_clip_degrees"]
    ):
        raise ValueError(
            "care avoidance.safe_fov_threshold_degrees must be positive "
            "and no greater than theta_clip_degrees"
        )

    target_direction["stale_timeout_seconds"] = float(
        target_direction.get("stale_timeout_seconds", 0.0)
    )
    if target_direction["stale_timeout_seconds"] <= 0.0:
        raise ValueError(
            "care target_direction.stale_timeout_seconds must be positive"
        )

    required_topics = (
        "cmd_dir_topic",
        "action_candidates_topic",
        "obstacle_points_topic",
        "all_obstacle_points_topic",
        "care_bev_topic",
    )
    missing = [key for key in required_topics if not topics.get(key)]
    if missing:
        raise ValueError(f"CARE topics are missing: {', '.join(missing)}")
    return {
        "runtime": runtime,
        "map": map_config,
        "avoidance": avoidance,
        "target_direction": target_direction,
        "topics": topics,
    }
