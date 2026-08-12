import os

import numpy as np
import pytest

from vnm_ros.care import decode_action_candidates, render_care_bev
from vnm_ros.utils.config import load_care_config, package_root


def test_decode_action_candidates_restores_shape_and_selection():
    data = [
        1,
        0,
        2,
        2,
        2,
        0.1,
        0.0,
        0.2,
        0.1,
        0.1,
        0.0,
        0.2,
        -0.1,
    ]

    decoded = decode_action_candidates(data)

    assert decoded["selected"] == 1
    assert decoded["waypoint_index"] == 0
    assert decoded["avoidance_active"] is False
    assert decoded["actions"].shape == (2, 2, 2)
    np.testing.assert_allclose(decoded["actions"][1, 1], [0.2, -0.1])


def test_decode_action_candidates_reads_appended_avoidance_state():
    data = [0, 0, 1, 1, 2, 0.2, 0.0, 1]

    decoded = decode_action_candidates(data)

    assert decoded["avoidance_active"] is True


@pytest.mark.parametrize(
    "data",
    (
        [],
        [0, 0, 1, 1, 2],
        [0, 0, 0, 1, 2, 0, 0],
        [0, 0, 1, 1, 2, float("nan"), 0],
    ),
)
def test_decode_action_candidates_rejects_invalid_messages(data):
    assert decode_action_candidates(data) is None


def test_render_care_bev_combines_obstacles_and_candidates():
    obstacles = np.asarray(
        [[2.0, 1.0], [2.0, -1.0], [5.0, 0.0]],
        dtype=np.float32,
    )
    candidates = {
        "selected": 0,
        "waypoint_index": 1,
        "actions": np.asarray(
            [
                [[0.2, 0.0], [0.4, 0.1]],
                [[0.2, 0.0], [0.4, -0.1]],
            ],
            dtype=np.float32,
        ),
    }

    image = render_care_bev(
        obstacles,
        candidates,
        map_width_m=6.0,
        map_height_m=4.0,
        image_size=(640, 640),
        all_obstacle_points=np.asarray(
            [[1.0, 0.5], [1.2, 0.5]],
            dtype=np.float32,
        ),
    )
    pixels = np.asarray(image)

    assert image.mode == "RGB"
    assert image.size == (640, 640)
    assert np.any(np.all(pixels == (225, 55, 55), axis=2))
    assert np.any(np.all(pixels == (125, 132, 142), axis=2))
    assert np.any(np.all(pixels == (255, 211, 40), axis=2))
    assert np.any(np.all(pixels == (64, 156, 255), axis=2))


def test_render_care_bev_changes_selected_path_color_during_avoidance():
    candidates = {
        "selected": 0,
        "waypoint_index": 1,
        "avoidance_active": True,
        "actions": np.asarray(
            [[[0.2, 0.0], [0.4, 0.1]]],
            dtype=np.float32,
        ),
    }

    image = render_care_bev(
        np.empty((0, 2), dtype=np.float32),
        candidates,
        map_width_m=3.0,
        map_height_m=1.2,
        image_size=(640, 640),
    )
    pixels = np.asarray(image)

    assert np.any(np.all(pixels == (225, 70, 235), axis=2))
    assert not np.any(np.all(pixels == (255, 211, 40), axis=2))


def test_default_care_config_uses_shared_robot_frame_and_topics():
    config = load_care_config(os.path.join(package_root(), "config"))

    assert config["runtime"]["robot_frame"] == "base_footprint"
    assert config["map"]["width_m"] == 3.0
    assert config["map"]["height_m"] == 1.2
    assert config["avoidance"] == {
        "require_obstacle_data": True,
        "num_action_samples": 8,
        "waypoint_index": 1,
        "maximum_forward_range_m": 1.0,
        "path_influence_radius_m": 0.40,
        "depth_offset_m": 0.0,
        "minimum_force_distance_m": 0.10,
        "force_balance_ratio_threshold": 0.30,
        "theta_clip_degrees": 45.0,
        "safe_fov_threshold_degrees": 30.0,
    }
    assert (
        config["topics"]["obstacle_points_topic"]
        == "/unidepth/obstacle_points"
    )
    assert (
        config["topics"]["all_obstacle_points_topic"]
        == "/unidepth/obstacle_points_all"
    )
    assert config["topics"]["care_bev_topic"] == "/vnm/care_bev"
