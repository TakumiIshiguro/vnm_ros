import numpy as np

from vnm_ros.care import CareRepulsiveAdjuster


def make_adjuster():
    return CareRepulsiveAdjuster(
        maximum_forward_range_m=1.0,
        path_influence_radius_m=0.40,
        depth_offset_m=0.0,
        minimum_force_distance_m=0.10,
        force_balance_ratio_threshold=0.30,
        theta_clip_degrees=45.0,
        safe_fov_threshold_degrees=30.0,
    )


def test_no_obstacles_preserves_original_trajectory():
    action = np.asarray([[0.2, 0.0], [0.4, 0.1]], dtype=np.float32)

    result = make_adjuster().adjust(action, np.empty((0, 2)))

    np.testing.assert_array_equal(result.action, action)
    assert result.rotation_angle == 0.0
    assert result.rotate_in_place is False
    assert result.reason == "no_nearby_obstacles"


def test_obstacle_on_left_rotates_trajectory_to_right():
    action = np.asarray([[0.2, 0.0], [0.4, 0.0]], dtype=np.float32)
    obstacles = np.asarray([[0.4, 0.2]], dtype=np.float32)

    result = make_adjuster().adjust(action, obstacles)

    assert result.rotation_angle < 0.0
    assert result.action[-1, 1] < 0.0
    assert result.reason == "repulsive_rotation"


def test_rotation_is_clipped_and_safe_fov_requests_in_place_turn():
    action = np.asarray([[0.2, 0.0], [0.4, 0.0]], dtype=np.float32)
    obstacles = np.asarray([[0.6, 0.0]], dtype=np.float32)

    result = make_adjuster().adjust(action, obstacles)

    assert np.isclose(abs(result.rotation_angle), np.pi / 4.0)
    assert np.isclose(abs(result.desired_heading), np.pi / 4.0)
    assert result.rotate_in_place is True


def test_obstacles_outside_care_range_are_ignored():
    action = np.asarray([[0.2, 0.0], [0.4, 0.0]], dtype=np.float32)
    obstacles = np.asarray([[1.1, 0.0], [-0.2, 0.0]], dtype=np.float32)

    result = make_adjuster().adjust(action, obstacles)

    np.testing.assert_array_equal(result.action, action)
    assert result.reason == "no_nearby_obstacles"


def test_lateral_bin_side_wall_points_do_not_rotate_straight_path():
    action = np.asarray([[0.2, 0.0], [0.4, 0.0]], dtype=np.float32)
    walls = np.asarray(
        [[0.1, 0.5], [0.1, -0.5]],
        dtype=np.float32,
    )

    result = make_adjuster().adjust(action, walls)

    np.testing.assert_array_equal(result.action, action)
    assert result.rotation_angle == 0.0
    assert result.reason == "no_nearby_obstacles"


def test_small_left_right_wall_imbalance_is_ignored():
    action = np.asarray([[0.2, 0.0]], dtype=np.float32)
    walls = np.asarray(
        [[0.2, 0.39], [0.2, -0.38]],
        dtype=np.float32,
    )

    result = make_adjuster().adjust(action, walls)

    np.testing.assert_array_equal(result.action, action)
    assert result.rotation_angle == 0.0
    assert result.reason == "balanced_repulsive_force"
    assert result.force_balance_ratio < 0.30


def test_one_sided_wall_force_remains_active():
    action = np.asarray([[0.2, 0.0]], dtype=np.float32)
    wall = np.asarray([[0.2, 0.39]], dtype=np.float32)

    result = make_adjuster().adjust(action, wall)

    assert result.rotation_angle < 0.0
    assert result.reason == "repulsive_rotation"
    assert result.force_balance_ratio == 1.0


def test_frontal_obstacle_within_one_meter_is_used():
    action = np.asarray([[0.2, 0.0], [0.4, 0.0]], dtype=np.float32)
    obstacle = np.asarray([[0.9, 0.0]], dtype=np.float32)

    result = make_adjuster().adjust(action, obstacle)

    assert result.reason == "repulsive_rotation"
    assert abs(result.rotation_angle) > 0.0


def test_obstacle_near_extended_curved_path_is_used():
    action = np.asarray([[0.2, 0.1], [0.4, 0.3]], dtype=np.float32)
    obstacle = np.asarray([[0.8, 0.7]], dtype=np.float32)

    result = make_adjuster().adjust(action, obstacle)

    assert result.reason == "repulsive_rotation"


def test_missing_depth_stop_generates_zero_trajectory():
    action = np.ones((8, 2), dtype=np.float32)

    result = make_adjuster().stop(action, "obstacle_data_missing_or_stale")

    np.testing.assert_array_equal(result.action, np.zeros_like(action))
    assert result.stopped is True
    assert result.reason == "obstacle_data_missing_or_stale"
