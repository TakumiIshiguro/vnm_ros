import pytest

from vnm_ros.control.joy_direction import direction_from_joy


def test_dpad_axis_selects_left_and_right():
    axes = [0.0] * 7
    axes[6] = 1.0
    assert direction_from_joy(axes, []) == "left"

    axes[6] = -1.0
    assert direction_from_joy(axes, []) == "right"


def test_left_stick_is_used_when_dpad_is_neutral():
    axes = [0.75] + [0.0] * 6
    assert direction_from_joy(axes, []) == "left"


def test_neutral_and_conflicting_buttons_select_straight():
    assert direction_from_joy([0.0] * 7, []) == "straight"
    assert direction_from_joy(
        [],
        [1, 1],
        left_button=0,
        right_button=1,
    ) == "straight"


def test_optional_button_mapping_has_priority_over_axes():
    assert direction_from_joy(
        [-1.0],
        [1],
        left_button=0,
    ) == "left"


def test_invalid_axis_threshold_is_rejected():
    with pytest.raises(ValueError):
        direction_from_joy([], [], axis_threshold=0.0)
