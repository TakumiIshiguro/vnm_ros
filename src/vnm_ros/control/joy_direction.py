DIRECTION_COMMANDS = {
    "straight": [1, 0, 0],
    "left": [0, 1, 0],
    "right": [0, 0, 1],
}


def _button_pressed(buttons, index):
    return 0 <= index < len(buttons) and bool(buttons[index])


def _axis_value(axes, index):
    if 0 <= index < len(axes):
        return float(axes[index])
    return 0.0


def direction_from_joy(
    axes,
    buttons,
    horizontal_axis=6,
    fallback_horizontal_axis=0,
    axis_threshold=0.5,
    invert_axis=False,
    straight_button=-1,
    left_button=-1,
    right_button=-1,
):
    if not 0.0 < axis_threshold <= 1.0:
        raise ValueError("axis_threshold must be in (0, 1]")

    pressed_directions = [
        direction
        for direction, index in (
            ("straight", straight_button),
            ("left", left_button),
            ("right", right_button),
        )
        if _button_pressed(buttons, int(index))
    ]
    if len(pressed_directions) == 1:
        return pressed_directions[0]
    if len(pressed_directions) > 1:
        return "straight"

    for axis_index in (horizontal_axis, fallback_horizontal_axis):
        value = _axis_value(axes, int(axis_index))
        if invert_axis:
            value = -value
        if value >= axis_threshold:
            return "left"
        if value <= -axis_threshold:
            return "right"
    return "straight"
