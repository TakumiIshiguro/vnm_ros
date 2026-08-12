import numpy as np
import pytest

from vnm_ros.care import select_base_action


def test_mean_base_action_averages_target_conditioned_samples():
    actions = np.asarray(
        [
            [[0.1, 0.0], [0.2, 0.1]],
            [[0.3, 0.2], [0.4, 0.3]],
        ],
        dtype=np.float32,
    )

    selected = select_base_action(actions, "mean")

    np.testing.assert_allclose(
        selected,
        [[0.2, 0.1], [0.3, 0.2]],
    )


def test_first_base_action_preserves_first_sample():
    actions = np.asarray(
        [
            [[0.1, 0.0], [0.2, 0.1]],
            [[0.3, 0.2], [0.4, 0.3]],
        ],
        dtype=np.float32,
    )

    selected = select_base_action(actions, "first")

    np.testing.assert_array_equal(selected, actions[0])
    assert selected is not actions[0]


@pytest.mark.parametrize("strategy", ("median", "", "unknown"))
def test_unknown_base_action_strategy_is_rejected(strategy):
    with pytest.raises(ValueError, match="base action strategy"):
        select_base_action(np.zeros((2, 3, 2), dtype=np.float32), strategy)


@pytest.mark.parametrize(
    "actions",
    (
        np.empty((0, 3, 2), dtype=np.float32),
        np.empty((2, 3), dtype=np.float32),
        np.full((2, 3, 2), np.nan, dtype=np.float32),
    ),
)
def test_invalid_actions_are_rejected(actions):
    with pytest.raises(ValueError):
        select_base_action(actions, "mean")
