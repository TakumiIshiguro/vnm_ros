import numpy as np


BASE_ACTION_STRATEGIES = ("mean", "first", "cmd_dir")

CMD_DIR_NAMES = ("straight", "left", "right")


def select_base_action(actions, strategy: str) -> np.ndarray:
    """Choose the target-conditioned trajectory that CARE will adjust."""
    actions = np.asarray(actions)
    if actions.ndim != 3 or actions.shape[0] == 0 or actions.shape[-1] < 2:
        raise ValueError(
            "actions must have shape (samples, horizon, dimensions>=2)"
        )
    if not np.isfinite(actions).all():
        raise ValueError("actions must contain only finite values")

    strategy = str(strategy).strip().lower()
    if strategy == "cmd_dir":
        raise ValueError(
            "the cmd_dir strategy needs the command; call "
            "select_base_action_by_cmd_dir instead"
        )
    if strategy == "mean":
        return actions.mean(axis=0)
    if strategy == "first":
        return actions[0].copy()
    raise ValueError(
        "base action strategy must be one of "
        + ", ".join(BASE_ACTION_STRATEGIES)
    )


def trajectory_thetas(actions: np.ndarray) -> np.ndarray:
    """Mean bearing of each sampled trajectory, in radians, left positive.

    Averaged over the horizon rather than taken at the endpoint so a single
    noisy far waypoint cannot decide the sample's direction. Matches
    CmdDirActionSelector._trajectory_thetas so the two selectors rank
    samples the same way.
    """
    angles = np.arctan2(actions[:, :, 1], actions[:, :, 0])
    return np.arctan2(
        np.sin(angles).mean(axis=1), np.cos(angles).mean(axis=1)
    )


def select_base_action_by_cmd_dir(actions, cmd_dir, target_theta_deg=None):
    """Pick the sample whose bearing is nearest the commanded direction's.

    Returns ``(action, sample_index)``; the index is -1 when the returned
    action is the synthesised mean rather than one of the samples.

    Only a direction with a target angle is selected on. Straight has one --
    zero -- because the teacher's own bearing under a straight command is a
    tight unimodal distribution about it (median -0.03 deg, IQR 1.43 deg
    over a_f+g+h). Commanded straight the model nonetheless steers into side
    openings; there its samples spread and the nearest-to-zero one is much
    closer to the teacher than their mean (at a_f[2382], -0.5 deg against a
    mean of -2.9 deg).

    Left and right have no such angle. Their teacher bearings are four times
    wider (IQR 6.53 / 5.38 deg) and span zero at the 5th/95th percentile,
    because the correct angle depends on where in the turn the robot is:
    about zero on approach, near 10 deg at the apex, back to zero on exit. A
    fixed target would start the turn early and then cap it. So they default
    to no target, which averages the samples as the mean strategy does.
    """
    actions = np.asarray(actions)
    if actions.ndim != 3 or actions.shape[0] == 0 or actions.shape[-1] < 2:
        raise ValueError(
            "actions must have shape (samples, horizon, dimensions>=2)"
        )
    if not np.isfinite(actions).all():
        raise ValueError("actions must contain only finite values")

    target = _target_theta(cmd_dir, target_theta_deg)
    if target is None:
        return actions.mean(axis=0), -1
    thetas = trajectory_thetas(actions[:, :, :2])
    index = int(np.argmin(np.abs(thetas - target)))
    return actions[index].copy(), index


def _target_theta(cmd_dir, target_theta_deg):
    if cmd_dir is None or target_theta_deg is None:
        return None
    cmd_dir = np.asarray(cmd_dir).reshape(-1)[:3]
    if cmd_dir.size < 3 or np.count_nonzero(cmd_dir > 0) != 1:
        return None
    name = CMD_DIR_NAMES[int(np.argmax(cmd_dir))]
    value = target_theta_deg.get(name)
    if value is None:
        return None
    return float(np.deg2rad(float(value)))
