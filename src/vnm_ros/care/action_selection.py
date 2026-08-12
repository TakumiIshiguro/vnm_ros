import numpy as np


BASE_ACTION_STRATEGIES = ("mean", "first")


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
    if strategy == "mean":
        return actions.mean(axis=0)
    if strategy == "first":
        return actions[0].copy()
    raise ValueError(
        "base action strategy must be one of "
        + ", ".join(BASE_ACTION_STRATEGIES)
    )
