import numpy as np


def cmd_dir_index(cmd_dir) -> int:
    return int(np.argmax(np.asarray(cmd_dir, dtype=np.float32)))


def same_cmd_dir(first, second) -> bool:
    return cmd_dir_index(first) == cmd_dir_index(second)


class CmdDirHoldFilter:
    def __init__(self, hold_samples_after_change: int = 0):
        self.hold_samples_after_change = int(hold_samples_after_change)
        self.stable_cmd_dir = None
        self.pending_cmd_dir = None
        self.pending_count = 0

    def reset(self):
        self.stable_cmd_dir = None
        self.pending_cmd_dir = None
        self.pending_count = 0

    def update(self, cmd_dir):
        cmd_dir = np.asarray(cmd_dir, dtype=np.float32)
        if cmd_dir.shape != (3,):
            raise ValueError(f"cmd_dir must have shape (3,), got {cmd_dir.shape}")
        if self.hold_samples_after_change <= 0:
            self.stable_cmd_dir = cmd_dir.copy()
            return cmd_dir.copy()
        if self.stable_cmd_dir is None:
            self.stable_cmd_dir = cmd_dir.copy()
            return self.stable_cmd_dir.copy()
        if same_cmd_dir(cmd_dir, self.stable_cmd_dir):
            self.pending_cmd_dir = None
            self.pending_count = 0
            return self.stable_cmd_dir.copy()
        if self.pending_cmd_dir is not None and same_cmd_dir(cmd_dir, self.pending_cmd_dir):
            self.pending_count += 1
        else:
            self.pending_cmd_dir = cmd_dir.copy()
            self.pending_count = 1
        if self.pending_count > self.hold_samples_after_change:
            self.stable_cmd_dir = self.pending_cmd_dir.copy()
            self.pending_cmd_dir = None
            self.pending_count = 0
        return self.stable_cmd_dir.copy()


def hold_cmd_dir_changes(cmd_dirs, hold_samples_after_change: int = 0) -> np.ndarray:
    cmd_dirs = np.asarray(cmd_dirs, dtype=np.float32)
    if cmd_dirs.ndim != 2 or cmd_dirs.shape[1] != 3:
        raise ValueError(f"cmd_dirs must have shape [N, 3], got {cmd_dirs.shape}")
    cmd_filter = CmdDirHoldFilter(hold_samples_after_change)
    return np.asarray([cmd_filter.update(cmd_dir) for cmd_dir in cmd_dirs], dtype=np.float32)
