import numpy as np


class SafetyFilter:
    def __init__(self, max_v: float, max_w: float):
        self.max_v = max_v
        self.max_w = max_w

    def clip(self, v: float, w: float):
        return (
            float(np.clip(v, 0.0, self.max_v)),
            float(np.clip(w, -self.max_w, self.max_w)),
        )

