from typing import Sequence, Tuple

import numpy as np

from vnm_ros.control.safety_filter import SafetyFilter

EPS = 1e-8


def clip_angle(theta: float) -> float:
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


class WaypointController:
    def __init__(
        self, dt: float, max_v: float, max_w: float, angular_gain: float = 1.0
    ):
        self.dt = dt
        # Multiplies the geometrically correct turn rate below. The model
        # predicts the conditional mean over the demonstrations, which
        # understates the sharpest turns, so following its waypoints exactly
        # undershoots the demonstrated turn radius; this compensates.
        self.angular_gain = float(angular_gain)
        self.safety = SafetyFilter(max_v=max_v, max_w=max_w)

    def command(
        self, waypoint: Sequence[float], waypoint_index: int = 0
    ) -> Tuple[float, float]:
        if len(waypoint) < 2:
            return 0.0, 0.0

        dx = float(waypoint[0])
        dy = float(waypoint[1])
        angular_dt = self.dt * (max(int(waypoint_index), 0) + 1)

        if len(waypoint) >= 4 and abs(dx) < EPS and abs(dy) < EPS:
            hx = float(waypoint[2])
            hy = float(waypoint[3])
            v = 0.0
            w = clip_angle(np.arctan2(hy, hx)) / angular_dt
        elif abs(dx) < EPS:
            v = 0.0
            w = np.sign(dy) * np.pi / (2 * angular_dt)
        else:
            v = dx / self.dt
            # On the constant-curvature arc that reaches the waypoint, the
            # waypoint's bearing is half the heading change, so the rate that
            # actually gets there is 2*bearing/angular_dt. Commanding
            # bearing/angular_dt steers at half the required rate.
            w = self.angular_gain * 2.0 * np.arctan(dy / dx) / angular_dt

        return self.safety.clip(v, w)
