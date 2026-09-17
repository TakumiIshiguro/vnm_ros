from typing import Sequence, Tuple

import numpy as np

from vnm_ros.control.safety_filter import SafetyFilter

EPS = 1e-8


def clip_angle(theta: float) -> float:
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


class WaypointController:
    """Stock NoMaD/ViNT waypoint controller.

    Turn rate is the waypoint's bearing divided by one control step, matching
    the reference implementation. Spreading it over the horizon to the chosen
    waypoint instead (dt * (waypoint_index + 1)) divided the commanded rate by
    that horizon, which left the robot tracking a 3.2 m radius where the
    demonstrations drove 1.2 m.
    """

    def __init__(self, dt: float, max_v: float, max_w: float):
        self.dt = dt
        self.safety = SafetyFilter(max_v=max_v, max_w=max_w)

    def command(self, waypoint: Sequence[float]) -> Tuple[float, float]:
        if len(waypoint) < 2:
            return 0.0, 0.0

        dx = float(waypoint[0])
        dy = float(waypoint[1])

        if len(waypoint) >= 4 and abs(dx) < EPS and abs(dy) < EPS:
            hx = float(waypoint[2])
            hy = float(waypoint[3])
            v = 0.0
            w = clip_angle(np.arctan2(hy, hx)) / self.dt
        elif abs(dx) < EPS:
            v = 0.0
            w = np.sign(dy) * np.pi / (2 * self.dt)
        else:
            v = dx / self.dt
            w = np.arctan(dy / dx) / self.dt

        return self.safety.clip(v, w)
