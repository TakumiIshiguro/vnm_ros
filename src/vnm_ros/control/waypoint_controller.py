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

    def __init__(
        self,
        dt: float,
        max_v: float,
        max_w: float,
        angular_deadband: float = 0.0,
        straight_max_w: float = 0.0,
    ):
        self.dt = dt
        if angular_deadband < 0.0:
            raise ValueError("angular_deadband must be non-negative")
        if straight_max_w < 0.0:
            raise ValueError("straight_max_w must be non-negative")
        self.angular_deadband = float(angular_deadband)
        self.straight_max_w = float(straight_max_w)
        self.angular_deadband_active = False
        self.safety = SafetyFilter(max_v=max_v, max_w=max_w)

    def command(
        self, waypoint: Sequence[float], straight: bool = False
    ) -> Tuple[float, float]:
        # Telemetry for this command only; do not retain it across stops.
        self.angular_deadband_active = False
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

        # A turn rate this small is the model's own straight-ahead bias, not
        # a correction worth making. Commanded straight the model aims a few
        # tenths of a degree off centre, which is only ~0.03 rad/s but never
        # changes sign, so the heading walks away at a couple of degrees a
        # second until something else pushes back. Measured over the real
        # dataset the two cases do not overlap: straight commands sit at a
        # median 0.04 rad/s and turns saturate max_w at 0.4, so a deadband
        # anywhere up to 0.08 zeroes most of the drift and none of the turns.
        #
        # It does leave genuine small corrections unmade until the error
        # grows past the band, trading a steady drift for a slow weave.
        # Zero disables it.
        if self.angular_deadband > 0.0 and abs(w) < self.angular_deadband:
            self.angular_deadband_active = bool(w != 0.0)
            w = 0.0

        # Commanded straight, the model still steers hard where a side
        # opening fills the view. Measured on a_f around the right-hand
        # opening at index 2376-2388, cmd_dir=straight produced up to
        # -0.22 rad/s -- over half max_w, and twice the median rate of a
        # real turn -- for about four seconds, which is enough to swing the
        # heading ~60 degrees into the opening. The recovery that follows is
        # weaker than the pull, so the net effect is a drift toward every
        # opening passed. Capping the rate while straight is commanded does
        # not stop the model reacting to the opening, it only bounds how far
        # it gets before the corridor comes back into view.
        #
        # The caller sets straight=False during active CARE avoidance so
        # obstacle steering retains the full max_w range. Zero disables it.
        if straight and self.straight_max_w > 0.0:
            w = float(np.clip(w, -self.straight_max_w, self.straight_max_w))
        return self.safety.clip(v, w)

    @staticmethod
    def lookahead_index(path: Sequence[Sequence[float]], lookahead_m: float) -> int:
        """Index of the first path point lookahead_m away, else the last."""
        distances = np.linalg.norm(np.asarray(path, dtype=np.float64)[:, :2], axis=1)
        beyond = np.flatnonzero(distances >= lookahead_m)
        return int(beyond[0]) if len(beyond) else len(distances) - 1

    def pure_pursuit(
        self, path: Sequence[Sequence[float]], lookahead_m: float
    ) -> Tuple[float, float]:
        """Follow the arc to where a robot-frame path crosses lookahead_m.

        The last point stands in for a shorter path, and an all-zero path
        stops the robot.
        """
        points = np.asarray(path, dtype=np.float64)[:, :2]
        index = self.lookahead_index(points, lookahead_m)
        target = points[index]
        if target @ target >= lookahead_m ** 2:
            start = points[index - 1] if index > 0 else np.zeros(2)
            step = target - start
            near = start @ step
            target = start + step * (
                -near
                + np.sqrt(
                    near ** 2 - (step @ step) * (start @ start - lookahead_m ** 2)
                )
            ) / (step @ step)
        distance_squared = target @ target
        if distance_squared < EPS:
            return 0.0, 0.0
        v = self.safety.max_v
        return self.safety.clip(v, 2.0 * v * target[1] / distance_squared)
