import unittest

import numpy as np

from vnm_ros.control.waypoint_controller import WaypointController


class WaypointControllerTest(unittest.TestCase):
    def test_waypoint_horizon_scales_only_angular_velocity(self):
        controller = WaypointController(dt=0.2, max_v=10.0, max_w=10.0)
        waypoint = [0.3, 0.03]

        one_step_v, one_step_w = controller.command(waypoint, waypoint_index=0)
        three_step_v, three_step_w = controller.command(waypoint, waypoint_index=2)

        self.assertAlmostEqual(three_step_v, one_step_v)
        self.assertAlmostEqual(three_step_w, one_step_w / 3.0)

    def test_negative_waypoint_index_uses_one_step_horizon(self):
        controller = WaypointController(dt=0.2, max_v=10.0, max_w=10.0)
        waypoint = [0.3, 0.03]

        negative_index_command = controller.command(waypoint, waypoint_index=-1)
        one_step_command = controller.command(waypoint, waypoint_index=0)

        np.testing.assert_allclose(negative_index_command, one_step_command)


if __name__ == "__main__":
    unittest.main()
