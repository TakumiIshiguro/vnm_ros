import unittest

import numpy as np

from vnm_ros.control.waypoint_controller import WaypointController


class WaypointControllerTest(unittest.TestCase):
    def test_turn_rate_is_bearing_over_one_control_step(self):
        controller = WaypointController(dt=0.2, max_v=10.0, max_w=10.0)
        waypoint = [0.3, 0.03]

        v, w = controller.command(waypoint)

        self.assertAlmostEqual(v, 0.3 / 0.2)
        self.assertAlmostEqual(w, np.arctan(0.03 / 0.3) / 0.2)

    def test_commands_are_clipped_to_the_configured_limits(self):
        controller = WaypointController(dt=0.2, max_v=0.25, max_w=0.4)

        v, w = controller.command([0.3, 0.3])

        self.assertAlmostEqual(v, 0.25)
        self.assertAlmostEqual(w, 0.4)

    def test_heading_is_used_only_when_the_waypoint_is_at_the_origin(self):
        controller = WaypointController(dt=0.2, max_v=10.0, max_w=10.0)

        v, w = controller.command([0.0, 0.0, 0.0, 1.0])

        self.assertAlmostEqual(v, 0.0)
        self.assertAlmostEqual(w, (np.pi / 2.0) / 0.2)

    def test_small_turn_rates_are_zeroed_by_the_deadband(self):
        controller = WaypointController(
            dt=0.25, max_v=0.25, max_w=0.4, angular_deadband=0.05
        )
        # ~0.5 deg of bearing, the model's straight-ahead bias: 0.035 rad/s.
        v, w = controller.command([1.0, np.tan(np.radians(0.5))])

        self.assertEqual(w, 0.0)
        self.assertGreater(v, 0.0)

    def test_the_deadband_leaves_real_turns_alone(self):
        controller = WaypointController(
            dt=0.25, max_v=0.25, max_w=0.4, angular_deadband=0.05
        )
        v, w = controller.command([1.0, np.tan(np.radians(12.0))])

        self.assertAlmostEqual(w, 0.4)

    def test_the_deadband_is_off_by_default(self):
        controller = WaypointController(dt=0.25, max_v=0.25, max_w=0.4)
        _, w = controller.command([1.0, np.tan(np.radians(0.5))])

        self.assertGreater(abs(w), 0.0)

    def test_a_negative_deadband_is_rejected(self):
        with self.assertRaises(ValueError):
            WaypointController(dt=0.25, max_v=0.25, max_w=0.4, angular_deadband=-0.1)


if __name__ == "__main__":
    unittest.main()
