import unittest

import numpy as np

from simulation import CONTROLLERS, DURATION, run_simulation, target_state


class SimulationTests(unittest.TestCase):
    def test_target_state_shapes_and_finiteness(self):
        for t in (0.0, 17.5, DURATION):
            position, velocity, acceleration = target_state(t)
            self.assertEqual(position.shape, (3,))
            self.assertEqual(velocity.shape, (3,))
            self.assertEqual(acceleration.shape, (3,))
            self.assertTrue(np.all(np.isfinite(position)))

    def test_every_controller_completes_sixty_seconds(self):
        for name in CONTROLLERS:
            with self.subTest(controller=name):
                result = run_simulation(name)
                self.assertAlmostEqual(float(result.time[-1]), 60.0)
                self.assertEqual(result.uav_position.shape[1], 3)
                self.assertTrue(np.all(np.isfinite(result.uav_position)))
                self.assertTrue(np.all(np.isfinite(result.control)))
                self.assertLess(result.metrics["rmse_total"], 8.0)
                self.assertGreaterEqual(result.metrics["tracking_rate"], 0.0)
                self.assertLessEqual(result.metrics["tracking_rate"], 100.0)
                self.assertGreaterEqual(result.metrics["accuracy"], 0.0)
                self.assertLessEqual(result.metrics["accuracy"], 100.0)

    def test_unknown_controller_rejected(self):
        with self.assertRaises(ValueError):
            run_simulation("bilinmeyen")


if __name__ == "__main__":
    unittest.main()
