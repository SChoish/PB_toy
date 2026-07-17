import unittest

import numpy as np

from concept.envs import ToyEnvConfig, ToyHazardEnv
from concept.fig_shared import chord_hits_hazard


class ChordHazardTest(unittest.TestCase):
    def setUp(self):
        self.env = ToyHazardEnv(ToyEnvConfig(hazard_center=(0.0, 0.0), hazard_radius=0.1))

    def test_detects_short_segment_inside_hazard(self):
        self.assertTrue(chord_hits_hazard(np.array([-0.01, 0.0]), np.array([0.01, 0.0]), self.env))

    def test_detects_tangent_as_collision(self):
        self.assertTrue(chord_hits_hazard(np.array([-1.0, 0.1]), np.array([1.0, 0.1]), self.env))

    def test_rejects_clear_segment(self):
        self.assertFalse(chord_hits_hazard(np.array([-1.0, 0.11]), np.array([1.0, 0.11]), self.env))


if __name__ == "__main__":
    unittest.main()
