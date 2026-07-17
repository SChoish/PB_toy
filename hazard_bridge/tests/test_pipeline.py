import unittest

import numpy as np

from toy_pathbridger.dataset import generate_dataset, validate_dataset
from toy_pathbridger.env import ToyEnv
from toy_pathbridger.learning import PinnedBridgeRegressor


class ToyPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ToyEnv(grid_size=41)
        cls.episodes = generate_dataset(cls.env, n_episodes=32, seed=11)

    def test_dataset_segments_avoid_hazards(self):
        report = validate_dataset(self.env, self.episodes)
        self.assertEqual(report["unsafe_transitions"], 0)
        self.assertGreater(report["transitions"], 100)

    def test_pinned_bridge_keeps_exact_endpoints(self):
        model = PinnedBridgeRegressor.fit(self.episodes, horizon=8)
        ep = next(ep for ep in self.episodes if len(ep.states) > 8)
        start, endpoint = ep.states[0], ep.states[8]
        bridge = model.predict(start, endpoint)
        np.testing.assert_allclose(bridge[0], start, atol=1e-12)
        np.testing.assert_allclose(bridge[-1], endpoint, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
