import unittest

import numpy as np

from hazard_env.env import ContinuousHazard2DEnv, Hazard2DConfig


class HazardConfigTest(unittest.TestCase):
    def test_rejects_negative_drag(self):
        with self.assertRaisesRegex(ValueError, "linear_drag"):
            Hazard2DConfig(linear_drag=-0.1).validate()

    def test_rejects_unphysical_wall_restitution(self):
        with self.assertRaisesRegex(ValueError, "wall_restitution"):
            Hazard2DConfig(wall_restitution=1.1).validate()

    def test_rejects_negative_sampling_distances(self):
        with self.assertRaisesRegex(ValueError, "min_start_goal_distance"):
            Hazard2DConfig(min_start_goal_distance=-0.1).validate()
        with self.assertRaisesRegex(ValueError, "spawn_clearance"):
            Hazard2DConfig(spawn_clearance=-0.1).validate()


class HazardEpisodeLifecycleTest(unittest.TestCase):
    def test_step_after_time_limit_requires_reset(self):
        env = ContinuousHazard2DEnv(
            config=Hazard2DConfig(max_episode_steps=1),
            observation_mode="state",
        )
        env.reset(
            seed=0,
            options={"position": (-0.8, -0.8), "goal": (0.8, 0.8)},
        )
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaisesRegex(RuntimeError, "after episode end"):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
        env.close()


if __name__ == "__main__":
    unittest.main()
