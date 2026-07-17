import unittest

import numpy as np

from hazard_env.env import GRAVITY_STRENGTHS, ContinuousHazard2DEnv, Hazard2DConfig


class HazardConfigTest(unittest.TestCase):
    def test_environment_mode_names_share_hazard_prefix(self):
        self.assertEqual(
            set(GRAVITY_STRENGTHS),
            {"hazard_plain", "hazard_grav", "hazard_anti_grav"},
        )

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

    def test_rejects_nonpositive_gravity_soft_min(self):
        with self.assertRaisesRegex(ValueError, "gravity_soft_min"):
            Hazard2DConfig(gravity_soft_min=0.0).validate()


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


class HazardGravityPhysicsTest(unittest.TestCase):
    def _rollout_zero_thrust(self, gravity_strength: float) -> float:
        env = ContinuousHazard2DEnv(
            config=Hazard2DConfig(gravity_strength=gravity_strength),
            observation_mode="state",
        )
        env.reset(
            seed=0,
            options={
                "position": np.array([-0.8, 0.0], dtype=np.float32),
                "goal": np.array([0.8, 0.55], dtype=np.float32),
                "velocity": np.zeros(2, dtype=np.float32),
            },
        )
        for _ in range(20):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
            if env.dead:
                break
        x = float(env.position[0])
        env.close()
        return x

    def test_positive_gravity_attracts_toward_hazard(self):
        self.assertGreater(self._rollout_zero_thrust(0.45), -0.8)

    def test_negative_gravity_repels_from_hazard(self):
        self.assertLess(self._rollout_zero_thrust(-0.45), -0.8)

    def test_zero_gravity_has_no_external_acceleration(self):
        env = ContinuousHazard2DEnv(
            config=Hazard2DConfig(gravity_strength=0.0),
            observation_mode="state",
        )
        np.testing.assert_array_equal(
            env._external_acceleration(np.array([-0.8, 0.0], dtype=np.float32)),
            np.zeros(2, dtype=np.float32),
        )
        env.close()

    def test_gravity_magnitude_follows_inverse_square_distance(self):
        env = ContinuousHazard2DEnv(
            config=Hazard2DConfig(gravity_strength=0.45),
            observation_mode="state",
        )
        center = np.asarray(env.config.hazard_center, dtype=np.float32)
        near = np.linalg.norm(
            env._external_acceleration(center + np.array([0.5, 0.0]))
        )
        far = np.linalg.norm(
            env._external_acceleration(center + np.array([1.0, 0.0]))
        )
        self.assertAlmostEqual(float(near / far), 4.0, places=5)
        env.close()


if __name__ == "__main__":
    unittest.main()
