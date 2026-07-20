import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from swingby import OrbitalSwingByEnv, black_hole_config, planet_config
from swingby.env import swingby_eval_rotation
from swingby.generate_dataset import collect_split
from swingby.policies import PolicyState, expert_action


class SwingByApiTest(unittest.TestCase):
    def test_gymnasium_checker_for_both_bodies(self):
        for config in (planet_config(), black_hole_config()):
            env = OrbitalSwingByEnv(config=config, observation_mode="goal_dict")
            check_env(env, skip_render_check=True)
            env.close()

    def test_all_fixed_tasks_reset_for_both_bodies(self):
        for config in (planet_config(), black_hole_config()):
            env = OrbitalSwingByEnv(config=config, observation_mode="state")
            for task_id in range(1, env.num_tasks + 1):
                observation, info = env.reset(options={"task_id": task_id})
                self.assertTrue(env.observation_space.contains(observation))
                self.assertEqual(info["task_id"], task_id)
                self.assertEqual(np.asarray(info["goal"]).shape, (4,))
                np.testing.assert_array_equal(
                    info["goal"][:2], info["goal_position"]
                )
                np.testing.assert_array_equal(
                    info["goal"][2:], info["goal_velocity"]
                )
                self.assertEqual(info["success"], info["is_success"])
            env.close()

    def test_rotated_task_reset_rotates_state_velocity_and_goal(self):
        env = OrbitalSwingByEnv(config=planet_config(), observation_mode="state_goal")
        base, _ = env.reset(options={"task_id": 3})
        angle = 0.17
        rotated, info = env.reset(
            options={"task_id": 3, "task_rotation": angle}
        )
        c, s = np.cos(angle), np.sin(angle)
        matrix = np.array([[c, -s], [s, c]])
        np.testing.assert_allclose(rotated[:2], matrix @ base[:2], atol=1e-5)
        np.testing.assert_allclose(rotated[2:4], matrix @ base[2:4], atol=1e-5)
        self.assertEqual(float(rotated[4]), float(base[4]))
        np.testing.assert_allclose(rotated[5:7], matrix @ base[5:7], atol=1e-5)
        np.testing.assert_allclose(rotated[7:9], matrix @ base[7:9], atol=1e-5)
        self.assertAlmostEqual(info["task_rotation"], angle)
        env.close()

    def test_swingby_eval_has_one_canonical_and_24_unique_heldout_variants(self):
        for task_id in range(1, 6):
            angles = np.array(
                [swingby_eval_rotation(task_id, i, 25) for i in range(25)]
            )
            self.assertEqual(int(np.isclose(angles, 0.0).sum()), 1)
            self.assertEqual(len(np.unique(angles)), 25)

    def test_fixed_tasks_are_outgoing_powered_flybys(self):
        env = OrbitalSwingByEnv(config=planet_config(), observation_mode="state")
        for task in env.task_infos:
            self.assertLess(float(task["init_xy"][0]), 0.0)
            self.assertGreater(float(task["goal_xy"][0]), 0.0)
        env.close()

    def test_gravity_matches_reported_potential_gradient(self):
        point = np.array([0.50, 0.20], dtype=np.float64)
        epsilon = 1e-5
        for config in (planet_config(), black_hole_config()):
            env = OrbitalSwingByEnv(config=config, observation_mode="state")
            numerical = np.zeros(2, dtype=np.float64)
            for axis in range(2):
                shift = np.zeros(2, dtype=np.float64)
                shift[axis] = epsilon
                numerical[axis] = -(
                    env._gravity_potential(point + shift)
                    - env._gravity_potential(point - shift)
                ) / (2.0 * epsilon)
            np.testing.assert_allclose(
                env.gravity_acceleration(point),
                numerical,
                rtol=2e-5,
                atol=2e-6,
            )
            env.close()

    def test_ballistic_integrator_conserves_orbital_invariants(self):
        for config in (planet_config(), black_hole_config()):
            env = OrbitalSwingByEnv(config=config, observation_mode="state")
            radius = 0.50
            if config.body_kind == "planet":
                denominator = (
                    radius * radius + config.gravity_softening**2
                )
                speed = np.sqrt(
                    config.gravitational_parameter
                    * radius
                    / denominator
                )
            else:
                speed = np.sqrt(
                    radius
                    * config.gravitational_parameter
                    / (radius - config.schwarzschild_radius) ** 2
                )
            positions, velocities, diagnostics = env.simulate_ballistic(
                np.array([radius, 0.0]),
                np.array([0.0, speed]),
                horizon_steps=300,
            )
            self.assertFalse(diagnostics["collided"])
            self.assertFalse(diagnostics["escaped"])
            energies = np.array(
                [
                    0.5 * np.dot(velocity, velocity)
                    + env._gravity_potential(position)
                    for position, velocity in zip(positions, velocities)
                ]
            )
            momenta = (
                positions[:, 0] * velocities[:, 1]
                - positions[:, 1] * velocities[:, 0]
            )
            relative_energy_range = np.ptp(energies) / abs(energies.mean())
            relative_momentum_range = np.ptp(momenta) / abs(momenta.mean())
            self.assertLess(relative_energy_range, 1e-4)
            self.assertLess(relative_momentum_range, 1e-4)
            env.close()

    def test_goal_event_stops_at_first_matching_crossing(self):
        config = planet_config(
            gravitational_parameter=0.0,
            dt=1.0,
            physics_substeps=1,
            max_speed=1.0,
        )
        env = OrbitalSwingByEnv(config=config, observation_mode="state")
        env.reset(
            options={
                "position": (-0.10, 0.50),
                "velocity": (0.30, 0.0),
                "goal": (0.0, 0.50),
                "goal_velocity": (0.30, 0.0),
            }
        )
        _, _, terminated, _, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        self.assertAlmostEqual(
            env.distance_to_goal, env.config.goal_radius, places=5
        )
        self.assertEqual(
            env.compute_reward(env.achieved_goal, env.desired_goal),
            env.config.success_reward,
        )
        env.close()

    def test_body_collision_stops_at_capture_surface(self):
        config = planet_config(
            gravitational_parameter=0.0,
            physics_substeps=1,
        )
        env = OrbitalSwingByEnv(config=config, observation_mode="state")
        env.reset(
            options={
                "position": (0.21, 0.0),
                "velocity": (-3.0, 0.0),
                "goal": (0.0, 0.50),
                "goal_velocity": (0.0, 0.0),
            }
        )
        _, _, terminated, _, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertTrue(info["dead"])
        capture_radius = config.body_radius + config.satellite_radius
        self.assertAlmostEqual(env.distance_to_body, capture_radius, places=6)
        env.close()

    def test_slow_capture_rejects_stationary_velocity(self):
        env = OrbitalSwingByEnv(config=planet_config(), observation_mode="state")
        env.reset(options={"task_id": 5})
        achieved = np.concatenate([env.goal, np.zeros(2, dtype=np.float32)])
        reward = env.compute_reward(achieved, env.desired_goal)
        self.assertLess(reward, env.config.success_reward)
        env.close()

    def test_expert_action_is_normalized(self):
        env = OrbitalSwingByEnv(config=planet_config(), observation_mode="state")
        env.reset(seed=0)
        action = expert_action(env, PolicyState())
        self.assertTrue(np.all(action >= env.action_space.low))
        self.assertTrue(np.all(action <= env.action_space.high))
        env.close()

    def test_fixed_eval_removes_trivial_t1_but_dataset_profile_is_frozen(self):
        eval_env = OrbitalSwingByEnv(
            config=planet_config(), observation_mode="state"
        )
        _, eval_info = eval_env.reset(options={"task_id": 1})
        correction = expert_action(eval_env, PolicyState())
        self.assertEqual(eval_info["task_profile"], "eval_fixed")
        self.assertEqual(eval_info["task_name"], "task1_shallow_powered_flyby")
        self.assertEqual(float(correction[1]), 1.0)
        eval_env.close()

        data_env = OrbitalSwingByEnv(
            config=planet_config(),
            observation_mode="state",
            task_profile="dataset",
        )
        _, data_info = data_env.reset(options={"task_id": 1})
        coast = expert_action(data_env, PolicyState())
        self.assertEqual(data_info["task_name"], "task1_coast_alignment")
        self.assertEqual(float(coast[1]), 0.0)
        data_env.close()

    def test_fixed_eval_mix_uses_dataset_t2_and_interpolated_hard_tasks(self):
        eval_env = OrbitalSwingByEnv(
            config=planet_config(), observation_mode="state"
        )
        data_env = OrbitalSwingByEnv(
            config=planet_config(),
            observation_mode="state",
            task_profile="dataset",
        )
        for key in ("init_xy", "init_velocity", "goal_xy", "goal_velocity"):
            np.testing.assert_allclose(
                eval_env.task_infos[1][key], data_env.task_infos[1][key]
            )
        self.assertEqual(
            eval_env.task_infos[4]["task_name"], "task5_mixed_far_capture"
        )
        self.assertEqual(
            eval_env.task_infos[3]["task_name"], "task4_heldout_deep_swingby"
        )
        self.assertFalse(
            np.allclose(
                eval_env.task_infos[4]["goal_xy"],
                data_env.task_infos[4]["goal_xy"],
            )
        )
        eval_env.close()
        data_env.close()

    def test_expert_completes_all_fixed_tasks_for_both_bodies(self):
        for config in (planet_config(), black_hole_config()):
            for task_id in range(1, 6):
                env = OrbitalSwingByEnv(config=config, observation_mode="state")
                env.reset(options={"task_id": task_id})
                policy_state = PolicyState()
                info: dict = {}
                for _ in range(config.max_episode_steps):
                    _, _, terminated, truncated, info = env.step(
                        expert_action(env, policy_state)
                    )
                    if terminated or truncated:
                        break
                self.assertTrue(
                    info["is_success"],
                    f"{config.body_kind=} {task_id=} {info=}",
                )
                env.close()

    def test_dataset_collector_uses_package_imports(self):
        data, stats = collect_split(
            env_name="swingby_planet",
            policy="expert",
            minimum_steps=30,
            seed=3,
            max_episode_steps=120,
            noise=0.1,
        )
        self.assertGreaterEqual(len(data["actions"]), 30)
        self.assertGreaterEqual(stats["episodes"], 1.0)


if __name__ == "__main__":
    unittest.main()
