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
        self.assertEqual(eval_info["task_name"], "task1_offset_intercept")
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

    def test_fixed_eval_mix_uses_dataset_t2_and_interpolated_t5(self):
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
