import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from swingby import OrbitalSwingByEnv, black_hole_config, planet_config
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

    def test_expert_action_is_normalized(self):
        env = OrbitalSwingByEnv(config=planet_config(), observation_mode="state")
        env.reset(seed=0)
        action = expert_action(env, PolicyState())
        self.assertTrue(np.all(action >= env.action_space.low))
        self.assertTrue(np.all(action <= env.action_space.high))
        env.close()

    def test_expert_preburns_only_for_a_ballistic_miss(self):
        env = OrbitalSwingByEnv(config=planet_config(), observation_mode="state")
        env.reset(options={"task_id": 1})
        coast = expert_action(env, PolicyState())
        self.assertEqual(float(coast[1]), 0.0)

        env.reset(options={"task_id": 2})
        preburn = expert_action(env, PolicyState())
        self.assertEqual(float(preburn[1]), 1.0)
        env.close()

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
