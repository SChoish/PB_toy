import pathlib
import tempfile
import unittest

import numpy as np

from car_race.datasets import (
    goal_success,
    load_car_race_dataset,
    load_car_race_dqc_dataset,
    load_car_race_trl_dataset,
)
from car_race.env import CarRaceConfig, CarRaceEnv, mode_config_kwargs
from car_race.generate_dataset import collect_split, expert_action


class LapGoalSemanticsTest(unittest.TestCase):
    def test_state_prefix_and_final_goal_disambiguate_finish(self):
        env = CarRaceEnv(
            CarRaceConfig(task_mode="lap"), observation_mode="goal_dict"
        )
        observation, info = env.reset(
            seed=0, options={"start_checkpoint": 0, "direction": 1}
        )
        state = observation["observation"]
        desired = observation["desired_goal"]
        np.testing.assert_allclose(state[:4], [0.575, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(desired, [0.575, 0.0, 1.0, 1.0])
        self.assertFalse(
            bool(goal_success(observation["achieved_goal"], desired))
        )
        self.assertFalse(np.array_equal(info["current_waypoint"], info["final_goal"]))
        env.close()

    def test_navigation_uses_zero_progress_and_direction(self):
        env = CarRaceEnv(observation_mode="goal_dict")
        observation, _ = env.reset(
            options={"position": (0.575, 0.0), "goal": (0.0, 0.575)}
        )
        np.testing.assert_array_equal(
            observation["observation"][2:4], np.zeros(2, dtype=np.float32)
        )
        np.testing.assert_allclose(
            observation["desired_goal"], [0.0, 0.575, 0.0, 0.0]
        )
        env.close()

    def test_goal_reward_requires_progress_and_direction(self):
        env = CarRaceEnv(CarRaceConfig(task_mode="lap"))
        desired = np.array([0.575, 0.0, 1.0, 1.0], dtype=np.float32)
        same_xy_at_start = np.array([0.575, 0.0, 0.0, 1.0], dtype=np.float32)
        wrong_direction = np.array([0.575, 0.0, 1.0, -1.0], dtype=np.float32)
        finished = np.array([0.575, 0.0, 1.0, 1.0], dtype=np.float32)
        self.assertLess(env.compute_reward(same_xy_at_start, desired), 0.0)
        self.assertLess(env.compute_reward(wrong_direction, desired), 0.0)
        self.assertGreater(env.compute_reward(finished, desired), 0.0)
        env.close()


class CollectorTest(unittest.TestCase):
    def test_expert_brakes_for_a_sharp_turn_at_max_speed(self):
        env = CarRaceEnv(observation_mode="state")
        env.reset(
            options={
                "position": (0.575, 0.0),
                "goal": (-0.575, 0.0),
                "heading": 0.0,
                "speed": env.config.max_speed,
            }
        )
        action = expert_action(env)
        self.assertGreater(abs(float(action[0])), 0.9)
        self.assertLess(float(action[1]), 0.0)
        env.close()

    def test_expert_completes_lap_with_external_fields(self):
        for mode in ("car_race_grav", "car_race_anti_grav", "car_race_ice"):
            env = CarRaceEnv(
                CarRaceConfig(
                    task_mode="lap",
                    max_episode_steps=500,
                    **mode_config_kwargs(mode),
                ),
                observation_mode="state",
            )
            env.reset(seed=0)
            terminated = truncated = False
            info: dict = {}
            step_count = 0
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(
                    expert_action(env)
                )
                step_count += 1
            self.assertTrue(info["is_success"], mode)
            if mode == "car_race_ice":
                # Guard against making the slippery lap effectively trivial.
                self.assertGreaterEqual(step_count, 200)
            env.close()

    def test_ice_expert_completes_navigation_across_seeds(self):
        for seed in range(5):
            env = CarRaceEnv(
                CarRaceConfig(
                    max_episode_steps=300,
                    **mode_config_kwargs("car_race_ice"),
                ),
                observation_mode="state",
            )
            env.reset(seed=seed)
            info: dict = {}
            for step_count in range(1, 301):
                _, _, terminated, truncated, info = env.step(
                    expert_action(env)
                )
                if terminated or truncated:
                    break
            self.assertTrue(info["is_success"], seed)
            self.assertFalse(info["dead"], seed)
            self.assertGreaterEqual(step_count, 30)
            env.close()

    def test_ice_expert_dataset_collection(self):
        data, stats = collect_split(
            env_name="car_race_ice",
            policy="expert",
            minimum_steps=120,
            seed=7,
            max_episode_steps=250,
            noise=0.08,
        )
        self.assertGreaterEqual(len(data["actions"]), 120)
        self.assertEqual(data["observations"].shape[1], 8)
        self.assertEqual(data["actions"].shape[1], 2)
        self.assertEqual(stats["death_rate"], 0.0)
        self.assertTrue(np.all(np.abs(data["actions"]) <= 1.0))

    def test_all_collected_episodes_have_terminal_boundaries(self):
        data, stats = collect_split(
            policy="expert",
            minimum_steps=100,
            seed=3,
            max_episode_steps=250,
            noise=0.08,
        )
        self.assertGreaterEqual(len(data["actions"]), 100)
        self.assertGreaterEqual(stats["goals_per_episode"], 0.0)
        for episode_id in np.unique(data["episode_ids"]):
            indices = np.flatnonzero(data["episode_ids"] == episode_id)
            self.assertTrue(data["terminals"][indices[-1]])
            self.assertFalse(np.any(data["terminals"][indices[:-1]]))

    def test_train_and_validation_seeds_produce_different_rollouts(self):
        train, _ = collect_split(
            policy="expert",
            minimum_steps=30,
            seed=5,
            max_episode_steps=200,
            noise=0.08,
        )
        validation, _ = collect_split(
            policy="expert",
            minimum_steps=30,
            seed=1_000_005,
            max_episode_steps=200,
            noise=0.08,
        )
        self.assertFalse(
            np.array_equal(train["observations"][0], validation["observations"][0])
        )

    def test_actions_are_already_normalized(self):
        data, _ = collect_split(
            policy="random",
            minimum_steps=30,
            seed=7,
            max_episode_steps=100,
            noise=0.08,
        )
        self.assertTrue(np.all(data["actions"] >= -1.0))
        self.assertTrue(np.all(data["actions"] <= 1.0))


class CarRaceDatasetLoaderTest(unittest.TestCase):
    def _write_dataset(self, directory: str) -> pathlib.Path:
        data, _ = collect_split(
            policy="expert",
            minimum_steps=180,
            seed=11,
            max_episode_steps=250,
            noise=0.08,
        )
        path = pathlib.Path(directory) / "lap.npz"
        np.savez_compressed(path, **data)
        return path

    def test_one_raw_file_is_shared_by_navigation_and_all_laps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_dataset(directory)
            raw = np.load(path)
            self.assertEqual(raw["observations"].shape[1], 8)
            self.assertNotIn("goals", raw.files)
            views = {
                task: load_car_race_dataset(path, task=task)
                for task in ("navigation", "lap_2p", "lap_4p", "lap_8p")
            }
        reference_actions = views["navigation"].actions
        for dataset in views.values():
            np.testing.assert_array_equal(dataset.actions, reference_actions)
            self.assertEqual(dataset.observations.shape[1], 10)

    def test_batch_shapes_match_gcrl_agents(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_car_race_dataset(
                self._write_dataset(directory), task="lap_2p"
            )
            batch = dataset.sample(np.random.default_rng(0), 32)
        self.assertEqual(batch["observations"].shape, (32, 10))
        self.assertEqual(batch["actions"].shape, (32, 2))
        self.assertEqual(batch["goals"].shape, (32, 4))
        self.assertEqual(batch["high_actor_targets"].shape, (32, 10))
        self.assertEqual(batch["path_observations"].shape, (32, 9, 10))
        self.assertEqual(batch["rewards"].shape, (32,))
        self.assertEqual(batch["masks"].shape, (32,))

    def test_future_relabeling_preserves_direction_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_car_race_dataset(
                self._write_dataset(directory),
                task="lap_2p",
                goal_relabel_prob=1.0,
                random_goal_prob=0.0,
            )
            batch = dataset.sample(np.random.default_rng(4), 256)
        np.testing.assert_array_equal(
            batch["goals"][:, 3], batch["observations"][:, 3]
        )
        self.assertTrue(
            np.all(batch["goals"][:, 2] + 1e-6 >= batch["observations"][:, 2])
        )

    def test_loader_keeps_actions_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_dataset(directory)
            raw_actions = np.load(path)["actions"]
            dataset = load_car_race_dataset(path, task="lap_2p")
        np.testing.assert_array_equal(dataset.actions, raw_actions)

    def test_trl_and_dqc_adapters_use_four_dimensional_goals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_dataset(directory)
            trl = load_car_race_trl_dataset(
                path,
                config={"actor_p_randomgoal": 0.0},
                task="lap_2p",
            )
            dqc = load_car_race_dqc_dataset(
                path,
                config={
                    "backup_horizon": 8,
                    "value_p_randomgoal": 0.0,
                    "discount": 0.99,
                },
                task="lap_2p",
            )
            rng = np.random.default_rng(3)
            trl_batch = trl.sample(rng, 16)
            dqc_batch = dqc.sample(rng, 16)
        self.assertEqual(trl_batch["value_goals"].shape, (16, 4))
        self.assertEqual(trl_batch["actor_goals"].shape, (16, 4))
        self.assertEqual(dqc_batch["high_value_goals"].shape, (16, 4))
        self.assertEqual(dqc_batch["high_value_action_chunks"].shape, (16, 16))

    def test_loader_rejects_episode_without_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_dataset(directory)
            raw = dict(np.load(path))
            raw["terminals"] = np.zeros_like(raw["terminals"])
            broken = pathlib.Path(directory) / "broken.npz"
            np.savez_compressed(broken, **raw)
            with self.assertRaisesRegex(ValueError, "does not end"):
                load_car_race_dataset(broken)


if __name__ == "__main__":
    unittest.main()
