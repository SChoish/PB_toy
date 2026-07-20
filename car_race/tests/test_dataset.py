import pathlib
import tempfile
import unittest

import numpy as np

from agents import DEFAULT_CONFIGS
from car_race.datasets import (
    CarRaceDQCDataset,
    CarRaceTRLDataset,
    goal_success,
    load_car_race_dataset,
    load_car_race_dqc_dataset,
    load_car_race_trl_dataset,
)
from car_race.env import CarRaceConfig, CarRaceEnv, mode_config_kwargs
from car_race.generate_dataset import collect_split, expert_action


class LapGoalSemanticsTest(unittest.TestCase):
    def test_lap_goal_is_the_active_waypoint(self):
        env = CarRaceEnv(
            CarRaceConfig(task_mode="lap"), observation_mode="goal_dict"
        )
        observation, info = env.reset(
            seed=0, options={"start_checkpoint": 0, "direction": 1}
        )
        state = observation["observation"]
        desired = observation["desired_goal"]
        np.testing.assert_allclose(state[:4], [0.575, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            desired,
            [0.575 / np.sqrt(2.0), 0.575 / np.sqrt(2.0), 0.125, 1.0],
            atol=1e-6,
        )
        self.assertEqual(state.shape, (14,))
        np.testing.assert_allclose(state[12:14], desired[:2])
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
        desired = np.array([0.0, 0.575, 0.125, 1.0], dtype=np.float32)
        same_xy_at_start = np.array([0.0, 0.575, 0.0, 1.0], dtype=np.float32)
        wrong_direction = np.array([0.0, 0.575, 0.125, -1.0], dtype=np.float32)
        finished = np.array([0.0, 0.575, 0.125, 1.0], dtype=np.float32)
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

    def test_expert_completes_every_lap_density_with_external_fields(self):
        for mode in ("car_race_grav", "car_race_anti_grav", "car_race_ice"):
            for checkpoint_count in range(2, 10):
                env = CarRaceEnv(
                    CarRaceConfig(
                        task_mode="lap",
                        checkpoint_count=checkpoint_count,
                        max_episode_steps=500,
                        **mode_config_kwargs(mode),
                    ),
                    observation_mode="state",
                )
                env.reset(
                    seed=0,
                    options={"start_checkpoint": 0, "direction": 1},
                )
                info: dict = {}
                for step_count in range(1, 501):
                    _, _, terminated, truncated, info = env.step(
                        expert_action(env)
                    )
                    if terminated or truncated:
                        break
                self.assertTrue(
                    info["is_success"], (mode, checkpoint_count, info)
                )
                if mode == "car_race_ice":
                    self.assertGreaterEqual(step_count, 150)
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

    def test_navigation_expert_is_one_successful_command_per_episode(self):
        data, stats = collect_split(
            policy="expert",
            minimum_steps=200,
            seed=23,
            max_episode_steps=300,
            noise=0.08,
            task="navigation",
        )
        self.assertEqual(stats["death_rate"], 0.0)
        self.assertEqual(stats["goals_per_episode"], 1.0)
        self.assertIn("commanded_goals", data)
        for episode_id in np.unique(data["episode_ids"]):
            indices = np.flatnonzero(data["episode_ids"] == episode_id)
            np.testing.assert_allclose(
                data["commanded_goals"][indices],
                np.broadcast_to(
                    data["commanded_goals"][indices[0]],
                    (len(indices), 4),
                ),
            )
            self.assertTrue(data["terminals"][indices[-1]])

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
            max_episode_steps=500,
            noise=0.08,
            task="lap",
        )
        path = pathlib.Path(directory) / "lap.npz"
        np.savez_compressed(path, **data)
        return path

    def test_one_raw_file_is_shared_by_navigation_and_lap1p_through_lap8p(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_dataset(directory)
            raw = np.load(path)
            self.assertEqual(raw["observations"].shape[1], 8)
            self.assertNotIn("goals", raw.files)
            views = {
                task: load_car_race_dataset(path, task=task)
                for task in (
                    "navigation",
                    *(f"lap_{n}p" for n in range(1, 9)),
                )
            }
        reference_actions = views["navigation"].actions
        for task, dataset in views.items():
            np.testing.assert_array_equal(dataset.actions, reference_actions)
            expected_dim = 10 if task == "navigation" else 14
            self.assertEqual(dataset.observations.shape[1], expected_dim)
            if task != "navigation":
                ring_count = int(task.split("_")[1][:-1]) + 1
                crossings = np.sum(
                    dataset.next_observations[:, 2]
                    > dataset.observations[:, 2] + 1e-6
                )
                self.assertEqual(
                    crossings,
                    len(np.unique(dataset.episode_ids)) * ring_count,
                )

    def test_ice_coarse_lap_view_truncates_post_success_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, _ = collect_split(
                policy="expert",
                minimum_steps=180,
                seed=5,
                max_episode_steps=500,
                noise=0.08,
                env_name="car_race_ice",
                task="lap",
            )
            path = pathlib.Path(directory) / "ice_lap.npz"
            np.savez_compressed(path, **raw)
            coarse = load_car_race_dataset(path, task="lap_1p")
            dense = load_car_race_dataset(path, task="lap_8p")

        self.assertLess(len(coarse), len(dense))
        for episode_id in np.unique(coarse.episode_ids):
            indices = np.flatnonzero(coarse.episode_ids == episode_id)
            self.assertTrue(coarse.terminals[indices[-1]])
            self.assertAlmostEqual(
                float(coarse.next_observations[indices[-1], 2]), 1.0
            )
        need = max(coarse.path_horizon, coarse.action_chunk_horizon)
        self.assertTrue(
            np.all(coarse.valid_indices + need - 1 <= coarse.episode_ends[coarse.valid_indices])
        )

    def test_batch_shapes_match_gcrl_agents(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_car_race_dataset(
                self._write_dataset(directory), task="lap_2p"
            )
            batch = dataset.sample(np.random.default_rng(0), 32)
        self.assertEqual(batch["observations"].shape, (32, 14))
        self.assertEqual(batch["actions"].shape, (32, 2))
        self.assertEqual(batch["goals"].shape, (32, 4))
        self.assertEqual(batch["high_actor_targets"].shape, (32, 14))
        self.assertEqual(batch["path_observations"].shape, (32, 9, 14))
        self.assertEqual(batch["rewards"].shape, (32,))
        self.assertEqual(batch["masks"].shape, (32,))

    def test_pb_value_goals_are_full_and_transitive_tuple_is_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_car_race_dataset(
                self._write_dataset(directory), task="lap_2p"
            )
            batch = dataset.sample(np.random.default_rng(9), 256)

        self.assertEqual(batch["goals"].shape, (256, 4))
        for key in (
            "subgoal_value_goals",
            "value_goals",
            "value_base_goals",
            "trans_v_left_goals",
            "trans_v_right_goals",
        ):
            self.assertEqual(batch[key].shape, (256, 14), key)
        np.testing.assert_array_equal(
            batch["trans_v_right_goals"], batch["value_goals"]
        )
        valid = batch["trans_v_valid_mask"] > 0
        self.assertTrue(
            np.all(
                batch["trans_v_split_offsets"][valid]
                < batch["value_offsets"][valid]
            )
        )
        self.assertTrue(
            np.any(
                np.all(
                    np.isclose(
                        batch["path_observations"][:, -1],
                        batch["path_observations"][:, -2],
                    ),
                    axis=1,
                )
            )
        )
        self.assertTrue(np.all(batch["value_base_offsets"] <= 5.0))
        np.testing.assert_allclose(
            batch["high_actor_targets"], batch["path_observations"][:, -1]
        )
        self.assertTrue(
            np.all(
                goal_success(
                    batch["subgoal_value_goals"][:, :4], batch["goals"]
                )
            )
        )


    def test_future_relabeling_preserves_direction_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_car_race_dataset(
                self._write_dataset(directory),
                task="lap_2p",
                goal_relabel_prob=1.0,
                checkpoint_goal_prob=0.0,
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
                    "backup_horizon": 25,
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
        self.assertEqual(dqc_batch["high_value_action_chunks"].shape, (16, 50))

    def test_loader_rejects_episode_without_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_dataset(directory)
            raw = dict(np.load(path))
            raw["terminals"] = np.zeros_like(raw["terminals"])
            broken = pathlib.Path(directory) / "broken.npz"
            np.savez_compressed(broken, **raw)
            with self.assertRaisesRegex(ValueError, "does not end"):
                load_car_race_dataset(broken)

    def test_pb_eval_subgoal_sample_counts(self):
        pbg = DEFAULT_CONFIGS["pbg"]()
        pbf = DEFAULT_CONFIGS["pbf"]()
        self.assertEqual(pbg["subgoal_eval_num_samples"], 1)
        self.assertEqual(pbf["subgoal_eval_num_samples"], 8)
        self.assertFalse(pbg["subgoal_include_mean"])
        self.assertFalse(pbf["subgoal_include_mean"])
        self.assertEqual(pbg["phi_goal_obs_indices"], (0, 1, 2, 3))
        self.assertEqual(pbf["phi_goal_obs_indices"], (0, 1, 2, 3))
        self.assertEqual(pbg["subgoal_value_goal_representation"], "full")

    def test_dqc_chunk_horizons(self):
        cfg = DEFAULT_CONFIGS["dqc"]()
        self.assertEqual(cfg["backup_horizon"], 25)
        self.assertEqual(cfg["policy_chunk_size"], 5)


class SequenceAdapterTimeIndexTest(unittest.TestCase):
    def setUp(self):
        self.observations = np.zeros((6, 4), dtype=np.float32)
        self.next_observations = np.zeros((6, 4), dtype=np.float32)
        self.observations[:, 0] = np.arange(6)
        self.next_observations[:, 0] = np.arange(1, 7)
        self.actions = np.stack(
            [np.arange(6), -np.arange(6)], axis=-1
        ).astype(np.float32)
        self.episode_ends = np.full(6, 5, dtype=np.int64)
        self.valid_indices = np.array([0], dtype=np.int64)

    def test_trl_offsets_match_regular_transition_states(self):
        dataset = CarRaceTRLDataset(
            self.observations,
            self.actions,
            self.next_observations,
            self.episode_ends,
            self.valid_indices,
            {"actor_p_randomgoal": 0.0},
        )
        batch = dataset.sample(np.random.default_rng(3), 1)
        value_offset = int(batch["value_offsets"][0])
        midpoint_offset = int(batch["value_midpoint_offsets"][0])
        self.assertEqual(float(batch["value_goals"][0, 0]), value_offset)
        self.assertEqual(
            float(batch["value_midpoint_observations"][0, 0]),
            midpoint_offset,
        )
        self.assertGreater(midpoint_offset, 0)
        self.assertLess(midpoint_offset, value_offset)

    def test_dqc_backup_state_is_after_exactly_backup_actions(self):
        discount = 0.99
        dataset = CarRaceDQCDataset(
            self.observations,
            self.actions,
            self.next_observations,
            self.episode_ends,
            self.valid_indices,
            {
                "backup_horizon": 3,
                "value_p_randomgoal": 0.0,
                "discount": discount,
            },
        )
        batch = dataset.sample(np.random.default_rng(4), 1)
        backup = int(batch["high_value_backup_horizon"][0])
        self.assertEqual(
            float(batch["high_value_next_observations"][0, 0]), backup
        )
        np.testing.assert_array_equal(batch["valids"], np.ones((1, 3)))
        if batch["high_value_masks"][0] == 0.0:
            self.assertAlmostEqual(
                float(batch["high_value_rewards"][0]), discount**backup
            )


if __name__ == "__main__":
    unittest.main()
