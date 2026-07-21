import pathlib
import tempfile
import unittest

import numpy as np

from car_parking.datasets import (
    CarParkingDQCDataset,
    CarParkingTRLDataset,
    load_car_parking_dataset,
    load_car_parking_dqc_dataset,
    load_car_parking_trl_dataset,
    reconstruct_states,
)


class CarParkingDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temporary.name) / "parking.npz"
        episode_length = 7
        count = 5 * episode_length
        observations = np.zeros((count, 8), dtype=np.float32)
        next_observations = np.zeros_like(observations)
        actions = np.zeros((count, 2), dtype=np.float32)
        goals = np.zeros((count, 5), dtype=np.float32)
        terminals = np.zeros(count, dtype=bool)
        successes = np.zeros(count, dtype=bool)
        deaths = np.zeros(count, dtype=bool)
        episode_ids = np.repeat(np.arange(5, dtype=np.int32), episode_length)
        task_ids = np.repeat(np.arange(1, 6, dtype=np.int8), episode_length)

        for episode in range(5):
            start = episode * episode_length
            base_x = float(10 * episode)
            observations[start : start + episode_length, 0] = (
                base_x + np.arange(episode_length)
            )
            next_observations[start : start + episode_length, 0] = (
                base_x + np.arange(1, episode_length + 1)
            )
            observations[start : start + episode_length, 2] = 1.0
            next_observations[start : start + episode_length, 2] = 1.0
            observations[start : start + episode_length, 6] = 1.0
            next_observations[start : start + episode_length, 6] = 1.0
            actions[start : start + episode_length, 0] = np.arange(episode_length)
            actions[start : start + episode_length] /= episode_length
            goals[start : start + episode_length] = [
                base_x + episode_length,
                0.0,
                1.0,
                0.0,
                1.0,
            ]
            terminals[start + episode_length - 1] = True
            successes[start + episode_length - 1] = True
            next_observations[start + episode_length - 1, 7] = 1.0

        np.savez_compressed(
            self.path,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            commanded_goals=goals,
            dataset_schema=np.asarray("parking_v2"),
            task_ids=task_ids,
            terminals=terminals,
            successes=successes,
            deaths=deaths,
            episode_ids=episode_ids,
            slot_lengths=np.full(count, 0.30, dtype=np.float32),
            slot_widths=np.full(count, 0.16, dtype=np.float32),
            car_length=np.asarray(0.18, dtype=np.float32),
            car_width=np.asarray(0.10, dtype=np.float32),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_reconstruction_changes_only_goal_relative_features(self):
        physical = np.array(
            [[0.0, 0.0, 1.0, 0.0, 0.1, -0.2, 0.75, 0.25]], dtype=np.float32
        )
        goals = np.array(
            [[0.0, 0.0, 1.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        states = reconstruct_states(
            np.broadcast_to(physical, (2, 8)),
            goals,
            np.full(2, 0.30),
            np.full(2, 0.16),
        )
        np.testing.assert_array_equal(states[0, :6], states[1, :6])
        self.assertEqual(states[0, 9], states[1, 9])
        self.assertEqual(states[0, 10], states[1, 10])
        self.assertFalse(np.array_equal(states[0, 6:9], states[1, 6:9]))
        self.assertEqual(states[0, 8], 1.0)
        self.assertAlmostEqual(float(states[1, 7]), -0.5)

    def test_loader_and_general_batch_contracts_are_goal_safe(self):
        data = load_car_parking_dataset(
            self.path,
            path_horizon=3,
            action_chunk_horizon=2,
            goal_relabel_prob=1.0,
        )
        batch = data.sample(np.random.default_rng(4), 512)
        self.assertEqual(data.observations.shape, (35, 11))
        self.assertEqual(batch["observations"].shape, (512, 11))
        self.assertEqual(batch["goals"].shape, (512, 5))
        self.assertEqual(batch["path_observations"].shape, (512, 4, 11))
        self.assertEqual(batch["action_chunk_actions"].shape, (512, 4))
        task_counts = np.bincount(
            batch["task_ids"].astype(np.int64), minlength=6
        )[1:]
        self.assertTrue(np.all(task_counts > 70))
        self.assertTrue(np.all(task_counts < 130))
        for key in (
            "subgoal_value_goals",
            "value_goals",
            "value_base_goals",
            "trans_v_left_goals",
            "trans_v_right_goals",
            "high_actor_targets",
            "action_chunk_next_observations",
        ):
            self.assertEqual(batch[key].shape, (512, 11), key)

        # Future-HER reward is attached only to the transition arriving at
        # the sampled future index.
        np.testing.assert_array_equal(
            batch["rewards"], (batch["value_offsets"] == 1).astype(np.float32)
        )
        # Every reconstructed full-state endpoint uses its own sampled goal.
        np.testing.assert_allclose(
            batch["value_goals"][:, :4], batch["goals"][:, :4], atol=1e-6
        )
        expected_distance = np.linalg.norm(
            batch["observations"][:, :2] - batch["goals"][:, :2], axis=1
        )
        np.testing.assert_allclose(batch["observations"][:, 6], expected_distance)

    def test_commanded_rewards_use_recorded_successes(self):
        data = load_car_parking_dataset(
            self.path,
            path_horizon=1,
            action_chunk_horizon=1,
            goal_relabel_prob=0.0,
        )
        data.valid_indices = np.array(
            [5, 6, 12, 13, 19, 20, 26, 27, 33, 34], dtype=np.int64
        )
        batch = data.sample(np.random.default_rng(1), 2000)
        self.assertTrue(np.all(batch["commanded_goal_mask"] == 1.0))
        terminal_state = np.mod(batch["observations"][:, 0], 10.0) == 6.0
        np.testing.assert_array_equal(
            batch["rewards"], terminal_state.astype(np.float32)
        )

    def test_paths_and_action_chunks_never_cross_episode_boundaries(self):
        data = load_car_parking_dataset(
            self.path,
            path_horizon=3,
            action_chunk_horizon=3,
            goal_relabel_prob=1.0,
        )
        self.assertFalse(np.any(data.valid_indices == 5))
        self.assertFalse(np.any(data.valid_indices == 6))
        batch = data.sample(np.random.default_rng(8), 1000)
        path_x = batch["path_observations"][:, :, 0]
        np.testing.assert_array_equal(
            np.floor(path_x[:, 0] / 10.0),
            np.floor(path_x[:, -1] / 10.0),
        )
        chunks = batch["action_chunk_actions"].reshape(1000, 3, 2)
        np.testing.assert_allclose(
            batch["action_chunk_next_observations"][:, 0],
            batch["observations"][:, 0] + 3.0,
        )
        self.assertTrue(np.all(chunks[:, :, 0] < 1.0))

    def test_trl_adapter_uses_strict_future_and_goal_safe_midpoints(self):
        data = load_car_parking_trl_dataset(
            self.path, config={"subgoal_steps": 2, "commanded_goal_prob": 0.5}
        )
        self.assertIsInstance(data, CarParkingTRLDataset)
        batch = data.sample(np.random.default_rng(2), 512)
        self.assertEqual(batch["observations"].shape, (512, 11))
        self.assertEqual(batch["value_goals"].shape, (512, 5))
        self.assertTrue(np.all(batch["value_offsets"] >= 2.0))
        self.assertTrue(np.all(batch["value_midpoint_offsets"] > 0.0))
        self.assertTrue(
            np.all(batch["value_midpoint_offsets"] < batch["value_offsets"])
        )
        expected = np.linalg.norm(
            batch["value_midpoint_observations"][:, :2]
            - batch["value_goals"][:, :2],
            axis=1,
        )
        np.testing.assert_allclose(
            batch["value_midpoint_observations"][:, 6], expected
        )
        np.testing.assert_allclose(
            batch["value_midpoint_actions"][:, 0],
            np.mod(batch["value_midpoint_observations"][:, 0], 10.0) / 7.0,
        )

    def test_dqc_adapter_shapes_and_full_chunks(self):
        data = load_car_parking_dqc_dataset(
            self.path,
            config={
                "backup_horizon": 3,
                "commanded_goal_prob": 0.5,
                "discount": 0.99,
            },
        )
        self.assertIsInstance(data, CarParkingDQCDataset)
        batch = data.sample(np.random.default_rng(3), 64)
        self.assertEqual(batch["high_value_goals"].shape, (64, 5))
        self.assertEqual(batch["high_value_next_observations"].shape, (64, 11))
        self.assertEqual(batch["high_value_action_chunks"].shape, (64, 6))
        np.testing.assert_array_equal(batch["valids"], np.ones((64, 3)))

    def test_rejects_unterminated_episode(self):
        with np.load(self.path) as archive:
            raw = {key: np.asarray(archive[key]) for key in archive.files}
        raw["terminals"] = np.zeros_like(raw["terminals"])
        broken = self.path.with_name("broken.npz")
        np.savez_compressed(broken, **raw)
        with self.assertRaisesRegex(ValueError, "does not end"):
            load_car_parking_dataset(broken)


if __name__ == "__main__":
    unittest.main()
