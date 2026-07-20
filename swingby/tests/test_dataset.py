import pathlib
import tempfile
import unittest

import numpy as np

from swingby.datasets import (
    SwingbyDQCDataset,
    SwingbyTRLDataset,
    CARTESIAN_ACTION_ENCODING,
    denormalize_actions,
    load_swingby_dataset,
    load_swingby_trl_dataset,
    normalize_actions,
)
from swingby.generate_dataset import collect_split
from swingby.env import SWINGBY_EVAL_ROTATION_RANGES, SWINGBY_TRAIN_ROTATION_RANGES


class SwingByDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = pathlib.Path(cls._tmp.name) / "swingby.npz"
        cls.raw, cls.stats = collect_split(
            env_name="swingby_planet",
            policy="expert",
            minimum_steps=500,
            seed=19,
            max_episode_steps=250,
            noise=0.1,
            dataset_mode="swingby",
        )
        np.savez_compressed(cls.path, **cls.raw)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_cartesian_thrust_round_trip_and_coast(self):
        actions = np.array(
            [[-np.pi, 1.0], [np.pi, 0.7], [1.2, 0.3], [-2.1, 0.0]],
            dtype=np.float32,
        )
        encoded = normalize_actions(
            actions, encoding=CARTESIAN_ACTION_ENCODING
        )
        decoded = denormalize_actions(
            encoded, encoding=CARTESIAN_ACTION_ENCODING
        )
        np.testing.assert_allclose(decoded[:, 1], actions[:, 1], atol=1e-6)
        np.testing.assert_allclose(encoded[-1], np.zeros(2), atol=1e-6)
        np.testing.assert_allclose(
            np.cos(decoded[:-1, 0]), np.cos(actions[:-1, 0]), atol=1e-6
        )
        np.testing.assert_allclose(
            np.sin(decoded[:-1, 0]), np.sin(actions[:-1, 0]), atol=1e-6
        )

    def test_swingby_balances_all_five_fixed_families(self):
        ids = self.raw["task_family_ids"]
        self.assertEqual(str(self.raw["dataset_schema"]), "swingby")
        self.assertEqual(str(self.raw["action_encoding"]), "cartesian_thrust")
        self.assertEqual(set(np.unique(ids)), {1, 2, 3, 4, 5})
        counts = np.asarray([(ids == task_id).sum() for task_id in range(1, 6)])
        self.assertTrue(np.all(counts >= 100), counts)
        for task_id in range(1, 6):
            self.assertTrue(self.raw["successes"][ids == task_id].any())

    def test_training_rotations_exclude_canonical_and_eval_bands(self):
        ids = self.raw["task_family_ids"]
        rotations = self.raw["task_rotations"]
        self.assertFalse(np.any(np.isclose(rotations, 0.0)))
        for task_id in range(1, 6):
            task_rotations = rotations[ids == task_id]
            train_ranges = SWINGBY_TRAIN_ROTATION_RANGES[task_id]
            eval_ranges = SWINGBY_EVAL_ROTATION_RANGES[task_id]
            in_train = np.zeros(len(task_rotations), dtype=bool)
            for low, high in train_ranges:
                in_train |= (task_rotations >= low) & (task_rotations <= high)
            self.assertTrue(np.all(in_train), (task_id, task_rotations))
            for train_low, train_high in train_ranges:
                for eval_low, eval_high in eval_ranges:
                    self.assertTrue(
                        train_high < eval_low or eval_high < train_low,
                        (task_id, train_ranges, eval_ranges),
                    )

    def test_default_sampler_uses_half_exact_commanded_goals(self):
        data = load_swingby_dataset(self.path, path_horizon=25)
        batch = data.sample(np.random.default_rng(0), 4000)
        commanded = batch["commanded_goal_mask"].astype(bool)
        self.assertGreater(float(commanded.mean()), 0.45)
        self.assertLess(float(commanded.mean()), 0.55)
        np.testing.assert_allclose(
            batch["value_goals"][commanded, :4],
            batch["goals"][commanded],
        )
        self.assertEqual(data.action_encoding, CARTESIAN_ACTION_ENCODING)
        need = max(data.path_horizon, data.action_chunk_horizon)
        self.assertTrue(
            np.all(data.valid_indices + need - 1 <= data.episode_ends[data.valid_indices])
        )

    def test_trl_commanded_goals_use_reached_trajectory_offsets(self):
        data = load_swingby_trl_dataset(
            self.path,
            config={"subgoal_steps": 25, "commanded_goal_prob": 1.0},
        )
        batch = data.sample(np.random.default_rng(4), 512)
        fixed_goals = np.unique(self.raw["commanded_goals"], axis=0)
        for goal in batch["actor_goals"]:
            self.assertTrue(
                np.any(np.all(np.isclose(fixed_goals, goal), axis=1))
            )
        self.assertTrue(np.all(batch["value_offsets"] > 0))


class SwingbySequenceAdapterTimeIndexTest(unittest.TestCase):
    def setUp(self):
        self.observations = np.zeros((6, 5), dtype=np.float32)
        self.next_observations = np.zeros((6, 5), dtype=np.float32)
        self.observations[:, 0] = np.arange(6)
        self.next_observations[:, 0] = np.arange(1, 7)
        self.actions = np.stack(
            [np.arange(6), -np.arange(6)], axis=-1
        ).astype(np.float32)
        self.commanded_goals = np.zeros((6, 4), dtype=np.float32)
        self.commanded_goal_indices = np.full(6, -1, dtype=np.int64)
        self.episode_ends = np.full(6, 5, dtype=np.int64)
        self.valid_indices = np.array([0], dtype=np.int64)

    def _make(self, cls, config):
        return cls(
            self.observations,
            self.actions,
            self.next_observations,
            self.commanded_goals,
            self.commanded_goal_indices,
            self.episode_ends,
            self.valid_indices,
            config,
        )

    def test_trl_and_dqc_offsets_match_regular_transition_states(self):
        trl = self._make(
            SwingbyTRLDataset,
            {"commanded_goal_prob": 0.0},
        )
        trl_batch = trl.sample(np.random.default_rng(3), 1)
        value_offset = int(trl_batch["value_offsets"][0])
        midpoint_offset = int(trl_batch["value_midpoint_offsets"][0])
        self.assertEqual(float(trl_batch["value_goals"][0, 0]), value_offset)
        self.assertEqual(
            float(trl_batch["value_midpoint_observations"][0, 0]),
            midpoint_offset,
        )
        self.assertGreater(midpoint_offset, 0)
        self.assertLess(midpoint_offset, value_offset)

        dqc = self._make(
            SwingbyDQCDataset,
            {
                "backup_horizon": 3,
                "commanded_goal_prob": 0.0,
                "discount": 0.99,
            },
        )
        dqc_batch = dqc.sample(np.random.default_rng(4), 1)
        backup = int(dqc_batch["high_value_backup_horizon"][0])
        self.assertEqual(
            float(dqc_batch["high_value_next_observations"][0, 0]), backup
        )
        np.testing.assert_array_equal(dqc_batch["valids"], np.ones((1, 3)))


if __name__ == "__main__":
    unittest.main()
