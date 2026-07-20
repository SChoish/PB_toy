import pathlib
import tempfile
import unittest

import numpy as np

from swingby.datasets import (
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


if __name__ == "__main__":
    unittest.main()
