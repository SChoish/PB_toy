import unittest

from car_parking import CarParkingEnv, rollout_expert


class ParkingExpertSmokeTest(unittest.TestCase):
    def test_canonical_tasks_succeed_through_real_env_steps(self):
        for task_id in range(1, 6):
            with self.subTest(task_id=task_id):
                env = CarParkingEnv()
                result = rollout_expert(env, task_id=task_id, seed=0)
                self.assertTrue(result.success)
                self.assertFalse(result.dead)
                self.assertFalse(result.collision)
                self.assertFalse(result.timeout)
                self.assertAlmostEqual(
                    result.minimum_health, env.config.initial_health
                )
                self.assertAlmostEqual(result.total_health_loss, 0.0)
                self.assertEqual(result.steps, env.elapsed_steps)
                self.assertGreater(len(result.actions), env.config.dwell_steps)
                env.close()


if __name__ == "__main__":
    unittest.main()
