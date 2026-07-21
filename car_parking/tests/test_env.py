import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from car_parking import (
    MANEUVERS,
    CarParkingConfig,
    CarParkingEnv,
    fixed_task_options,
    register_environment,
)
from car_parking.env import _parking_row_geometry


class CarParkingConfigTest(unittest.TestCase):
    def test_rejects_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "car_width"):
            CarParkingConfig(car_length=0.1, car_width=0.2).validate()

    def test_rejects_unknown_maneuver(self):
        with self.assertRaisesRegex(ValueError, "maneuver"):
            CarParkingConfig(maneuver="sideways").validate()  # type: ignore[arg-type]

    def test_rejects_implausibly_narrow_aisle(self):
        with self.assertRaisesRegex(ValueError, "aisle_width"):
            CarParkingConfig(aisle_width=0.20).validate()


class CarParkingApiTest(unittest.TestCase):
    def test_gymnasium_checker_for_every_maneuver(self):
        for maneuver in (*MANEUVERS, "mixed"):
            env = CarParkingEnv(
                CarParkingConfig(maneuver=maneuver),
                observation_mode="goal_dict",
            )
            check_env(env, skip_render_check=True)
            env.close()

    def test_seed_reproduces_layout_and_state(self):
        env = CarParkingEnv(observation_mode="state_goal")
        observation_a, info_a = env.reset(seed=17)
        observation_b, info_b = env.reset(seed=17)
        np.testing.assert_array_equal(observation_a, observation_b)
        self.assertEqual(info_a["maneuver"], info_b["maneuver"])
        env.close()

    def test_five_fixed_tasks_cover_parking_styles(self):
        names = [fixed_task_options(task_id)["maneuver"] for task_id in range(1, 6)]
        self.assertEqual(
            names,
            ["parallel", "parallel", "t_reverse", "t_forward", "angled"],
        )

    def test_registration(self):
        register_environment()
        for env_id in (
            "CarParking-v0",
            "CarParkingParallel-v0",
            "CarParkingTForward-v0",
            "CarParkingTReverse-v0",
            "CarParkingAngled-v0",
        ):
            env = gym.make(env_id)
            observation, _ = env.reset(seed=0)
            self.assertTrue(env.observation_space.contains(observation))
            env.close()

    def test_rgb_render(self):
        env = CarParkingEnv(render_mode="rgb_array", render_size=128)
        env.reset(options={"task_id": 1})
        frame = env.render()
        self.assertEqual(frame.shape, (128, 128, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(len(np.unique(frame.reshape(-1, 3), axis=0)), 4)
        env.close()

    def test_default_aisle_has_a_physical_opposite_boundary(self):
        env = CarParkingEnv(CarParkingConfig(maneuver="parallel"))
        _, info = env.reset(options={"variant": 1})
        islands = [box for box in env.layout.obstacles if box.kind == "island"]
        self.assertEqual(len(islands), 1)
        self.assertEqual(info["aisle_width"], env.config.aisle_width)
        self.assertEqual(env.config.aisle_width, 0.48)
        env.close()

    def test_fixed_tasks_use_lane_consistent_start_and_equal_sized_cars(self):
        env = CarParkingEnv()
        for task_id in range(1, 6):
            with self.subTest(task_id=task_id):
                env.reset(options={"task_id": task_id})
                geometry = _parking_row_geometry(
                    env.layout.slot, env.config.aisle_width
                )
                approach_lane_center = geometry[4]
                self.assertAlmostEqual(
                    float(env.position[1]), approach_lane_center
                )
                heading_delta = (
                    env.heading - env.layout.start_heading + np.pi
                ) % (2.0 * np.pi) - np.pi
                self.assertAlmostEqual(heading_delta, 0.0)
                parked = [
                    box
                    for box in env.layout.obstacles
                    if box.kind == "vehicle"
                ]
                self.assertEqual(len(parked), 2)
                for vehicle in parked:
                    self.assertEqual(vehicle.length, env.config.car_length)
                    self.assertEqual(vehicle.width, env.config.car_width)
                self.assertFalse(env._collides(env.vehicle_box))
        env.close()


class CarParkingDynamicsTest(unittest.TestCase):
    def test_throttle_moves_car_forward(self):
        env = CarParkingEnv(CarParkingConfig(maneuver="parallel"))
        env.reset(options={"variant": 1})
        x_before = float(env.position[0])
        for _ in range(8):
            env.step(np.array([0.0, 1.0], dtype=np.float32))
        self.assertGreater(env.position[0], x_before)
        env.close()

    def test_reverse_steering_uses_bicycle_sign(self):
        env = CarParkingEnv(CarParkingConfig(maneuver="parallel"))
        env.reset(options={"variant": 1, "speed": -0.1})
        heading_before = env.heading
        env.step(np.array([1.0, -0.2], dtype=np.float32))
        self.assertLess(env.heading, heading_before)
        env.close()

    def test_collision_terminates_episode(self):
        env = CarParkingEnv(CarParkingConfig(maneuver="parallel"))
        env.reset(options={"variant": 1})
        env.position = np.array([-0.80, -0.88], dtype=np.float32)
        env.heading = -np.pi / 2.0
        env.speed = 0.2
        _, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["collision"])
        self.assertEqual(reward, env.config.collision_penalty)
        with self.assertRaisesRegex(RuntimeError, "after episode end"):
            env.step(np.zeros(2, dtype=np.float32))
        env.close()


class CarParkingSuccessTest(unittest.TestCase):
    def test_pose_requires_full_containment_orientation_and_low_speed(self):
        env = CarParkingEnv(CarParkingConfig(maneuver="parallel"))
        env.reset(options={"variant": 1})
        env.position = np.asarray(env.layout.slot.center, dtype=np.float32)
        env.heading = env.layout.slot.heading
        env.speed = 0.0
        self.assertTrue(env.fully_inside_slot)
        self.assertTrue(env.parked_pose)

        env.heading += 2.0 * env.config.orientation_tolerance
        self.assertFalse(env.parked_pose)
        env.heading = env.layout.slot.heading
        env.speed = 2.0 * env.config.parked_speed_tolerance
        self.assertFalse(env.parked_pose)
        env.close()

    def test_success_needs_dwell_steps(self):
        config = CarParkingConfig(maneuver="parallel", dwell_steps=3)
        env = CarParkingEnv(config)
        env.reset(options={"variant": 1})
        env.position = np.asarray(env.layout.slot.center, dtype=np.float32)
        env.heading = env.layout.slot.heading
        for step in range(config.dwell_steps):
            _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
            self.assertEqual(info["dwell_count"], step + 1)
        self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        env.close()

    def test_goal_reward_supports_batches(self):
        env = CarParkingEnv(CarParkingConfig(maneuver="parallel"))
        env.reset(options={"variant": 1})
        desired = env.desired_goal
        achieved = np.stack([desired, desired + np.array([1.0, 0.0, 0.0, 0.0])])
        rewards = env.compute_reward(achieved, np.broadcast_to(desired, achieved.shape))
        self.assertEqual(rewards.shape, (2,))
        self.assertGreater(rewards[0], rewards[1])
        env.close()


if __name__ == "__main__":
    unittest.main()
