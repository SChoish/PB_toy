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

    def test_continuous_slot_shift_changes_goal_without_changing_task(self):
        env = CarParkingEnv()
        _, base = env.reset(options={"task_id": 3})
        base_start = env.position.copy()
        _, shifted = env.reset(
            options={"task_id": 3, "slot_shift_x": 0.025}
        )
        self.assertAlmostEqual(shifted["goal"][0] - base["goal"][0], 0.025)
        np.testing.assert_allclose(env.position, base_start)
        self.assertEqual(shifted["task_id"], 3)
        env.close()

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
        self.assertEqual(frame.shape, (env._render_view()[4], 128, 3))
        self.assertLess(frame.shape[0], frame.shape[1])
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

    def test_collision_causes_damage_and_repeated_impact_depletes_health(self):
        env = CarParkingEnv(
            CarParkingConfig(maneuver="parallel"),
            observation_mode="state",
        )
        env.reset(options={"variant": 1})
        env.position = np.array([-0.80, -0.88], dtype=np.float32)
        env.heading = -np.pi / 2.0
        env.speed = 0.2
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["collision"])
        self.assertFalse(info["dead"])
        self.assertLess(info["health"], env.config.initial_health)
        self.assertGreater(info["health_loss"], 0.0)
        self.assertGreater(info["step_impulse"], 0.0)
        self.assertAlmostEqual(observation[9], info["health"])
        self.assertLess(reward, 0.0)

        env.position = np.array([-0.80, -0.88], dtype=np.float32)
        env.heading = -np.pi / 2.0
        env.speed = 0.2
        _, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["collision"])
        self.assertTrue(info["dead"])
        self.assertEqual(info["health"], 0.0)
        self.assertEqual(info["termination_reason"], "health_depleted")
        self.assertEqual(reward, env.config.death_penalty)
        with self.assertRaisesRegex(RuntimeError, "after episode end"):
            env.step(np.zeros(2, dtype=np.float32))
        env.close()

    def test_immediate_collision_termination_is_an_ablation(self):
        config = CarParkingConfig(
            maneuver="parallel",
            terminate_on_collision=True,
        )
        env = CarParkingEnv(config)
        env.reset(options={"variant": 1})
        env.position = np.array([-0.80, -0.88], dtype=np.float32)
        env.heading = -np.pi / 2.0
        env.speed = 0.05
        _, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["collision"])
        self.assertTrue(info["dead"])
        self.assertGreater(info["health"], 0.0)
        self.assertEqual(info["termination_reason"], "collision")
        self.assertEqual(reward, config.death_penalty)
        env.close()

    def test_reset_restores_or_accepts_valid_health(self):
        env = CarParkingEnv(
            CarParkingConfig(maneuver="parallel"),
            observation_mode="state",
        )
        observation, info = env.reset(
            options={"variant": 1, "health": 0.4}
        )
        self.assertAlmostEqual(observation[9], 0.4)
        self.assertAlmostEqual(info["health"], 0.4)
        observation, info = env.reset(options={"variant": 1})
        self.assertAlmostEqual(
            observation[9], env.config.initial_health
        )
        self.assertAlmostEqual(
            info["health"], env.config.initial_health
        )
        with self.assertRaisesRegex(ValueError, "health"):
            env.reset(options={"variant": 1, "health": 0.0})
        env.close()


class CarParkingSuccessTest(unittest.TestCase):
    def test_containment_bonus_is_awarded_only_once(self):
        config = CarParkingConfig(maneuver="parallel", dwell_steps=4)
        env = CarParkingEnv(config)
        env.reset(options={"variant": 1})
        env.position = np.asarray(env.layout.slot.center, dtype=np.float32)
        env.heading = env.layout.slot.heading
        _, first_reward, _, _, _ = env.step(
            np.zeros(2, dtype=np.float32)
        )
        _, second_reward, _, _, _ = env.step(
            np.zeros(2, dtype=np.float32)
        )
        self.assertAlmostEqual(
            first_reward - second_reward,
            config.containment_bonus,
        )
        env.close()

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
        env = CarParkingEnv(config, observation_mode="state")
        env.reset(options={"variant": 1})
        env.position = np.asarray(env.layout.slot.center, dtype=np.float32)
        env.heading = env.layout.slot.heading
        desired = env.desired_goal.copy()
        self.assertEqual(env.achieved_goal[-1], 0.0)
        self.assertLess(
            env.compute_reward(env.achieved_goal, desired),
            config.success_reward,
        )
        for step in range(config.dwell_steps):
            observation, _, terminated, _, info = env.step(
                np.zeros(2, dtype=np.float32)
            )
            self.assertEqual(info["dwell_count"], step + 1)
            self.assertAlmostEqual(
                observation[-1], (step + 1) / config.dwell_steps
            )
        self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        self.assertEqual(env.achieved_goal[-1], 1.0)
        self.assertEqual(
            env.compute_reward(env.achieved_goal, desired),
            config.success_reward,
        )
        env.close()

    def test_goal_reward_supports_batches(self):
        env = CarParkingEnv(CarParkingConfig(maneuver="parallel"))
        env.reset(options={"variant": 1})
        desired = env.desired_goal
        achieved = np.stack([desired, desired + np.array([1.0, 0.0, 0.0, 0.0, 0.0])])
        rewards = env.compute_reward(achieved, np.broadcast_to(desired, achieved.shape))
        self.assertEqual(rewards.shape, (2,))
        self.assertGreater(rewards[0], rewards[1])
        env.close()


if __name__ == "__main__":
    unittest.main()
