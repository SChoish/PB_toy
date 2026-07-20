import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from car_race import (
    CORNERING_GRIPS,
    EXTERNAL_DRAGS,
    GRAVITY_STRENGTHS,
    LONGITUDINAL_GRIPS,
    MAX_EXTERNAL_SPEEDS,
    ROLLING_DRAGS,
    STEERING_RESPONSES,
    CarRaceConfig,
    CarRaceEnv,
    mode_config_kwargs,
    register_environment,
)


class CarRaceConfigTest(unittest.TestCase):
    def test_rejects_invalid_hazard_radii(self):
        with self.assertRaisesRegex(ValueError, "hazard radii"):
            CarRaceConfig(inner_hazard_radius=0.96).validate()

    def test_rejects_track_inside_hazard(self):
        with self.assertRaisesRegex(ValueError, "track_radius"):
            CarRaceConfig(track_radius=0.1).validate()

    def test_rejects_nonpositive_damage_capacity(self):
        with self.assertRaisesRegex(ValueError, "damage_capacity"):
            CarRaceConfig(damage_capacity=0.0).validate()

    def test_rejects_invalid_tire_response(self):
        invalid = (
            ("cornering_grip", {"cornering_grip": 1.1}),
            ("longitudinal_grip", {"longitudinal_grip": 0.0}),
            ("steering_response", {"steering_response": 0.0}),
        )
        for message, kwargs in invalid:
            with self.subTest(parameter=message):
                with self.assertRaisesRegex(ValueError, message):
                    CarRaceConfig(**kwargs).validate()


class CarRaceApiTest(unittest.TestCase):
    def test_gymnasium_checker_for_both_tasks_and_surfaces(self):
        for task_mode in ("navigation", "lap"):
            for surface_mode in ("car_race_plain", "car_race_ice"):
                env = CarRaceEnv(
                    CarRaceConfig(
                        task_mode=task_mode,
                        **mode_config_kwargs(surface_mode),
                    ),
                    observation_mode="goal_dict",
                )
                check_env(env, skip_render_check=True)
                env.close()

    def test_all_lap_densities_are_registered(self):
        register_environment()
        for waypoint_count in range(1, 9):
            for prefix in (
                "CarRace",
                "CarRacePlain",
                "CarRaceGrav",
                "CarRaceAntiGrav",
                "CarRaceIce",
            ):
                env = gym.make(f"{prefix}Lap{waypoint_count}p-v0")
                observation, _ = env.reset(seed=waypoint_count)
                self.assertTrue(env.observation_space.contains(observation))
                env.close()

    def test_seed_reproduces_navigation_task(self):
        env = CarRaceEnv(observation_mode="state_goal")
        obs_a, _ = env.reset(seed=17)
        obs_b, _ = env.reset(seed=17)
        np.testing.assert_array_equal(obs_a, obs_b)
        env.close()

    def test_step_after_time_limit_requires_reset(self):
        env = CarRaceEnv(
            CarRaceConfig(max_episode_steps=1), observation_mode="state"
        )
        env.reset(seed=0)
        _, _, terminated, truncated, _ = env.step(
            np.zeros(2, dtype=np.float32)
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaisesRegex(RuntimeError, "after episode end"):
            env.step(np.zeros(2, dtype=np.float32))
        env.close()

    def test_registration(self):
        register_environment()
        for env_id in (
            "CarRaceNavigation-v0",
            "CarRaceLap1p-v0",
            "CarRaceLap2p-v0",
            "CarRaceLap4p-v0",
            "CarRaceLap8p-v0",
            "CarRaceGravLap2p-v0",
            "CarRaceAntiGravLap4p-v0",
            "CarRaceIceNavigation-v0",
            "CarRaceIceLap8p-v0",
        ):
            env = gym.make(env_id)
            observation, _ = env.reset(seed=0)
            self.assertTrue(env.observation_space.contains(observation))
            env.close()


class CarRaceDynamicsTest(unittest.TestCase):
    def test_throttle_moves_car_forward(self):
        env = CarRaceEnv(observation_mode="state")
        env.reset(
            options={
                "position": (0.55, 0.0),
                "goal": (0.0, 0.55),
                "heading": 0.0,
            }
        )
        x_before = float(env.position[0])
        for _ in range(5):
            env.step(np.array([0.0, 1.0], dtype=np.float32))
        self.assertGreater(float(env.position[0]), x_before)
        self.assertGreater(env.speed, 0.0)
        env.close()

    def test_steering_changes_heading(self):
        env = CarRaceEnv(observation_mode="state")
        env.reset(
            options={
                "position": (0.55, 0.0),
                "goal": (0.0, 0.55),
                "heading": np.pi / 2.0,
                "speed": 0.3,
            }
        )
        heading_before = env.heading
        env.step(np.array([0.7, 0.0], dtype=np.float32))
        self.assertGreater(env.heading, heading_before)
        env.close()

    def test_speed_is_clipped(self):
        env = CarRaceEnv(
            CarRaceConfig(max_episode_steps=300), observation_mode="state"
        )
        env.reset(
            options={
                "position": (0.55, 0.0),
                "goal": (-0.55, 0.0),
                "heading": np.pi / 2.0,
            }
        )
        for _ in range(100):
            env.step(np.array([0.25, 1.0], dtype=np.float32))
            if env.dead:
                break
            self.assertLessEqual(env.speed, env.config.max_speed)
        env.close()

    def test_square_wall_stops_car(self):
        env = CarRaceEnv(observation_mode="state")
        env.reset(
            options={
                "position": (0.55, 0.0),
                "goal": (0.0, 0.55),
            }
        )
        env.position = np.array([0.944, 0.0], dtype=np.float32)
        env.heading = 0.0
        env.speed = 0.8
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(info["wall_collision"])
        self.assertAlmostEqual(env.speed, 0.0)
        self.assertLessEqual(
            env.position[0], env.config.arena_high - env.config.collision_radius
        )
        env.close()


class CarRaceDamageTest(unittest.TestCase):
    def _ram_inner_hazard(self, speed: float) -> tuple[float, float, float]:
        """Hit the inner ring once at a controlled normal speed."""
        env = CarRaceEnv(
            CarRaceConfig(rolling_drag=0.0, max_episode_steps=20),
            observation_mode="state",
        )
        env.reset(
            options={
                "position": (0.55, 0.0),
                "goal": (0.0, 0.55),
                "heading": np.pi,
                "speed": 0.0,
            }
        )
        # Place just outside the ring so the next step is a single impact.
        inner = env.config.inner_hazard_radius + env.config.collision_radius
        env.position = np.array([inner + 0.02, 0.0], dtype=np.float32)
        env.heading = np.pi
        env.speed = speed
        env.external_velocity[:] = 0.0
        health_before = env.health
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        radius = float(np.linalg.norm(env.position))
        loss = health_before - env.health
        impulse = float(info["step_impulse"])
        env.close()
        return loss, impulse, radius

    def test_center_and_outer_regions_are_hazards(self):
        env = CarRaceEnv(observation_mode="state")
        env.reset(seed=0)
        env.position = np.zeros(2, dtype=np.float32)
        self.assertTrue(env.in_hazard)
        env.position = np.array([0.55, 0.0], dtype=np.float32)
        self.assertFalse(env.in_hazard)
        env.position = np.array([0.91, 0.0], dtype=np.float32)
        self.assertTrue(env.in_hazard)
        env.close()

    def test_hazard_collision_blocks_penetration(self):
        _, impulse, radius = self._ram_inner_hazard(0.8)
        env = CarRaceEnv(observation_mode="state")
        inner_limit = (
            env.config.inner_hazard_radius + env.config.collision_radius
        )
        env.close()
        self.assertGreaterEqual(radius, inner_limit - 1e-5)
        self.assertGreater(impulse, 0.0)

    def test_impact_speed_increases_impulse_and_health_loss(self):
        slow_loss, slow_impulse, _ = self._ram_inner_hazard(0.25)
        fast_loss, fast_impulse, _ = self._ram_inner_hazard(0.8)
        self.assertGreater(fast_impulse, slow_impulse)
        self.assertGreater(fast_loss, slow_loss)

    def test_resting_against_hazard_does_not_drain_health(self):
        env = CarRaceEnv(observation_mode="state")
        env.reset(
            options={
                "position": (0.55, 0.0),
                "goal": (0.0, 0.55),
                "heading": 0.0,
                "speed": 0.0,
            }
        )
        # Place gently on the boundary with no inward velocity.
        inner = env.config.inner_hazard_radius + env.config.collision_radius
        env.position = np.array([inner, 0.0], dtype=np.float32)
        env.speed = 0.0
        env.external_velocity[:] = 0.0
        health_before = env.health
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertEqual(info["step_impulse"], 0.0)
        self.assertAlmostEqual(env.health, health_before)
        env.close()

    def test_health_does_not_recover_in_safe_region(self):
        env = CarRaceEnv(observation_mode="state")
        env.reset(seed=0, options={"health": 0.5})
        for _ in range(5):
            env.step(np.zeros(2, dtype=np.float32))
        self.assertAlmostEqual(env.health, 0.5)
        env.close()

    def test_health_zero_terminates_episode(self):
        env = CarRaceEnv(
            CarRaceConfig(
                max_episode_steps=200,
                impact_impulse_scale=1.2,
            ),
            observation_mode="state",
        )
        env.reset(
            options={
                "position": (0.55, 0.0),
                "goal": (0.0, 0.55),
                "heading": np.pi,
            }
        )
        reward = 0.0
        info: dict = {}
        for _ in range(120):
            # Keep ramming the inner ring until health is gone.
            env.heading = np.pi
            env.speed = 0.8
            _, reward, terminated, _, info = env.step(
                np.array([0.0, 1.0], dtype=np.float32)
            )
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertTrue(info["dead"])
        self.assertEqual(info["termination_reason"], "health_depleted")
        self.assertEqual(env.health, 0.0)
        self.assertEqual(reward, env.config.death_penalty)
        env.close()


class CarRaceGravityTest(unittest.TestCase):
    def _zero_throttle_x(self, strength: float) -> float:
        env = CarRaceEnv(
            CarRaceConfig(gravity_strength=strength), observation_mode="state"
        )
        env.reset(
            options={
                "position": (0.575, 0.0),
                "goal": (0.0, 0.575),
                "heading": np.pi / 2.0,
            }
        )
        for _ in range(10):
            env.step(np.zeros(2, dtype=np.float32))
        x = float(env.position[0])
        env.close()
        return x

    def test_gravity_moves_stationary_car_inward(self):
        self.assertLess(
            self._zero_throttle_x(GRAVITY_STRENGTHS["car_race_grav"]),
            0.575,
        )

    def test_anti_gravity_moves_stationary_car_outward(self):
        self.assertGreater(
            self._zero_throttle_x(
                GRAVITY_STRENGTHS["car_race_anti_grav"]
            ),
            0.575,
        )

    def test_plain_has_no_world_drift(self):
        self.assertAlmostEqual(
            self._zero_throttle_x(GRAVITY_STRENGTHS["car_race_plain"]),
            0.575,
            places=6,
        )


class CarRaceIceTest(unittest.TestCase):
    def _coast_state(self, rolling_drag: float, *, steps: int = 8) -> tuple[float, float]:
        env = CarRaceEnv(
            CarRaceConfig(rolling_drag=rolling_drag),
            observation_mode="state",
        )
        env.reset(
            options={
                "position": (0.575, 0.0),
                "goal": (0.0, 0.575),
                "heading": 0.0,
            }
        )
        start = np.asarray(env.position, dtype=np.float64).copy()
        env.speed = 0.60
        for _ in range(steps):
            env.step(np.zeros(2, dtype=np.float32))
        speed = float(env.speed)
        dist = float(np.linalg.norm(np.asarray(env.position) - start))
        env.close()
        return speed, dist

    def test_ice_mode_uses_low_rolling_drag(self):
        kwargs = mode_config_kwargs("car_race_ice")
        self.assertEqual(kwargs["gravity_strength"], 0.0)
        self.assertLess(
            kwargs["rolling_drag"], ROLLING_DRAGS["car_race_plain"]
        )
        self.assertEqual(kwargs["rolling_drag"], ROLLING_DRAGS["car_race_ice"])
        self.assertEqual(kwargs["cornering_grip"], CORNERING_GRIPS["car_race_ice"])
        self.assertLess(
            kwargs["cornering_grip"], CORNERING_GRIPS["car_race_plain"]
        )
        self.assertEqual(kwargs["external_drag"], EXTERNAL_DRAGS["car_race_ice"])
        self.assertEqual(
            kwargs["longitudinal_grip"], LONGITUDINAL_GRIPS["car_race_ice"]
        )
        self.assertLess(
            kwargs["longitudinal_grip"], LONGITUDINAL_GRIPS["car_race_plain"]
        )
        self.assertEqual(
            kwargs["steering_response"], STEERING_RESPONSES["car_race_ice"]
        )
        self.assertEqual(
            kwargs["max_external_speed"], MAX_EXTERNAL_SPEEDS["car_race_ice"]
        )

    def test_ice_coasts_farther_than_plain(self):
        # Short free-roll window before either car hits the outer hazard.
        plain_speed, plain_dist = self._coast_state(ROLLING_DRAGS["car_race_plain"])
        ice_speed, ice_dist = self._coast_state(ROLLING_DRAGS["car_race_ice"])
        self.assertGreater(plain_speed, 0.0)
        self.assertGreater(ice_speed, plain_speed)
        self.assertGreater(ice_dist, plain_dist)

    def _turning_state(self, mode: str) -> tuple[float, float]:
        env = CarRaceEnv(
            CarRaceConfig(max_episode_steps=30, **mode_config_kwargs(mode)),
            observation_mode="state",
        )
        env.reset(
            options={
                "position": (0.575, 0.0),
                "goal": (-0.575, 0.0),
                "heading": np.pi / 2.0,
                "speed": 0.60,
            }
        )
        for _ in range(5):
            env.step(np.array([1.0, 0.0], dtype=np.float32))
        forward = np.array([np.cos(env.heading), np.sin(env.heading)])
        world_velocity = env.speed * forward + env.external_velocity
        cross = forward[0] * world_velocity[1] - forward[1] * world_velocity[0]
        slip_angle = abs(float(np.arctan2(cross, np.dot(forward, world_velocity))))
        external_speed = float(np.linalg.norm(env.external_velocity))
        env.close()
        return slip_angle, external_speed

    def test_ice_retains_lateral_momentum_during_turn(self):
        plain_angle, plain_external = self._turning_state("car_race_plain")
        ice_angle, ice_external = self._turning_state("car_race_ice")
        self.assertLess(np.degrees(plain_angle), 1.0)
        self.assertGreater(np.degrees(ice_angle), 20.0)
        self.assertGreater(ice_external, plain_external + 0.20)

    def test_ice_has_less_braking_grip(self):
        speeds: dict[str, float] = {}
        for mode in ("car_race_plain", "car_race_ice"):
            env = CarRaceEnv(
                CarRaceConfig(max_episode_steps=30, **mode_config_kwargs(mode)),
                observation_mode="state",
            )
            env.reset(
                options={
                    "position": (0.575, 0.0),
                    "goal": (-0.575, 0.0),
                    "heading": 0.0,
                    "speed": 0.60,
                }
            )
            for _ in range(5):
                env.step(np.array([0.0, -1.0], dtype=np.float32))
            speeds[mode] = float(env.speed)
            env.close()
        self.assertGreater(
            speeds["car_race_ice"], speeds["car_race_plain"] + 0.10
        )


class CarRaceTaskTest(unittest.TestCase):
    def test_navigation_goal_terminates_successfully(self):
        env = CarRaceEnv(observation_mode="goal_dict")
        env.reset(
            options={"position": (0.55, 0.0), "goal": (0.0, 0.55)}
        )
        env.position = env.goal.copy()
        _, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertGreater(reward, 0.0)
        env.close()

    def test_lap_2p_has_three_ring_points_including_finish(self):
        # 2p: two waypoints besides spawn; including return-to-start => 3 ring points.
        env = CarRaceEnv(
            CarRaceConfig(task_mode="lap", checkpoint_count=3),
            observation_mode="state_goal",
        )
        self.assertEqual(len(env._checkpoints), 3)
        env.reset(
            seed=0, options={"start_checkpoint": 0, "direction": 1}
        )
        start = env.goal.copy()
        for completed in range(1, 4):
            env.position = env.current_waypoint.copy()
            _, _, terminated, _, info = env.step(
                np.zeros(2, dtype=np.float32)
            )
            self.assertEqual(info["checkpoints_completed"], completed)
            self.assertEqual(float(env.state[11]), 1.0)
            if completed < 3:
                self.assertFalse(terminated)
                np.testing.assert_allclose(
                    info["goal"][:2], env.current_waypoint, atol=1e-6
                )
                self.assertAlmostEqual(
                    float(info["goal"][2]), (completed + 1) / 3
                )
            else:
                self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        np.testing.assert_allclose(info["final_goal"][:2], start[:2], atol=1e-5)
        self.assertEqual(info["termination_reason"], "lap_complete")
        env.close()

    def test_lap_requires_ordered_checkpoints(self):
        env = CarRaceEnv(
            CarRaceConfig(task_mode="lap", checkpoint_count=9),
            observation_mode="state_goal",
        )
        env.reset(
            seed=0, options={"start_checkpoint": 0, "direction": 1}
        )
        for completed in range(1, env.config.checkpoint_count + 1):
            env.position = env.current_waypoint.copy()
            _, _, terminated, _, info = env.step(
                np.zeros(2, dtype=np.float32)
            )
            self.assertEqual(info["checkpoints_completed"], completed)
            if completed < env.config.checkpoint_count:
                self.assertFalse(terminated)
            else:
                self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["termination_reason"], "lap_complete")
        self.assertEqual(info["lap_progress"], 1.0)
        env.close()

    def test_lap_direction_is_seeded_and_binary(self):
        env = CarRaceEnv(CarRaceConfig(task_mode="lap"))
        _, info_a = env.reset(seed=9)
        _, info_b = env.reset(seed=9)
        self.assertEqual(info_a["lap_direction"], info_b["lap_direction"])
        self.assertIn(info_a["lap_direction"], (-1, 1))
        env.close()

    def test_dense_lap_checkpoint_does_not_use_next_target_distance(self):
        env = CarRaceEnv(
            CarRaceConfig(task_mode="lap", reward_mode="dense")
        )
        env.reset(options={"start_checkpoint": 0, "direction": 1})
        env.position = env.current_waypoint.copy()
        _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(info["checkpoint_crossed"])
        self.assertGreater(reward, 0.0)
        env.close()


class CarRaceRenderTest(unittest.TestCase):
    def test_rgb_render_shape_and_health_bar(self):
        env = CarRaceEnv(render_mode="rgb_array", render_size=128)
        env.reset(seed=0)
        frame_full = env.render()
        env.health = 0.25
        env._display_health = 0.25
        frame_low = env.render()
        self.assertEqual(frame_full.shape, (128, 128, 3))
        self.assertEqual(frame_full.dtype, np.uint8)
        self.assertFalse(np.array_equal(frame_full, frame_low))
        env.close()

    def test_health_bar_eases_toward_true_health(self):
        env = CarRaceEnv(render_mode="rgb_array", render_size=128)
        env.reset(seed=0)
        env.health = 0.2
        before = env._display_health
        env._update_display_health(env.config.dt)
        self.assertLess(env._display_health, before)
        self.assertGreater(env._display_health, env.health)
        env.close()


if __name__ == "__main__":
    unittest.main()
