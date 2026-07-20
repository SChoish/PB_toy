"""Fuel-limited 2-D orbital navigation with a planet or black hole.

State
-----
    [x, y, vx, vy, fuel_fraction]

Goal
----
    [goal_x, goal_y, goal_vx, goal_vy]

Actions
-------
    action[0]: inertial thrust angle in radians, in [-pi, pi]
    action[1]: throttle in [0, 1]
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from .config import (
        ObservationMode,
        OrbitalSwingByConfig,
        TaskMode,
        black_hole_config,
        planet_config,
    )
    from .render import OrbitalSwingByRenderer
except ImportError:  # script-style: `cd swingby && python ...`
    from config import (
        ObservationMode,
        OrbitalSwingByConfig,
        TaskMode,
        black_hole_config,
        planet_config,
    )
    from render import OrbitalSwingByRenderer

Array = np.ndarray

# Train uses inner rotation bands; evaluation uses the canonical task once and
# held-out outer bands. Central-force dynamics are rotation-equivariant, so this
# changes geometry without changing the intended burn/capture skill.
SWINGBY_TRAIN_ROTATION_RANGES: dict[int, tuple[tuple[float, float], ...]] = {
    1: ((-0.18, -0.04), (0.04, 0.12)),
    2: ((-0.18, -0.04),),
    3: ((0.04, 0.18),),
    4: ((-0.18, -0.04),),
    5: ((0.04, 0.18),),
}
SWINGBY_EVAL_ROTATION_RANGES: dict[int, tuple[tuple[float, float], ...]] = {
    1: ((-0.30, -0.21), (0.15, 0.18)),
    2: ((-0.30, -0.21),),
    3: ((0.21, 0.30),),
    4: ((-0.30, -0.21),),
    5: ((0.21, 0.30),),
}

TaskProfile = Literal["eval_fixed", "dataset"]

# Keep the already-generated canonical swingby datasets reproducible. Evaluation uses
# a separate task table with changed initial states and goals only.
DATASET_TASKS: tuple[dict[str, Any], ...] = (
    {
        "task_name": "task1_coast_alignment",
        "difficulty": "easy",
        "init_xy": (-0.82, 0.58),
        "init_velocity": (0.39, -0.01),
        "goal_xy": (-0.18, 0.54),
        "goal_velocity": (0.39, -0.03),
    },
    {
        "task_name": "task2_upper_flyby",
        "difficulty": "medium",
        "init_xy": (-0.88, 0.46),
        "init_velocity": (0.49, -0.04),
        "goal_xy": (0.80, -0.28),
        "goal_velocity": (0.42, -0.20),
    },
    {
        "task_name": "task3_lower_flyby",
        "difficulty": "medium",
        "init_xy": (-0.88, -0.46),
        "init_velocity": (0.49, 0.04),
        "goal_xy": (0.80, 0.28),
        "goal_velocity": (0.42, 0.20),
    },
    {
        "task_name": "task4_deep_swingby",
        "difficulty": "hard",
        "init_xy": (-0.91, 0.54),
        "init_velocity": (0.52, -0.13),
        "goal_xy": (0.83, -0.58),
        "goal_velocity": (0.34, -0.27),
    },
    {
        "task_name": "task5_fuel_limited_capture",
        "difficulty": "hardest",
        "init_xy": (-0.90, -0.45),
        "init_velocity": (0.56, -0.04),
        "goal_xy": (0.72, 0.58),
        "goal_velocity": (0.18, 0.18),
    },
)

HARD_TASKS: tuple[dict[str, Any], ...] = (
    {
        "task_name": "task1_offset_intercept",
        "difficulty": "medium",
        "init_xy": (-0.754, 0.502),
        "init_velocity": (0.256, -0.070),
        "goal_xy": (-0.481, 0.495),
        "goal_velocity": (0.570, -0.089),
    },
    {
        "task_name": "task2_upper_transfer",
        "difficulty": "medium-hard",
        "init_xy": (-0.901, 0.570),
        "init_velocity": (0.444, 0.054),
        "goal_xy": (0.814, -0.159),
        "goal_velocity": (0.395, -0.231),
    },
    {
        "task_name": "task3_lower_transfer",
        "difficulty": "hard",
        "init_xy": (-0.874, -0.504),
        "init_velocity": (0.472, 0.052),
        "goal_xy": (0.859, 0.340),
        "goal_velocity": (0.367, 0.243),
    },
    {
        "task_name": "task4_deep_velocity_match",
        "difficulty": "hard",
        "init_xy": (-0.838, 0.501),
        "init_velocity": (0.517, -0.077),
        "goal_xy": (0.793, -0.525),
        "goal_velocity": (0.184, -0.174),
    },
    {
        "task_name": "task5_far_capture",
        "difficulty": "hardest",
        "init_xy": (-0.861, -0.465),
        "init_velocity": (0.539, -0.043),
        "goal_xy": (0.780, 0.679),
        "goal_velocity": (0.202, 0.147),
    },

)

# Final benchmark: mix the solvable hard tasks with one dataset goal task,
# plus a validated 77.5% hard interpolation for T5. On the reference 5k HIQL
# checkpoint this yields 0.504 mean success while the expert remains 250/250.
FIXED_EVAL_TASKS: tuple[dict[str, Any], ...] = (
    HARD_TASKS[0],
    DATASET_TASKS[1],
    HARD_TASKS[2],
    HARD_TASKS[3],
    {
        "task_name": "task5_mixed_far_capture",
        "difficulty": "hardest",
        "init_xy": (-0.869775, -0.461625),
        "init_velocity": (0.543725, -0.042325),
        "goal_xy": (0.766500, 0.656725),
        "goal_velocity": (0.197050, 0.154425),
    },
)


def rotate_task_vector(vector: Array, angle: float) -> Array:
    value = np.asarray(vector, dtype=np.float32)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.array(
        [cosine * value[0] - sine * value[1],
         sine * value[0] + cosine * value[1]],
        dtype=np.float32,
    )


def sample_swingby_train_rotation(
    rng: np.random.Generator, task_id: int
) -> float:
    ranges = SWINGBY_TRAIN_ROTATION_RANGES[int(task_id)]
    low, high = ranges[int(rng.integers(0, len(ranges)))]
    return float(rng.uniform(low, high))


def swingby_eval_rotation(
    task_id: int, variant_index: int, num_variants: int
) -> float:
    if variant_index <= 0 or num_variants <= 1:
        return 0.0
    ranges = SWINGBY_EVAL_ROTATION_RANGES[int(task_id)]
    offset = int(variant_index) - 1
    range_index = offset % len(ranges)
    rank = offset // len(ranges)
    count = (max(0, int(num_variants) - 2 - range_index) // len(ranges)) + 1
    fraction = (rank + 0.5) / max(count, 1)
    low, high = ranges[range_index]
    return float(low + fraction * (high - low))


class OrbitalSwingByEnv(OrbitalSwingByRenderer, gym.Env):
    """Fuel-limited orbital navigation around a planet or black hole.

    The central body is fixed in the inertial frame. Gravity can bend the
    satellite trajectory and reduce required thrust, but a fixed body cannot
    provide a true inertial-frame energy gain as a moving planet would.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        config: OrbitalSwingByConfig | None = None,
        observation_mode: ObservationMode = "state_goal",
        render_mode: str | None = None,
        render_size: int = 640,
        terminate_at_goal: bool = True,
        task_profile: TaskProfile = "eval_fixed",
    ) -> None:
        super().__init__()

        raw = config or OrbitalSwingByConfig()
        raw.validate()
        self.config = raw.normalized()

        if observation_mode not in ("state", "state_goal", "goal_dict"):
            raise ValueError(f"Unknown observation_mode: {observation_mode}")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if render_size < 128:
            raise ValueError("render_size must be at least 128")

        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.render_size = int(render_size)
        self.terminate_at_goal = bool(terminate_at_goal)
        if task_profile not in ("eval_fixed", "dataset"):
            raise ValueError(f"Unknown task_profile: {task_profile!r}")
        self.task_profile: TaskProfile = task_profile
        self._dtype = np.float32

        cfg = self.config
        self._body_center = np.asarray(cfg.body_center, dtype=self._dtype)

        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0], dtype=self._dtype),
            high=np.array([np.pi, 1.0], dtype=self._dtype),
            dtype=self._dtype,
        )

        state_low = np.array(
            [cfg.arena_low, cfg.arena_low, -cfg.max_speed, -cfg.max_speed, 0.0],
            dtype=self._dtype,
        )
        state_high = np.array(
            [cfg.arena_high, cfg.arena_high, cfg.max_speed, cfg.max_speed, 1.0],
            dtype=self._dtype,
        )
        goal_low = np.array(
            [cfg.arena_low, cfg.arena_low, -cfg.max_speed, -cfg.max_speed],
            dtype=self._dtype,
        )
        goal_high = np.array(
            [cfg.arena_high, cfg.arena_high, cfg.max_speed, cfg.max_speed],
            dtype=self._dtype,
        )

        if observation_mode == "state":
            self.observation_space = spaces.Box(
                low=state_low, high=state_high, dtype=self._dtype
            )
        elif observation_mode == "state_goal":
            self.observation_space = spaces.Box(
                low=np.concatenate([state_low, goal_low]),
                high=np.concatenate([state_high, goal_high]),
                dtype=self._dtype,
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=state_low, high=state_high, dtype=self._dtype
                    ),
                    "achieved_goal": spaces.Box(
                        low=goal_low, high=goal_high, dtype=self._dtype
                    ),
                    "desired_goal": spaces.Box(
                        low=goal_low, high=goal_high, dtype=self._dtype
                    ),
                }
            )

        self.position = np.zeros(2, dtype=self._dtype)
        self.velocity = np.zeros(2, dtype=self._dtype)
        self.goal = np.zeros(2, dtype=self._dtype)
        self.goal_velocity = np.zeros(2, dtype=self._dtype)
        self.fuel = float(cfg.initial_fuel)
        self.elapsed_steps = 0
        self.dead = False
        self.escaped = False
        self.success = False
        self.total_fuel_used = 0.0
        self.total_delta_v = 0.0
        self.closest_approach = np.inf
        self.cur_task_id: int | None = None
        self.cur_task_info: dict[str, Any] | None = None
        self.cur_task_rotation = 0.0

        self._last_action_angle = 0.0
        self._last_actual_throttle = 0.0
        self._trail: list[Array] = []

        self._init_renderer()
        self.set_tasks()
        self.num_tasks = len(self.task_infos)

    # ------------------------------------------------------------------
    # Public state properties
    # ------------------------------------------------------------------

    @property
    def fuel_fraction(self) -> float:
        return float(self.fuel / self.config.fuel_capacity)

    @property
    def mass(self) -> float:
        return self._current_mass()

    @property
    def state(self) -> Array:
        return np.array(
            [
                self.position[0],
                self.position[1],
                self.velocity[0],
                self.velocity[1],
                self.fuel_fraction,
            ],
            dtype=self._dtype,
        )

    @property
    def achieved_goal(self) -> Array:
        return np.concatenate([self.position, self.velocity]).astype(
            self._dtype, copy=True
        )

    @property
    def desired_goal(self) -> Array:
        return np.concatenate([self.goal, self.goal_velocity]).astype(
            self._dtype, copy=True
        )

    @property
    def body_center(self) -> Array:
        return self._body_center.copy()

    @property
    def distance_to_body(self) -> float:
        return float(np.linalg.norm(self.position - self._body_center))

    @property
    def signed_distance_to_body(self) -> float:
        return float(
            self.distance_to_body
            - self.config.body_radius
            - self.config.satellite_radius
        )

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.position - self.goal))

    @property
    def velocity_error(self) -> float:
        return float(np.linalg.norm(self.velocity - self.goal_velocity))

    @property
    def specific_orbital_energy(self) -> float:
        speed_sq = float(np.dot(self.velocity, self.velocity))
        return 0.5 * speed_sq + self._gravity_potential(self.position)

    @property
    def specific_angular_momentum(self) -> float:
        relative = self.position - self._body_center
        return float(relative[0] * self.velocity[1] - relative[1] * self.velocity[0])

    @property
    def delta_v_remaining(self) -> float:
        """Rocket-equation style remaining Δv estimate under current mass."""
        fuel_mass = self.config.fuel_mass_scale * self.fuel
        if fuel_mass <= 0.0:
            return 0.0
        exhaust_speed = self.config.max_thrust_force / max(
            self.config.fuel_burn_rate * self.config.fuel_mass_scale, 1e-8
        )
        return float(exhaust_speed * np.log1p(fuel_mass / max(self.config.dry_mass, 1e-8)))

    def gravity_acceleration(self, position: Array | None = None) -> Array:
        """Public wrapper around the central-body gravity field."""
        return self._gravity_acceleration(
            self.position if position is None else position
        )

    def predict_ballistic_trajectory(self, horizon_steps: int = 48) -> Array:
        """Integrate a thrust-free coast for rendering / diagnostics."""
        positions, _, _ = self.simulate_ballistic(
            self.position,
            self.velocity,
            horizon_steps=horizon_steps,
        )
        return positions

    def simulate_ballistic(
        self,
        position: Array,
        velocity: Array,
        *,
        horizon_steps: int = 200,
    ) -> tuple[Array, Array, dict[str, Any]]:
        """Thrust-free rollout used by task sampling and experts.

        Returns positions ``[T+1, 2]``, velocities ``[T+1, 2]``, and a small
        diagnostics dict (collision / escape / periapsis index).
        """
        position = np.asarray(position, dtype=np.float64).copy()
        velocity = np.asarray(velocity, dtype=np.float64).copy()
        positions = [position.copy()]
        velocities = [velocity.copy()]
        sub_dt = self.config.dt / self.config.physics_substeps
        lethal = self.config.body_radius + self.config.satellite_radius
        collided = False
        escaped = False
        closest_index = 0
        closest_distance = float(np.linalg.norm(position - self._body_center))

        for macro in range(max(1, int(horizon_steps))):
            for _ in range(self.config.physics_substeps):
                accel_start = self._gravity_acceleration(position).astype(np.float64)
                velocity_half = velocity + 0.5 * accel_start * sub_dt
                candidate = position + velocity_half * sub_dt
                if self._segment_hits_circle(
                    position, candidate, self._body_center, lethal
                ):
                    collided = True
                    break
                if self._segment_leaves_square(position, candidate):
                    escaped = True
                    break
                position = candidate
                accel_end = self._gravity_acceleration(position).astype(np.float64)
                velocity = velocity_half + 0.5 * accel_end * sub_dt
                if not np.all(np.isfinite(velocity)):
                    escaped = True
                    break
            if collided or escaped:
                break
            positions.append(position.copy())
            velocities.append(velocity.copy())
            distance = float(np.linalg.norm(position - self._body_center))
            if distance < closest_distance:
                closest_distance = distance
                closest_index = len(positions) - 1

        info = {
            "collided": collided,
            "escaped": escaped,
            "periapsis_index": int(closest_index),
            "closest_approach": float(closest_distance),
        }
        return (
            np.asarray(positions, dtype=self._dtype),
            np.asarray(velocities, dtype=self._dtype),
            info,
        )

    def sample_ballistic_goal(
        self,
        *,
        min_delay_steps: int = 25,
        max_delay_steps: int = 160,
    ) -> tuple[Array, Array] | None:
        """Pick a future coasting state as a reachable outgoing goal."""
        positions, velocities, info = self.simulate_ballistic(
            self.position,
            self.velocity,
            horizon_steps=max_delay_steps + 5,
        )
        if info["collided"] or positions.shape[0] < min_delay_steps + 2:
            return None
        peri = int(info["periapsis_index"])
        low = max(min_delay_steps, peri + 1)
        high = min(positions.shape[0] - 1, max_delay_steps)
        if high < low:
            return None
        index = int(self.np_random.integers(low, high + 1))
        goal = positions[index]
        goal_velocity = velocities[index]
        if not self._is_safe_spawn(goal):
            return None
        if np.linalg.norm(goal - self.position) <= self.config.goal_radius:
            return None
        return goal.astype(self._dtype), goal_velocity.astype(self._dtype)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})

        task_id = options.get("task_id")
        explicit_position = options.get("position")
        explicit_goal = options.get("goal")

        if task_id is not None:
            if explicit_position is not None or explicit_goal is not None:
                raise ValueError("task_id cannot be combined with position/goal")
            task_id = int(task_id)
            if not 1 <= task_id <= self.num_tasks:
                raise ValueError(f"task_id must be in [1, {self.num_tasks}]")
            self.cur_task_id = task_id
            self.cur_task_info = dict(self.task_infos[task_id - 1])
            self.cur_task_rotation = float(options.get("task_rotation", 0.0))
            if not np.isfinite(self.cur_task_rotation):
                raise ValueError("task_rotation must be finite")
            position = self._validate_safe_point(
                rotate_task_vector(
                    self.cur_task_info["init_xy"], self.cur_task_rotation
                ),
                "task init_xy",
            )
            velocity = self._validate_velocity(
                rotate_task_vector(
                    self.cur_task_info["init_velocity"], self.cur_task_rotation
                ),
                "task init_velocity",
            )
            goal = self._validate_safe_point(
                rotate_task_vector(
                    self.cur_task_info["goal_xy"], self.cur_task_rotation
                ),
                "task goal_xy",
            )
            goal_velocity = self._validate_velocity(
                rotate_task_vector(
                    self.cur_task_info["goal_velocity"], self.cur_task_rotation
                ),
                "task goal_velocity",
            )
        else:
            self.cur_task_id = None
            self.cur_task_info = None
            self.cur_task_rotation = 0.0
            if "task_rotation" in options:
                raise ValueError("task_rotation requires task_id")
            task_mode = options.get("task_mode", self.config.task_mode)
            if task_mode not in ("random", "swingby", "orbit_transfer"):
                raise ValueError(f"Unknown task_mode: {task_mode}")

            if explicit_position is None and explicit_goal is None:
                position, velocity, goal, goal_velocity = self._sample_task(task_mode)
            else:
                if explicit_position is None:
                    goal = self._validate_safe_point(explicit_goal, "goal")
                    position = self._sample_safe_point(
                        min_distance_from=goal,
                        minimum_distance=self.config.min_start_goal_distance,
                    )
                else:
                    position = self._validate_safe_point(explicit_position, "position")

                if explicit_goal is None:
                    goal = self._sample_safe_point(
                        min_distance_from=position,
                        minimum_distance=self.config.min_start_goal_distance,
                    )
                else:
                    goal = self._validate_safe_point(explicit_goal, "goal")

                velocity = self._validate_velocity(
                    options.get("velocity", np.zeros(2)), "velocity"
                )
                goal_velocity = self._validate_velocity(
                    options.get("goal_velocity", np.zeros(2)), "goal_velocity"
                )

        fuel = float(options.get("fuel", self.config.initial_fuel))
        if not np.isfinite(fuel) or not 0.0 <= fuel <= self.config.fuel_capacity:
            raise ValueError("fuel must be finite and in [0, fuel_capacity]")

        self.position = position.astype(self._dtype, copy=True)
        self.velocity = velocity.astype(self._dtype, copy=True)
        self.goal = goal.astype(self._dtype, copy=True)
        self.goal_velocity = goal_velocity.astype(self._dtype, copy=True)
        self.fuel = fuel
        self.elapsed_steps = 0
        self.dead = False
        self.escaped = False
        self.success = False
        self.total_fuel_used = 0.0
        self.total_delta_v = 0.0
        self.closest_approach = self.distance_to_body
        self._last_action_angle = float(np.arctan2(self.velocity[1], self.velocity[0]))
        self._last_actual_throttle = 0.0
        self._trail = [self.position.copy()]

        observation = self._get_observation()
        info = self._get_info(termination_reason=None, fuel_used_step=0.0)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self,
        action: Array,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        episode_over = (
            self.dead
            or self.escaped
            or (self.success and self.terminate_at_goal)
            or self.elapsed_steps >= self.config.max_episode_steps
        )
        if episode_over:
            raise RuntimeError("step() called after episode end; call reset() first")

        action = np.asarray(action, dtype=self._dtype)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector with shape (2,)")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        angle = float(action[0])
        requested_throttle = float(action[1])
        self._last_action_angle = angle

        previous_distance = self.distance_to_goal
        previous_velocity_error = self.velocity_error
        fuel_before = self.fuel
        delta_v_step = 0.0
        body_hit = False
        escaped = False
        goal_crossed = False

        sub_dt = self.config.dt / self.config.physics_substeps

        for _ in range(self.config.physics_substeps):
            actual_throttle, fuel_used = self._consume_fuel(
                requested_throttle,
                sub_dt,
            )
            self._last_actual_throttle = actual_throttle
            self.total_fuel_used += fuel_used

            direction = np.array([np.cos(angle), np.sin(angle)], dtype=self._dtype)
            thrust_accel = (
                self.config.max_thrust_force
                * actual_throttle
                / self.mass
                * direction
            ).astype(self._dtype)

            accel_start = self._gravity_acceleration(self.position) + thrust_accel
            velocity_half = self.velocity + 0.5 * accel_start * sub_dt
            old_position = self.position.copy()
            candidate = old_position + velocity_half * sub_dt

            lethal_radius = self.config.body_radius + self.config.satellite_radius
            if self._segment_hits_circle(
                old_position,
                candidate,
                self._body_center,
                lethal_radius,
            ):
                self.position = candidate.astype(self._dtype, copy=False)
                self.velocity = velocity_half.astype(self._dtype, copy=False)
                body_hit = True
                break

            if self._segment_leaves_square(old_position, candidate):
                self.position = candidate.astype(self._dtype, copy=False)
                self.velocity = velocity_half.astype(self._dtype, copy=False)
                escaped = True
                break

            goal_crossed = goal_crossed or self._segment_hits_circle(
                old_position,
                candidate,
                self.goal,
                self.config.goal_radius,
            )

            self.position = candidate.astype(self._dtype, copy=False)
            accel_end = self._gravity_acceleration(self.position) + thrust_accel
            new_velocity = velocity_half + 0.5 * accel_end * sub_dt
            self.velocity = new_velocity.astype(self._dtype, copy=False)

            speed = float(np.linalg.norm(self.velocity))
            if not np.isfinite(speed) or speed > self.config.max_speed:
                escaped = True
                break

            delta_v_step += float(np.linalg.norm(thrust_accel)) * sub_dt
            self.closest_approach = min(self.closest_approach, self.distance_to_body)

        self.elapsed_steps += 1
        self.dead = bool(body_hit)
        self.escaped = bool(escaped)

        position_success = goal_crossed or self.distance_to_goal <= self.config.goal_radius
        velocity_success = (
            True
            if not self.config.goal_requires_velocity_match
            else self.velocity_error <= self.config.goal_velocity_tolerance
        )
        self.success = bool(
            not self.dead
            and not self.escaped
            and position_success
            and velocity_success
        )

        self.total_delta_v += delta_v_step
        self._trail.append(self.position.copy())
        if len(self._trail) > 900:
            self._trail = self._trail[-900:]

        terminated = self.dead or self.escaped or (
            self.success and self.terminate_at_goal
        )
        truncated = (
            self.elapsed_steps >= self.config.max_episode_steps and not terminated
        )

        fuel_used_step = fuel_before - self.fuel
        reward = self._compute_step_reward(
            previous_distance=previous_distance,
            previous_velocity_error=previous_velocity_error,
            fuel_used_step=fuel_used_step,
        )

        if self.dead:
            reason = (
                "body_collision"
                if self.config.body_kind == "planet"
                else "event_horizon"
            )
        elif self.escaped:
            reason = "escaped_arena"
        elif self.success and self.terminate_at_goal:
            reason = "goal"
        elif truncated:
            reason = "time_limit"
        else:
            reason = None

        info = self._get_info(
            termination_reason=reason,
            fuel_used_step=fuel_used_step,
        )
        info["action_angle"] = angle
        info["requested_throttle"] = requested_throttle
        info["actual_throttle"] = self._last_actual_throttle
        info["delta_v_step"] = delta_v_step

        observation = self._get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def set_goal(
        self,
        goal: Array,
        goal_velocity: Array | None = None,
    ) -> None:
        validated_goal = self._validate_safe_point(goal, "goal")
        if np.linalg.norm(validated_goal - self.position) <= self.config.goal_radius:
            raise ValueError("New goal coincides with the satellite")
        if goal_velocity is None:
            validated_velocity = np.zeros(2, dtype=self._dtype)
        else:
            validated_velocity = self._validate_velocity(
                goal_velocity, "goal_velocity"
            )
        self.goal = validated_goal
        self.goal_velocity = validated_velocity
        self.success = False

    def sample_safe_point(self, **kwargs: Any) -> Array:
        return self._sample_safe_point(**kwargs)

    def compute_reward(
        self,
        achieved_goal: Array,
        desired_goal: Array,
        info: Mapping[str, Any] | np.ndarray | list | tuple | None = None,
    ) -> Array | float:
        """Vectorized sparse goal reward for HER-style relabeling."""
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        if achieved.shape[-1] != 4 or desired.shape[-1] != 4:
            raise ValueError("goals must end in dimension 4: [x, y, vx, vy]")

        position_distance = np.linalg.norm(
            achieved[..., :2] - desired[..., :2], axis=-1
        )
        velocity_distance = np.linalg.norm(
            achieved[..., 2:] - desired[..., 2:], axis=-1
        )
        if self.config.goal_requires_velocity_match:
            success = (position_distance <= self.config.goal_radius) & (
                velocity_distance <= self.config.goal_velocity_tolerance
            )
        else:
            success = position_distance <= self.config.goal_radius

        rewards = np.where(
            success,
            self.config.success_reward,
            -self.config.step_penalty,
        ).astype(np.float32)

        terminal_mask = self._terminal_mask_from_info(info, rewards.shape)
        if terminal_mask is not None:
            rewards = np.where(
                terminal_mask,
                self.config.body_collision_penalty,
                rewards,
            ).astype(np.float32)

        return float(rewards) if rewards.ndim == 0 else rewards

    def close(self) -> None:
        self.close_renderer()

    # ------------------------------------------------------------------
    # Fixed tasks and sampling
    # ------------------------------------------------------------------

    def set_tasks(self) -> None:
        """Define deterministic evaluation flybys ordered by difficulty."""
        tasks = (
            FIXED_EVAL_TASKS
            if self.task_profile == "eval_fixed"
            else DATASET_TASKS
        )
        self.task_infos = []
        for task in tasks:
            self.task_infos.append(
                {
                    **task,
                    "init_xy": np.asarray(task["init_xy"], dtype=np.float32),
                    "init_velocity": np.asarray(
                        task["init_velocity"], dtype=np.float32
                    ),
                    "goal_xy": np.asarray(task["goal_xy"], dtype=np.float32),
                    "goal_velocity": np.asarray(
                        task["goal_velocity"], dtype=np.float32
                    ),
                }
            )

    def _sample_task(
        self,
        task_mode: TaskMode,
    ) -> tuple[Array, Array, Array, Array]:
        if task_mode == "swingby":
            return self._sample_swingby_task()
        if task_mode == "orbit_transfer":
            return self._sample_orbit_transfer_task()

        position = self._sample_safe_point()
        goal = self._sample_safe_point(
            min_distance_from=position,
            minimum_distance=self.config.min_start_goal_distance,
        )
        velocity = self._sample_velocity(
            self.config.initial_speed_low,
            self.config.initial_speed_high,
        )
        goal_velocity = self._sample_velocity(
            self.config.target_speed_low,
            self.config.target_speed_high,
        )
        return position, velocity, goal, goal_velocity

    def _sample_swingby_task(self) -> tuple[Array, Array, Array, Array]:
        """Sample an incoming coast whose outgoing ballistic state is the goal."""
        for _ in range(80):
            side = float(self.np_random.choice((-1.0, 1.0)))
            start_y = side * self.np_random.uniform(0.34, 0.62)
            position = np.array(
                [self.np_random.uniform(-0.94, -0.80), start_y],
                dtype=self._dtype,
            )
            if not self._is_safe_spawn(position):
                continue

            speed = self.np_random.uniform(
                self.config.initial_speed_low,
                self.config.initial_speed_high,
            )
            velocity = np.array(
                [speed, -side * self.np_random.uniform(0.00, 0.14)],
                dtype=self._dtype,
            )

            positions, velocities, info = self.simulate_ballistic(
                position,
                velocity,
                horizon_steps=220,
            )
            if info["collided"] or positions.shape[0] < 40:
                continue

            peri = int(info["periapsis_index"])
            clearance = (
                self.config.body_radius
                + self.config.satellite_radius
                + 0.04
            )
            if info["closest_approach"] < clearance:
                continue
            if peri < 8 or peri > positions.shape[0] - 12:
                continue

            # Prefer an outgoing branch state after periapsis.
            low = peri + max(8, positions.shape[0] // 12)
            high = min(positions.shape[0] - 1, peri + max(40, positions.shape[0] // 3))
            if high < low:
                continue
            index = int(self.np_random.integers(low, high + 1))
            goal = positions[index]
            goal_velocity = velocities[index]
            if not self._is_safe_spawn(goal):
                continue
            if np.linalg.norm(goal - position) < self.config.min_start_goal_distance * 0.45:
                continue
            return (
                position.astype(self._dtype),
                velocity.astype(self._dtype),
                goal.astype(self._dtype),
                goal_velocity.astype(self._dtype),
            )

        # Fallback: keep a simple geometric flyby if sampling fails.
        side = float(self.np_random.choice((-1.0, 1.0)))
        position = np.array([-0.88, side * 0.48], dtype=self._dtype)
        velocity = np.array([0.48, -side * 0.05], dtype=self._dtype)
        goal = np.array([0.78, -side * 0.30], dtype=self._dtype)
        goal_velocity = np.array([0.40, -side * 0.18], dtype=self._dtype)
        return position, velocity, goal, goal_velocity

    def _sample_orbit_transfer_task(self) -> tuple[Array, Array, Array, Array]:
        body = self._body_center
        inner_min = self.config.body_radius + self.config.satellite_radius + 0.16
        outer_max = min(0.76, self.config.arena_high - 0.12)
        r0 = self.np_random.uniform(inner_min, min(0.48, outer_max - 0.10))
        rg = self.np_random.uniform(max(r0 + 0.16, 0.52), outer_max)
        theta0 = self.np_random.uniform(-np.pi, np.pi)
        thetag = theta0 + self.np_random.uniform(0.8, 2.4)
        direction = float(self.np_random.choice((-1.0, 1.0)))

        position = body + r0 * np.array(
            [np.cos(theta0), np.sin(theta0)], dtype=self._dtype
        )
        goal = body + rg * np.array(
            [np.cos(thetag), np.sin(thetag)], dtype=self._dtype
        )

        v0_mag = np.sqrt(max(self.config.gravitational_parameter / r0, 1e-8))
        vg_mag = np.sqrt(max(self.config.gravitational_parameter / rg, 1e-8))
        tangent0 = direction * np.array(
            [-np.sin(theta0), np.cos(theta0)], dtype=self._dtype
        )
        tangentg = direction * np.array(
            [-np.sin(thetag), np.cos(thetag)], dtype=self._dtype
        )
        velocity = 0.94 * v0_mag * tangent0
        goal_velocity = vg_mag * tangentg
        return (
            position.astype(self._dtype),
            velocity.astype(self._dtype),
            goal.astype(self._dtype),
            goal_velocity.astype(self._dtype),
        )

    def _sample_safe_point(
        self,
        *,
        min_distance_from: Array | None = None,
        minimum_distance: float = 0.0,
    ) -> Array:
        margin = self.config.satellite_radius + self.config.spawn_clearance
        low = self.config.arena_low + margin
        high = self.config.arena_high - margin
        body_clearance = (
            self.config.body_radius
            + self.config.satellite_radius
            + self.config.spawn_clearance
        )

        for _ in range(10_000):
            point = self.np_random.uniform(low, high, size=2).astype(self._dtype)
            if np.linalg.norm(point - self._body_center) <= body_clearance:
                continue
            if min_distance_from is not None and np.linalg.norm(
                point - np.asarray(min_distance_from)
            ) < minimum_distance:
                continue
            return point
        raise RuntimeError("Could not sample a safe point")

    def _sample_velocity(self, low: float, high: float) -> Array:
        speed = self.np_random.uniform(low, high)
        angle = self.np_random.uniform(-np.pi, np.pi)
        return speed * np.array([np.cos(angle), np.sin(angle)], dtype=self._dtype)

    def _validate_safe_point(self, point: Any, name: str) -> Array:
        value = np.asarray(point, dtype=self._dtype)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite vector with shape (2,)")
        if not self._is_safe_spawn(value):
            raise ValueError(f"{name} is outside the arena or too close to the body")
        return value.copy()

    def _validate_velocity(self, velocity: Any, name: str) -> Array:
        value = np.asarray(velocity, dtype=self._dtype)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite vector with shape (2,)")
        if np.linalg.norm(value) > self.config.max_speed:
            raise ValueError(f"{name} exceeds max_speed")
        return value.copy()

    def _is_safe_spawn(self, point: Array) -> bool:
        margin = self.config.satellite_radius + self.config.spawn_clearance
        if np.any(point <= self.config.arena_low + margin) or np.any(
            point >= self.config.arena_high - margin
        ):
            return False
        clearance = (
            self.config.body_radius
            + self.config.satellite_radius
            + self.config.spawn_clearance
        )
        return bool(np.linalg.norm(point - self._body_center) > clearance)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _gravity_acceleration(self, position: Array) -> Array:
        delta = self._body_center - np.asarray(position, dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12 or self.config.gravitational_parameter == 0.0:
            return np.zeros(2, dtype=self._dtype)

        direction = delta / distance
        mu = float(self.config.gravitational_parameter)
        model = self.config.canonical_gravity_model

        if model == "newtonian":
            softened_sq = distance * distance + self.config.gravity_softening**2
            magnitude = mu / softened_sq
        else:
            gap = max(
                distance - self.config.schwarzschild_radius,
                self.config.gravity_softening,
            )
            magnitude = mu / (gap * gap)

        magnitude = min(magnitude, self.config.max_gravity_acceleration)
        return (magnitude * direction).astype(self._dtype)

    def _gravity_potential(self, position: Array) -> float:
        distance = float(np.linalg.norm(np.asarray(position) - self._body_center))
        mu = float(self.config.gravitational_parameter)
        if distance <= 1e-12:
            return -np.inf
        model = self.config.canonical_gravity_model
        if model == "newtonian":
            softened = np.sqrt(distance**2 + self.config.gravity_softening**2)
            return float(-mu / softened)
        gap = max(
            distance - self.config.schwarzschild_radius,
            self.config.gravity_softening,
        )
        return float(-mu / gap)

    def _current_mass(self) -> float:
        return float(self.config.dry_mass + self.config.fuel_mass_scale * self.fuel)

    def _consume_fuel(
        self,
        requested_throttle: float,
        dt: float,
    ) -> tuple[float, float]:
        if requested_throttle <= 0.0 or self.fuel <= 0.0:
            return 0.0, 0.0

        full_throttle_fuel = self.config.fuel_burn_rate * dt
        requested_fuel = full_throttle_fuel * requested_throttle
        if requested_fuel <= self.fuel:
            actual_throttle = requested_throttle
            fuel_used = requested_fuel
        else:
            actual_throttle = self.fuel / max(full_throttle_fuel, 1e-12)
            fuel_used = self.fuel

        self.fuel = max(0.0, self.fuel - fuel_used)
        return float(actual_throttle), float(fuel_used)

    def _segment_leaves_square(self, start: Array, end: Array) -> bool:
        del start
        margin = self.config.satellite_radius
        low = self.config.arena_low + margin
        high = self.config.arena_high - margin
        return bool(np.any(end < low) or np.any(end > high))

    @staticmethod
    def _segment_hits_circle(
        start: Array,
        end: Array,
        center: Array,
        radius: float,
    ) -> bool:
        segment = end - start
        squared_length = float(np.dot(segment, segment))
        if squared_length <= 1e-16:
            closest = start
        else:
            t = float(np.dot(center - start, segment) / squared_length)
            t = float(np.clip(t, 0.0, 1.0))
            closest = start + t * segment
        return bool(np.linalg.norm(closest - center) <= radius)

    # ------------------------------------------------------------------
    # Reward, observation, and info
    # ------------------------------------------------------------------

    def _compute_step_reward(
        self,
        *,
        previous_distance: float,
        previous_velocity_error: float,
        fuel_used_step: float,
    ) -> float:
        if self.dead:
            return float(self.config.body_collision_penalty)
        if self.escaped:
            return float(self.config.escape_penalty)
        if self.success:
            return float(
                self.config.success_reward
                + self.config.remaining_fuel_bonus * self.fuel_fraction
            )

        reward = -self.config.step_penalty
        reward -= self.config.fuel_penalty_scale * fuel_used_step

        if self.config.reward_mode == "dense":
            reward += self.config.progress_scale * (
                previous_distance - self.distance_to_goal
            )
            if self.config.goal_requires_velocity_match:
                reward += self.config.velocity_progress_scale * (
                    previous_velocity_error - self.velocity_error
                )
        return float(reward)

    def _get_observation(self) -> Any:
        state = self.state
        if self.observation_mode == "state":
            return state
        if self.observation_mode == "state_goal":
            return np.concatenate([state, self.desired_goal]).astype(
                self._dtype, copy=False
            )
        return {
            "observation": state,
            "achieved_goal": self.achieved_goal,
            "desired_goal": self.desired_goal,
        }

    def _get_info(
        self,
        termination_reason: str | None,
        fuel_used_step: float,
    ) -> dict[str, Any]:
        gravity = self._gravity_acceleration(self.position)
        info = {
            "is_success": bool(self.success),
            "dead": bool(self.dead),
            "escaped": bool(self.escaped),
            "termination_reason": termination_reason,
            "elapsed_steps": int(self.elapsed_steps),
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "speed": float(np.linalg.norm(self.velocity)),
            "goal": self.goal.copy(),
            "goal_velocity": self.goal_velocity.copy(),
            "distance_to_goal": self.distance_to_goal,
            "velocity_error": self.velocity_error,
            "relative_speed_to_goal": self.velocity_error,
            "fuel": float(self.fuel),
            "fuel_fraction": self.fuel_fraction,
            "fuel_used_step": float(fuel_used_step),
            "fuel_used": float(fuel_used_step),
            "fuel_burn": float(fuel_used_step),
            "total_fuel_used": float(self.total_fuel_used),
            "total_delta_v": float(self.total_delta_v),
            "delta_v_remaining": self.delta_v_remaining,
            "body_center": self._body_center.copy(),
            "body_radius": float(self.config.body_radius),
            "distance_to_body": self.distance_to_body,
            "signed_distance_to_body": self.signed_distance_to_body,
            "closest_approach": float(self.closest_approach),
            "specific_orbital_energy": self.specific_orbital_energy,
            "orbital_energy": self.specific_orbital_energy,
            "specific_angular_momentum": self.specific_angular_momentum,
            "angular_momentum": self.specific_angular_momentum,
            "gravity_acceleration": gravity.copy(),
            "gravity_model": self.config.gravity_model,
            "body_kind": self.config.body_kind,
            "task_id": self.cur_task_id,
            "task_rotation": float(self.cur_task_rotation),
            "task_profile": self.task_profile,
            "task_name": (
                None
                if self.cur_task_info is None
                else self.cur_task_info.get("task_name")
            ),
        }
        return info

    @staticmethod
    def _terminal_mask_from_info(
        info: Mapping[str, Any] | np.ndarray | list | tuple | None,
        batch_shape: tuple[int, ...],
    ) -> Array | None:
        if info is None:
            return None
        if isinstance(info, Mapping):
            terminal = bool(info.get("dead", False) or info.get("escaped", False))
            if batch_shape == ():
                return np.asarray(terminal, dtype=bool)
            return np.full(batch_shape, terminal, dtype=bool)
        if isinstance(info, np.ndarray):
            mask = np.asarray(info, dtype=bool)
            if mask.shape != batch_shape:
                raise ValueError("info mask shape does not match reward batch")
            return mask
        if isinstance(info, (list, tuple)):
            values = [
                bool(item.get("dead", False) or item.get("escaped", False))
                if isinstance(item, Mapping)
                else bool(item)
                for item in info
            ]
            mask = np.asarray(values, dtype=bool)
            if mask.shape != batch_shape:
                try:
                    mask = mask.reshape(batch_shape)
                except ValueError as exc:
                    raise ValueError(
                        "info sequence shape does not match reward batch"
                    ) from exc
            return mask
        raise TypeError(f"Unsupported info type: {type(info)!r}")


# Documentation-facing alias.
OrbitalSwingbyEnv = OrbitalSwingByEnv


def register_environments() -> None:
    """Register Gymnasium environment IDs once."""
    from gymnasium.envs.registration import register, registry

    entries = {
        "OrbitalSwingBy2D-v0": planet_config(),
        "BlackHoleSwingBy2D-v0": black_hole_config(),
        "OrbitalSwingby-Planet-v0": planet_config(),
        "OrbitalSwingby-BlackHole-v0": black_hole_config(),
    }
    for environment_id, cfg in entries.items():
        if environment_id in registry:
            continue
        register(
            id=environment_id,
            entry_point="orbital_swingby_env:OrbitalSwingByEnv",
            kwargs={
                "config": cfg,
                "observation_mode": "state_goal",
            },
            max_episode_steps=None,
        )


register_environment = register_environments
