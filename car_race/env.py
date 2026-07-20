"""Continuous annular car environment with cumulative hazard damage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Array = np.ndarray
TaskMode = Literal["navigation", "lap"]
ObservationMode = Literal["state", "state_goal", "goal_dict"]
RewardMode = Literal["sparse", "dense"]
CarRaceMode = Literal[
    "car_race_plain",
    "car_race_grav",
    "car_race_anti_grav",
    "car_race_ice",
]

# Mode catalog. Gravity variants only change the radial field. Ice keeps
# gravity off, lowers rolling friction, and retains lateral momentum while
# the chassis turns.
GRAVITY_STRENGTHS: dict[CarRaceMode, float] = {
    "car_race_plain": 0.0,
    "car_race_grav": 0.15,
    "car_race_anti_grav": -0.15,
    "car_race_ice": 0.0,
}

# Default ``CarRaceConfig.rolling_drag`` is 0.40; ice is intentionally slippy.
ROLLING_DRAGS: dict[CarRaceMode, float] = {
    "car_race_plain": 0.40,
    "car_race_grav": 0.40,
    "car_race_anti_grav": 0.40,
    "car_race_ice": 0.08,
}

# Fraction of each steering-induced velocity rotation that the tires can
# realize immediately. One is the no-slip bicycle model; lower values leave
# more of the previous travel direction in ``external_velocity`` as drift.
CORNERING_GRIPS: dict[CarRaceMode, float] = {
    "car_race_plain": 1.0,
    "car_race_grav": 1.0,
    "car_race_anti_grav": 1.0,
    "car_race_ice": 0.15,
}

# Longitudinal tire force controls both acceleration and braking. Ice needs a
# longer distance to build or shed drive speed.
LONGITUDINAL_GRIPS: dict[CarRaceMode, float] = {
    "car_race_plain": 1.0,
    "car_race_grav": 1.0,
    "car_race_anti_grav": 1.0,
    "car_race_ice": 0.45,
}

# Chassis yaw response is reduced separately from momentum retention. This
# produces wide, deliberate turns without making the expert uncontrollable.
STEERING_RESPONSES: dict[CarRaceMode, float] = {
    "car_race_plain": 1.0,
    "car_race_grav": 1.0,
    "car_race_anti_grav": 1.0,
    "car_race_ice": 0.465,
}

# Ice retains the lateral momentum longer than asphalt. In field modes this
# coefficient continues to damp gravity-induced velocity as before.
EXTERNAL_DRAGS: dict[CarRaceMode, float] = {
    "car_race_plain": 1.2,
    "car_race_grav": 1.2,
    "car_race_anti_grav": 1.2,
    "car_race_ice": 0.30,
}

MAX_EXTERNAL_SPEEDS: dict[CarRaceMode, float] = {
    "car_race_plain": 0.30,
    "car_race_grav": 0.30,
    "car_race_anti_grav": 0.30,
    "car_race_ice": 0.45,
}

NUM_FIXED_TASKS = 5


def fixed_task_options(
    task_mode: TaskMode,
    task_id: int,
    *,
    checkpoint_count: int = 8,
    track_radius: float = 0.575,
) -> dict[str, Any]:
    """Return one of the five environment-owned evaluation tasks."""
    task_id = int(task_id)
    if not 1 <= task_id <= NUM_FIXED_TASKS:
        raise ValueError(f"task_id must be in [1, {NUM_FIXED_TASKS}]")
    if task_mode == "navigation":
        radius = float(track_radius)
        diagonal = radius * 0.7071
        specifications = (
            ((radius, 0.0), (0.0, radius)),
            ((0.0, radius), (-radius, 0.0)),
            ((-radius, 0.0), (0.0, -radius)),
            ((0.0, -radius), (radius, 0.0)),
            ((diagonal, diagonal), (-diagonal, -diagonal)),
        )
        position, goal = specifications[task_id - 1]
        return {"position": position, "goal": goal}
    if task_mode == "lap":
        specifications = (
            (0, 1),
            (0, -1),
            (checkpoint_count // 2, 1),
            (max(1, checkpoint_count // 3), -1),
            (checkpoint_count - 1, 1),
        )
        start_checkpoint, direction = specifications[task_id - 1]
        return {
            "start_checkpoint": int(start_checkpoint % checkpoint_count),
            "direction": int(direction),
        }
    raise ValueError(f"Unknown task_mode: {task_mode}")


def mode_config_kwargs(mode: CarRaceMode) -> dict[str, float]:
    """Physics overrides for a named CarRace mode."""
    if mode not in GRAVITY_STRENGTHS:
        raise ValueError(
            f"Unknown CarRace mode {mode!r}; "
            f"choose from {tuple(GRAVITY_STRENGTHS)}"
        )
    return {
        "gravity_strength": float(GRAVITY_STRENGTHS[mode]),
        "rolling_drag": float(ROLLING_DRAGS[mode]),
        "cornering_grip": float(CORNERING_GRIPS[mode]),
        "longitudinal_grip": float(LONGITUDINAL_GRIPS[mode]),
        "steering_response": float(STEERING_RESPONSES[mode]),
        "external_drag": float(EXTERNAL_DRAGS[mode]),
        "max_external_speed": float(MAX_EXTERNAL_SPEEDS[mode]),
    }


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class CarRaceConfig:
    """Geometry, dynamics, damage, and task parameters."""

    # Map geometry.
    arena_low: float = -1.0
    arena_high: float = 1.0
    inner_hazard_radius: float = 0.20
    outer_hazard_radius: float = 0.95
    car_length: float = 0.10
    car_width: float = 0.055
    collision_radius: float = 0.055
    goal_radius: float = 0.07
    spawn_clearance: float = 0.04

    # Kinematic bicycle dynamics.
    dt: float = 0.05
    physics_substeps: int = 5
    wheelbase: float = 0.075
    max_steer_angle: float = 0.60
    max_acceleration: float = 1.5
    max_speed: float = 0.80
    max_reverse_speed: float = 0.35
    rolling_drag: float = 0.40
    cornering_grip: float = 1.0
    longitudinal_grip: float = 1.0
    steering_response: float = 1.0
    wall_restitution: float = 0.0
    # Signed inverse-square world-space field around the origin.  Positive
    # attracts inward and negative repels outward.
    gravity_strength: float = 0.0
    gravity_soft_min: float = 0.12
    external_drag: float = 1.2
    max_external_speed: float = 0.30

    # Impact damage on hazard bounce: health -= impulse / damage_capacity,
    # where impulse = impact_impulse_scale * (1 + hazard_restitution) * |v_n|.
    # At drive-only max speed, defaults remove about 68% of health. With the
    # maximum aligned external velocity, the loss rises to about 94%.
    initial_health: float = 1.0
    damage_capacity: float = 1.0
    hazard_restitution: float = 0.55
    impact_impulse_scale: float = 0.55
    min_impact_speed: float = 0.03
    # Visual-only smoothing for the health bar fill/color.
    health_bar_tau: float = 0.05

    # Tasks.
    task_mode: TaskMode = "navigation"
    checkpoint_count: int = 8
    track_radius: float = 0.575
    min_start_goal_distance: float = 0.65
    max_episode_steps: int = 500

    # Reward.
    reward_mode: RewardMode = "sparse"
    step_penalty: float = 0.005
    control_cost: float = 0.001
    progress_scale: float = 1.0
    checkpoint_reward: float = 0.20
    success_reward: float = 1.0
    death_penalty: float = -2.5
    damage_penalty_scale: float = 0.05

    def validate(self) -> None:
        if not self.arena_low < self.arena_high:
            raise ValueError("arena_low must be smaller than arena_high")
        if not 0.0 < self.inner_hazard_radius < self.outer_hazard_radius:
            raise ValueError("hazard radii must satisfy 0 < inner < outer")
        if self.outer_hazard_radius > min(
            abs(self.arena_low), abs(self.arena_high)
        ):
            raise ValueError("outer_hazard_radius must fit inside the arena")
        if self.collision_radius <= 0.0:
            raise ValueError("collision_radius must be positive")
        if self.car_length <= 0.0 or self.car_width <= 0.0:
            raise ValueError("car dimensions must be positive")
        safe_low = self.inner_hazard_radius + self.collision_radius
        safe_high = self.outer_hazard_radius - self.collision_radius
        if safe_low >= safe_high:
            raise ValueError("hazards leave no safe annulus for the car")
        if not safe_low < self.track_radius < safe_high:
            raise ValueError("track_radius must lie in the safe annulus")
        if self.goal_radius <= 0.0 or self.spawn_clearance < 0.0:
            raise ValueError("goal_radius must be positive and clearance non-negative")
        if self.dt <= 0.0 or self.physics_substeps < 1:
            raise ValueError("dt must be positive and physics_substeps >= 1")
        if self.wheelbase <= 0.0 or self.max_steer_angle <= 0.0:
            raise ValueError("wheelbase and max_steer_angle must be positive")
        if self.max_acceleration <= 0.0 or self.max_speed <= 0.0:
            raise ValueError("acceleration and max_speed must be positive")
        if self.max_reverse_speed < 0.0 or self.rolling_drag < 0.0:
            raise ValueError("reverse speed and rolling_drag must be non-negative")
        if not 0.0 <= self.cornering_grip <= 1.0:
            raise ValueError("cornering_grip must be in [0, 1]")
        if not 0.0 < self.longitudinal_grip <= 1.0:
            raise ValueError("longitudinal_grip must be in (0, 1]")
        if not 0.0 < self.steering_response <= 1.0:
            raise ValueError("steering_response must be in (0, 1]")
        if not 0.0 <= self.wall_restitution <= 1.0:
            raise ValueError("wall_restitution must be in [0, 1]")
        if not 0.0 <= self.hazard_restitution <= 1.0:
            raise ValueError("hazard_restitution must be in [0, 1]")
        if self.gravity_soft_min <= 0.0:
            raise ValueError("gravity_soft_min must be positive")
        if self.external_drag < 0.0 or self.max_external_speed <= 0.0:
            raise ValueError("external_drag must be non-negative and speed positive")
        if self.initial_health <= 0.0 or self.damage_capacity <= 0.0:
            raise ValueError("health and damage_capacity must be positive")
        if self.impact_impulse_scale < 0.0 or self.min_impact_speed < 0.0:
            raise ValueError("impact impulse parameters must be non-negative")
        if self.health_bar_tau <= 0.0:
            raise ValueError("health_bar_tau must be positive")
        if self.task_mode not in ("navigation", "lap"):
            raise ValueError(f"Unknown task_mode: {self.task_mode}")
        if self.reward_mode not in ("sparse", "dense"):
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
        if self.checkpoint_count < 2:
            raise ValueError("checkpoint_count must be at least 2")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")


class CarRaceEnv(gym.Env):
    """Goal navigation and one-lap racing on an annular safe track.

    Actions are normalized ``[steering, throttle/brake]`` in ``[-1, 1]^2``.
    State observations start with
    ``[x, y, task_progress, direction, cos(heading), sin(heading), speed,
    health, external_velocity_x, external_velocity_y]``.  Lap observations
    append ``[waypoint_index, waypoint_reached, waypoint_x, waypoint_y]``.
    The first four coordinates form the achieved-goal representation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    num_tasks = NUM_FIXED_TASKS

    def __init__(
        self,
        config: CarRaceConfig | None = None,
        observation_mode: ObservationMode = "state_goal",
        render_mode: str | None = None,
        render_size: int = 512,
        terminate_on_success: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or CarRaceConfig()
        self.config.validate()
        if observation_mode not in ("state", "state_goal", "goal_dict"):
            raise ValueError(f"Unknown observation_mode: {observation_mode}")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if render_size < 64:
            raise ValueError("render_size must be at least 64")

        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.render_size = int(render_size)
        self.terminate_on_success = bool(terminate_on_success)
        self._dtype = np.float32

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=self._dtype)
        state_low = np.array(
            [
                self.config.arena_low,
                self.config.arena_low,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -self.config.max_reverse_speed,
                0.0,
                -self.config.max_external_speed,
                -self.config.max_external_speed,
            ],
            dtype=self._dtype,
        )
        state_high = np.array(
            [
                self.config.arena_high,
                self.config.arena_high,
                1.0,
                1.0,
                1.0,
                1.0,
                self.config.max_speed,
                self.config.initial_health,
                self.config.max_external_speed,
                self.config.max_external_speed,
            ],
            dtype=self._dtype,
        )
        if self.config.task_mode == "lap":
            state_low = np.concatenate(
                [
                    state_low,
                    np.array(
                        [0.0, 0.0, self.config.arena_low, self.config.arena_low],
                        dtype=self._dtype,
                    ),
                ]
            )
            state_high = np.concatenate(
                [
                    state_high,
                    np.array(
                        [1.0, 1.0, self.config.arena_high, self.config.arena_high],
                        dtype=self._dtype,
                    ),
                ]
            )
        goal_low = np.array(
            [self.config.arena_low, self.config.arena_low, 0.0, -1.0],
            dtype=self._dtype,
        )
        goal_high = np.array(
            [self.config.arena_high, self.config.arena_high, 1.0, 1.0],
            dtype=self._dtype,
        )
        if observation_mode == "state":
            self.observation_space = spaces.Box(state_low, state_high, dtype=self._dtype)
        elif observation_mode == "state_goal":
            self.observation_space = spaces.Box(
                np.concatenate([state_low, goal_low]),
                np.concatenate([state_high, goal_high]),
                dtype=self._dtype,
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        state_low, state_high, dtype=self._dtype
                    ),
                    "achieved_goal": spaces.Box(
                        goal_low, goal_high, dtype=self._dtype
                    ),
                    "desired_goal": spaces.Box(
                        goal_low, goal_high, dtype=self._dtype
                    ),
                }
            )

        self.position = np.zeros(2, dtype=self._dtype)
        self.heading = 0.0
        self.speed = 0.0
        self.external_velocity = np.zeros(2, dtype=self._dtype)
        self.health = float(self.config.initial_health)
        self._display_health = float(self.config.initial_health)
        self.goal = np.zeros(2, dtype=self._dtype)
        self.current_waypoint = np.zeros(2, dtype=self._dtype)
        self.elapsed_steps = 0
        self.total_impulse = 0.0
        self.dead = False
        self.success = False
        self._hazard_contact = False
        self._lap_direction = 0
        self._lap_start_index = 0
        self._lap_completed = 0
        self._checkpoint_index = 0
        self._waypoint_reached = False
        self.cur_task_id: int | None = None
        self._checkpoints = self._build_checkpoints()
        self._human_figure: Any = None
        self._human_axis: Any = None
        self._human_image: Any = None

    @property
    def state(self) -> Array:
        physical = np.array(
            [
                self.position[0],
                self.position[1],
                self.task_progress,
                float(self._lap_direction),
                np.cos(self.heading),
                np.sin(self.heading),
                self.speed,
                self.health,
                self.external_velocity[0],
                self.external_velocity[1],
            ],
            dtype=self._dtype,
        )
        if self.config.task_mode != "lap":
            return physical
        active_ordinal = (
            min(self._lap_completed + 1, self.config.checkpoint_count)
            % self.config.checkpoint_count
        )
        waypoint_index = float(
            active_ordinal / max(self.config.checkpoint_count - 1, 1)
        )
        task = np.array(
            [
                waypoint_index,
                float(self._waypoint_reached),
                self.current_waypoint[0],
                self.current_waypoint[1],
            ],
            dtype=self._dtype,
        )
        return np.concatenate([physical, task]).astype(self._dtype)

    @property
    def task_progress(self) -> float:
        if self.config.task_mode == "lap":
            return float(self._lap_completed / self.config.checkpoint_count)
        return 0.0

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.position - self.goal))

    @property
    def distance_to_waypoint(self) -> float:
        return float(np.linalg.norm(self.position - self.current_waypoint))

    @property
    def achieved_goal(self) -> Array:
        return self.state[:4].copy()

    @property
    def desired_goal(self) -> Array:
        if self.config.task_mode == "lap":
            target_progress = min(
                (self._lap_completed + 1) / self.config.checkpoint_count,
                1.0,
            )
            return np.array(
                [
                    self.current_waypoint[0],
                    self.current_waypoint[1],
                    target_progress,
                    float(self._lap_direction),
                ],
                dtype=self._dtype,
            )
        return np.array(
            [self.goal[0], self.goal[1], 0.0, 0.0], dtype=self._dtype
        )

    @property
    def in_hazard(self) -> bool:
        radius = float(np.linalg.norm(self.position))
        return bool(
            radius - self.config.collision_radius
            <= self.config.inner_hazard_radius
            or radius + self.config.collision_radius
            >= self.config.outer_hazard_radius
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})

        task_id = options.pop("task_id", None)
        if task_id is not None:
            task_keys = (
                ("position", "goal")
                if self.config.task_mode == "navigation"
                else ("start_checkpoint", "direction")
            )
            conflicts = [key for key in task_keys if key in options]
            if conflicts:
                raise ValueError(
                    "task_id cannot be combined with " + ", ".join(conflicts)
                )
            self.cur_task_id = int(task_id)
            options.update(
                fixed_task_options(
                    self.config.task_mode,
                    self.cur_task_id,
                    checkpoint_count=self.config.checkpoint_count,
                    track_radius=self.config.track_radius,
                )
            )
        else:
            self.cur_task_id = None

        self.elapsed_steps = 0
        self.total_impulse = 0.0
        self.dead = False
        self.success = False
        self._hazard_contact = False
        self._waypoint_reached = False
        self.health = float(options.get("health", self.config.initial_health))
        if not 0.0 < self.health <= self.config.initial_health:
            raise ValueError("health must be in (0, initial_health]")
        self._display_health = float(self.health)

        if self.config.task_mode == "navigation":
            position, goal = self._reset_navigation(options)
            waypoint = goal
            default_heading = float(
                np.arctan2(goal[1] - position[1], goal[0] - position[0])
            )
            self._lap_direction = 0
            self._lap_completed = 0
            self._checkpoint_index = 0
        else:
            position, goal, waypoint, default_heading = self._reset_lap(options)

        self.position = position.astype(self._dtype, copy=True)
        self.goal = goal.astype(self._dtype, copy=True)
        self.current_waypoint = waypoint.astype(self._dtype, copy=True)
        self.heading = _wrap_angle(float(options.get("heading", default_heading)))
        self.speed = float(options.get("speed", 0.0))
        if not -self.config.max_reverse_speed <= self.speed <= self.config.max_speed:
            raise ValueError("initial speed is outside configured limits")
        external_velocity = np.asarray(
            options.get("external_velocity", np.zeros(2)), dtype=self._dtype
        )
        if external_velocity.shape != (2,) or not np.all(
            np.isfinite(external_velocity)
        ):
            raise ValueError("external_velocity must be a finite shape-(2,) vector")
        self.external_velocity = self._clip_external_velocity(external_velocity)

        observation = self._get_observation()
        info = self._get_info(None, step_impulse=0.0, checkpoint_crossed=False)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _reset_navigation(self, options: dict[str, Any]) -> tuple[Array, Array]:
        explicit_position = options.get("position")
        explicit_goal = options.get("goal")
        if explicit_position is None and explicit_goal is None:
            for _ in range(10_000):
                position = self._sample_safe_point()
                goal = self._sample_safe_point()
                if (
                    np.linalg.norm(position - goal)
                    >= self.config.min_start_goal_distance
                ):
                    return position, goal
            raise RuntimeError("Could not sample a navigation task")
        if explicit_position is None:
            goal = self._validate_safe_point(explicit_goal, "goal")
            position = self._sample_safe_point(
                min_distance_from=goal,
                minimum_distance=self.config.min_start_goal_distance,
            )
        elif explicit_goal is None:
            position = self._validate_safe_point(explicit_position, "position")
            goal = self._sample_safe_point(
                min_distance_from=position,
                minimum_distance=self.config.min_start_goal_distance,
            )
        else:
            position = self._validate_safe_point(explicit_position, "position")
            goal = self._validate_safe_point(explicit_goal, "goal")
        if np.linalg.norm(position - goal) <= self.config.goal_radius:
            raise ValueError("Initial position is already inside the goal")
        return position, goal

    def _reset_lap(
        self, options: dict[str, Any]
    ) -> tuple[Array, Array, Array, float]:
        direction = options.get("direction")
        if direction is None:
            direction = int(self.np_random.choice((-1, 1)))
        direction = int(direction)
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        start_index = int(
            options.get(
                "start_checkpoint",
                self.np_random.integers(0, self.config.checkpoint_count),
            )
        )
        if not 0 <= start_index < self.config.checkpoint_count:
            raise ValueError("start_checkpoint is out of range")

        self._lap_direction = direction
        self._lap_start_index = start_index
        self._lap_completed = 0
        self._checkpoint_index = (
            start_index + direction
        ) % self.config.checkpoint_count
        default_position = self._checkpoints[start_index]
        position = self._validate_safe_point(
            options.get("position", default_position), "position"
        )
        final_goal = self._checkpoints[start_index]
        waypoint = self._checkpoints[self._checkpoint_index]
        radial_angle = 2.0 * np.pi * start_index / self.config.checkpoint_count
        default_heading = radial_angle + direction * np.pi / 2.0
        return position, final_goal, waypoint, default_heading

    def step(
        self, action: Array
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if (
            self.dead
            or (self.success and self.terminate_on_success)
            or self.elapsed_steps >= self.config.max_episode_steps
        ):
            raise RuntimeError("step() called after episode end; call reset() first")

        action = np.asarray(action, dtype=self._dtype)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector with shape (2,)")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        steer = float(action[0]) * self.config.max_steer_angle
        throttle = float(action[1])

        previous_distance = self.distance_to_waypoint
        step_impulse = 0.0
        wall_collision = False
        hazard_contact = False
        sub_dt = self.config.dt / self.config.physics_substeps
        external_drag_factor = float(
            np.exp(-self.config.external_drag * sub_dt)
        )

        for _ in range(self.config.physics_substeps):
            acceleration = (
                self.config.max_acceleration
                * self.config.longitudinal_grip
                * throttle
            )
            acceleration -= self.config.rolling_drag * self.speed
            self.speed = float(
                np.clip(
                    self.speed + acceleration * sub_dt,
                    -self.config.max_reverse_speed,
                    self.config.max_speed,
                )
            )
            old_forward = np.array(
                [np.cos(self.heading), np.sin(self.heading)], dtype=self._dtype
            )
            yaw_rate = (
                self.config.steering_response
                * self.speed
                * np.tan(steer)
                / self.config.wheelbase
            )
            self.heading = _wrap_angle(self.heading + yaw_rate * sub_dt)
            forward = np.array(
                [np.cos(self.heading), np.sin(self.heading)], dtype=self._dtype
            )
            # A no-slip tire rotates the full drive velocity with the chassis.
            # On ice, retain part of the old direction as world-space lateral
            # momentum. The existing observed external-velocity channel keeps
            # this state Markov without changing observation or dataset shapes.
            lost_grip = 1.0 - self.config.cornering_grip
            if lost_grip > 0.0 and self.speed != 0.0:
                self.external_velocity = (
                    self.external_velocity
                    + lost_grip * self.speed * (old_forward - forward)
                )
            self.external_velocity = (
                self.external_velocity
                + self._external_acceleration(self.position) * sub_dt
            )
            self.external_velocity *= external_drag_factor
            self.external_velocity = self._clip_external_velocity(
                self.external_velocity
            )
            world_velocity = self.speed * forward + self.external_velocity
            candidate = self.position + world_velocity * sub_dt
            candidate, wall_hit = self._resolve_wall_collision(candidate)
            wall_collision = wall_collision or wall_hit
            candidate, hazard_hit, impulse = self._resolve_hazard_collision(
                candidate
            )
            self.position = candidate.astype(self._dtype, copy=False)

            if hazard_hit:
                hazard_contact = True
            if impulse > 0.0:
                step_impulse += impulse
                self.health = max(
                    0.0,
                    self.health - impulse / self.config.damage_capacity,
                )
                if self.health <= 0.0:
                    self.dead = True
                    break

        self.elapsed_steps += 1
        self.total_impulse += step_impulse
        self._hazard_contact = hazard_contact
        self._update_display_health(self.config.dt)
        checkpoint_crossed = False
        self._waypoint_reached = False
        distance_after_motion = self.distance_to_waypoint

        if not self.dead:
            if self.config.task_mode == "navigation":
                self.success = distance_after_motion <= self.config.goal_radius
            elif distance_after_motion <= self.config.goal_radius:
                checkpoint_crossed = True
                self._waypoint_reached = True
                self._lap_completed += 1
                if self._lap_completed >= self.config.checkpoint_count:
                    self.success = True
                else:
                    self._checkpoint_index = (
                        self._checkpoint_index + self._lap_direction
                    ) % self.config.checkpoint_count
                    self.current_waypoint = self._checkpoints[
                        self._checkpoint_index
                    ].copy()

        terminated = self.dead or (self.success and self.terminate_on_success)
        truncated = self.elapsed_steps >= self.config.max_episode_steps and not terminated
        reward = self._compute_step_reward(
            previous_distance=previous_distance,
            current_target_distance=distance_after_motion,
            action=action,
            step_impulse=step_impulse,
            checkpoint_crossed=checkpoint_crossed,
        )
        if self.dead:
            reason = "health_depleted"
        elif self.success and self.terminate_on_success:
            reason = "goal" if self.config.task_mode == "navigation" else "lap_complete"
        elif truncated:
            reason = "time_limit"
        else:
            reason = None

        info = self._get_info(
            reason,
            step_impulse=step_impulse,
            checkpoint_crossed=checkpoint_crossed,
        )
        info["wall_collision"] = wall_collision
        info["steer_angle"] = steer
        info["throttle"] = throttle
        observation = self._get_observation()
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def set_goal(self, goal: Array) -> None:
        if self.config.task_mode != "navigation":
            raise RuntimeError("set_goal() is only valid in navigation mode")
        validated = self._validate_safe_point(goal, "goal")
        if np.linalg.norm(validated - self.position) <= self.config.goal_radius:
            raise ValueError("New goal coincides with the car")
        self.goal = validated
        self.current_waypoint = validated.copy()
        self.success = False

    def sample_safe_point(self, **kwargs: Any) -> Array:
        return self._sample_safe_point(**kwargs)

    def compute_reward(
        self,
        achieved_goal: Array,
        desired_goal: Array,
        info: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
    ) -> Array | float:
        achieved = np.asarray(achieved_goal, dtype=self._dtype)
        desired = np.asarray(desired_goal, dtype=self._dtype)
        if achieved.shape[-1] != 4 or desired.shape[-1] != 4:
            raise ValueError("achieved_goal and desired_goal must end in dimension 4")
        spatial_success = (
            np.linalg.norm(achieved[..., :2] - desired[..., :2], axis=-1)
            <= self.config.goal_radius
        )
        progress_success = achieved[..., 2] + 1e-6 >= desired[..., 2]
        direction_success = np.isclose(
            achieved[..., 3], desired[..., 3], atol=1e-6
        )
        success = spatial_success & progress_success & direction_success
        rewards = np.where(
            success,
            self.config.success_reward,
            -self.config.step_penalty,
        ).astype(self._dtype)
        if info is not None:
            if isinstance(info, Mapping):
                dead = np.asarray(bool(info.get("dead", False)))
            else:
                dead = np.asarray([bool(item.get("dead", False)) for item in info])
            rewards = np.where(dead, self.config.death_penalty, rewards).astype(
                self._dtype
            )
        return float(rewards) if rewards.ndim == 0 else rewards

    def _compute_step_reward(
        self,
        *,
        previous_distance: float,
        current_target_distance: float,
        action: Array,
        step_impulse: float,
        checkpoint_crossed: bool,
    ) -> float:
        if self.dead:
            return self.config.death_penalty
        reward = -self.config.step_penalty
        reward -= self.config.control_cost * float(np.dot(action, action))
        reward -= (
            self.config.damage_penalty_scale
            * step_impulse
            / self.config.damage_capacity
        )
        if self.config.reward_mode == "dense":
            reward += self.config.progress_scale * (
                previous_distance - current_target_distance
            )
        if self.config.task_mode == "navigation" and self.success:
            reward += self.config.success_reward
        elif self.config.task_mode == "lap" and checkpoint_crossed:
            reward += self.config.checkpoint_reward
            if self.success:
                reward += self.config.success_reward
        return float(reward)

    def _sample_safe_point(
        self,
        *,
        min_distance_from: Array | None = None,
        minimum_distance: float = 0.0,
    ) -> Array:
        inner = (
            self.config.inner_hazard_radius
            + self.config.collision_radius
            + self.config.spawn_clearance
        )
        outer = (
            self.config.outer_hazard_radius
            - self.config.collision_radius
            - self.config.spawn_clearance
        )
        if inner >= outer:
            raise ValueError("spawn_clearance leaves no sampling region")
        for _ in range(10_000):
            radius = np.sqrt(self.np_random.uniform(inner**2, outer**2))
            angle = self.np_random.uniform(-np.pi, np.pi)
            point = radius * np.array([np.cos(angle), np.sin(angle)])
            if min_distance_from is not None and np.linalg.norm(
                point - np.asarray(min_distance_from)
            ) < minimum_distance:
                continue
            return point.astype(self._dtype)
        raise RuntimeError("Could not sample a safe point")

    def _validate_safe_point(self, point: Any, name: str) -> Array:
        value = np.asarray(point, dtype=self._dtype)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite vector with shape (2,)")
        radius = float(np.linalg.norm(value))
        low = self.config.inner_hazard_radius + self.config.collision_radius
        high = self.config.outer_hazard_radius - self.config.collision_radius
        if not low < radius < high:
            raise ValueError(f"{name} must lie in the collision-free annulus")
        return value.copy()

    def _external_acceleration(self, position: Array) -> Array:
        """Signed inverse-square acceleration toward the world origin."""
        strength = float(self.config.gravity_strength)
        if strength == 0.0:
            return np.zeros(2, dtype=self._dtype)
        delta = -np.asarray(position, dtype=self._dtype)
        distance = float(np.linalg.norm(delta))
        softened = max(distance, self.config.gravity_soft_min)
        return (delta * (strength / softened**3)).astype(
            self._dtype, copy=False
        )

    def _clip_external_velocity(self, velocity: Array) -> Array:
        velocity = np.asarray(velocity, dtype=self._dtype)
        magnitude = float(np.linalg.norm(velocity))
        if magnitude > self.config.max_external_speed:
            velocity = velocity * (
                self.config.max_external_speed / magnitude
            )
        return velocity.astype(self._dtype, copy=False)

    def _update_display_health(self, dt: float) -> None:
        """Exponentially ease the rendered health bar toward true health."""
        alpha = 1.0 - float(np.exp(-dt / self.config.health_bar_tau))
        self._display_health = float(
            self._display_health
            + alpha * (self.health - self._display_health)
        )
        if self.health <= 0.0 and self._display_health < 0.01:
            self._display_health = 0.0

    def _resolve_wall_collision(self, candidate: Array) -> tuple[Array, bool]:
        low = self.config.arena_low + self.config.collision_radius
        high = self.config.arena_high - self.config.collision_radius
        clipped = np.clip(candidate, low, high)
        hit_axes = clipped != candidate
        hit = bool(np.any(hit_axes))
        if hit:
            self.speed = -self.speed * self.config.wall_restitution
            self.external_velocity[hit_axes] *= -self.config.wall_restitution
        return clipped, hit

    def _resolve_hazard_collision(
        self, candidate: Array
    ) -> tuple[Array, bool, float]:
        """Keep the car in the safe annulus and apply bounce impact impulse.

        Returns ``(position, contacted, impulse)``.  Impulse is nonzero only when
        the car hits the ring with normal speed above ``min_impact_speed``.
        """
        candidate = np.asarray(candidate, dtype=self._dtype).copy()
        radius = float(np.linalg.norm(candidate))
        inner = self.config.inner_hazard_radius + self.config.collision_radius
        outer = self.config.outer_hazard_radius - self.config.collision_radius
        if radius < 1e-8:
            candidate = np.array([inner, 0.0], dtype=self._dtype)
            radius = inner

        hit_inner = radius < inner
        hit_outer = radius > outer
        if not (hit_inner or hit_outer):
            return candidate, False, 0.0

        target = inner if hit_inner else outer
        radial = candidate / radius
        candidate = (radial * target).astype(self._dtype, copy=False)

        forward = np.array(
            [np.cos(self.heading), np.sin(self.heading)], dtype=self._dtype
        )
        world_velocity = self.speed * forward + self.external_velocity
        radial_speed = float(np.dot(world_velocity, radial))
        into_wall = (hit_inner and radial_speed < 0.0) or (
            hit_outer and radial_speed > 0.0
        )
        impulse = 0.0
        if into_wall:
            into_speed = abs(radial_speed)
            restitution = float(self.config.hazard_restitution)
            if into_speed >= self.config.min_impact_speed:
                impulse = float(
                    self.config.impact_impulse_scale
                    * (1.0 + restitution)
                    * into_speed
                )
            world_velocity = (
                world_velocity - (1.0 + restitution) * radial_speed * radial
            )
            self.speed = float(
                np.clip(
                    np.dot(world_velocity, forward),
                    -self.config.max_reverse_speed,
                    self.config.max_speed,
                )
            )
            self.external_velocity = self._clip_external_velocity(
                world_velocity - self.speed * forward
            )
        return candidate, True, impulse

    def _build_checkpoints(self) -> Array:
        angles = np.linspace(
            0.0, 2.0 * np.pi, self.config.checkpoint_count, endpoint=False
        )
        return (
            self.config.track_radius
            * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        ).astype(self._dtype)

    def _get_observation(self) -> Any:
        state = self.state
        if self.observation_mode == "state":
            return state
        if self.observation_mode == "state_goal":
            return np.concatenate([state, self.desired_goal]).astype(self._dtype)
        return {
            "observation": state,
            "achieved_goal": self.achieved_goal,
            "desired_goal": self.desired_goal,
        }

    def _get_info(
        self,
        termination_reason: str | None,
        *,
        step_impulse: float,
        checkpoint_crossed: bool,
    ) -> dict[str, Any]:
        return {
            "is_success": bool(self.success),
            "success": bool(self.success),
            "dead": bool(self.dead),
            "termination_reason": termination_reason,
            "position": self.position.copy(),
            "heading": float(self.heading),
            "speed": float(self.speed),
            "external_velocity": self.external_velocity.copy(),
            "gravity_strength": float(self.config.gravity_strength),
            "rolling_drag": float(self.config.rolling_drag),
            "cornering_grip": float(self.config.cornering_grip),
            "longitudinal_grip": float(self.config.longitudinal_grip),
            "steering_response": float(self.config.steering_response),
            "health": float(self.health),
            "step_impulse": float(step_impulse),
            "total_impulse": float(self.total_impulse),
            "hazard_contact": bool(self._hazard_contact),
            "distance_to_goal": self.distance_to_goal,
            "distance_to_waypoint": self.distance_to_waypoint,
            "goal": self.desired_goal,
            "final_goal": self.goal.copy(),
            "current_waypoint": self.current_waypoint.copy(),
            "elapsed_steps": int(self.elapsed_steps),
            "task_mode": self.config.task_mode,
            "task_id": self.cur_task_id,
            "checkpoint_index": (
                int(self._checkpoint_index)
                if self.config.task_mode == "lap"
                else None
            ),
            "checkpoint_crossed": bool(checkpoint_crossed),
            "checkpoints_completed": int(self._lap_completed),
            "lap_progress": self.task_progress,
            "lap_direction": (
                int(self._lap_direction)
                if self.config.task_mode == "lap"
                else None
            ),
        }

    def render(self) -> Array | None:
        frame = self._render_rgb()
        if self.render_mode == "rgb_array":
            return frame
        if self.render_mode == "human":
            self._render_human(frame)
        return None

    def close(self) -> None:
        if self._human_figure is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._human_figure)
            except ImportError:
                pass
        self._human_figure = None
        self._human_axis = None
        self._human_image = None

    def _render_rgb(self) -> Array:
        """Render a race-track themed frame without affecting environment state.

        The renderer intentionally derives every visual element from the existing
        environment state/configuration.  Physics, rewards, observations, task
        logic, and collision geometry are untouched.
        """
        size = self.render_size
        low = self.config.arena_low
        high = self.config.arena_high
        span = high - low

        xs = np.linspace(low, high, size)
        ys = np.linspace(high, low, size)
        xx, yy = np.meshgrid(xs, ys)
        rr = np.sqrt(xx**2 + yy**2)
        angle = np.mod(np.arctan2(yy, xx), 2.0 * np.pi)

        inner = self.config.inner_hazard_radius
        outer = self.config.outer_hazard_radius
        track = (rr >= inner) & (rr <= outer)
        icy = (
            float(self.config.rolling_drag) <= 0.15
            and abs(float(self.config.gravity_strength)) <= 1e-12
        )

        # ------------------------------------------------------------------
        # Stadium ground. Ice mode replaces grass with a snow-covered arena;
        # this branch is visual-only and leaves every physical parameter intact.
        # ------------------------------------------------------------------
        frame = np.empty((size, size, 3), dtype=np.uint8)
        infield = rr < inner

        if icy:
            snow_tiles = (
                np.floor((xx - low) * 10.0) + np.floor((yy - low) * 10.0)
            ).astype(np.int32)
            snow_grain = (
                4.2 * np.sin(27.0 * xx + 13.0 * yy)
                + 2.8 * np.cos(43.0 * xx - 31.0 * yy)
                + 1.8 * (snow_tiles % 2)
            )
            snow_shadow = 8.0 * np.clip((rr - outer) / max(high - outer, 1e-6), 0.0, 1.0)
            snow_base = 224.0 + snow_grain - snow_shadow
            frame[..., 0] = np.clip(snow_base * 0.90, 180, 232).astype(np.uint8)
            frame[..., 1] = np.clip(snow_base * 0.98, 202, 244).astype(np.uint8)
            frame[..., 2] = np.clip(snow_base * 1.07, 220, 255).astype(np.uint8)

            # The inaccessible center becomes a compact frozen island.
            radial_frost = (
                7.0 * np.sin(31.0 * rr + 4.0 * np.cos(6.0 * angle))
                + 3.0 * np.cos(19.0 * rr - 5.0 * angle)
            )
            frame[..., 0][infield] = np.clip(
                181.0 + 0.45 * radial_frost[infield], 164, 204
            ).astype(np.uint8)
            frame[..., 1][infield] = np.clip(
                218.0 + 0.65 * radial_frost[infield], 198, 236
            ).astype(np.uint8)
            frame[..., 2][infield] = np.clip(
                239.0 + 0.45 * radial_frost[infield], 222, 252
            ).astype(np.uint8)
        else:
            grass_tiles = (
                np.floor((xx - low) * 8.5) + np.floor((yy - low) * 8.5)
            ).astype(np.int32)
            grass_wave = 4.0 * np.sin(16.0 * xx + 6.0 * yy)
            grass_base = 73.0 + 7.0 * (grass_tiles % 2) + grass_wave

            frame[..., 0] = np.clip(grass_base * 0.48, 22, 55).astype(np.uint8)
            frame[..., 1] = np.clip(grass_base * 1.18, 72, 126).astype(np.uint8)
            frame[..., 2] = np.clip(grass_base * 0.58, 32, 72).astype(np.uint8)

            # Central infield has a slightly brighter radial mowing pattern.
            radial_mow = 6.0 * np.sin(26.0 * rr + 5.0 * np.cos(5.0 * angle))
            frame[..., 0][infield] = np.clip(
                39.0 + radial_mow[infield], 28, 52
            ).astype(np.uint8)
            frame[..., 1][infield] = np.clip(
                105.0 + radial_mow[infield], 82, 126
            ).astype(np.uint8)
            frame[..., 2][infield] = np.clip(
                55.0 + 0.6 * radial_mow[infield], 42, 68
            ).astype(np.uint8)

        # ------------------------------------------------------------------
        # Asphalt: subtle aggregate, rubbered racing line, and lane markings.
        # ------------------------------------------------------------------
        aggregate = (
            3.0 * np.sin(91.0 * xx + 47.0 * yy)
            + 2.0 * np.sin(143.0 * xx - 71.0 * yy)
            + 1.5 * np.cos(211.0 * xx + 17.0 * yy)
        )
        rubber = 8.0 * np.exp(
            -((rr - self.config.track_radius) / 0.075) ** 2
        )
        if icy:
            # Layered blue ice: diffuse texture, radial tint, and glassy sheen.
            frost_texture = (
                4.0 * np.sin(35.0 * xx + 21.0 * yy)
                + 2.4 * np.cos(69.0 * xx - 37.0 * yy)
            )
            radial_tint = 8.0 * np.exp(
                -((rr - self.config.track_radius) / 0.24) ** 2
            )
            asphalt = np.clip(
                126.0 + 0.48 * aggregate + frost_texture + radial_tint,
                100,
                158,
            )
            frame[..., 0][track] = np.clip(asphalt[track] * 0.82, 0, 255).astype(
                np.uint8
            )
            frame[..., 1][track] = np.clip(asphalt[track] * 1.08, 0, 255).astype(
                np.uint8
            )
            frame[..., 2][track] = np.clip(asphalt[track] * 1.34, 0, 255).astype(
                np.uint8
            )

            # Broad specular ribbons make the surface read as polished ice.
            sheen_field = (
                np.sin(3.0 * angle + 10.0 * rr + 0.45)
                + 0.35 * np.sin(7.0 * angle - 4.0 * rr)
            )
            sheen = track & (sheen_field > 0.88)
            self._alpha_blend_mask(
                frame,
                sheen,
                np.array([223, 250, 255], dtype=np.uint8),
                0.24,
            )

            # Sparse deterministic hairline cracks; they are purely decorative.
            crack_a = np.abs(
                np.sin(23.0 * xx + 17.0 * yy + 1.7 * np.sin(8.0 * yy))
            ) < 0.020
            crack_b = np.abs(
                np.sin(29.0 * xx - 19.0 * yy + 1.4 * np.sin(7.0 * xx))
            ) < 0.016
            crack_gate = (
                np.sin(9.0 * xx + 13.0 * yy) > 0.30
            ) | (
                np.cos(11.0 * xx - 7.0 * yy) > 0.55
            )
            cracks = (
                track
                & (crack_a | crack_b)
                & crack_gate
                & (rr > inner + 0.065)
                & (rr < outer - 0.065)
            )
            self._alpha_blend_mask(
                frame,
                cracks,
                np.array([218, 249, 255], dtype=np.uint8),
                0.58,
            )
        else:
            asphalt = np.clip(62.0 + aggregate - rubber, 42, 72)
            frame[..., 0][track] = np.clip(asphalt[track] * 0.94, 0, 255).astype(
                np.uint8
            )
            frame[..., 1][track] = np.clip(asphalt[track], 0, 255).astype(
                np.uint8
            )
            frame[..., 2][track] = np.clip(asphalt[track] * 1.04, 0, 255).astype(
                np.uint8
            )

        # Pale edge lines separate the racing surface from the curbs.
        edge_width = max(0.006, 1.5 * span / size)
        inner_edge = np.abs(rr - (inner + 0.050)) <= edge_width
        outer_edge = np.abs(rr - (outer - 0.050)) <= edge_width
        frame[inner_edge | outer_edge] = np.array(
            [226, 250, 255] if icy else [224, 226, 218],
            dtype=np.uint8,
        )

        # Red/white racing curbs around both hazardous boundaries.
        curb_width = min(0.045, 0.12 * (outer - inner))
        curb_segment = (
            np.floor(angle / (2.0 * np.pi / 36.0)).astype(np.int32) % 2
        )
        inner_curb = (rr >= inner) & (rr <= inner + curb_width)
        outer_curb = (rr <= outer) & (rr >= outer - curb_width)
        curbs = inner_curb | outer_curb
        red_curb = curbs & (curb_segment == 0)
        white_curb = curbs & (curb_segment == 1)
        if icy:
            frame[red_curb] = np.array([55, 160, 216], dtype=np.uint8)
            frame[white_curb] = np.array([233, 250, 255], dtype=np.uint8)
        else:
            frame[red_curb] = np.array([205, 42, 45], dtype=np.uint8)
            frame[white_curb] = np.array([238, 235, 220], dtype=np.uint8)

        # Tire/rubber barriers visually communicate the collision boundary.
        wall_width = max(0.010, 2.4 * span / size)
        inner_wall = np.abs(rr - inner) <= wall_width
        outer_wall = np.abs(rr - outer) <= wall_width
        frame[inner_wall | outer_wall] = np.array(
            [31, 67, 91] if icy else [19, 22, 27],
            dtype=np.uint8,
        )
        if icy:
            ice_cap_width = max(0.004, 0.42 * wall_width)
            inner_ice_cap = np.abs(rr - inner) <= ice_cap_width
            outer_ice_cap = np.abs(rr - outer) <= ice_cap_width
            frame[inner_ice_cap | outer_ice_cap] = np.array(
                [171, 231, 247], dtype=np.uint8
            )

        # Dashed guide line around the nominal racing radius.
        dash_period = 2.0 * np.pi / 30.0
        dash = np.mod(angle, dash_period) < 0.56 * dash_period
        guide = (
            np.abs(rr - self.config.track_radius)
            <= max(0.005, 1.2 * span / size)
        ) & dash
        if icy:
            guide_glow = (
                np.abs(rr - self.config.track_radius)
                <= max(0.012, 2.8 * span / size)
            ) & dash
            self._alpha_blend_mask(
                frame,
                guide_glow,
                np.array([112, 225, 250], dtype=np.uint8),
                0.18,
            )
            frame[guide] = np.array([195, 247, 255], dtype=np.uint8)
        else:
            frame[guide] = np.array([222, 190, 78], dtype=np.uint8)

        # Start/finish checkerboard for lap tasks.  This is visual only and is
        # aligned with the existing lap start checkpoint.
        if self.config.task_mode == "lap":
            start_angle = (
                2.0
                * np.pi
                * self._lap_start_index
                / self.config.checkpoint_count
            )
            radial_axis = np.array(
                [np.cos(start_angle), np.sin(start_angle)], dtype=np.float32
            )
            tangent_axis = np.array(
                [-np.sin(start_angle), np.cos(start_angle)], dtype=np.float32
            )
            radial_coord = radial_axis[0] * xx + radial_axis[1] * yy
            tangent_coord = tangent_axis[0] * xx + tangent_axis[1] * yy
            finish_half_width = 0.025
            finish = (
                (np.abs(tangent_coord) <= finish_half_width)
                & (radial_coord >= inner + curb_width)
                & (radial_coord <= outer - curb_width)
            )
            radial_bin = np.floor(
                (radial_coord - inner - curb_width) / 0.038
            ).astype(np.int32)
            tangent_bin = np.floor(
                (tangent_coord + finish_half_width) / 0.0125
            ).astype(np.int32)
            checker = (radial_bin + tangent_bin) % 2 == 0
            frame[finish & checker] = np.array([244, 244, 238], dtype=np.uint8)
            frame[finish & ~checker] = np.array([18, 20, 24], dtype=np.uint8)

        # Barrier studs and stadium lights add trackside depth.
        stud_radius = max(0.006, 1.6 * span / size)
        for index, theta in enumerate(
            np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
        ):
            outer_stud = (outer + 0.018) * np.array(
                [np.cos(theta), np.sin(theta)], dtype=np.float32
            )
            if icy:
                stud_color = np.array(
                    [186, 246, 255] if index % 2 == 0 else [69, 177, 229],
                    dtype=np.uint8,
                )
                self._blend_disk(
                    frame,
                    outer_stud,
                    stud_radius * 2.2,
                    np.array([112, 222, 255], dtype=np.uint8),
                    0.14,
                )
            else:
                stud_color = np.array(
                    [235, 205, 92] if index % 2 == 0 else [190, 200, 210],
                    dtype=np.uint8,
                )
            self._draw_disk(frame, outer_stud, stud_radius, stud_color)

        # Central source indicator only.
        # Nothing is drawn on the track, car, or HUD.
        if abs(float(self.config.gravity_strength)) > 1e-12:
            self._draw_center_gravity_source(frame)
        elif icy:
            self._draw_center_ice_emblem(frame)
        else:
            # Existing sponsor emblem for the plain environment (ring only).
            emblem_outer = max(0.035, inner * 0.32)
            emblem_inner = emblem_outer * 0.57

            self._blend_ring(
                frame,
                np.zeros(2, dtype=np.float32),
                emblem_outer,
                emblem_inner,
                np.array([238, 206, 63], dtype=np.uint8),
                0.85,
            )

        # ------------------------------------------------------------------
        # Goals/checkpoints styled as luminous race gates.
        # ------------------------------------------------------------------
        if self.config.task_mode == "lap":
            for index, checkpoint in enumerate(self._checkpoints):
                active = index == self._checkpoint_index
                if icy:
                    color = np.array(
                        [75, 244, 232] if active else [95, 176, 242],
                        dtype=np.uint8,
                    )
                else:
                    color = np.array(
                        [55, 226, 126] if active else [240, 182, 55],
                        dtype=np.uint8,
                    )
                if active:
                    self._blend_disk(
                        frame,
                        checkpoint,
                        self.config.goal_radius * 1.55,
                        color,
                        0.18,
                    )
                self._blend_ring(
                    frame,
                    checkpoint,
                    self.config.goal_radius * (0.82 if active else 0.68),
                    self.config.goal_radius * (0.48 if active else 0.43),
                    color,
                    0.95 if active else 0.82,
                )
                self._draw_disk(
                    frame,
                    checkpoint,
                    self.config.goal_radius * 0.20,
                    np.array([244, 246, 238], dtype=np.uint8),
                )
        else:
            goal_green = np.array(
                [74, 242, 222] if icy else [46, 222, 112],
                dtype=np.uint8,
            )
            pulse = 1.0 + 0.06 * np.sin(0.22 * self.elapsed_steps)
            self._blend_disk(
                frame,
                self.goal,
                self.config.goal_radius * 1.75 * pulse,
                goal_green,
                0.18,
            )
            self._blend_ring(
                frame,
                self.goal,
                self.config.goal_radius * 1.18,
                self.config.goal_radius * 0.82,
                np.array([245, 247, 239], dtype=np.uint8),
                0.96,
            )
            self._draw_checkered_disk(
                frame,
                self.goal,
                self.config.goal_radius * 0.80,
                goal_green,
                np.array([19, 24, 28], dtype=np.uint8),
            )

        # ------------------------------------------------------------------
        # Race car: shadow, wheels, bodywork, cockpit, stripe, and lights.
        # ------------------------------------------------------------------
        dx = xx - float(self.position[0])
        dy = yy - float(self.position[1])
        c, s = np.cos(self.heading), np.sin(self.heading)
        longitudinal = c * dx + s * dy
        lateral = -s * dx + c * dy
        half_length = self.config.car_length / 2.0
        half_width = self.config.car_width / 2.0

        shadow_dx = xx - float(self.position[0] + 0.012)
        shadow_dy = yy - float(self.position[1] - 0.014)
        shadow_long = c * shadow_dx + s * shadow_dy
        shadow_lat = -s * shadow_dx + c * shadow_dy
        shadow_mask = (
            (np.abs(shadow_long) <= half_length * 1.05)
            & (np.abs(shadow_lat) <= half_width * 1.18)
        )
        self._alpha_blend_mask(
            frame,
            shadow_mask,
            np.array([0, 0, 0], dtype=np.uint8),
            0.30,
        )

        if icy:
            self._draw_ice_motion_effects(frame, c=c, s=s)

        # Four dark wheels slightly outside the body shell.
        wheel_long_half = max(0.006, self.config.car_length * 0.13)
        wheel_lat_half = max(0.004, self.config.car_width * 0.12)
        for wheel_long in (-0.29 * self.config.car_length, 0.29 * self.config.car_length):
            for wheel_lat in (-0.56 * self.config.car_width, 0.56 * self.config.car_width):
                wheel = (
                    np.abs(longitudinal - wheel_long) <= wheel_long_half
                ) & (np.abs(lateral - wheel_lat) <= wheel_lat_half)
                frame[wheel] = np.array([11, 13, 16], dtype=np.uint8)

        # Tapered silhouette, then a slightly smaller body for an outlined look.
        nose_fraction = np.clip(
            (longitudinal + half_length) / max(self.config.car_length, 1e-8),
            0.0,
            1.0,
        )
        width_scale = 1.0 - 0.24 * np.clip(
            (nose_fraction - 0.58) / 0.42, 0.0, 1.0
        )
        outline = (
            (np.abs(longitudinal) <= half_length * 1.03)
            & (np.abs(lateral) <= half_width * 1.10 * width_scale)
        )
        frame[outline] = np.array([13, 20, 32], dtype=np.uint8)

        body = (
            (np.abs(longitudinal) <= half_length * 0.94)
            & (np.abs(lateral) <= half_width * 0.92 * width_scale)
        )
        if self.dead:
            body_color = np.array([61, 64, 69], dtype=np.uint8)
        elif self._hazard_contact:
            body_color = np.array([225, 82, 45], dtype=np.uint8)
        else:
            body_color = np.array(
                [34, 132, 218] if icy else [34, 111, 230],
                dtype=np.uint8,
            )
        frame[body] = body_color

        # Rear wing.
        wing = (
            np.abs(longitudinal + 0.82 * half_length)
            <= max(0.004, 0.07 * self.config.car_length)
        ) & (np.abs(lateral) <= 1.10 * half_width)
        frame[wing] = np.array([12, 21, 36], dtype=np.uint8)

        # Bright center racing stripe.
        stripe = (
            (longitudinal >= -0.78 * half_length)
            & (longitudinal <= 0.86 * half_length)
            & (np.abs(lateral) <= max(0.0035, 0.12 * half_width))
        )
        frame[stripe] = np.array(
            [174, 247, 255] if icy else [238, 236, 218],
            dtype=np.uint8,
        )

        # Cockpit canopy.
        cockpit = (
            ((longitudinal + 0.10 * half_length) / (0.43 * half_length + 1e-8))
            ** 2
            + (lateral / (0.58 * half_width + 1e-8)) ** 2
            <= 1.0
        )
        frame[cockpit] = np.array(
            [18, 55, 78] if icy else [24, 44, 67],
            dtype=np.uint8,
        )
        canopy_glint = cockpit & (lateral < -0.12 * half_width)
        self._alpha_blend_mask(
            frame,
            canopy_glint,
            np.array([105, 195, 229], dtype=np.uint8),
            0.33,
        )

        # Headlights at the front and taillights at the rear.
        headlight = (
            (longitudinal >= 0.72 * half_length)
            & (longitudinal <= 0.98 * half_length)
            & (np.abs(lateral) >= 0.42 * half_width)
            & (np.abs(lateral) <= 0.78 * half_width)
        )
        frame[headlight] = np.array([250, 236, 142], dtype=np.uint8)
        taillight = (
            (longitudinal <= -0.72 * half_length)
            & (longitudinal >= -0.98 * half_length)
            & (np.abs(lateral) >= 0.38 * half_width)
            & (np.abs(lateral) <= 0.80 * half_width)
        )
        frame[taillight] = np.array([232, 47, 51], dtype=np.uint8)

        # Dead car gets a clear red X while preserving the same physical state.
        if self.dead:
            forward = np.array([c, s], dtype=np.float32)
            right = np.array([-s, c], dtype=np.float32)
            p1 = self.position - 0.72 * half_length * forward - 0.70 * half_width * right
            p2 = self.position + 0.72 * half_length * forward + 0.70 * half_width * right
            p3 = self.position - 0.72 * half_length * forward + 0.70 * half_width * right
            p4 = self.position + 0.72 * half_length * forward - 0.70 * half_width * right
            dead_color = np.array([224, 48, 52], dtype=np.uint8)
            self._draw_line(
                frame,
                self._world_to_pixel(p1),
                self._world_to_pixel(p2),
                dead_color,
                max(2, size // 190),
            )
            self._draw_line(
                frame,
                self._world_to_pixel(p3),
                self._world_to_pixel(p4),
                dead_color,
                max(2, size // 190),
            )

        # ------------------------------------------------------------------
        # Racing HUD: segmented health, drift/slip meter, and lap progress.
        # ------------------------------------------------------------------
        panel_margin = max(8, size // 42)
        panel_height = max(31, size // 13)
        panel_width = max(140, size // 3)
        self._blend_rect(
            frame,
            panel_margin,
            panel_margin,
            min(size - panel_margin, panel_margin + panel_width),
            min(size - panel_margin, panel_margin + panel_height),
            np.array([7, 24, 39] if icy else [9, 14, 20], dtype=np.uint8),
            0.84 if icy else 0.80,
        )

        health_frac = float(
            np.clip(
                self._display_health / self.config.initial_health,
                0.0,
                1.0,
            )
        )
        segment_count = 12
        segment_gap = max(2, size // 240)
        bar_x0 = panel_margin + max(10, size // 75)
        bar_y0 = panel_margin + max(8, size // 80)
        bar_x1 = panel_margin + panel_width - max(10, size // 75)
        health_h = max(8, size // 48)
        usable_w = bar_x1 - bar_x0 - (segment_count - 1) * segment_gap
        segment_w = max(2, usable_w // segment_count)
        lit_segments = int(np.ceil(health_frac * segment_count - 1e-8))
        health_color = (
            health_frac * np.array([52, 218, 112], dtype=np.float32)
            + (1.0 - health_frac)
            * np.array([232, 51, 48], dtype=np.float32)
        ).astype(np.uint8)
        for i in range(segment_count):
            x0 = bar_x0 + i * (segment_w + segment_gap)
            x1 = min(bar_x1, x0 + segment_w)
            color = (
                health_color
                if i < lit_segments
                else np.array([44, 50, 57], dtype=np.uint8)
            )
            frame[bar_y0 : bar_y0 + health_h, x0:x1] = color

        # --------------------------------------------------------------
        # Drift / slip meter below the health segments.
        #
        # The meter is centered:
        #   left drift  <- [ fill | center |      ]
        #   right drift -> [      | center | fill ]
        #
        # This is visual only and does not modify environment state.
        # --------------------------------------------------------------
        forward = np.array(
            [np.cos(self.heading), np.sin(self.heading)],
            dtype=np.float32,
        )

        # Vehicle-local right direction.
        right = np.array(
            [np.sin(self.heading), -np.cos(self.heading)],
            dtype=np.float32,
        )

        # Actual world-space velocity contains both commanded vehicle motion
        # and gravity-induced external motion.
        world_velocity = self.speed * forward + self.external_velocity

        longitudinal_speed = float(np.dot(world_velocity, forward))
        lateral_speed = float(np.dot(world_velocity, right))
        world_speed = float(np.linalg.norm(world_velocity))

        # At very low velocity, the slip angle is numerically unstable and
        # visually uninformative, so show the neutral state.
        if world_speed < 0.025:
            slip_angle = 0.0
        else:
            slip_angle = float(
                np.arctan2(
                    lateral_speed,
                    abs(longitudinal_speed) + 0.04,
                )
            )

        # 45 degrees or more is treated as maximum slip.
        max_visual_slip = np.deg2rad(45.0)
        slip_frac = float(
            np.clip(
                abs(slip_angle) / max_visual_slip,
                0.0,
                1.0,
            )
        )

        drift_y0 = bar_y0 + health_h + max(6, size // 80)
        drift_h = max(6, size // 72)

        # Meter background.
        frame[
            drift_y0 : drift_y0 + drift_h,
            bar_x0:bar_x1,
        ] = np.array([37, 43, 50], dtype=np.uint8)

        center_x = (bar_x0 + bar_x1) // 2
        half_width = max(1, (bar_x1 - bar_x0) // 2)

        # Neutral center marker.
        center_marker_width = max(2, size // 240)
        frame[
            drift_y0 - 1 : drift_y0 + drift_h + 1,
            center_x - center_marker_width : center_x + center_marker_width + 1,
        ] = np.array([226, 230, 232], dtype=np.uint8)

        # Safe drift is green; severe drift approaches red.
        safe_color = np.array(
            [65, 214, 244] if icy else [56, 211, 126],
            dtype=np.float32,
        )
        danger_color = np.array([242, 66, 56], dtype=np.float32)

        slip_color = (
            (1.0 - slip_frac) * safe_color
            + slip_frac * danger_color
        ).astype(np.uint8)

        fill_width = int(half_width * slip_frac)

        if fill_width > 0:
            if lateral_speed < 0.0:
                # Sliding toward vehicle-local left.
                x0 = max(bar_x0, center_x - fill_width)
                x1 = center_x

            else:
                # Sliding toward vehicle-local right.
                x0 = center_x
                x1 = min(bar_x1, center_x + fill_width)

            frame[
                drift_y0 : drift_y0 + drift_h,
                x0:x1,
            ] = slip_color

        # Optional small end caps make the bilateral scale easier to read.
        cap_width = max(1, size // 300)

        frame[
            drift_y0 : drift_y0 + drift_h,
            bar_x0 : bar_x0 + cap_width,
        ] = np.array([105, 116, 126], dtype=np.uint8)

        frame[
            drift_y0 : drift_y0 + drift_h,
            bar_x1 - cap_width : bar_x1,
        ] = np.array([105, 116, 126], dtype=np.uint8)

        if self.config.task_mode == "lap":
            dot_radius = max(2, size // 120)
            dot_spacing = dot_radius * 3
            total_width = self.config.checkpoint_count * dot_spacing
            dot_y = panel_margin + panel_height // 2
            dot_x0 = size - panel_margin - total_width
            for i in range(self.config.checkpoint_count):
                if icy:
                    color = np.array(
                        [78, 241, 229]
                        if i == self._checkpoint_index
                        else [94, 169, 229],
                        dtype=np.uint8,
                    )
                else:
                    color = np.array(
                        [52, 225, 122]
                        if i == self._checkpoint_index
                        else [224, 172, 58],
                        dtype=np.uint8,
                    )
                self._draw_pixel_disk(
                    frame,
                    (dot_x0 + i * dot_spacing, dot_y),
                    dot_radius,
                    color,
                )

        # Dark outer frame with small race accents.
        border = max(3, size // 150)
        frame[:border] = np.array([15, 18, 22], dtype=np.uint8)
        frame[-border:] = np.array([15, 18, 22], dtype=np.uint8)
        frame[:, :border] = np.array([15, 18, 22], dtype=np.uint8)
        frame[:, -border:] = np.array([15, 18, 22], dtype=np.uint8)
        accent = max(8, size // 18)
        accent_primary = np.array(
            [61, 180, 230] if icy else [214, 42, 46],
            dtype=np.uint8,
        )
        accent_secondary = np.array(
            [224, 250, 255] if icy else [238, 237, 226],
            dtype=np.uint8,
        )
        frame[:border, :accent] = accent_primary
        frame[:border, accent : 2 * accent] = accent_secondary
        frame[-border:, -accent:] = accent_primary
        frame[-border:, -2 * accent : -accent] = accent_secondary
        return frame


    def _draw_center_ice_emblem(self, frame: Array) -> None:
        """Draw a compact snowflake generator inside the center island.

        The emblem is visual-only.  It deliberately stays inside the inaccessible
        infield so the ice mode is obvious without adding observations on-track.
        """
        center = np.zeros(2, dtype=np.float32)
        inner = float(self.config.inner_hazard_radius)

        self._blend_disk(
            frame,
            center,
            inner * 0.76,
            np.array([106, 206, 239], dtype=np.uint8),
            0.17,
        )
        self._blend_ring(
            frame,
            center,
            outer_radius=inner * 0.69,
            inner_radius=inner * 0.59,
            color=np.array([199, 244, 255], dtype=np.uint8),
            alpha=0.76,
        )
        self._blend_ring(
            frame,
            center,
            outer_radius=inner * 0.43,
            inner_radius=inner * 0.37,
            color=np.array([83, 178, 226], dtype=np.uint8),
            alpha=0.68,
        )
        self._blend_disk(
            frame,
            center,
            inner * 0.16,
            np.array([225, 252, 255], dtype=np.uint8),
            0.93,
        )

        line_color = np.array([223, 251, 255], dtype=np.uint8)
        branch_color = np.array([126, 218, 246], dtype=np.uint8)
        thickness = max(2, self.render_size // 250)

        # Three diameters form a six-armed snowflake.
        for theta in (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0):
            axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            normal = np.array([-axis[1], axis[0]], dtype=np.float32)
            self._draw_line(
                frame,
                self._world_to_pixel(-axis * (inner * 0.51)),
                self._world_to_pixel(axis * (inner * 0.51)),
                line_color,
                thickness,
            )

            for sign in (-1.0, 1.0):
                outward = sign * axis
                anchor = outward * (inner * 0.29)
                for branch_sign in (-1.0, 1.0):
                    branch_end = (
                        anchor
                        + outward * (inner * 0.14)
                        + branch_sign * normal * (inner * 0.10)
                    )
                    self._draw_line(
                        frame,
                        self._world_to_pixel(anchor),
                        self._world_to_pixel(branch_end),
                        branch_color,
                        thickness,
                    )

    def _draw_ice_motion_effects(
        self,
        frame: Array,
        *,
        c: float,
        s: float,
    ) -> None:
        """Draw short skate marks and powder spray behind a moving ice car."""
        forward = np.array([c, s], dtype=np.float32)
        right = np.array([-s, c], dtype=np.float32)
        world_velocity = self.speed * forward + self.external_velocity
        speed = float(np.linalg.norm(world_velocity))
        if speed < 0.045:
            return

        travel = world_velocity / max(speed, 1e-8)
        trail = -travel
        speed_fraction = float(
            np.clip(
                speed / max(self.config.max_speed, 1e-8),
                0.0,
                1.0,
            )
        )
        tail = self.position - 0.48 * self.config.car_length * forward
        trail_length = 0.045 + 0.13 * speed_fraction

        # Twin dark grooves underneath a pale scraped-ice highlight.
        for side in (-1.0, 1.0):
            start = tail + side * (0.34 * self.config.car_width) * right
            end = start + trail_length * trail
            self._draw_line(
                frame,
                self._world_to_pixel(start),
                self._world_to_pixel(end),
                np.array([76, 137, 164], dtype=np.uint8),
                max(1, self.render_size // 360),
            )
            highlight_end = start + 0.72 * trail_length * trail
            self._draw_line(
                frame,
                self._world_to_pixel(start),
                self._world_to_pixel(highlight_end),
                np.array([194, 239, 249], dtype=np.uint8),
                max(1, self.render_size // 430),
            )

        lateral_motion = float(np.dot(world_velocity, right))
        spray_bias = np.sign(lateral_motion) * min(abs(lateral_motion) * 0.12, 0.022)
        phase = 0.17 * float(self.elapsed_steps)
        for i in range(7):
            distance = trail_length * (0.20 + 0.10 * i)
            side = -1.0 if i % 2 == 0 else 1.0
            lateral = (
                side * (0.007 + 0.0025 * i)
                + spray_bias
                + 0.004 * np.sin(phase + 1.7 * i)
            )
            center = tail + distance * trail + lateral * right
            radius = 0.006 + 0.002 * (i % 3) + 0.004 * speed_fraction
            self._blend_disk(
                frame,
                center,
                radius,
                np.array([220, 249, 255], dtype=np.uint8),
                0.18 + 0.04 * (6 - i),
            )

    def _draw_center_gravity_source(self, frame: Array) -> None:
        """Draw gravity only as a source indicator in the center island.

        Positive gravity:
            cool-colored core with arrows pointing inward.

        Negative gravity:
            warm-colored core with arrows pointing outward.

        This method modifies rendering only. It does not alter dynamics,
        rewards, observations, or environment state.
        """
        strength = float(self.config.gravity_strength)

        if abs(strength) <= 1e-12:
            return

        inward = strength > 0.0
        center = np.zeros(2, dtype=np.float32)
        inner = float(self.config.inner_hazard_radius)

        # Normalize visual intensity around the configured default |g| = 0.35.
        strength_fraction = float(
            np.clip(abs(strength) / 0.35, 0.0, 1.0)
        )

        if inward:
            # Gravity: dark well with cool rings.
            outer_glow = np.array([70, 168, 240], dtype=np.uint8)
            ring_color = np.array([104, 210, 255], dtype=np.uint8)
            core_color = np.array([20, 28, 67], dtype=np.uint8)
            arrow_color = np.array([205, 242, 255], dtype=np.uint8)
        else:
            # Anti-gravity: energized warm source.
            outer_glow = np.array([255, 116, 72], dtype=np.uint8)
            ring_color = np.array([255, 166, 82], dtype=np.uint8)
            core_color = np.array([93, 24, 54], dtype=np.uint8)
            arrow_color = np.array([255, 230, 166], dtype=np.uint8)

        # Keep every visual element strictly inside the center island.
        source_radius = inner * 0.80

        # Subtle outer source halo.
        self._blend_disk(
            frame,
            center,
            source_radius,
            outer_glow,
            0.08 + 0.08 * strength_fraction,
        )

        # Metallic/source boundary.
        self._blend_ring(
            frame,
            center,
            outer_radius=inner * 0.76,
            inner_radius=inner * 0.66,
            color=ring_color,
            alpha=0.58 + 0.20 * strength_fraction,
        )

        # Inner field rings.
        self._blend_ring(
            frame,
            center,
            outer_radius=inner * 0.56,
            inner_radius=inner * 0.50,
            color=ring_color,
            alpha=0.48 + 0.18 * strength_fraction,
        )

        self._blend_ring(
            frame,
            center,
            outer_radius=inner * 0.39,
            inner_radius=inner * 0.34,
            color=ring_color,
            alpha=0.42 + 0.18 * strength_fraction,
        )

        # Central core.
        self._blend_disk(
            frame,
            center,
            inner * 0.27,
            core_color,
            0.92,
        )

        self._blend_ring(
            frame,
            center,
            outer_radius=inner * 0.29,
            inner_radius=inner * 0.24,
            color=arrow_color,
            alpha=0.72,
        )

        # Four directional arrows contained within the center island.
        # The direction, not just the color, distinguishes gravity modes.
        for theta in np.linspace(
            0.0,
            2.0 * np.pi,
            4,
            endpoint=False,
        ):
            radial = np.array(
                [np.cos(theta), np.sin(theta)],
                dtype=np.float32,
            )

            if inward:
                # Arrows move from the outer source ring toward the core.
                start = radial * (inner * 0.63)
                end = radial * (inner * 0.38)
            else:
                # Arrows move from the core toward the outer source ring.
                start = radial * (inner * 0.34)
                end = radial * (inner * 0.62)

            self._draw_center_arrow(
                frame,
                start,
                end,
                arrow_color,
                thickness=max(2, self.render_size // 240),
            )

    def _draw_center_arrow(
        self,
        frame: Array,
        start: Array,
        end: Array,
        color: Array,
        thickness: int,
    ) -> None:
        """Draw a small arrow in world coordinates."""
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)

        displacement = end - start
        length = float(np.linalg.norm(displacement))

        if length <= 1e-8:
            return

        direction = displacement / length
        perpendicular = np.array(
            [-direction[1], direction[0]],
            dtype=np.float32,
        )

        self._draw_line(
            frame,
            self._world_to_pixel(start),
            self._world_to_pixel(end),
            color,
            thickness,
        )

        # Arrowhead size in world coordinates.
        head_length = min(0.021, length * 0.38)
        head_width = head_length * 0.60

        left = (
            end
            - head_length * direction
            + head_width * perpendicular
        )
        right = (
            end
            - head_length * direction
            - head_width * perpendicular
        )

        self._draw_line(
            frame,
            self._world_to_pixel(end),
            self._world_to_pixel(left),
            color,
            thickness,
        )

        self._draw_line(
            frame,
            self._world_to_pixel(end),
            self._world_to_pixel(right),
            color,
            thickness,
        )

    @staticmethod
    def _alpha_blend_mask(
        frame: Array,
        mask: Array,
        color: Array,
        alpha: float,
    ) -> None:
        if not np.any(mask):
            return
        a = float(np.clip(alpha, 0.0, 1.0))
        blended = (1.0 - a) * frame[mask].astype(np.float32) + a * color
        frame[mask] = np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _blend_rect(
        frame: Array,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: Array,
        alpha: float,
    ) -> None:
        height, width = frame.shape[:2]
        x0 = int(np.clip(x0, 0, width))
        x1 = int(np.clip(x1, 0, width))
        y0 = int(np.clip(y0, 0, height))
        y1 = int(np.clip(y1, 0, height))
        if x0 >= x1 or y0 >= y1:
            return
        a = float(np.clip(alpha, 0.0, 1.0))
        region = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = np.clip(
            (1.0 - a) * region + a * color,
            0,
            255,
        ).astype(np.uint8)

    def _blend_disk(
        self,
        frame: Array,
        center: Array,
        radius: float,
        color: Array,
        alpha: float,
    ) -> None:
        cx, cy = self._world_to_pixel(center)
        pixel_radius = max(
            1,
            int(round(radius * (self.render_size - 1) / (
                self.config.arena_high - self.config.arena_low
            ))),
        )
        yy, xx = np.ogrid[: self.render_size, : self.render_size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= pixel_radius**2
        self._alpha_blend_mask(frame, mask, color, alpha)

    def _blend_ring(
        self,
        frame: Array,
        center: Array,
        outer_radius: float,
        inner_radius: float,
        color: Array,
        alpha: float,
    ) -> None:
        cx, cy = self._world_to_pixel(center)
        scale = (self.render_size - 1) / (
            self.config.arena_high - self.config.arena_low
        )
        outer_px = max(1, int(round(outer_radius * scale)))
        inner_px = max(0, int(round(inner_radius * scale)))
        yy, xx = np.ogrid[: self.render_size, : self.render_size]
        distance_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = (distance_sq <= outer_px**2) & (distance_sq >= inner_px**2)
        self._alpha_blend_mask(frame, mask, color, alpha)

    def _draw_checkered_disk(
        self,
        frame: Array,
        center: Array,
        radius: float,
        color_a: Array,
        color_b: Array,
    ) -> None:
        cx, cy = self._world_to_pixel(center)
        scale = (self.render_size - 1) / (
            self.config.arena_high - self.config.arena_low
        )
        pixel_radius = max(2, int(round(radius * scale)))
        yy, xx = np.ogrid[: self.render_size, : self.render_size]
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= pixel_radius**2
        tile = max(2, pixel_radius // 3)
        checker = ((xx // tile) + (yy // tile)) % 2 == 0
        frame[circle & checker] = color_a
        frame[circle & ~checker] = color_b

    @staticmethod
    def _draw_pixel_disk(
        frame: Array,
        center: tuple[int, int],
        radius: int,
        color: Array,
    ) -> None:
        cx, cy = center
        height, width = frame.shape[:2]
        yy, xx = np.ogrid[:height, :width]
        frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = color

    def _draw_disk(
        self, frame: Array, center: Array, radius: float, color: Array
    ) -> None:
        cx, cy = self._world_to_pixel(center)
        pixel_radius = max(
            1,
            int(
                round(
                    radius
                    * (self.render_size - 1)
                    / (self.config.arena_high - self.config.arena_low)
                )
            ),
        )
        yy, xx = np.ogrid[: self.render_size, : self.render_size]
        frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= pixel_radius**2] = color

    def _render_human(self, frame: Array) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("render_mode='human' requires matplotlib") from exc
        if self._human_figure is None:
            plt.ion()
            self._human_figure, self._human_axis = plt.subplots(figsize=(6, 6))
            self._human_image = self._human_axis.imshow(frame)
            self._human_axis.set_axis_off()
            self._human_figure.tight_layout(pad=0)
        else:
            self._human_image.set_data(frame)
        self._human_figure.canvas.draw_idle()
        self._human_figure.canvas.flush_events()
        plt.pause(1.0 / self.metadata["render_fps"])

    def _world_to_pixel(self, point: Array) -> tuple[int, int]:
        scale = (self.render_size - 1) / (
            self.config.arena_high - self.config.arena_low
        )
        col = int(round((float(point[0]) - self.config.arena_low) * scale))
        row = int(round((self.config.arena_high - float(point[1])) * scale))
        return (
            int(np.clip(col, 0, self.render_size - 1)),
            int(np.clip(row, 0, self.render_size - 1)),
        )

    @staticmethod
    def _draw_line(
        image: Array,
        start: tuple[int, int],
        end: tuple[int, int],
        color: Array,
        thickness: int,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(int)
        ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(int)
        radius = max(0, thickness // 2)
        height, width = image.shape[:2]
        for x, y in zip(xs, ys, strict=False):
            image[
                max(0, y - radius) : min(height, y + radius + 1),
                max(0, x - radius) : min(width, x + radius + 1),
            ] = color


def register_environment() -> None:
    """Register plain, grav, anti-grav, and ice task variants."""
    from gymnasium.envs.registration import register, registry

    # (task_mode, mode, checkpoint_count)
    registrations: dict[str, tuple[str, CarRaceMode, int]] = {
        "CarRaceNavigation-v0": ("navigation", "car_race_plain", 8),
        "CarRacePlainNavigation-v0": ("navigation", "car_race_plain", 8),
        "CarRaceGravNavigation-v0": ("navigation", "car_race_grav", 8),
        "CarRaceAntiGravNavigation-v0": (
            "navigation",
            "car_race_anti_grav",
            8,
        ),
        "CarRaceIceNavigation-v0": ("navigation", "car_race_ice", 8),
        # Np => N waypoints besides spawn; ring size / required passes = N+1.
        "CarRaceLap-v0": ("lap", "car_race_plain", 9),
        "CarRaceLap1p-v0": ("lap", "car_race_plain", 2),
        "CarRaceLap2p-v0": ("lap", "car_race_plain", 3),
        "CarRaceLap4p-v0": ("lap", "car_race_plain", 5),
        "CarRaceLap8p-v0": ("lap", "car_race_plain", 9),
        "CarRacePlainLap1p-v0": ("lap", "car_race_plain", 2),
        "CarRacePlainLap2p-v0": ("lap", "car_race_plain", 3),
        "CarRacePlainLap4p-v0": ("lap", "car_race_plain", 5),
        "CarRacePlainLap8p-v0": ("lap", "car_race_plain", 9),
        "CarRaceGravLap1p-v0": ("lap", "car_race_grav", 2),
        "CarRaceGravLap2p-v0": ("lap", "car_race_grav", 3),
        "CarRaceGravLap4p-v0": ("lap", "car_race_grav", 5),
        "CarRaceGravLap8p-v0": ("lap", "car_race_grav", 9),
        "CarRaceAntiGravLap1p-v0": ("lap", "car_race_anti_grav", 2),
        "CarRaceAntiGravLap2p-v0": ("lap", "car_race_anti_grav", 3),
        "CarRaceAntiGravLap4p-v0": ("lap", "car_race_anti_grav", 5),
        "CarRaceAntiGravLap8p-v0": ("lap", "car_race_anti_grav", 9),
        "CarRaceIceLap1p-v0": ("lap", "car_race_ice", 2),
        "CarRaceIceLap2p-v0": ("lap", "car_race_ice", 3),
        "CarRaceIceLap4p-v0": ("lap", "car_race_ice", 5),
        "CarRaceIceLap8p-v0": ("lap", "car_race_ice", 9),
    }
    prefixes: dict[str, CarRaceMode] = {
        "CarRace": "car_race_plain",
        "CarRacePlain": "car_race_plain",
        "CarRaceGrav": "car_race_grav",
        "CarRaceAntiGrav": "car_race_anti_grav",
        "CarRaceIce": "car_race_ice",
    }
    for prefix, mode in prefixes.items():
        for waypoint_count in range(1, 9):
            registrations[f"{prefix}Lap{waypoint_count}p-v0"] = (
                "lap",
                mode,
                waypoint_count + 1,
            )

    for env_id, (task_mode, mode, checkpoint_count) in registrations.items():
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="car_race.env:CarRaceEnv",
            kwargs={
                "config": CarRaceConfig(
                    task_mode=task_mode,  # type: ignore[arg-type]
                    checkpoint_count=checkpoint_count,
                    **mode_config_kwargs(mode),
                )
            },
            max_episode_steps=None,
        )
