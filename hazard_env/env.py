"""Continuous 2-D point-mass environment with a lethal circular hazard.

Workspace
---------
The agent's center moves in the continuous square [-1, 1]^2. The square boundary
is a hard wall. Entering the circular hazard terminates the episode immediately.

State
-----
    position: (x, y)
    velocity: (vx, vy)

Action
------
    action[0]: thrust angle in radians, in [-pi, pi]
    action[1]: thrust magnitude, in [0, 1]

The environment follows the Gymnasium API. It can expose either a flat state-goal
observation or a goal-conditioned Dict observation suitable for hindsight relabeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Array = np.ndarray
ObservationMode = Literal["state", "state_goal", "goal_dict"]
RewardMode = Literal["sparse", "dense"]
TaskMode = Literal["random", "cross_hazard"]


@dataclass(frozen=True)
class Hazard2DConfig:
    """Physical and task parameters for :class:`ContinuousHazard2DEnv`."""

    # Geometry.
    arena_low: float = -1.0
    arena_high: float = 1.0
    hazard_center: tuple[float, float] = (0.05, -0.03)
    hazard_radius: float = 0.17
    agent_radius: float = 0.025
    goal_radius: float = 0.08

    # Point-mass dynamics.
    dt: float = 0.05
    physics_substeps: int = 5
    max_acceleration: float = 3.0
    max_speed: float = 1.0
    linear_drag: float = 1.1
    wall_restitution: float = 0.25

    # Episode and task sampling.
    max_episode_steps: int = 300
    min_start_goal_distance: float = 0.9
    spawn_clearance: float = 0.08
    task_mode: TaskMode = "random"

    # Reward.
    reward_mode: RewardMode = "sparse"
    step_penalty: float = 0.01
    control_cost: float = 0.002
    progress_scale: float = 1.0
    success_reward: float = 1.0
    death_penalty: float = -1.0

    def validate(self) -> None:
        if not self.arena_low < self.arena_high:
            raise ValueError("arena_low must be smaller than arena_high")
        if self.hazard_radius <= 0.0:
            raise ValueError("hazard_radius must be positive")
        if self.agent_radius <= 0.0:
            raise ValueError("agent_radius must be positive")
        if self.goal_radius <= 0.0:
            raise ValueError("goal_radius must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.physics_substeps < 1:
            raise ValueError("physics_substeps must be at least 1")
        if self.max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be positive")
        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if self.linear_drag < 0.0:
            raise ValueError("linear_drag must be non-negative")
        if not 0.0 <= self.wall_restitution <= 1.0:
            raise ValueError("wall_restitution must be in [0, 1]")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")
        if self.min_start_goal_distance < 0.0:
            raise ValueError("min_start_goal_distance must be non-negative")
        if self.spawn_clearance < 0.0:
            raise ValueError("spawn_clearance must be non-negative")
        if self.reward_mode not in ("sparse", "dense"):
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
        if self.task_mode not in ("random", "cross_hazard"):
            raise ValueError(f"Unknown task_mode: {self.task_mode}")

        c = np.asarray(self.hazard_center, dtype=np.float64)
        if c.shape != (2,):
            raise ValueError("hazard_center must have two coordinates")

        inner_low = self.arena_low + self.agent_radius
        inner_high = self.arena_high - self.agent_radius
        if np.any(c - self.hazard_radius < inner_low) or np.any(
            c + self.hazard_radius > inner_high
        ):
            raise ValueError("The hazard must fit completely inside the arena")


class ContinuousHazard2DEnv(gym.Env):
    """A continuous point-mass navigation environment with a lethal hazard.

    Parameters
    ----------
    config:
        Dynamics, geometry, reward, and sampling parameters.
    observation_mode:
        ``"state"`` returns ``[x, y, vx, vy]``.
        ``"state_goal"`` returns ``[x, y, vx, vy, gx, gy]``.
        ``"goal_dict"`` returns a Gymnasium goal dictionary with keys
        ``observation``, ``achieved_goal``, and ``desired_goal``.
    render_mode:
        ``None``, ``"rgb_array"``, or ``"human"``.
    render_size:
        Width and height of the square RGB frame.
    terminate_at_goal:
        If ``True`` (default), reaching the goal terminates the episode.
        If ``False``, success is reported in ``info`` only so a collector can
        resample a new goal mid-episode (OGBench-style navigate).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        config: Hazard2DConfig | None = None,
        observation_mode: ObservationMode = "state_goal",
        render_mode: str | None = None,
        render_size: int = 512,
        terminate_at_goal: bool = True,
    ) -> None:
        super().__init__()

        self.config = config or Hazard2DConfig()
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
        self.terminate_at_goal = bool(terminate_at_goal)

        cfg = self.config
        self._dtype = np.float32
        self._hazard_center = np.asarray(cfg.hazard_center, dtype=self._dtype)
        self._inner_low = float(cfg.arena_low + cfg.agent_radius)
        self._inner_high = float(cfg.arena_high - cfg.agent_radius)

        # Physical action units: angle [rad], normalized thrust magnitude [0, 1].
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0], dtype=self._dtype),
            high=np.array([np.pi, 1.0], dtype=self._dtype),
            dtype=self._dtype,
        )

        state_low = np.array(
            [cfg.arena_low, cfg.arena_low, -cfg.max_speed, -cfg.max_speed],
            dtype=self._dtype,
        )
        state_high = np.array(
            [cfg.arena_high, cfg.arena_high, cfg.max_speed, cfg.max_speed],
            dtype=self._dtype,
        )
        goal_low = np.full(2, cfg.arena_low, dtype=self._dtype)
        goal_high = np.full(2, cfg.arena_high, dtype=self._dtype)

        if observation_mode == "state":
            self.observation_space = spaces.Box(
                low=state_low,
                high=state_high,
                dtype=self._dtype,
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
                        low=state_low,
                        high=state_high,
                        dtype=self._dtype,
                    ),
                    "achieved_goal": spaces.Box(
                        low=goal_low,
                        high=goal_high,
                        dtype=self._dtype,
                    ),
                    "desired_goal": spaces.Box(
                        low=goal_low,
                        high=goal_high,
                        dtype=self._dtype,
                    ),
                }
            )

        self.position = np.zeros(2, dtype=self._dtype)
        self.velocity = np.zeros(2, dtype=self._dtype)
        self.goal = np.zeros(2, dtype=self._dtype)
        self.elapsed_steps = 0
        self.dead = False
        self.success = False
        self.cur_task_id: int | None = None
        self.cur_task_info: dict[str, Any] | None = None

        # Lazily-created Matplotlib objects for human rendering.
        self._human_figure: Any | None = None
        self._human_axis: Any | None = None
        self._human_image: Any | None = None

        self.set_tasks()
        self.num_tasks = len(self.task_infos)
    @property
    def state(self) -> Array:
        """Return a copy of ``[x, y, vx, vy]``."""
        return np.concatenate([self.position, self.velocity]).astype(
            self._dtype, copy=True
        )

    @property
    def hazard_center(self) -> Array:
        return self._hazard_center.copy()

    @property
    def hazard_radius(self) -> float:
        return float(self.config.hazard_radius)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment.

        Supported ``options`` keys
        --------------------------
        ``position``:
            Explicit initial position, shape ``(2,)``.
        ``velocity``:
            Explicit initial velocity, shape ``(2,)``. Defaults to zero.
        ``goal``:
            Explicit goal position, shape ``(2,)``.
        ``task_mode``:
            Override the configured sampling mode for this reset.
        ``task_id``:
            Integer in ``[1, num_tasks]`` selecting a fixed evaluation task.
            Mutually exclusive with explicit ``position`` / ``goal``.
        """
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
                raise ValueError(f"task_id must be in [1, {self.num_tasks}], got {task_id}")
            self.cur_task_id = task_id
            self.cur_task_info = dict(self.task_infos[task_id - 1])
            position = self._validate_safe_point(
                self.cur_task_info["init_xy"], "task init_xy"
            )
            goal = self._validate_safe_point(
                self.cur_task_info["goal_xy"], "task goal_xy"
            )
            if np.linalg.norm(position - goal) < self.config.goal_radius:
                raise ValueError("Task init coincides with the goal region")
        else:
            self.cur_task_id = None
            self.cur_task_info = None

            task_mode = options.get("task_mode", self.config.task_mode)
            if task_mode not in ("random", "cross_hazard"):
                raise ValueError(f"Unknown task_mode: {task_mode}")

            if explicit_position is None and explicit_goal is None:
                position, goal = self._sample_task(task_mode)
            elif explicit_position is None:
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
                if np.linalg.norm(position - goal) < self.config.goal_radius:
                    raise ValueError("Initial position is already inside the goal region")

        velocity = np.asarray(
            options.get("velocity", np.zeros(2)), dtype=self._dtype
        )
        if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity must be a finite vector with shape (2,)")
        velocity = self._clip_speed(velocity)

        self.position = position.astype(self._dtype, copy=True)
        self.velocity = velocity.astype(self._dtype, copy=True)
        self.goal = goal.astype(self._dtype, copy=True)
        self.elapsed_steps = 0
        self.dead = False
        self.success = False

        observation = self._get_observation()
        info = self._get_info(termination_reason=None)

        if self.render_mode == "human":
            self.render()
        return observation, info

    def set_tasks(self) -> None:
        """Define fixed evaluation tasks ordered by increasing difficulty."""
        # (init_xy, goal_xy, short_name, difficulty)
        tasks = [
            (
                (-0.70, 0.55),
                (-0.25, 0.60),
                "same_side_short",
                "easy",
            ),
            (
                (-0.75, 0.55),
                (0.75, 0.60),
                "same_side_long",
                "easy_medium",
            ),
            (
                (-0.75, 0.12),
                (0.75, 0.08),
                "skim_detour",
                "medium",
            ),
            (
                (-0.75, -0.05),
                (0.75, -0.05),
                "cross_hazard",
                "hard",
            ),
            (
                (-0.80, -0.70),
                (0.80, 0.70),
                "cross_corners",
                "hardest",
            ),
        ]
        self.task_infos = []
        for i, (init_xy, goal_xy, name, difficulty) in enumerate(tasks):
            self.task_infos.append(
                {
                    "task_name": f"task{i + 1}_{name}",
                    "init_xy": np.asarray(init_xy, dtype=np.float32),
                    "goal_xy": np.asarray(goal_xy, dtype=np.float32),
                    "difficulty": difficulty,
                }
            )
    def step(
        self, action: Array
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        episode_over = (
            self.dead
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
        thrust = float(action[1])
        acceleration = (
            self.config.max_acceleration
            * thrust
            * np.array([np.cos(angle), np.sin(angle)], dtype=self._dtype)
        )

        previous_distance = self.distance_to_goal
        collided_with_wall = False
        death = False

        sub_dt = self.config.dt / self.config.physics_substeps
        drag_factor = float(np.exp(-self.config.linear_drag * sub_dt))

        for _ in range(self.config.physics_substeps):
            # Semi-implicit Euler integration.
            self.velocity = self.velocity + acceleration * sub_dt
            self.velocity = self.velocity * drag_factor
            self.velocity = self._clip_speed(self.velocity)

            old_position = self.position.copy()
            candidate_position = old_position + self.velocity * sub_dt

            # Continuous segment-circle test avoids tunneling through the hazard.
            lethal_radius = self.config.hazard_radius + self.config.agent_radius
            if self._segment_hits_circle(
                old_position,
                candidate_position,
                self._hazard_center,
                lethal_radius,
            ):
                self.position = candidate_position.astype(self._dtype, copy=False)
                death = True
                break

            candidate_position, wall_hit = self._resolve_wall_collision(
                candidate_position
            )
            collided_with_wall = collided_with_wall or wall_hit
            self.position = candidate_position.astype(self._dtype, copy=False)

        self.elapsed_steps += 1
        self.dead = bool(death)
        self.success = bool(
            not self.dead and self.distance_to_goal <= self.config.goal_radius
        )

        terminated = self.dead or (self.success and self.terminate_at_goal)
        truncated = self.elapsed_steps >= self.config.max_episode_steps and not terminated

        reward = self._compute_step_reward(
            previous_distance=previous_distance,
            thrust=thrust,
        )

        if self.dead:
            termination_reason = "hazard"
        elif self.success and self.terminate_at_goal:
            termination_reason = "goal"
        elif truncated:
            termination_reason = "time_limit"
        else:
            termination_reason = None

        info = self._get_info(termination_reason=termination_reason)
        info["wall_collision"] = collided_with_wall
        info["action_angle"] = angle
        info["action_force"] = thrust

        observation = self._get_observation()
        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    def set_goal(self, goal: Array) -> None:
        """Replace the desired goal without resetting the agent state."""
        validated = self._validate_safe_point(goal, "goal")
        if np.linalg.norm(validated - self.position) < self.config.goal_radius:
            raise ValueError("New goal coincides with the current agent position")
        self.goal = validated.astype(self._dtype, copy=True)
        self.success = False

    def sample_safe_point(self, **kwargs: Any) -> Array:
        """Public wrapper around safe-point sampling (for dataset collectors)."""
        return self._sample_safe_point(**kwargs)

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.position - self.goal))

    @property
    def signed_distance_to_hazard(self) -> float:
        """Agent-surface to hazard-surface signed distance.

        Positive means separated; zero means contact; negative means overlap.
        """
        center_distance = np.linalg.norm(self.position - self._hazard_center)
        return float(
            center_distance
            - self.config.hazard_radius
            - self.config.agent_radius
        )

    def compute_reward(
        self,
        achieved_goal: Array,
        desired_goal: Array,
        info: Mapping[str, Any] | np.ndarray | list | tuple | None = None,
    ) -> Array | float:
        """Vectorized goal reward helper for hindsight relabeling.

        Evaluates goal achievement from ``achieved_goal`` / ``desired_goal``.
        Hazard deaths can be injected via ``info``:

        * mapping with ``dead`` (scalar)
        * sequence of mappings, one per batch row
        * boolean array aligned with the batch
        """
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        distances = np.linalg.norm(achieved - desired, axis=-1)
        rewards = np.where(
            distances <= self.config.goal_radius,
            self.config.success_reward,
            -self.config.step_penalty,
        ).astype(np.float32)

        dead_mask = self._dead_mask_from_info(info, batch_shape=rewards.shape)
        if dead_mask is not None:
            rewards = np.where(dead_mask, self.config.death_penalty, rewards).astype(
                np.float32
            )

        if rewards.ndim == 0:
            return float(rewards)
        return rewards

    @staticmethod
    def _dead_mask_from_info(
        info: Mapping[str, Any] | np.ndarray | list | tuple | None,
        *,
        batch_shape: tuple[int, ...],
    ) -> Array | None:
        if info is None:
            return None

        if isinstance(info, Mapping):
            dead = bool(info.get("dead", False))
            if batch_shape == ():
                return np.asarray(dead, dtype=bool)
            return np.full(batch_shape, dead, dtype=bool)

        if isinstance(info, np.ndarray):
            mask = np.asarray(info, dtype=bool)
            if mask.shape != batch_shape:
                raise ValueError(
                    f"info dead mask shape {mask.shape} != batch shape {batch_shape}"
                )
            return mask

        if isinstance(info, (list, tuple)):
            if batch_shape == ():
                raise ValueError("sequence info is only valid for batched rewards")
            n = int(np.prod(batch_shape)) if batch_shape else 1
            if len(info) != n and len(info) != batch_shape[0]:
                raise ValueError(
                    f"info length {len(info)} incompatible with batch shape {batch_shape}"
                )
            if len(info) == batch_shape[0] and len(batch_shape) == 1:
                return np.asarray(
                    [bool(item.get("dead", False)) if isinstance(item, Mapping) else bool(item)
                     for item in info],
                    dtype=bool,
                )
            # Flattened sequence matching prod(batch_shape).
            vals = [
                bool(item.get("dead", False)) if isinstance(item, Mapping) else bool(item)
                for item in info
            ]
            return np.asarray(vals, dtype=bool).reshape(batch_shape)

        raise TypeError(f"Unsupported info type for compute_reward: {type(info)!r}")

    def render(self) -> Array | None:
        frame = self._render_rgb()
        if self.render_mode == "rgb_array":
            return frame
        if self.render_mode == "human":
            self._render_human(frame)
            return None
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

    # ------------------------------------------------------------------
    # Sampling and validation
    # ------------------------------------------------------------------

    def _sample_task(self, task_mode: TaskMode) -> tuple[Array, Array]:
        lethal_radius = self.config.hazard_radius + self.config.agent_radius
        for _ in range(10_000):
            if task_mode == "random":
                position = self._sample_safe_point()
                goal = self._sample_safe_point()
            else:
                # Opposite sides + y near the hazard so the straight chord is lethal,
                # while northern/southern detours remain feasible.
                position = self._sample_safe_point(region="left", near_hazard_y=True)
                goal = self._sample_safe_point(region="right", near_hazard_y=True)

            if (
                np.linalg.norm(position - goal)
                < self.config.min_start_goal_distance
            ):
                continue
            if task_mode == "cross_hazard" and not self._segment_hits_circle(
                position,
                goal,
                self._hazard_center,
                lethal_radius,
            ):
                continue
            return position, goal
        raise RuntimeError("Could not sample a valid start-goal pair")

    def _sample_safe_point(
        self,
        *,
        region: Literal["left", "right"] | None = None,
        near_hazard_y: bool = False,
        min_distance_from: Array | None = None,
        minimum_distance: float = 0.0,
    ) -> Array:
        margin = self.config.agent_radius + self.config.spawn_clearance
        low = self.config.arena_low + margin
        high = self.config.arena_high - margin
        hazard_y = float(self._hazard_center[1])

        for _ in range(10_000):
            if region is None:
                point = self.np_random.uniform(low, high, size=2)
            else:
                if region == "left":
                    x_low, x_high = low, min(-0.45, high)
                else:
                    x_low, x_high = max(0.45, low), high
                if x_low >= x_high:
                    raise ValueError("Arena is too small for cross_hazard sampling")
                if near_hazard_y:
                    # Tight vertical band around the hazard; still room to detour.
                    y_low = max(low, hazard_y - 0.28)
                    y_high = min(high, hazard_y + 0.28)
                else:
                    y_low, y_high = -0.65, 0.65
                    y_low = max(low, y_low)
                    y_high = min(high, y_high)
                if y_low >= y_high:
                    y_low, y_high = low, high
                point = np.array(
                    [
                        self.np_random.uniform(x_low, x_high),
                        self.np_random.uniform(y_low, y_high),
                    ]
                )

            point = np.asarray(point, dtype=self._dtype)
            if not self._is_safe_spawn(point):
                continue
            if min_distance_from is not None and (
                np.linalg.norm(point - min_distance_from) < minimum_distance
            ):
                continue
            return point

        raise RuntimeError("Could not sample a safe point")

    def _validate_safe_point(self, point: Any, name: str) -> Array:
        point = np.asarray(point, dtype=self._dtype)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{name} must be a finite vector with shape (2,)")
        if not self._is_safe_spawn(point):
            raise ValueError(f"{name} is outside the arena or too close to the hazard")
        return point

    def _is_safe_spawn(self, point: Array) -> bool:
        margin = self.config.agent_radius + self.config.spawn_clearance
        if np.any(point < self.config.arena_low + margin) or np.any(
            point > self.config.arena_high - margin
        ):
            return False

        hazard_clearance = (
            self.config.hazard_radius
            + self.config.agent_radius
            + self.config.spawn_clearance
        )
        return bool(np.linalg.norm(point - self._hazard_center) > hazard_clearance)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _clip_speed(self, velocity: Array) -> Array:
        speed = float(np.linalg.norm(velocity))
        if speed > self.config.max_speed:
            velocity = velocity * (self.config.max_speed / speed)
        return np.asarray(velocity, dtype=self._dtype)

    def _resolve_wall_collision(self, candidate: Array) -> tuple[Array, bool]:
        candidate = np.asarray(candidate, dtype=self._dtype).copy()
        hit = False
        for axis in range(2):
            if candidate[axis] < self._inner_low:
                candidate[axis] = self._inner_low
                if self.velocity[axis] < 0.0:
                    self.velocity[axis] *= -self.config.wall_restitution
                hit = True
            elif candidate[axis] > self._inner_high:
                candidate[axis] = self._inner_high
                if self.velocity[axis] > 0.0:
                    self.velocity[axis] *= -self.config.wall_restitution
                hit = True
        return candidate, hit

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
    # Reward and observations
    # ------------------------------------------------------------------

    def _compute_step_reward(
        self,
        *,
        previous_distance: float,
        thrust: float,
    ) -> float:
        if self.dead:
            return self.config.death_penalty
        if self.success:
            return self.config.success_reward

        reward = -self.config.step_penalty
        reward -= self.config.control_cost * thrust * thrust

        if self.config.reward_mode == "dense":
            progress = previous_distance - self.distance_to_goal
            reward += self.config.progress_scale * progress
        return float(reward)

    def _get_observation(self) -> Any:
        state = self.state
        if self.observation_mode == "state":
            return state
        if self.observation_mode == "state_goal":
            return np.concatenate([state, self.goal]).astype(
                self._dtype, copy=False
            )
        return {
            "observation": state,
            "achieved_goal": self.position.astype(self._dtype, copy=True),
            "desired_goal": self.goal.astype(self._dtype, copy=True),
        }

    def _get_info(self, termination_reason: str | None) -> dict[str, Any]:
        return {
            "is_success": bool(self.success),
            "dead": bool(self.dead),
            "termination_reason": termination_reason,
            "distance_to_goal": self.distance_to_goal,
            "signed_distance_to_hazard": self.signed_distance_to_hazard,
            "elapsed_steps": int(self.elapsed_steps),
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "goal": self.goal.copy(),
            "hazard_center": self._hazard_center.copy(),
            "hazard_radius": float(self.config.hazard_radius),
            "task_id": self.cur_task_id,
            "task_name": (
                None
                if self.cur_task_info is None
                else self.cur_task_info.get("task_name")
            ),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_rgb(self) -> Array:
        size = self.render_size
        frame = np.full((size, size, 3), 247, dtype=np.uint8)

        # Coordinate grid in environment units; y is flipped for image rows.
        xs = np.linspace(
            self.config.arena_low,
            self.config.arena_high,
            size,
            dtype=np.float32,
        )
        ys = np.linspace(
            self.config.arena_high,
            self.config.arena_low,
            size,
            dtype=np.float32,
        )
        xx, yy = np.meshgrid(xs, ys)

        # Hazard and goal disks.
        hazard_mask = (
            (xx - self._hazard_center[0]) ** 2
            + (yy - self._hazard_center[1]) ** 2
            <= self.config.hazard_radius**2
        )
        frame[hazard_mask] = np.array([225, 55, 55], dtype=np.uint8)

        goal_mask = (
            (xx - self.goal[0]) ** 2 + (yy - self.goal[1]) ** 2
            <= self.config.goal_radius**2
        )
        frame[goal_mask] = np.array([70, 180, 95], dtype=np.uint8)

        agent_mask = (
            (xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2
            <= self.config.agent_radius**2
        )
        agent_color = [40, 90, 210] if not self.dead else [60, 60, 60]
        frame[agent_mask] = np.array(agent_color, dtype=np.uint8)

        # Arena border.
        border = max(2, size // 180)
        frame[:border] = 30
        frame[-border:] = 30
        frame[:, :border] = 30
        frame[:, -border:] = 30

        # Velocity arrow.
        start = self._world_to_pixel(self.position)
        velocity_endpoint = self.position + 0.20 * self.velocity
        end = self._world_to_pixel(velocity_endpoint)
        self._draw_line(frame, start, end, np.array([25, 25, 25], dtype=np.uint8), 2)
        return frame

    def _render_human(self, frame: Array) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "render_mode='human' requires matplotlib; install it with "
                "`pip install matplotlib`."
            ) from exc

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
        col = int(round((point[0] - self.config.arena_low) * scale))
        row = int(round((self.config.arena_high - point[1]) * scale))
        col = int(np.clip(col, 0, self.render_size - 1))
        row = int(np.clip(row, 0, self.render_size - 1))
        return col, row

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
            x_lo = max(0, x - radius)
            x_hi = min(width, x + radius + 1)
            y_lo = max(0, y - radius)
            y_hi = min(height, y + radius + 1)
            image[y_lo:y_hi, x_lo:x_hi] = color


def register_environment() -> None:
    """Register ``ContinuousHazard2D-v0`` with Gymnasium once.

    Requires ``toy_examples`` (the parent of this package) to be on ``PYTHONPATH``::

        export PYTHONPATH=/path/to/toy_examples:$PYTHONPATH
        python -c "from hazard_env import register_environment; register_environment()"
    """
    from gymnasium.envs.registration import register, registry

    environment_id = "ContinuousHazard2D-v0"
    if environment_id not in registry:
        register(
            id=environment_id,
            entry_point="hazard_env.env:ContinuousHazard2DEnv",
            kwargs={"observation_mode": "state_goal"},
            max_episode_steps=None,  # env handles its own time limit via truncated
        )


def _smoke_test() -> None:
    """Run Gymnasium's checker plus deterministic hazard/success rollouts."""
    from gymnasium.utils.env_checker import check_env

    env = ContinuousHazard2DEnv(observation_mode="goal_dict")
    check_env(env, skip_render_check=True)

    # A direct thrust through the central hazard must terminate as death.
    env.reset(
        seed=0,
        options={
            "position": np.array([-0.75, -0.03], dtype=np.float32),
            "goal": np.array([0.75, -0.03], dtype=np.float32),
        },
    )
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )
    assert info["dead"] and info["termination_reason"] == "hazard"

    # A safe upper route should be able to avoid the hazard and reach the goal.
    env.reset(
        seed=1,
        options={
            "position": np.array([-0.75, -0.45], dtype=np.float32),
            "goal": np.array([0.75, 0.45], dtype=np.float32),
        },
    )
    waypoints = [
        np.array([-0.35, 0.48], dtype=np.float32),
        np.array([0.35, 0.58], dtype=np.float32),
        env.goal.copy(),
    ]
    terminated = truncated = False
    for waypoint in waypoints:
        for _ in range(180):
            delta = waypoint - env.position
            tolerance = 0.04 if np.array_equal(waypoint, env.goal) else 0.10
            if np.linalg.norm(delta) < tolerance:
                break
            angle = np.arctan2(delta[1], delta[0])
            _, _, terminated, truncated, info = env.step(
                np.array([angle, 0.75], dtype=np.float32)
            )
            if terminated or truncated:
                break
        if terminated or truncated:
            break

    assert not info.get("dead", False), "Safe rollout unexpectedly hit the hazard"
    assert info.get("is_success", False), "Safe rollout did not reach the goal"

    # terminate_at_goal=False keeps the episode alive after success.
    nav = ContinuousHazard2DEnv(
        observation_mode="state",
        terminate_at_goal=False,
        config=Hazard2DConfig(max_episode_steps=80),
    )
    nav.reset(
        seed=2,
        options={
            "position": np.array([-0.6, 0.55], dtype=np.float32),
            "goal": np.array([-0.35, 0.55], dtype=np.float32),
        },
    )
    for _ in range(60):
        delta = nav.goal - nav.position
        angle = float(np.arctan2(delta[1], delta[0]))
        _, _, terminated, truncated, info = nav.step(
            np.array([angle, 0.55], dtype=np.float32)
        )
        if info["is_success"]:
            assert not terminated
            nav.set_goal(np.array([0.55, 0.55], dtype=np.float32))
            assert not nav.success
            break
        if terminated or truncated:
            break
    else:
        raise AssertionError("navigate-style goal reach failed")
    nav.close()

    # Fixed evaluation tasks 1–5: safe spawns + chord difficulty ladder.
    tasks_env = ContinuousHazard2DEnv(observation_mode="state")
    assert tasks_env.num_tasks == 5
    lethal = tasks_env.config.hazard_radius + tasks_env.config.agent_radius
    expected_hits = {1: False, 2: False, 3: True, 4: True, 5: True}
    for task_id, should_hit in expected_hits.items():
        _, info = tasks_env.reset(seed=0, options={"task_id": task_id})
        assert info["task_id"] == task_id
        assert info["task_name"] is not None
        assert np.allclose(info["position"], tasks_env.task_infos[task_id - 1]["init_xy"])
        assert np.allclose(info["goal"], tasks_env.task_infos[task_id - 1]["goal_xy"])
        hits = ContinuousHazard2DEnv._segment_hits_circle(
            info["position"],
            info["goal"],
            info["hazard_center"],
            lethal,
        )
        assert hits == should_hit, (
            f"task {task_id} chord hit={hits}, expected {should_hit}"
        )
    try:
        tasks_env.reset(options={"task_id": 1, "position": np.zeros(2)})
        raise AssertionError("task_id+position should raise")
    except ValueError:
        pass
    tasks_env.close()

    # cross_hazard samples must have a lethal straight chord.
    cross = ContinuousHazard2DEnv(
        config=Hazard2DConfig(task_mode="cross_hazard"),
        observation_mode="state_goal",
    )
    for seed in range(8):
        _, info = cross.reset(seed=seed)
        lethal = cross.config.hazard_radius + cross.config.agent_radius
        assert ContinuousHazard2DEnv._segment_hits_circle(
            info["position"],
            info["goal"],
            info["hazard_center"],
            lethal,
        ), f"cross_hazard seed={seed} chord misses hazard"

    # Batched HER reward keeps shape when injecting deaths.
    ach = np.zeros((4, 2), dtype=np.float32)
    des = np.full((4, 2), 0.5, dtype=np.float32)
    dead_info = [{"dead": i == 1} for i in range(4)]
    rewards = env.compute_reward(ach, des, info=dead_info)
    assert np.shape(rewards) == (4,)
    assert float(rewards[1]) == env.config.death_penalty

    env.close()
    cross.close()
    print("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
