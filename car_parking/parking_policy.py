"""Path tracking expert policy that drives only through ``env.step``."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .env import CarParkingEnv, _wrap_angle
from .hybrid_astar import HybridAStarPlanner, PathPoint, PlannerConfig


@dataclass
class RolloutResult:
    task_id: int | None
    success: bool
    dead: bool
    collision: bool
    timeout: bool
    minimum_health: float
    total_health_loss: float
    steps: int
    replans: int
    path_points: int
    actions: list[np.ndarray]
    infos: list[dict[str, Any]]


class ParkingExpertPolicy:
    """Pure-pursuit path tracker with explicit braking and gear changes."""

    def __init__(
        self,
        env: CarParkingEnv,
        *,
        planner_config: PlannerConfig | None = None,
        allow_replan: bool = True,
    ) -> None:
        self.env = env
        self.planner_config = planner_config
        self.allow_replan = allow_replan
        self.path: list[PathPoint] = []
        self.index = 0
        self.replans = 0
        self._off_path_steps = 0
        self._segment_progress = 0.0
        self._last_position = self.env.position.copy()
        self._final_gear: int | None = None

    def reset(
        self, path: list[PathPoint] | None = None
    ) -> list[PathPoint]:
        """Plan a path or resynchronize the tracker to a supplied path."""
        self.path = (
            list(path)
            if path is not None
            else HybridAStarPlanner(self.env, self.planner_config).plan()
        )
        if not self.path:
            raise ValueError("parking expert path must not be empty")
        self.index = 0
        self.replans = 0
        self._off_path_steps = 0
        self._segment_progress = 0.0
        self._last_position = self.env.position.copy()
        self._final_gear = None
        return self.path

    def action(self) -> np.ndarray:
        if not self.path:
            self.reset()

        travelled = float(np.linalg.norm(self.env.position - self._last_position))
        self._last_position = self.env.position.copy()
        self._segment_progress += travelled
        while self.index < len(self.path) - 1:
            segment_length = math.hypot(
                self.path[self.index + 1].x - self.path[self.index].x,
                self.path[self.index + 1].y - self.path[self.index].y,
            )
            if self._segment_progress < segment_length:
                break
            self._segment_progress -= segment_length
            self.index += 1
        distance_from_path = math.hypot(
            float(self.env.position[0]) - self.path[self.index].x,
            float(self.env.position[1]) - self.path[self.index].y,
        )
        self._off_path_steps = (
            self._off_path_steps + 1 if distance_from_path > 0.075 else 0
        )
        if (
            self.allow_replan
            and self._off_path_steps >= 6
            and self.replans < 1
            and abs(self.env.speed) < 0.05
        ):
            self.path = HybridAStarPlanner(
                self.env, self.planner_config
            ).plan()
            self.index = 0
            self.replans += 1
            self._off_path_steps = 0
            self._segment_progress = 0.0
            self._last_position = self.env.position.copy()

        gear = self.path[min(self.index + 1, len(self.path) - 1)].gear
        if self.index >= len(self.path) - 2:
            goal_delta = (
                np.asarray(self.env.layout.slot.center) - self.env.position
            )
            longitudinal_error = float(
                goal_delta[0] * math.cos(self.env.heading)
                + goal_delta[1] * math.sin(self.env.heading)
            )
            if self._final_gear is None:
                self._final_gear = 1 if longitudinal_error >= 0.0 else -1
            elif longitudinal_error * self._final_gear < -0.008:
                self._final_gear *= -1
            gear = self._final_gear
        # Never request the opposite gear while the car is still rolling.
        if self.env.speed * gear < -0.006:
            return np.array(
                [0.0, float(np.clip(-5.0 * self.env.speed, -0.8, 0.8))],
                dtype=np.float32,
            )
        desired_steer = self.path[
            min(self.index + 2, len(self.path) - 1)
        ].steer
        steering_error = (
            desired_steer
            - self.env.steering / self.env.config.max_steer_angle
        )
        if abs(steering_error) > 0.65:
            if abs(self.env.speed) > 0.008:
                return np.array(
                    [
                        desired_steer,
                        float(np.clip(-5.0 * self.env.speed, -0.8, 0.8)),
                    ],
                    dtype=np.float32,
                )
            return np.array([desired_steer, 0.0], dtype=np.float32)

        remaining = self._remaining_distance()
        goal_distance = self.env.distance_to_goal
        final_phase = (
            self.index >= len(self.path) - 12
            or remaining < 0.10
            or goal_distance < 0.10
        )

        if final_phase and self.env.fully_inside_slot:
            if abs(self.env.speed) > 0.010:
                brake = float(np.clip(-5.0 * self.env.speed, -0.8, 0.8))
                return np.array([0.0, brake], dtype=np.float32)
            return np.zeros(2, dtype=np.float32)

        target_index = self._lookahead_index(gear)
        target = self.path[target_index]
        if self.index >= len(self.path) - 2:
            target_yaw = self.env.layout.slot.heading
        else:
            target_yaw = target.yaw

        reference = self.path[self.index]
        dx = float(self.env.position[0]) - reference.x
        dy = float(self.env.position[1]) - reference.y
        cross_track = -math.sin(reference.yaw) * dx + math.cos(reference.yaw) * dy
        heading_error = float(_wrap_angle(target_yaw - self.env.heading))
        heading_gain = 3.0 if self.index >= len(self.path) - 2 else 1.55
        cross_track_gain = 0.80
        curvature_steer = desired_steer * self.env.config.max_steer_angle
        curvature_steer += gear * (
            heading_gain * heading_error
            - math.atan2(
                cross_track_gain * cross_track,
                max(abs(self.env.speed), 0.070),
            )
        )
        steering = float(
            np.clip(
                curvature_steer / self.env.config.max_steer_angle,
                -1.0,
                1.0,
            )
        )

        cruise = 0.20 if gear > 0 else 0.15
        if self.index >= len(self.path) - 8:
            cruise = min(cruise, 0.030)
        if remaining < 0.24:
            cruise = min(cruise, 0.040 + 0.25 * remaining)
        if goal_distance < 0.10:
            cruise = min(cruise, 0.022 + 0.30 * goal_distance)
        if self.index >= len(self.path) - 2 and not self.env.fully_inside_slot:
            cruise = 0.055
        target_speed = gear * cruise
        speed_error = target_speed - self.env.speed
        throttle = float(np.clip(4.5 * speed_error, -1.0, 1.0))
        return np.array([steering, throttle], dtype=np.float32)

    def _nearest_index(self) -> int:
        stop = min(len(self.path), self.index + 35)
        current_gear = self.path[self.index].gear
        for index in range(self.index + 1, stop):
            if self.path[index].gear != current_gear:
                stop = index
                break
        position = self.env.position
        return min(
            range(self.index, stop),
            key=lambda i: (self.path[i].x - position[0]) ** 2
            + (self.path[i].y - position[1]) ** 2,
        )

    def _lookahead_index(self, gear: int) -> int:
        position = self.env.position
        target = min(self.index + 1, len(self.path) - 1)
        for index in range(target, len(self.path)):
            if self.path[index].gear != gear:
                break
            target = index
            distance = math.hypot(
                self.path[index].x - float(position[0]),
                self.path[index].y - float(position[1]),
            )
            if distance >= 0.055:
                break
        return target

    def _remaining_distance(self) -> float:
        if self.index >= len(self.path) - 1:
            return self.env.distance_to_goal
        total = math.hypot(
            self.path[self.index].x - float(self.env.position[0]),
            self.path[self.index].y - float(self.env.position[1]),
        )
        for first, second in zip(
            self.path[self.index :], self.path[self.index + 1 :]
        ):
            total += math.hypot(second.x - first.x, second.y - first.y)
        return total



def rollout_expert(
    env: CarParkingEnv,
    *,
    task_id: int | None = None,
    seed: int = 0,
    reset_options: dict[str, Any] | None = None,
    planner_config: PlannerConfig | None = None,
) -> RolloutResult:
    """Reset, plan, and execute one expert episode using actual dynamics."""
    options = dict(reset_options or {})
    if task_id is not None:
        options["task_id"] = task_id
    env.reset(seed=seed, options=options)
    policy = ParkingExpertPolicy(env, planner_config=planner_config)
    path = policy.reset()
    actions: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = policy.action()
        _, _, terminated, truncated, info = env.step(action)
        actions.append(action.copy())
        infos.append(info)
    return RolloutResult(
        task_id=task_id,
        success=bool(info.get("success")),
        dead=bool(info.get("dead")),
        collision=any(bool(step_info.get("collision")) for step_info in infos),
        timeout=bool(truncated),
        minimum_health=min(
            (float(step_info["health"]) for step_info in infos),
            default=float(env.health),
        ),
        total_health_loss=sum(
            float(step_info.get("health_loss", 0.0)) for step_info in infos
        ),
        steps=len(actions),
        replans=policy.replans,
        path_points=len(path),
        actions=actions,
        infos=infos,
    )
