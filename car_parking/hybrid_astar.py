"""Deterministic collision-aware lattice planner for car parking."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Iterable

import numpy as np

from .env import CarParkingEnv, OrientedBox, _box_corners, _wrap_angle

_PATH_CACHE: dict[tuple[object, ...], tuple["PathPoint", ...]] = {}


@dataclass(frozen=True)
class PathPoint:
    """A sampled vehicle pose and the gear used to reach the next point."""

    x: float
    y: float
    yaw: float
    gear: int
    steer: float = 0.0


@dataclass(frozen=True)
class PlannerConfig:
    xy_resolution: float = 0.035
    yaw_resolution: float = math.radians(10.0)
    primitive_length: float = 0.07
    collision_step: float = 0.01
    clearance_margin: float = 0.012
    steering_samples: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    goal_position_tolerance: float = 0.030
    goal_yaw_tolerance: float = math.radians(8.0)
    heuristic_weight: float = 2.5
    gear_switch_cost: float = 0.22
    reverse_cost: float = 0.06
    steering_cost: float = 0.012
    steering_change_cost: float = 0.018
    max_expansions: int = 180_000


@dataclass
class _Node:
    x: float
    y: float
    yaw: float
    gear: int
    steer: float
    cost: float
    parent: int | None
    samples: tuple[PathPoint, ...]


class HybridAStarPlanner:
    """Hybrid-A* style search over bicycle-model motion primitives.

    Search keys are discretized, while every expanded node and collision
    sample retains a continuous pose.  Collision checks use the environment's
    exact oriented vehicle box and obstacle test.
    """

    def __init__(
        self, env: CarParkingEnv, config: PlannerConfig | None = None
    ) -> None:
        self.env = env
        self.config = config or PlannerConfig()

    def plan(
        self,
        start: tuple[float, float, float] | None = None,
        goal: tuple[float, float, float] | None = None,
    ) -> list[PathPoint]:
        if start is None:
            start = (
                float(self.env.position[0]),
                float(self.env.position[1]),
                float(self.env.heading),
            )
        if goal is None:
            goal = (
                float(self.env.layout.slot.center[0]),
                float(self.env.layout.slot.center[1]),
                float(self.env.layout.slot.heading),
            )
        cache_key = self._cache_key(start, goal)
        cached = _PATH_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)
        if self._collides(*start):
            raise ValueError("start pose is in collision")

        counter = itertools.count()
        nodes: list[_Node] = []
        best: dict[tuple[int, int, int, int], float] = {}
        queue: list[tuple[float, int, int]] = []

        # Both initial gears are inserted so the first motion is not charged
        # as a gear switch. Duplicate continuous poses are harmless because
        # gear is part of the lattice key.
        for gear in (1, -1):
            node = _Node(*start, gear, 0.0, 0.0, None, ())
            index = len(nodes)
            nodes.append(node)
            key = self._key(node.x, node.y, node.yaw, gear)
            best[key] = 0.0
            heapq.heappush(
                queue,
                (
                    self.config.heuristic_weight
                    * self._heuristic(node.x, node.y, node.yaw, goal),
                    next(counter),
                    index,
                ),
            )

        goal_index: int | None = None
        expansions = 0
        while queue and expansions < self.config.max_expansions:
            _, _, index = heapq.heappop(queue)
            node = nodes[index]
            key = self._key(node.x, node.y, node.yaw, node.gear)
            if node.cost > best.get(key, math.inf) + 1e-9:
                continue
            expansions += 1
            if self._at_goal(node.x, node.y, node.yaw, goal):
                goal_index = index
                break

            for gear in (1, -1):
                for steer_normalized in self.config.steering_samples:
                    primitive = self._primitive(node, gear, steer_normalized)
                    if primitive is None:
                        continue
                    x, y, yaw, samples = primitive
                    switch = gear != node.gear
                    edge_cost = self.config.primitive_length
                    edge_cost += self.config.reverse_cost if gear < 0 else 0.0
                    edge_cost += self.config.gear_switch_cost if switch else 0.0
                    edge_cost += self.config.steering_cost * abs(steer_normalized)
                    edge_cost += self.config.steering_change_cost * abs(
                        steer_normalized - node.steer
                    )
                    cost = node.cost + edge_cost
                    child_key = self._key(x, y, yaw, gear)
                    if cost >= best.get(child_key, math.inf) - 1e-9:
                        continue
                    best[child_key] = cost
                    child = _Node(
                        x,
                        y,
                        yaw,
                        gear,
                        steer_normalized,
                        cost,
                        index,
                        samples,
                    )
                    child_index = len(nodes)
                    nodes.append(child)
                    priority = (
                        cost
                        + self.config.heuristic_weight
                        * self._heuristic(x, y, yaw, goal)
                    )
                    heapq.heappush(
                        queue, (priority, next(counter), child_index)
                    )

        if goal_index is None:
            raise RuntimeError(
                f"no parking path found after {expansions} expansions"
            )
        path = self._reconstruct(nodes, goal_index)
        _PATH_CACHE[cache_key] = tuple(path)
        return path

    def _cache_key(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
    ) -> tuple[object, ...]:
        obstacles = tuple(
            (
                box.center,
                box.length,
                box.width,
                box.heading,
                box.kind,
            )
            for box in self.env.layout.obstacles
        )
        geometry = (
            self.env.config.car_length,
            self.env.config.car_width,
            self.env.config.wheelbase,
            self.env.config.max_steer_angle,
            self.env.config.collision_margin,
            self.env.config.slot_margin,
        )
        return (self.config, start, goal, self.env.layout.slot, obstacles, geometry)

    def _primitive(
        self, node: _Node, gear: int, steer_normalized: float
    ) -> tuple[float, float, float, tuple[PathPoint, ...]] | None:
        count = max(
            1, int(math.ceil(self.config.primitive_length / self.config.collision_step))
        )
        distance = gear * self.config.primitive_length / count
        steer = steer_normalized * self.env.config.max_steer_angle
        x, y, yaw = node.x, node.y, node.yaw
        samples: list[PathPoint] = []
        for _ in range(count):
            yaw_delta = distance * math.tan(steer) / self.env.config.wheelbase
            mid_yaw = yaw + 0.5 * yaw_delta
            x += distance * math.cos(mid_yaw)
            y += distance * math.sin(mid_yaw)
            yaw = float(_wrap_angle(yaw + yaw_delta))
            if self._collides(x, y, yaw):
                return None
            samples.append(PathPoint(x, y, yaw, gear, steer_normalized))
        return x, y, yaw, tuple(samples)

    def _collides(self, x: float, y: float, yaw: float) -> bool:
        box = OrientedBox(
            (x, y),
            self.env.config.car_length + 2.0 * self.config.clearance_margin,
            self.env.config.car_width + 2.0 * self.config.clearance_margin,
            yaw,
        )
        return self.env._collides(box)

    def _key(
        self, x: float, y: float, yaw: float, gear: int
    ) -> tuple[int, int, int, int]:
        cfg = self.config
        yaw_bins = int(round(2.0 * math.pi / cfg.yaw_resolution))
        return (
            int(round(x / cfg.xy_resolution)),
            int(round(y / cfg.xy_resolution)),
            int(round(float(_wrap_angle(yaw)) / cfg.yaw_resolution)) % yaw_bins,
            gear,
        )

    def _heuristic(
        self, x: float, y: float, yaw: float, goal: tuple[float, float, float]
    ) -> float:
        distance = math.hypot(x - goal[0], y - goal[1])
        yaw_error = abs(float(_wrap_angle(yaw - goal[2])))
        # The yaw term is deliberately below the minimum-turn-radius lower
        # bound; it guides search without making the heuristic aggressive.
        return distance + 0.08 * yaw_error

    def _at_goal(
        self, x: float, y: float, yaw: float, goal: tuple[float, float, float]
    ) -> bool:
        if (
            math.hypot(x - goal[0], y - goal[1])
            > self.config.goal_position_tolerance
            or abs(float(_wrap_angle(yaw - goal[2])))
            > self.config.goal_yaw_tolerance
        ):
            return False
        vehicle = OrientedBox(
            (x, y),
            self.env.config.car_length,
            self.env.config.car_width,
            yaw,
        )
        corners = _box_corners(vehicle)
        slot = self.env.layout.slot
        center = np.asarray(slot.center)
        c, s = math.cos(slot.heading), math.sin(slot.heading)
        local = (corners - center) @ np.array([[c, -s], [s, c]])
        half_l = slot.length / 2.0 - self.env.config.slot_margin
        half_w = slot.width / 2.0 - self.env.config.slot_margin
        return bool(
            np.all(np.abs(local[:, 0]) <= half_l + 1e-8)
            and np.all(np.abs(local[:, 1]) <= half_w + 1e-8)
        )

    @staticmethod
    def _reconstruct(nodes: list[_Node], goal_index: int) -> list[PathPoint]:
        segments: list[tuple[PathPoint, ...]] = []
        index: int | None = goal_index
        while index is not None:
            node = nodes[index]
            if node.samples:
                segments.append(node.samples)
            index = node.parent
        segments.reverse()
        first = nodes[0]
        path = [
            PathPoint(
                first.x,
                first.y,
                first.yaw,
                segments[0][0].gear,
                segments[0][0].steer,
            )
        ]
        for segment in segments:
            path.extend(segment)
        return path


def path_is_collision_free(
    env: CarParkingEnv, path: Iterable[PathPoint]
) -> bool:
    """Return whether all sampled path poses pass environment collision checks."""
    planner = HybridAStarPlanner(env)
    return all(not planner._collides(point.x, point.y, point.yaw) for point in path)

