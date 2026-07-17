from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CircleHazard:
    center: tuple[float, float]
    radius: float
    name: str = "hazard"


class ToyEnv:
    """Continuous unit-square environment with circular hazards.

    Planning is done on a grid, while trajectories and collision checks remain
    continuous.  ``clearance`` inflates every hazard for safer demonstrations.
    """

    def __init__(
        self,
        hazards: Iterable[CircleHazard] | None = None,
        goal: tuple[float, float] = (0.88, 0.84),
        clearance: float = 0.018,
        grid_size: int = 61,
    ) -> None:
        self.hazards = tuple(hazards or (
            CircleHazard((0.46, 0.48), 0.125, "central"),
            CircleHazard((0.70, 0.69), 0.105, "upper-right"),
        ))
        self.goal = np.asarray(goal, dtype=float)
        self.clearance = float(clearance)
        self.grid_size = int(grid_size)
        if not self.is_safe(self.goal):
            raise ValueError("goal must be outside every hazard")

    def is_safe(self, point: np.ndarray | tuple[float, float], margin: float = 0.0) -> bool:
        p = np.asarray(point, dtype=float)
        if np.any(p < 0.0) or np.any(p > 1.0):
            return False
        for h in self.hazards:
            if np.linalg.norm(p - h.center) <= h.radius + self.clearance + margin:
                return False
        return True

    def segment_is_safe(self, a: np.ndarray, b: np.ndarray, margin: float = 0.0) -> bool:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        denom = float(d @ d)
        for h in self.hazards:
            c = np.asarray(h.center)
            t = 0.0 if denom == 0.0 else float(np.clip(((c - a) @ d) / denom, 0.0, 1.0))
            if np.linalg.norm(a + t * d - c) <= h.radius + self.clearance + margin:
                return False
        return self.is_safe(b, margin)

    def sample_safe(self, rng: np.random.Generator, low: float = 0.06, high: float = 0.94) -> np.ndarray:
        for _ in range(10_000):
            p = rng.uniform(low, high, size=2)
            if self.is_safe(p, margin=0.01):
                return p
        raise RuntimeError("could not sample a safe point")

    def _to_cell(self, p: np.ndarray) -> tuple[int, int]:
        q = np.clip(np.rint(np.asarray(p) * (self.grid_size - 1)), 0, self.grid_size - 1)
        return int(q[0]), int(q[1])

    def _to_point(self, cell: tuple[int, int]) -> np.ndarray:
        return np.asarray(cell, dtype=float) / (self.grid_size - 1)

    def plan(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        """A* path followed by line-of-sight shortcutting."""
        start, target = np.asarray(start, float), np.asarray(target, float)
        if not self.is_safe(start) or not self.is_safe(target):
            raise ValueError("start and target must be safe")
        s, g = self._to_cell(start), self._to_cell(target)
        moves = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, s)]
        parent: dict[tuple[int, int], tuple[int, int] | None] = {s: None}
        cost = {s: 0.0}
        while frontier:
            _, cur = heapq.heappop(frontier)
            if cur == g:
                break
            for dx, dy in moves:
                nxt = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size):
                    continue
                if not self.is_safe(self._to_point(nxt), margin=0.004):
                    continue
                new_cost = cost[cur] + math.hypot(dx, dy)
                if new_cost < cost.get(nxt, float("inf")):
                    cost[nxt] = new_cost
                    parent[nxt] = cur
                    heuristic = math.hypot(nxt[0] - g[0], nxt[1] - g[1])
                    heapq.heappush(frontier, (new_cost + heuristic, nxt))
        if g not in parent:
            raise RuntimeError("no safe path found")
        cells = []
        cur: tuple[int, int] | None = g
        while cur is not None:
            cells.append(cur)
            cur = parent[cur]
        raw = [start, *[self._to_point(c) for c in reversed(cells[1:-1])], target]

        short = [raw[0]]
        i = 0
        while i < len(raw) - 1:
            j = len(raw) - 1
            while j > i + 1 and not self.segment_is_safe(raw[i], raw[j], margin=0.004):
                j -= 1
            short.append(raw[j])
            i = j
        return np.asarray(short)

    def resample_path(self, path: np.ndarray, step_size: float = 0.028) -> np.ndarray:
        points = [np.asarray(path[0], float)]
        for endpoint in path[1:]:
            a = points[-1]
            distance = float(np.linalg.norm(endpoint - a))
            n = max(1, int(math.ceil(distance / step_size)))
            for k in range(1, n + 1):
                points.append(a + (endpoint - a) * (k / n))
        return np.asarray(points)

    def shield_step(self, state: np.ndarray, proposed: np.ndarray, max_step: float = 0.033) -> np.ndarray:
        """Clip a learned step and rotate it if it intersects a hazard."""
        state, proposed = np.asarray(state, float), np.asarray(proposed, float)
        delta = proposed - state
        norm = float(np.linalg.norm(delta))
        if norm > max_step:
            delta *= max_step / norm
        for angle in (0, 18, -18, 36, -36, 54, -54, 72, -72, 90, -90, 120, -120, 180):
            r = math.radians(angle)
            rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
            candidate = np.clip(state + rot @ delta, 0.005, 0.995)
            if self.segment_is_safe(state, candidate, margin=0.003):
                return candidate
        return state.copy()

    def value(self, points: np.ndarray) -> np.ndarray:
        """Hazard-aware true value: negative shortest-path length to the goal."""
        pts = np.atleast_2d(points)
        values = []
        for p in pts:
            if not self.is_safe(p):
                values.append(-2.0)
                continue
            try:
                path = self.plan(p, self.goal)
                length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                values.append(-length)
            except RuntimeError:
                values.append(-2.0)
        return np.asarray(values)
