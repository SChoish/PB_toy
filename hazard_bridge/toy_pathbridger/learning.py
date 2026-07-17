from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .dataset import Episode
from .env import ToyEnv


def features(x: np.ndarray) -> np.ndarray:
    """Quadratic feature map used by both tiny learned components."""
    x = np.atleast_2d(x)
    cols = [np.ones(len(x))]
    cols.extend(x[:, i] for i in range(x.shape[1]))
    cols.extend(x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1]))
    return np.column_stack(cols)


@dataclass
class RidgeModel:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, reg: float = 2e-3) -> "RidgeModel":
        mean, scale = x.mean(0), x.std(0) + 1e-6
        phi = features((x - mean) / scale)
        penalty = reg * np.eye(phi.shape[1])
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(phi.T @ phi + penalty, phi.T @ y)
        return cls(weights, mean, scale)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return features((np.atleast_2d(x) - self.mean) / self.scale) @ self.weights


@dataclass
class LearnedPathBridger:
    subgoal: RidgeModel
    bridge: RidgeModel
    lookahead: int

    @classmethod
    def fit(cls, env: ToyEnv, episodes: list[Episode], lookahead: int = 7) -> "LearnedPathBridger":
        sub_x, sub_y, bridge_x, bridge_y = [], [], [], []
        for ep in episodes:
            values = env.value(ep.states)
            for i in range(len(ep.states) - 1):
                j = min(i + lookahead, len(ep.states) - 1)
                state, future = ep.states[i], ep.states[j]
                sub_x.append([state[0], state[1], values[i], ep.target[0], ep.target[1]])
                sub_y.append(future)
                bridge_x.append([state[0], state[1], future[0], future[1]])
                bridge_y.append(ep.states[i + 1] - state)
        return cls(
            RidgeModel.fit(np.asarray(sub_x), np.asarray(sub_y), reg=8e-3),
            RidgeModel.fit(np.asarray(bridge_x), np.asarray(bridge_y), reg=3e-3),
            lookahead,
        )

    def estimate_subgoal(self, env: ToyEnv, state: np.ndarray, target: np.ndarray) -> np.ndarray:
        v = env.value(state)[0]
        x = np.array([state[0], state[1], v, target[0], target[1]])
        learned = np.clip(self.subgoal.predict(x)[0], 0.0, 1.0)
        delta = learned - state
        distance = min(float(np.linalg.norm(delta)), 0.22)
        if distance < 0.06:
            delta = target - state
            distance = min(float(np.linalg.norm(delta)), 0.18)
        base = math.atan2(delta[1], delta[0])
        # The estimator remains learned; true value only ranks a small local
        # candidate set, analogous to transitive subgoal selection.
        candidates = [learned]
        for offset in (0, 22, -22, 44, -44, 66, -66, 90, -90):
            a = base + math.radians(offset)
            candidate = np.clip(state + distance * np.array([math.cos(a), math.sin(a)]), 0.0, 1.0)
            if env.segment_is_safe(state, candidate, margin=0.003):
                candidates.append(candidate)
        values = env.value(np.asarray(candidates))
        return np.asarray(candidates[int(np.argmax(values))])

    def rollout(
        self, env: ToyEnv, start: np.ndarray, target: np.ndarray | None = None, max_steps: int = 90
    ) -> tuple[np.ndarray, np.ndarray]:
        target = env.goal if target is None else np.asarray(target)
        states = [np.asarray(start, float)]
        subgoals = []
        for t in range(max_steps):
            s = states[-1]
            sg = self.estimate_subgoal(env, s, target)
            # Refresh marks are retained for plotting; the model itself is queried every step.
            if t % self.lookahead == 0:
                subgoals.append(sg)
            delta = self.bridge.predict([s[0], s[1], sg[0], sg[1]])[0]
            directed = (sg - s) / max(np.linalg.norm(sg - s), 1e-8) * 0.028
            delta = 0.35 * delta + 0.65 * directed
            nxt = env.shield_step(s, s + delta)
            states.append(nxt)
            if np.linalg.norm(nxt - target) < 0.045:
                states.append(target.copy())
                break
            if np.linalg.norm(nxt - s) < 1e-6:
                break
        return np.asarray(states), np.asarray(subgoals)


def evaluate(env: ToyEnv, model: LearnedPathBridger, episodes: list[Episode]) -> dict[str, float | int]:
    sub_errors, bridge_errors = [], []
    successful, safe_rollouts = 0, 0
    for ep in episodes:
        vals = env.value(ep.states)
        for i in range(len(ep.states) - 1):
            j = min(i + model.lookahead, len(ep.states) - 1)
            pred_sg = model.subgoal.predict([ep.states[i, 0], ep.states[i, 1], vals[i], *ep.target])[0]
            sub_errors.append(float(np.linalg.norm(pred_sg - ep.states[j])))
            pred_d = model.bridge.predict([*ep.states[i], *ep.states[j]])[0]
            bridge_errors.append(float(np.linalg.norm(pred_d - (ep.states[i + 1] - ep.states[i]))))
    starts = [np.array([0.10, y]) for y in (0.12, 0.24, 0.36, 0.72, 0.86)]
    for start in starts:
        rollout, _ = model.rollout(env, start)
        safe = all(env.segment_is_safe(a, b) for a, b in zip(rollout[:-1], rollout[1:]))
        safe_rollouts += int(safe)
        successful += int(np.linalg.norm(rollout[-1] - env.goal) < 0.05)
    return {
        "subgoal_rmse_distance": float(np.sqrt(np.mean(np.square(sub_errors)))),
        "bridge_rmse_step": float(np.sqrt(np.mean(np.square(bridge_errors)))),
        "rollout_successes": successful,
        "rollout_trials": len(starts),
        "safe_rollouts": safe_rollouts,
    }


@dataclass
class PinnedBridgeRegressor:
    """Non-parametric regression of paper-style endpoint-pinned interiors.

    The model stores residual templates from real K-step dataset windows and
    performs locally weighted regression in (start, endpoint displacement)
    space.  Endpoint schedules guarantee exact pinning even after averaging.
    """

    descriptors: np.ndarray
    residuals: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    horizon: int

    @classmethod
    def fit(cls, episodes: list[Episode], horizon: int = 14) -> "PinnedBridgeRegressor":
        descriptors, residuals = [], []
        u = np.linspace(0.0, 1.0, horizon + 1)
        alpha = u**0.8
        mask = u * (1.0 - u)
        for ep in episodes:
            if len(ep.states) <= horizon:
                continue
            for i in range(0, len(ep.states) - horizon, max(1, horizon // 4)):
                window = ep.states[i : i + horizon + 1]
                delta_k = window[-1] - window[0]
                actual = window - window[0]
                r = np.zeros_like(actual)
                r[1:-1] = (actual[1:-1] - alpha[1:-1, None] * delta_k) / mask[1:-1, None]
                descriptors.append(np.r_[window[0], delta_k])
                residuals.append(r)
        x = np.asarray(descriptors)
        if len(x) == 0:
            raise ValueError("dataset has no complete bridge windows")
        return cls(x, np.asarray(residuals), x.mean(0), x.std(0) + 0.04, horizon)

    def predict(self, start: np.ndarray, endpoint: np.ndarray, neighbors: int = 10) -> np.ndarray:
        start, endpoint = np.asarray(start), np.asarray(endpoint)
        query = np.r_[start, endpoint - start]
        distances = np.linalg.norm((self.descriptors - query) / self.scale, axis=1)
        ids = np.argsort(distances)[: min(neighbors, len(distances))]
        weights = np.exp(-0.7 * np.square(distances[ids] / max(distances[ids][0] + 0.7, 0.7)))
        weights /= weights.sum()
        residual = np.tensordot(weights, self.residuals[ids], axes=(0, 0))
        u = np.linspace(0.0, 1.0, self.horizon + 1)
        alpha = u**0.8
        mask = u * (1.0 - u)
        delta = endpoint - start
        path = start + alpha[:, None] * delta + mask[:, None] * residual
        path[0], path[-1] = start, endpoint
        return path

    def nearest_windows(self, start: np.ndarray, endpoint: np.ndarray, count: int = 8) -> np.ndarray:
        query = np.r_[start, endpoint - start]
        distances = np.linalg.norm((self.descriptors - query) / self.scale, axis=1)
        ids = np.argsort(distances)[:count]
        u = np.linspace(0.0, 1.0, self.horizon + 1)
        alpha, mask = u**0.8, u * (1.0 - u)
        windows = []
        for idx in ids:
            s, d = self.descriptors[idx, :2], self.descriptors[idx, 2:]
            windows.append(s + alpha[:, None] * d + mask[:, None] * self.residuals[idx])
        return np.asarray(windows)


def select_bridge_example(
    env: ToyEnv, model: PinnedBridgeRegressor, episodes: list[Episode]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find a real endpoint pair where direct control fails but the bridge is safe."""
    candidates = []
    k = model.horizon
    for ep in episodes:
        for i in range(max(0, len(ep.states) - k)):
            window = ep.states[i : i + k + 1]
            if len(window) != k + 1:
                continue
            s, z = window[0], window[-1]
            if env.segment_is_safe(s, z, margin=0.0):
                continue
            bridge = model.predict(s, z)
            safe = all(env.segment_is_safe(a, b) for a, b in zip(bridge[:-1], bridge[1:]))
            if safe:
                length = float(np.linalg.norm(np.diff(window, axis=0), axis=1).sum())
                candidates.append((length, s, z, bridge))
    if not candidates:
        raise RuntimeError("could not find a bridge-vs-straight-line example")
    _, start, endpoint, bridge = max(candidates, key=lambda item: item[0])
    support = model.nearest_windows(start, endpoint)
    return start, endpoint, bridge, support
