from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .env import ToyEnv


@dataclass
class Episode:
    kind: str
    states: np.ndarray
    target: np.ndarray

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "target": self.target.round(6).tolist(),
            "states": self.states.round(6).tolist(),
        }


def generate_dataset(env: ToyEnv, n_episodes: int = 180, seed: int = 7) -> list[Episode]:
    """Generate safe demonstrations: 55% reaching, 25% partial, 20% roaming.

    Seeds a few bottom-left → goal corridor demos so the concept figure can find
    a clear bridge-vs-straight example near the hazard gap.
    """
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []

    corridor_starts = [
        np.array([0.10, 0.12]),
        np.array([0.12, 0.22]),
        np.array([0.08, 0.30]),
        np.array([0.18, 0.16]),
        np.array([0.14, 0.40]),
    ]
    for start in corridor_starts:
        if not env.is_safe(start, margin=0.01):
            continue
        path = env.resample_path(env.plan(start, env.goal), step_size=0.028)
        episodes.append(Episode("goal_reaching", path, env.goal.copy()))

    for i in range(n_episodes):
        # Bias ~30% of starts into the lower-left quadrant (figure corridor).
        if rng.random() < 0.30:
            start = None
            for _ in range(200):
                cand = rng.uniform([0.05, 0.05], [0.35, 0.45], size=2)
                if env.is_safe(cand, margin=0.01):
                    start = cand
                    break
            if start is None:
                start = env.sample_safe(rng)
        else:
            start = env.sample_safe(rng)
        u = rng.random()
        if u < 0.55:
            kind, target = "goal_reaching", env.goal.copy()
        elif u < 0.80:
            kind, target = "partial", env.goal.copy()
        else:
            kind, target = "roaming", env.sample_safe(rng)
        path = env.resample_path(env.plan(start, target), step_size=float(rng.uniform(0.022, 0.034)))
        if kind == "partial" and len(path) > 8:
            keep = int(rng.integers(max(5, len(path) // 3), max(6, 3 * len(path) // 4)))
            path = path[:keep]
        # Small perpendicular noise creates dataset diversity while retaining safety.
        noisy = [path[0]]
        for p in path[1:-1]:
            q = np.clip(p + rng.normal(0.0, 0.0025, size=2), 0.0, 1.0)
            noisy.append(q if env.segment_is_safe(noisy[-1], q, margin=0.001) else p)
        if len(path) > 1:
            noisy.append(path[-1])
        states = np.asarray(noisy)
        episodes.append(Episode(kind, states, np.asarray(target)))
    return episodes


def validate_dataset(env: ToyEnv, episodes: list[Episode]) -> dict[str, float | int]:
    transitions = 0
    unsafe = 0
    step_lengths: list[float] = []
    for ep in episodes:
        for a, b in zip(ep.states[:-1], ep.states[1:]):
            transitions += 1
            step_lengths.append(float(np.linalg.norm(b - a)))
            unsafe += int(not env.segment_is_safe(a, b))
    kinds = {k: sum(ep.kind == k for ep in episodes) for k in ("goal_reaching", "partial", "roaming")}
    return {
        "episodes": len(episodes),
        "transitions": transitions,
        "unsafe_transitions": unsafe,
        "mean_step": float(np.mean(step_lengths)),
        "max_step": float(np.max(step_lengths)),
        **{f"{k}_episodes": v for k, v in kinds.items()},
    }


def save_dataset(path: str | Path, env: ToyEnv, episodes: list[Episode], seed: int) -> None:
    payload = {
        "schema": "toy-pathbridger-v1",
        "seed": seed,
        "bounds": [[0.0, 0.0], [1.0, 1.0]],
        "goal": env.goal.tolist(),
        "hazards": [dict(name=h.name, center=list(h.center), radius=h.radius) for h in env.hazards],
        "episodes": [ep.as_dict() for ep in episodes],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
