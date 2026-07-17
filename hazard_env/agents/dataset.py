"""Navigate-dataset loader with simple geometric goal relabeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def normalize_actions(actions: np.ndarray) -> np.ndarray:
    """Map env actions (angle, thrust) -> roughly [-1, 1]^2."""
    actions = np.asarray(actions, dtype=np.float32)
    out = np.empty_like(actions)
    out[..., 0] = actions[..., 0] / np.pi
    out[..., 1] = actions[..., 1] * 2.0 - 1.0
    return np.clip(out, -1.0, 1.0)


def denormalize_actions(actions: np.ndarray) -> np.ndarray:
    """Map network actions back to (angle ∈ [-π, π], thrust ∈ [0, 1])."""
    actions = np.asarray(actions, dtype=np.float32)
    out = np.empty_like(actions)
    out[..., 0] = np.clip(actions[..., 0], -1.0, 1.0) * np.pi
    out[..., 1] = (np.clip(actions[..., 1], -1.0, 1.0) + 1.0) * 0.5
    return out


@dataclass
class TransitionDataset:
    """Flat transition store for offline GCRL toy training."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    future_observations: np.ndarray

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, np.ndarray]:
        idx = rng.integers(0, len(self), size=batch_size)
        goals = self.future_observations[idx]
        next_xy = self.next_observations[idx, :2]
        goal_xy = goals[:, :2]
        success = (np.linalg.norm(next_xy - goal_xy, axis=-1) <= 0.08).astype(np.float32)
        masks = (1.0 - success) * self.masks[idx]
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "next_observations": self.next_observations[idx],
            "terminals": self.terminals[idx],
            "rewards": success.astype(np.float32),
            "masks": masks.astype(np.float32),
            "goals": goals,
            "actor_goals": goals,
            "value_goals": goals,
            "low_actor_goals": goals,
            "high_actor_goals": goals,
            "high_actor_targets": self.future_observations[idx],
            "subgoals": self.future_observations[idx],
        }


def _episode_bounds(terminals: np.ndarray) -> list[tuple[int, int]]:
    ends = np.flatnonzero(terminals)
    bounds: list[tuple[int, int]] = []
    start = 0
    for end in ends:
        bounds.append((start, int(end) + 1))
        start = int(end) + 1
    if start < len(terminals):
        bounds.append((start, len(terminals)))
    return bounds


def load_navigate_dataset(
    path: str | Path,
    *,
    subgoal_steps: int = 8,
    goal_relabel_prob: float = 0.8,
    seed: int = 0,
) -> TransitionDataset:
    """Load OGBench-style compact navigate npz into transitions."""
    path = Path(path)
    raw = np.load(path)
    obs = np.asarray(raw["observations"], dtype=np.float32)
    actions = normalize_actions(np.asarray(raw["actions"], dtype=np.float32))
    terminals = np.asarray(raw["terminals"], dtype=bool)

    valid = ~terminals
    valid[-1] = False
    idxs = np.flatnonzero(valid)
    next_idxs = idxs + 1

    observations = obs[idxs]
    next_observations = obs[next_idxs]
    acts = actions[idxs]
    terms = terminals[idxs]
    masks = (1.0 - terms.astype(np.float32)).astype(np.float32)

    rng = np.random.default_rng(seed)
    future = np.empty_like(observations)
    bounds = _episode_bounds(terminals)

    # Optional commanded goals from collection (xy only).
    commanded: np.ndarray | None = None
    if "goals" in raw.files:
        stored = np.asarray(raw["goals"], dtype=np.float32)
        if stored.shape[-1] == 2 and observations.shape[-1] >= 2:
            commanded = np.concatenate(
                [stored[idxs], np.zeros((len(idxs), observations.shape[-1] - 2), dtype=np.float32)],
                axis=-1,
            )

    for start, end in bounds:
        mask = (idxs >= start) & (idxs < end)
        rows = np.flatnonzero(mask)
        ep_last = end - 1
        for r, t in zip(rows, idxs[mask], strict=False):
            sub_t = min(t + subgoal_steps, ep_last)
            if rng.random() < goal_relabel_prob:
                future[r] = obs[sub_t]
            elif commanded is not None:
                future[r] = commanded[r]
            else:
                future[r] = obs[ep_last]

    return TransitionDataset(
        observations=observations,
        actions=acts,
        next_observations=next_observations,
        terminals=terms,
        rewards=np.zeros(len(idxs), dtype=np.float32),
        masks=masks,
        future_observations=future.astype(np.float32),
    )
