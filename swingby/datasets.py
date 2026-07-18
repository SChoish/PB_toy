"""Offline GCRL dataset loader for OrbitalSwingBy NPZ trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def normalize_actions(actions: np.ndarray) -> np.ndarray:
    """Map env actions (angle, throttle) -> roughly [-1, 1]^2."""
    actions = np.asarray(actions, dtype=np.float32)
    out = np.empty_like(actions)
    out[..., 0] = actions[..., 0] / np.pi
    out[..., 1] = actions[..., 1] * 2.0 - 1.0
    return np.clip(out, -1.0, 1.0)


def denormalize_actions(actions: np.ndarray) -> np.ndarray:
    """Map network actions back to (angle ∈ [-π, π], throttle ∈ [0, 1])."""
    actions = np.asarray(actions, dtype=np.float32)
    out = np.empty_like(actions)
    out[..., 0] = np.clip(actions[..., 0], -1.0, 1.0) * np.pi
    out[..., 1] = (np.clip(actions[..., 1], -1.0, 1.0) + 1.0) * 0.5
    return out


def goal_success(
    achieved_xy_v: np.ndarray,
    desired: np.ndarray,
    *,
    goal_radius: float = 0.075,
    goal_velocity_tolerance: float = 0.35,
) -> np.ndarray:
    """Position + velocity match on the 4-D commanded goal."""
    achieved = np.asarray(achieved_xy_v, dtype=np.float32)[..., :4]
    desired = np.asarray(desired, dtype=np.float32)[..., :4]
    pos = np.linalg.norm(achieved[..., :2] - desired[..., :2], axis=-1)
    vel = np.linalg.norm(achieved[..., 2:4] - desired[..., 2:4], axis=-1)
    return (pos <= goal_radius) & (vel <= goal_velocity_tolerance)


def _infer_episode_ids(terminals: np.ndarray) -> np.ndarray:
    episode_ids = np.zeros(len(terminals), dtype=np.int32)
    episode = 0
    for index in range(len(terminals)):
        episode_ids[index] = episode
        if terminals[index]:
            episode += 1
    return episode_ids


def _episode_end_lookup(
    episode_ids: np.ndarray, terminals: np.ndarray
) -> np.ndarray:
    if len(episode_ids) == 0:
        raise ValueError("dataset is empty")
    if np.any(np.diff(episode_ids) < 0):
        raise ValueError("episode_ids must be monotone")
    ends = np.empty(len(episode_ids), dtype=np.int64)
    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        if not terminals[indices[-1]]:
            raise ValueError(f"episode {episode_id} does not end at a terminal")
        ends[indices] = indices[-1]
    return ends


@dataclass
class SwingbyDataset:
    """Transition store with geometric goal / path relabeling."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    commanded_goals: np.ndarray
    episode_ids: np.ndarray
    episode_ends: np.ndarray
    valid_indices: np.ndarray
    path_horizon: int = 8
    goal_relabel_prob: float = 0.8
    random_goal_prob: float = 0.1
    goal_radius: float = 0.075
    goal_velocity_tolerance: float = 0.35

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.path_horizon)

    def _future_index(self, rng: np.random.Generator, index: int) -> int:
        end = int(self.episode_ends[index])
        if index >= end:
            return index
        return int(rng.integers(index, end + 1))

    def _path(self, index: int) -> np.ndarray:
        k = self.path_horizon
        end = int(self.episode_ends[index])
        if index + k - 1 > end:
            raise ValueError(
                f"path start {index} crosses episode end {end} for K={k}"
            )
        path = np.empty(
            (k + 1, self.observations.shape[-1]), dtype=np.float32
        )
        path[0] = self.observations[index]
        for offset in range(1, k + 1):
            transition_index = index + offset - 1
            path[offset] = self.next_observations[transition_index]
        return path

    def _state_at_offset(self, index: int, offset: int) -> np.ndarray:
        if offset == 0:
            return self.observations[index]
        return self.next_observations[index + offset - 1]

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        idxs = self.valid_indices[
            rng.integers(0, len(self.valid_indices), size=batch_size)
        ]
        goals = np.empty((batch_size, 4), dtype=np.float32)
        subgoal_value_goals = np.empty(
            (batch_size, self.observations.shape[-1]), dtype=np.float32
        )
        value_goals = np.empty_like(subgoal_value_goals)
        split_states = np.empty_like(subgoal_value_goals)
        base_states = np.empty_like(subgoal_value_goals)
        split_offsets = np.zeros(batch_size, dtype=np.int64)
        value_offsets = np.zeros(batch_size, dtype=np.int64)
        base_offsets = np.zeros(batch_size, dtype=np.int64)
        transitive_valid = np.zeros(batch_size, dtype=np.float32)
        paths = np.empty(
            (
                batch_size,
                self.path_horizon + 1,
                self.observations.shape[-1],
            ),
            dtype=np.float32,
        )

        for row, index in enumerate(idxs):
            paths[row] = self._path(int(index))
            draw = float(rng.random())
            if draw < self.random_goal_prob:
                goal_index = int(rng.integers(0, len(self.valid_indices)))
                random_index = int(
                    self.valid_indices[goal_index]
                )
                goals[row] = self.next_observations[random_index, :4]
                subgoal_value_goals[row] = self.next_observations[
                    random_index
                ]
            elif draw < self.random_goal_prob + self.goal_relabel_prob:
                goal_index = self._future_index(rng, int(index))
                goals[row] = self.next_observations[goal_index, :4]
                subgoal_value_goals[row] = self.next_observations[goal_index]
            else:
                goals[row] = self.commanded_goals[index]
                terminal_index = int(self.episode_ends[index])
                subgoal_value_goals[row] = self.next_observations[terminal_index]

            value_goal_index = self._future_index(rng, int(index))
            total_offset = int(value_goal_index - index + 1)
            value_offsets[row] = total_offset
            value_goals[row] = self.next_observations[value_goal_index]
            if total_offset >= 2:
                split_offset = int(rng.integers(1, total_offset))
                split_offsets[row] = split_offset
                split_states[row] = self._state_at_offset(
                    int(index), split_offset
                )
                transitive_valid[row] = 1.0
            else:
                split_states[row] = self.observations[index]

            max_base_offset = min(
                self.path_horizon,
                int(self.episode_ends[index] - index + 1),
            )
            base_offset = int(rng.integers(1, max_base_offset + 1))
            base_offsets[row] = base_offset
            base_states[row] = self._state_at_offset(
                int(index), base_offset
            )

        next_observations = self.next_observations[idxs]
        successes = goal_success(
            next_observations,
            goals,
            goal_radius=self.goal_radius,
            goal_velocity_tolerance=self.goal_velocity_tolerance,
        ).astype(np.float32)
        masks = (
            (1.0 - successes)
            * (1.0 - self.terminals[idxs].astype(np.float32))
        ).astype(np.float32)
        path_goal_states = paths[:, -1]

        return {
            "observations": self.observations[idxs].astype(np.float32),
            "actions": self.actions[idxs].astype(np.float32),
            "next_observations": next_observations.astype(np.float32),
            "terminals": self.terminals[idxs].astype(np.float32),
            "rewards": successes,
            "masks": masks,
            "goals": goals,
            "actor_goals": goals,
            "subgoal_value_goals": subgoal_value_goals,
            "value_goals": value_goals,
            "value_base_goals": base_states.astype(np.float32),
            "value_base_offsets": base_offsets.astype(np.float32),
            "trans_v_split_observations": split_states.astype(np.float32),
            "trans_v_left_goals": split_states.astype(np.float32),
            "trans_v_right_goals": value_goals,
            "trans_v_valid_mask": transitive_valid,
            "trans_v_split_offsets": split_offsets.astype(np.float32),
            "value_offsets": value_offsets.astype(np.float32),
            "low_actor_goals": path_goal_states.astype(np.float32),
            "high_actor_goals": goals,
            "high_actor_targets": path_goal_states.astype(np.float32),
            "subgoals": path_goal_states.astype(np.float32),
            "path_observations": paths.astype(np.float32),
        }


@dataclass
class SwingbyGoalSequenceDataset:
    """Episode-aware base used by TRL and DQC adapters."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    episode_ends: np.ndarray
    valid_indices: np.ndarray
    config: dict

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def _sample_indices(
        self, rng: np.random.Generator, batch_size: int
    ) -> np.ndarray:
        return self.valid_indices[
            rng.integers(0, len(self.valid_indices), size=batch_size)
        ]

    def _future_indices(
        self, rng: np.random.Generator, indices: np.ndarray
    ) -> np.ndarray:
        result = np.empty_like(indices)
        for row, index in enumerate(indices):
            end = int(self.episode_ends[index])
            result[row] = int(rng.integers(int(index) + 1, end + 1))
        return result

    def _goal_indices(
        self,
        rng: np.random.Generator,
        indices: np.ndarray,
        *,
        prefix: str,
    ) -> np.ndarray:
        future = self._future_indices(rng, indices)
        random_probability = float(
            self.config.get(f"{prefix}_p_randomgoal", 0.0)
        )
        if random_probability <= 0.0:
            return future
        random_indices = rng.integers(0, len(self), size=len(indices))
        return np.where(
            rng.random(len(indices)) < random_probability,
            random_indices,
            future,
        )


@dataclass
class SwingbyTRLDataset(SwingbyGoalSequenceDataset):
    """TRL midpoint batches with 4-D phase-space goals."""

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        indices = self._sample_indices(rng, batch_size)
        value_goal_indices = self._future_indices(rng, indices)
        actor_goal_indices = self._goal_indices(
            rng, indices, prefix="actor"
        )
        midpoint_indices = np.asarray(
            [
                rng.integers(int(start), int(goal))
                for start, goal in zip(
                    indices, value_goal_indices, strict=False
                )
            ],
            dtype=np.int64,
        )
        return {
            "observations": self.observations[indices].astype(np.float32),
            "actions": self.actions[indices].astype(np.float32),
            "next_observations": self.next_observations[indices].astype(
                np.float32
            ),
            "value_goals": self.next_observations[
                value_goal_indices, :4
            ].astype(np.float32),
            "actor_goals": self.next_observations[
                actor_goal_indices, :4
            ].astype(np.float32),
            "value_offsets": (value_goal_indices - indices).astype(
                np.float32
            ),
            "value_midpoint_offsets": (
                midpoint_indices - indices
            ).astype(np.float32),
            "value_midpoint_observations": self.observations[
                midpoint_indices
            ].astype(np.float32),
            "value_midpoint_goals": self.observations[
                midpoint_indices, :4
            ].astype(np.float32),
            "value_midpoint_actions": self.actions[midpoint_indices].astype(
                np.float32
            ),
        }


@dataclass
class SwingbyDQCDataset(SwingbyGoalSequenceDataset):
    """DQC action-chunk batches."""

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        indices = self._sample_indices(rng, batch_size)
        goals = self._goal_indices(rng, indices, prefix="value")
        horizon = int(self.config.get("backup_horizon", 8))
        ends = self.episode_ends[indices]
        backup = np.minimum(horizon, ends - indices)
        goal_offsets = goals - indices
        backup = np.where(
            (goal_offsets >= 0) & (goal_offsets < backup),
            goal_offsets,
            backup,
        ).astype(np.int64)
        next_indices = indices + backup
        chunk_indices = indices[:, None] + np.arange(horizon)[None]
        chunks = self.actions[chunk_indices].reshape(batch_size, -1)
        valids = (chunk_indices < ends[:, None]).astype(np.float32)
        success = (backup < horizon).astype(np.float32)
        discount = float(self.config.get("discount", 0.99))
        rewards = np.power(discount, backup).astype(np.float32) * success
        return {
            "observations": self.observations[indices].astype(np.float32),
            "actions": self.actions[indices].astype(np.float32),
            "next_observations": self.next_observations[indices].astype(
                np.float32
            ),
            "high_value_goals": self.next_observations[goals, :4].astype(
                np.float32
            ),
            "high_value_next_observations": self.observations[
                next_indices
            ].astype(np.float32),
            "high_value_action_chunks": chunks.astype(np.float32),
            "high_value_backup_horizon": backup.astype(np.float32),
            "high_value_rewards": rewards,
            "high_value_masks": (1.0 - success).astype(np.float32),
            "valids": valids,
        }


def load_swingby_dataset(
    path: str | Path,
    *,
    path_horizon: int = 8,
    goal_relabel_prob: float = 0.8,
    random_goal_prob: float = 0.1,
    goal_radius: float = 0.075,
    goal_velocity_tolerance: float = 0.35,
) -> SwingbyDataset:
    """Load a swingby generate_dataset NPZ into a GCRL transition store."""
    if path_horizon < 1:
        raise ValueError("path_horizon must be at least 1")
    if goal_relabel_prob < 0.0 or random_goal_prob < 0.0:
        raise ValueError("goal sampling probabilities must be non-negative")
    if goal_relabel_prob + random_goal_prob > 1.0:
        raise ValueError("goal sampling probabilities must sum to at most 1")

    raw = np.load(Path(path))
    required = {
        "observations",
        "actions",
        "next_observations",
        "terminals",
        "commanded_goals",
    }
    missing = required.difference(raw.files)
    if missing:
        raise ValueError(f"dataset is missing fields: {sorted(missing)}")

    observations = np.asarray(raw["observations"], dtype=np.float32)
    actions = normalize_actions(np.asarray(raw["actions"], dtype=np.float32))
    next_observations = np.asarray(raw["next_observations"], dtype=np.float32)
    terminals = np.asarray(raw["terminals"], dtype=bool)
    commanded_goals = np.asarray(raw["commanded_goals"], dtype=np.float32)

    if observations.ndim != 2 or observations.shape[1] != 5:
        raise ValueError("observations must have shape (N, 5)")
    if next_observations.shape != observations.shape:
        raise ValueError("next_observations must match observations")
    if actions.shape != (len(observations), 2):
        raise ValueError("actions must have shape (N, 2)")
    if commanded_goals.shape != (len(observations), 4):
        raise ValueError("commanded_goals must have shape (N, 4)")
    if terminals.shape != (len(observations),):
        raise ValueError("terminals must have shape (N,)")

    episode_ids = (
        np.asarray(raw["episode_ids"], dtype=np.int32)
        if "episode_ids" in raw.files
        else _infer_episode_ids(terminals)
    )
    if episode_ids.shape != terminals.shape:
        raise ValueError("episode_ids must have shape (N,)")
    episode_ends = _episode_end_lookup(episode_ids, terminals)
    indices = np.arange(len(observations), dtype=np.int64)
    valid_indices = indices[
        indices + int(path_horizon) - 1 <= episode_ends
    ]
    if len(valid_indices) == 0:
        raise ValueError(
            f"no episode contains a full K={path_horizon} path"
        )
    return SwingbyDataset(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        terminals=terminals,
        commanded_goals=commanded_goals,
        episode_ids=episode_ids,
        episode_ends=episode_ends,
        valid_indices=valid_indices,
        path_horizon=int(path_horizon),
        goal_relabel_prob=float(goal_relabel_prob),
        random_goal_prob=float(random_goal_prob),
        goal_radius=float(goal_radius),
        goal_velocity_tolerance=float(goal_velocity_tolerance),
    )


def _load_goal_sequence_base(
    path: str | Path, *, config: dict, min_future: int
) -> SwingbyGoalSequenceDataset:
    data = load_swingby_dataset(
        path, path_horizon=int(config.get("subgoal_steps", 8))
    )
    valid_parts = []
    for episode_id in np.unique(data.episode_ids):
        indices = np.flatnonzero(data.episode_ids == episode_id)
        end = int(indices[-1])
        last = end + 1 - min_future
        if last > int(indices[0]):
            valid_parts.append(
                np.arange(int(indices[0]), last, dtype=np.int64)
            )
    if not valid_parts:
        raise ValueError(f"no valid indices for TRL/DQC in {path}")
    return SwingbyGoalSequenceDataset(
        observations=data.observations,
        actions=data.actions,
        next_observations=data.next_observations,
        episode_ends=data.episode_ends,
        valid_indices=np.concatenate(valid_parts),
        config=dict(config),
    )


def load_swingby_trl_dataset(
    path: str | Path, *, config: dict
) -> SwingbyTRLDataset:
    base = _load_goal_sequence_base(path, config=config, min_future=2)
    return SwingbyTRLDataset(
        observations=base.observations,
        actions=base.actions,
        next_observations=base.next_observations,
        episode_ends=base.episode_ends,
        valid_indices=base.valid_indices,
        config=base.config,
    )


def load_swingby_dqc_dataset(
    path: str | Path, *, config: dict
) -> SwingbyDQCDataset:
    horizon = int(config.get("backup_horizon", 8))
    base = _load_goal_sequence_base(path, config=config, min_future=horizon)
    return SwingbyDQCDataset(
        observations=base.observations,
        actions=base.actions,
        next_observations=base.next_observations,
        episode_ends=base.episode_ends,
        valid_indices=base.valid_indices,
        config=base.config,
    )
