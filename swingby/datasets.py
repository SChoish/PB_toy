"""Offline GCRL dataset loader for OrbitalSwingBy NPZ trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


LEGACY_ACTION_ENCODING = "angle_throttle"
CARTESIAN_ACTION_ENCODING = "cartesian_thrust"
SWINGBY_SCHEMA = "swingby"
SWINGBY_SCHEMAS = frozenset({"taskmix_v2", SWINGBY_SCHEMA})


def read_dataset_metadata(path: str | Path) -> dict[str, str]:
    with np.load(Path(path)) as raw:
        schema = (
            str(np.asarray(raw["dataset_schema"]).item())
            if "dataset_schema" in raw.files
            else "ballistic_v1"
        )
        encoding = (
            str(np.asarray(raw["action_encoding"]).item())
            if "action_encoding" in raw.files
            else LEGACY_ACTION_ENCODING
        )
    return {"dataset_schema": schema, "action_encoding": encoding}


def normalize_actions(
    actions: np.ndarray, *, encoding: str = LEGACY_ACTION_ENCODING
) -> np.ndarray:
    """Map environment actions to the network's continuous 2-D action space."""
    actions = np.asarray(actions, dtype=np.float32)
    out = np.empty_like(actions)
    if encoding == CARTESIAN_ACTION_ENCODING:
        throttle = np.clip(actions[..., 1], 0.0, 1.0)
        out[..., 0] = throttle * np.cos(actions[..., 0])
        out[..., 1] = throttle * np.sin(actions[..., 0])
    elif encoding == LEGACY_ACTION_ENCODING:
        out[..., 0] = actions[..., 0] / np.pi
        out[..., 1] = actions[..., 1] * 2.0 - 1.0
    else:
        raise ValueError(f"Unknown action encoding: {encoding!r}")
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def denormalize_actions(
    actions: np.ndarray, *, encoding: str = LEGACY_ACTION_ENCODING
) -> np.ndarray:
    """Map network actions back to environment ``(angle, throttle)``."""
    actions = np.asarray(actions, dtype=np.float32)
    out = np.empty_like(actions)
    if encoding == CARTESIAN_ACTION_ENCODING:
        x = np.clip(actions[..., 0], -1.0, 1.0)
        y = np.clip(actions[..., 1], -1.0, 1.0)
        throttle = np.clip(np.sqrt(x * x + y * y), 0.0, 1.0)
        out[..., 0] = np.where(throttle > 1e-8, np.arctan2(y, x), 0.0)
        out[..., 1] = throttle
    elif encoding == LEGACY_ACTION_ENCODING:
        out[..., 0] = np.clip(actions[..., 0], -1.0, 1.0) * np.pi
        out[..., 1] = (np.clip(actions[..., 1], -1.0, 1.0) + 1.0) * 0.5
    else:
        raise ValueError(f"Unknown action encoding: {encoding!r}")
    return out.astype(np.float32)


def goal_success(
    achieved_xy_v: np.ndarray,
    desired: np.ndarray,
    *,
    goal_radius: float = 0.075,
    goal_velocity_tolerance: float = 0.35,
    goal_min_speed_ratio: float = 0.50,
    goal_min_velocity_alignment: float = 0.75,
) -> np.ndarray:
    """Position + velocity match on the 4-D commanded goal."""
    achieved = np.asarray(achieved_xy_v, dtype=np.float32)[..., :4]
    desired = np.asarray(desired, dtype=np.float32)[..., :4]
    pos = np.linalg.norm(achieved[..., :2] - desired[..., :2], axis=-1)
    actual_velocity = achieved[..., 2:4]
    target_velocity = desired[..., 2:4]
    vel = np.linalg.norm(actual_velocity - target_velocity, axis=-1)
    actual_speed = np.linalg.norm(actual_velocity, axis=-1)
    target_speed = np.linalg.norm(target_velocity, axis=-1)
    aligned = np.sum(actual_velocity * target_velocity, axis=-1) >= (
        goal_min_velocity_alignment * actual_speed * target_speed
    )
    fast_enough = actual_speed >= goal_min_speed_ratio * target_speed
    directed = np.where(target_speed <= 1e-8, True, aligned & fast_enough)
    return (
        (pos <= goal_radius)
        & (vel <= goal_velocity_tolerance)
        & directed
    )


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
    successes: np.ndarray
    commanded_goal_indices: np.ndarray
    episode_ids: np.ndarray
    episode_ends: np.ndarray
    valid_indices: np.ndarray
    path_horizon: int = 8
    action_chunk_horizon: int = 5
    goal_relabel_prob: float = 1.0
    random_goal_prob: float = 0.0
    goal_radius: float = 0.075
    goal_velocity_tolerance: float = 0.35
    goal_min_speed_ratio: float = 0.50
    goal_min_velocity_alignment: float = 0.75
    value_base_horizon: int = 5
    dataset_schema: str = "ballistic_v1"
    action_encoding: str = LEGACY_ACTION_ENCODING

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
        path = np.empty(
            (k + 1, self.observations.shape[-1]), dtype=np.float32
        )
        path[0] = self.observations[index]
        for offset in range(1, k + 1):
            transition_index = min(index + offset - 1, end)
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
        commanded_mask = np.zeros(batch_size, dtype=np.float32)
        paths = np.empty(
            (batch_size, self.path_horizon + 1, self.observations.shape[-1]),
            dtype=np.float32,
        )

        for row, raw_index in enumerate(idxs):
            index = int(raw_index)
            paths[row] = self._path(index)
            draw = float(rng.random())
            command_index = int(self.commanded_goal_indices[index])
            can_command = command_index >= index
            random_sample = draw < self.random_goal_prob
            future_sample = (
                draw < self.random_goal_prob + self.goal_relabel_prob
                or not can_command
            )

            if random_sample:
                random_index = int(
                    self.valid_indices[rng.integers(0, len(self.valid_indices))]
                )
                goals[row] = self.next_observations[random_index, :4]
                subgoal_value_goals[row] = self.next_observations[random_index]
                value_goal_index = self._future_index(rng, index)
                value_goals[row] = self.next_observations[value_goal_index]
                path_target_offset = self.path_horizon
            elif future_sample:
                value_goal_index = self._future_index(rng, index)
                goals[row] = self.next_observations[value_goal_index, :4]
                value_goals[row] = self.next_observations[value_goal_index]
                subgoal_value_goals[row] = value_goals[row]
                path_target_offset = min(
                    self.path_horizon, value_goal_index - index + 1
                )
            else:
                value_goal_index = command_index
                goals[row] = self.commanded_goals[index]
                full_goal = self.next_observations[value_goal_index].copy()
                full_goal[:4] = goals[row]
                value_goals[row] = full_goal
                subgoal_value_goals[row] = full_goal
                commanded_mask[row] = 1.0
                path_target_offset = min(
                    self.path_horizon, value_goal_index - index + 1
                )

            if path_target_offset < self.path_horizon:
                paths[row, path_target_offset:] = paths[row, path_target_offset]

            total_offset = int(value_goal_index - index + 1)
            value_offsets[row] = total_offset
            if total_offset >= 2:
                split_offset = int(rng.integers(1, total_offset))
                split_offsets[row] = split_offset
                split_states[row] = self._state_at_offset(index, split_offset)
                transitive_valid[row] = 1.0
            else:
                split_states[row] = self.observations[index]

            max_base_offset = min(
                self.value_base_horizon,
                int(self.episode_ends[index] - index + 1),
            )
            base_offset = int(rng.integers(1, max_base_offset + 1))
            base_offsets[row] = base_offset
            base_states[row] = self._state_at_offset(index, base_offset)

        next_observations = self.next_observations[idxs]
        batch_successes = goal_success(
            next_observations,
            goals,
            goal_radius=self.goal_radius,
            goal_velocity_tolerance=self.goal_velocity_tolerance,
            goal_min_speed_ratio=self.goal_min_speed_ratio,
            goal_min_velocity_alignment=self.goal_min_velocity_alignment,
        ).astype(np.float32)
        masks = (
            (1.0 - batch_successes)
            * (1.0 - self.terminals[idxs].astype(np.float32))
        ).astype(np.float32)
        path_goal_states = paths[:, -1]

        h_a = max(1, int(self.action_chunk_horizon))
        action_dim = int(self.actions.shape[-1])
        action_chunks = np.zeros((batch_size, h_a, action_dim), dtype=np.float32)
        action_chunk_next = np.empty_like(subgoal_value_goals)
        for row, raw_index in enumerate(idxs):
            index = int(raw_index)
            end = int(self.episode_ends[index])
            for t in range(h_a):
                action_chunks[row, t] = self.actions[min(index + t, end)]
            action_chunk_next[row] = self.next_observations[min(index + h_a, end)]

        return {
            "observations": self.observations[idxs].astype(np.float32),
            "actions": self.actions[idxs].astype(np.float32),
            "next_observations": next_observations.astype(np.float32),
            "terminals": self.terminals[idxs].astype(np.float32),
            "rewards": batch_successes,
            "masks": masks,
            "goals": goals,
            "actor_goals": goals,
            "commanded_goal_mask": commanded_mask,
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
            "action_chunk_actions": action_chunks.reshape(batch_size, -1),
            "action_chunk_next_observations": action_chunk_next.astype(np.float32),
        }


@dataclass
class SwingbyGoalSequenceDataset:
    """Episode-aware base used by TRL and DQC adapters."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    commanded_goals: np.ndarray
    commanded_goal_indices: np.ndarray
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

    def _goal_targets(
        self, rng: np.random.Generator, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = self._future_indices(rng, indices)
        goals = self.next_observations[targets, :4].copy()
        probability = float(self.config.get("commanded_goal_prob", 0.0))
        if probability <= 0.0:
            return targets, goals
        command_targets = self.commanded_goal_indices[indices]
        use_command = (
            (command_targets > indices)
            & (rng.random(len(indices)) < probability)
        )
        targets = np.where(use_command, command_targets, targets)
        goals[use_command] = self.commanded_goals[indices[use_command]]
        return targets, goals


@dataclass
class SwingbyTRLDataset(SwingbyGoalSequenceDataset):
    """TRL midpoint batches with 4-D phase-space goals."""

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        indices = self._sample_indices(rng, batch_size)
        value_goal_indices, goal_values = self._goal_targets(rng, indices)
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
            "value_goals": goal_values.astype(np.float32),
            "actor_goals": goal_values.astype(np.float32),
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
        goals, goal_values = self._goal_targets(rng, indices)
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
            "high_value_goals": goal_values.astype(np.float32),
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
    action_chunk_horizon: int = 5,
    goal_relabel_prob: float | None = None,
    random_goal_prob: float = 0.0,
    value_base_horizon: int = 5,
    goal_radius: float = 0.075,
    goal_velocity_tolerance: float = 0.35,
    goal_min_speed_ratio: float = 0.50,
    goal_min_velocity_alignment: float = 0.75,
) -> SwingbyDataset:
    """Load a swingby generate_dataset NPZ into a GCRL transition store."""
    if path_horizon < 1:
        raise ValueError("path_horizon must be at least 1")
    if action_chunk_horizon < 1:
        raise ValueError("action_chunk_horizon must be at least 1")

    raw = np.load(Path(path))
    dataset_schema = (
        str(np.asarray(raw["dataset_schema"]).item())
        if "dataset_schema" in raw.files
        else "ballistic_v1"
    )
    action_encoding = (
        str(np.asarray(raw["action_encoding"]).item())
        if "action_encoding" in raw.files
        else LEGACY_ACTION_ENCODING
    )
    if goal_relabel_prob is None:
        goal_relabel_prob = 0.5 if dataset_schema in SWINGBY_SCHEMAS else 1.0
    if goal_relabel_prob < 0.0 or random_goal_prob < 0.0:
        raise ValueError("goal sampling probabilities must be non-negative")
    if goal_relabel_prob + random_goal_prob > 1.0:
        raise ValueError("goal sampling probabilities must sum to at most 1")
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
    actions = normalize_actions(
        np.asarray(raw["actions"], dtype=np.float32),
        encoding=action_encoding,
    )
    next_observations = np.asarray(raw["next_observations"], dtype=np.float32)
    terminals = np.asarray(raw["terminals"], dtype=bool)
    commanded_goals = np.asarray(raw["commanded_goals"], dtype=np.float32)
    # Recompute canonical success labels from the current goal contract.
    # Older files used only a loose Euclidean velocity tolerance.
    successes = goal_success(
        next_observations,
        commanded_goals,
        goal_radius=goal_radius,
        goal_velocity_tolerance=goal_velocity_tolerance,
        goal_min_speed_ratio=goal_min_speed_ratio,
        goal_min_velocity_alignment=goal_min_velocity_alignment,
    )

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
    if successes.shape != terminals.shape:
        raise ValueError("successes must have shape (N,)")

    episode_ids = (
        np.asarray(raw["episode_ids"], dtype=np.int32)
        if "episode_ids" in raw.files
        else _infer_episode_ids(terminals)
    )
    if episode_ids.shape != terminals.shape:
        raise ValueError("episode_ids must have shape (N,)")
    episode_ends = _episode_end_lookup(episode_ids, terminals)
    commanded_goal_indices = np.full(len(observations), -1, dtype=np.int64)
    segment_start = 0
    for index in range(1, len(observations) + 1):
        boundary = index == len(observations) or (
            episode_ids[index] != episode_ids[index - 1]
            or np.any(commanded_goals[index] != commanded_goals[index - 1])
        )
        if not boundary:
            continue
        reached = np.flatnonzero(successes[segment_start:index])
        if len(reached):
            target = segment_start + int(reached[0])
            commanded_goal_indices[segment_start : target + 1] = target
        segment_start = index
    indices = np.arange(len(observations), dtype=np.int64)
    # All path/action-chunk samples must be composed of real transitions.
    # Terminal padding is still used inside a path shortened to a sampled goal,
    # but never to fabricate behavior after the environment has terminated.
    need = max(int(path_horizon), int(action_chunk_horizon))
    valid_indices = indices[indices + need - 1 <= episode_ends]
    if len(valid_indices) == 0:
        raise ValueError(
            f"no episode contains a valid K={path_horizon} / h_a={action_chunk_horizon} window"
        )
    return SwingbyDataset(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        terminals=terminals,
        commanded_goals=commanded_goals,
        successes=successes,
        commanded_goal_indices=commanded_goal_indices,
        episode_ids=episode_ids,
        episode_ends=episode_ends,
        valid_indices=valid_indices,
        path_horizon=int(path_horizon),
        action_chunk_horizon=int(action_chunk_horizon),
        goal_relabel_prob=float(goal_relabel_prob),
        random_goal_prob=float(random_goal_prob),
        goal_radius=float(goal_radius),
        goal_velocity_tolerance=float(goal_velocity_tolerance),
        goal_min_speed_ratio=float(goal_min_speed_ratio),
        goal_min_velocity_alignment=float(goal_min_velocity_alignment),
        value_base_horizon=int(value_base_horizon),
        dataset_schema=dataset_schema,
        action_encoding=action_encoding,
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
    sequence_config = dict(config)
    sequence_config.setdefault(
        "commanded_goal_prob", 0.5 if data.dataset_schema in SWINGBY_SCHEMAS else 0.0
    )
    return SwingbyGoalSequenceDataset(
        observations=data.observations,
        actions=data.actions,
        next_observations=data.next_observations,
        commanded_goals=data.commanded_goals,
        commanded_goal_indices=data.commanded_goal_indices,
        episode_ends=data.episode_ends,
        valid_indices=np.concatenate(valid_parts),
        config=sequence_config,
    )


def load_swingby_trl_dataset(
    path: str | Path, *, config: dict
) -> SwingbyTRLDataset:
    base = _load_goal_sequence_base(path, config=config, min_future=2)
    return SwingbyTRLDataset(
        observations=base.observations,
        actions=base.actions,
        next_observations=base.next_observations,
        commanded_goals=base.commanded_goals,
        commanded_goal_indices=base.commanded_goal_indices,
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
        commanded_goals=base.commanded_goals,
        commanded_goal_indices=base.commanded_goal_indices,
        episode_ends=base.episode_ends,
        valid_indices=base.valid_indices,
        config=base.config,
    )
