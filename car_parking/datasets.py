"""Goal-safe offline GCRL dataset loaders for CarParking trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def achieved_goals(physical_states: np.ndarray) -> np.ndarray:
    """Map raw Markov physics to [x, y, cos(yaw), sin(yaw), reached]."""
    physical = np.asarray(physical_states, dtype=np.float32)
    if physical.shape[-1] != 8:
        raise ValueError("physical states must end in dimension 8")
    return np.concatenate([physical[..., :4], physical[..., 7:8]], axis=-1)


def reconstruct_states(
    physical_states: np.ndarray,
    goals: np.ndarray,
    slot_lengths: np.ndarray,
    slot_widths: np.ndarray,
    *,
    car_length: float = 0.18,
    car_width: float = 0.10,
    slot_margin: float = 0.008,
) -> np.ndarray:
    """Reconstruct 11-D states without retaining features from another goal."""
    physical = np.asarray(physical_states, dtype=np.float32)
    goal = np.asarray(goals, dtype=np.float32)
    if physical.shape[-1] != 8 or goal.shape[-1] != 5:
        raise ValueError("physical states and goals must end in dimensions 8 and 5")
    lead_shape = np.broadcast_shapes(physical.shape[:-1], goal.shape[:-1])
    physical = np.broadcast_to(physical, (*lead_shape, 8))
    goal = np.broadcast_to(goal, (*lead_shape, 5))
    lengths = np.broadcast_to(np.asarray(slot_lengths, dtype=np.float32), lead_shape)
    widths = np.broadcast_to(np.asarray(slot_widths, dtype=np.float32), lead_shape)

    state = np.empty((*lead_shape, 11), dtype=np.float32)
    state[..., :6] = physical[..., :6]
    delta_xy = physical[..., :2] - goal[..., :2]
    state[..., 6] = np.linalg.norm(delta_xy, axis=-1)

    state_yaw = np.arctan2(physical[..., 3], physical[..., 2])
    goal_yaw = np.arctan2(goal[..., 3], goal[..., 2])
    yaw_error = _wrap_angle(state_yaw - goal_yaw)
    state[..., 7] = yaw_error / np.pi

    c = np.cos(goal_yaw)
    s = np.sin(goal_yaw)
    local_x = c * delta_xy[..., 0] + s * delta_xy[..., 1]
    local_y = -s * delta_xy[..., 0] + c * delta_xy[..., 1]
    relative_yaw = state_yaw - goal_yaw
    abs_c = np.abs(np.cos(relative_yaw))
    abs_s = np.abs(np.sin(relative_yaw))
    car_extent_x = 0.5 * (abs_c * car_length + abs_s * car_width)
    car_extent_y = 0.5 * (abs_s * car_length + abs_c * car_width)
    half_l = 0.5 * lengths - slot_margin
    half_w = 0.5 * widths - slot_margin
    state[..., 8] = (
        (np.abs(local_x) + car_extent_x <= half_l + 1e-8)
        & (np.abs(local_y) + car_extent_y <= half_w + 1e-8)
    ).astype(np.float32)
    state[..., 9] = physical[..., 6]
    state[..., 10] = physical[..., 7]
    return state


def _infer_episode_ids(terminals: np.ndarray) -> np.ndarray:
    result = np.zeros(len(terminals), dtype=np.int32)
    episode = 0
    for index, terminal in enumerate(terminals):
        result[index] = episode
        episode += int(terminal)
    return result


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
        if np.any(terminals[indices[:-1]]):
            raise ValueError(f"episode {episode_id} has an early terminal")
        ends[indices] = indices[-1]
    return ends


@dataclass
class CarParkingDataset:
    """Transition store shared by HIQL, TR-HIQL, PBG, and PBF."""

    raw_observations: np.ndarray
    actions: np.ndarray
    raw_next_observations: np.ndarray
    observations: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    deaths: np.ndarray
    successes: np.ndarray
    commanded_goals: np.ndarray
    slot_lengths: np.ndarray
    slot_widths: np.ndarray
    episode_ids: np.ndarray
    task_ids: np.ndarray
    episode_ends: np.ndarray
    success_indices: np.ndarray
    valid_indices: np.ndarray
    path_horizon: int = 8
    action_chunk_horizon: int = 5
    goal_relabel_prob: float = 0.5
    value_base_horizon: int = 5
    car_length: float = 0.18
    car_width: float = 0.10
    slot_margin: float = 0.008

    def __len__(self) -> int:
        return int(len(self.raw_observations))

    @property
    def horizon(self) -> int:
        return int(self.path_horizon)

    def _state(self, physical: np.ndarray, goal: np.ndarray, index: int) -> np.ndarray:
        return reconstruct_states(
            physical,
            goal,
            self.slot_lengths[index],
            self.slot_widths[index],
            car_length=self.car_length,
            car_width=self.car_width,
            slot_margin=self.slot_margin,
        )

    def _physical_at_offset(self, index: int, offset: int) -> np.ndarray:
        if offset == 0:
            return self.raw_observations[index]
        return self.raw_next_observations[index + offset - 1]

    def _future_index(self, rng: np.random.Generator, index: int) -> int:
        return int(rng.integers(index, int(self.episode_ends[index]) + 1))

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        sampled_tasks = rng.integers(1, 6, size=batch_size)
        indices = np.asarray(
            [
                rng.choice(self.valid_indices[self.task_ids[self.valid_indices] == task])
                for task in sampled_tasks
            ],
            dtype=np.int64,
        )
        state_dim = 11
        goals = np.empty((batch_size, 5), dtype=np.float32)
        obs = np.empty((batch_size, state_dim), dtype=np.float32)
        next_obs = np.empty_like(obs)
        target_states = np.empty_like(obs)
        base_states = np.empty_like(obs)
        split_states = np.empty_like(obs)
        paths = np.empty(
            (batch_size, self.path_horizon + 1, state_dim), dtype=np.float32
        )
        value_offsets = np.empty(batch_size, dtype=np.int64)
        base_offsets = np.empty(batch_size, dtype=np.int64)
        split_offsets = np.zeros(batch_size, dtype=np.int64)
        transitive_valid = np.zeros(batch_size, dtype=np.float32)
        commanded_mask = np.zeros(batch_size, dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)

        h_a = int(self.action_chunk_horizon)
        action_chunks = np.empty(
            (batch_size, h_a, self.actions.shape[1]), dtype=np.float32
        )
        chunk_next = np.empty_like(obs)

        for row, raw_index in enumerate(indices):
            index = int(raw_index)
            use_future = bool(rng.random() < self.goal_relabel_prob)
            success_index = int(self.success_indices[index])
            if use_future or success_index < index:
                target = self._future_index(rng, index)
                goal = achieved_goals(self.raw_next_observations[target]).copy()
                rewards[row] = float(target == index)
            else:
                target = success_index
                goal = self.commanded_goals[index].copy()
                commanded_mask[row] = 1.0
                rewards[row] = float(self.successes[index])

            goals[row] = goal
            obs[row] = self._state(self.raw_observations[index], goal, index)
            next_obs[row] = self._state(self.raw_next_observations[index], goal, index)
            target_states[row] = self._state(
                self.raw_next_observations[target], goal, target
            )
            total_offset = target - index + 1
            value_offsets[row] = total_offset

            for offset in range(self.path_horizon + 1):
                effective = min(offset, total_offset)
                physical = self._physical_at_offset(index, effective)
                state_index = index if effective == 0 else index + effective - 1
                paths[row, offset] = self._state(physical, goal, state_index)

            if total_offset >= 2:
                split = int(rng.integers(1, total_offset))
                split_offsets[row] = split
                split_states[row] = self._state(
                    self._physical_at_offset(index, split),
                    goal,
                    index + split - 1,
                )
                transitive_valid[row] = 1.0
            else:
                split_states[row] = obs[row]

            max_base = min(
                self.value_base_horizon,
                int(self.episode_ends[index] - index + 1),
            )
            base = int(rng.integers(1, max_base + 1))
            base_offsets[row] = base
            base_states[row] = self._state(
                self._physical_at_offset(index, base),
                goal,
                index + base - 1,
            )

            chunk_indices = index + np.arange(h_a)
            action_chunks[row] = self.actions[chunk_indices]
            chunk_target = index + h_a - 1
            chunk_next[row] = self._state(
                self.raw_next_observations[chunk_target], goal, chunk_target
            )

        masks = (
            (1.0 - rewards) * (1.0 - self.deaths[indices].astype(np.float32))
        ).astype(np.float32)
        path_targets = paths[:, -1]
        return {
            "observations": obs,
            "actions": self.actions[indices].astype(np.float32),
            "next_observations": next_obs,
            "terminals": self.terminals[indices].astype(np.float32),
            "task_ids": self.task_ids[indices].astype(np.int32),
            "rewards": rewards,
            "masks": masks,
            "goals": goals,
            "actor_goals": goals,
            "commanded_goal_mask": commanded_mask,
            "subgoal_value_goals": target_states,
            "value_goals": target_states,
            "value_base_goals": base_states,
            "value_base_offsets": base_offsets.astype(np.float32),
            "trans_v_split_observations": split_states,
            "trans_v_left_goals": split_states,
            "trans_v_right_goals": target_states,
            "trans_v_valid_mask": transitive_valid,
            "trans_v_split_offsets": split_offsets.astype(np.float32),
            "value_offsets": value_offsets.astype(np.float32),
            "low_actor_goals": path_targets,
            "high_actor_goals": goals,
            "high_actor_targets": path_targets,
            "subgoals": path_targets,
            "path_observations": paths,
            "action_chunk_actions": action_chunks.reshape(batch_size, -1),
            "action_chunk_next_observations": chunk_next,
        }


@dataclass
class _CarParkingSequenceDataset:
    raw_observations: np.ndarray
    actions: np.ndarray
    raw_next_observations: np.ndarray
    commanded_goals: np.ndarray
    slot_lengths: np.ndarray
    slot_widths: np.ndarray
    success_indices: np.ndarray
    episode_ends: np.ndarray
    task_ids: np.ndarray
    valid_indices: np.ndarray
    config: dict
    car_length: float = 0.18
    car_width: float = 0.10
    slot_margin: float = 0.008

    def __len__(self) -> int:
        return int(len(self.raw_observations))

    def _sample_indices(
        self, rng: np.random.Generator, batch_size: int
    ) -> np.ndarray:
        sampled_tasks = rng.integers(1, 6, size=batch_size)
        return np.asarray(
            [
                rng.choice(self.valid_indices[self.task_ids[self.valid_indices] == task])
                for task in sampled_tasks
            ],
            dtype=np.int64,
        )

    def _future_indices(
        self, rng: np.random.Generator, indices: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            [
                rng.integers(int(index) + 1, int(self.episode_ends[index]) + 1)
                for index in indices
            ],
            dtype=np.int64,
        )

    def _goal_targets(
        self, rng: np.random.Generator, indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = self._future_indices(rng, indices)
        goals = achieved_goals(self.raw_next_observations[targets]).copy()
        probability = float(self.config.get("commanded_goal_prob", 0.5))
        command_targets = self.success_indices[indices]
        use_command = (
            (command_targets > indices)
            & (rng.random(len(indices)) >= 1.0 - probability)
        )
        targets[use_command] = command_targets[use_command]
        goals[use_command] = self.commanded_goals[indices[use_command]]
        return targets, goals

    def _states(
        self, physical: np.ndarray, goals: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        return reconstruct_states(
            physical,
            goals,
            self.slot_lengths[indices],
            self.slot_widths[indices],
            car_length=self.car_length,
            car_width=self.car_width,
            slot_margin=self.slot_margin,
        )


@dataclass
class CarParkingTRLDataset(_CarParkingSequenceDataset):
    """TRL midpoint adapter with per-sample goal-safe observations."""

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        indices = self._sample_indices(rng, batch_size)
        targets, goals = self._goal_targets(rng, indices)
        offsets = targets - indices + 1
        midpoint_offsets = np.asarray(
            [rng.integers(1, int(offset)) for offset in offsets],
            dtype=np.int64,
        )
        midpoint_indices = indices + midpoint_offsets
        midpoint_physical = self.raw_observations[midpoint_indices]
        observations = self._states(self.raw_observations[indices], goals, indices)
        next_observations = self._states(
            self.raw_next_observations[indices], goals, indices
        )
        midpoint_observations = self._states(
            midpoint_physical, goals, midpoint_indices
        )
        return {
            "observations": observations,
            "actions": self.actions[indices].astype(np.float32),
            "next_observations": next_observations,
            "value_goals": goals.astype(np.float32),
            "actor_goals": goals.astype(np.float32),
            "value_offsets": offsets.astype(np.float32),
            "value_midpoint_offsets": midpoint_offsets.astype(np.float32),
            "value_midpoint_observations": midpoint_observations,
            "value_midpoint_goals": np.concatenate(
                [midpoint_observations[:, :4], midpoint_observations[:, 10:11]],
                axis=-1,
            ),
            "value_midpoint_actions": self.actions[
                midpoint_indices
            ].astype(np.float32),
        }


@dataclass
class CarParkingDQCDataset(_CarParkingSequenceDataset):
    """DQC action-chunk adapter with episode-safe goal reconstruction."""

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        indices = self._sample_indices(rng, batch_size)
        targets, goals = self._goal_targets(rng, indices)
        horizon = int(self.config.get("backup_horizon", 8))
        goal_offsets = targets - indices + 1
        backup = np.minimum(horizon, goal_offsets).astype(np.int64)
        reaches_goal = goal_offsets <= horizon
        next_indices = indices + backup - 1
        chunk_indices = indices[:, None] + np.arange(horizon)[None]
        chunks = self.actions[chunk_indices].reshape(batch_size, -1)
        observations = self._states(self.raw_observations[indices], goals, indices)
        next_observations = self._states(
            self.raw_next_observations[indices], goals, indices
        )
        backup_observations = self._states(
            self.raw_next_observations[next_indices], goals, next_indices
        )
        discount = float(self.config.get("discount", 0.99))
        rewards = np.power(discount, backup).astype(np.float32) * reaches_goal
        return {
            "observations": observations,
            "actions": self.actions[indices].astype(np.float32),
            "next_observations": next_observations,
            "high_value_goals": goals.astype(np.float32),
            "high_value_next_observations": backup_observations,
            "high_value_action_chunks": chunks.astype(np.float32),
            "high_value_backup_horizon": backup.astype(np.float32),
            "high_value_rewards": rewards.astype(np.float32),
            "high_value_masks": (1.0 - reaches_goal.astype(np.float32)),
            "valids": np.ones((batch_size, horizon), dtype=np.float32),
        }


def _scalar(raw: dict[str, np.ndarray], key: str, default: float) -> float:
    return float(np.asarray(raw[key]).item()) if key in raw else float(default)


def load_car_parking_dataset(
    path: str | Path,
    *,
    path_horizon: int = 8,
    action_chunk_horizon: int = 5,
    goal_relabel_prob: float = 0.5,
    value_base_horizon: int = 5,
) -> CarParkingDataset:
    """Load raw 8-D parking physics and create a goal-safe sampler."""
    if path_horizon < 1 or action_chunk_horizon < 1:
        raise ValueError("path and action-chunk horizons must be at least 1")
    if not 0.0 <= goal_relabel_prob <= 1.0:
        raise ValueError("goal_relabel_prob must be in [0, 1]")
    required = {
        "observations",
        "actions",
        "next_observations",
        "commanded_goals",
        "terminals",
        "successes",
        "deaths",
        "task_ids",
        "slot_lengths",
        "slot_widths",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"dataset is missing fields: {sorted(missing)}")
        raw = {key: np.asarray(archive[key]) for key in archive.files}
    schema = (
        str(np.asarray(raw["dataset_schema"]).item())
        if "dataset_schema" in raw
        else "legacy"
    )
    if schema != "parking_v2":
        raise ValueError(f"dataset schema {schema!r} is stale; regenerate parking_v2")

    physical = np.asarray(raw["observations"], dtype=np.float32)
    next_physical = np.asarray(raw["next_observations"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.float32)
    goals = np.asarray(raw["commanded_goals"], dtype=np.float32)
    terminals = np.asarray(raw["terminals"], dtype=bool)
    successes = np.asarray(raw["successes"], dtype=bool)
    deaths = np.asarray(raw["deaths"], dtype=bool)
    task_ids = np.asarray(raw["task_ids"], dtype=np.int8)
    lengths = np.asarray(raw["slot_lengths"], dtype=np.float32)
    widths = np.asarray(raw["slot_widths"], dtype=np.float32)
    count = len(physical)
    if physical.shape != (count, 8) or next_physical.shape != (count, 8):
        raise ValueError("observations and next_observations must have shape (N, 8)")
    if actions.shape != (count, 2):
        raise ValueError("actions must have shape (N, 2)")
    if np.any(np.abs(actions) > 1.00001):
        raise ValueError("actions must already be normalized to [-1, 1]")
    if goals.shape != (count, 5):
        raise ValueError("commanded_goals must have shape (N, 5)")
    for name, value in (
        ("terminals", terminals),
        ("successes", successes),
        ("deaths", deaths),
        ("task_ids", task_ids),
        ("slot_lengths", lengths),
        ("slot_widths", widths),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape (N,)")
    if np.any((task_ids < 1) | (task_ids > 5)):
        raise ValueError("task_ids must be in [1, 5]")
    episode_ids = (
        np.asarray(raw["episode_ids"], dtype=np.int32)
        if "episode_ids" in raw
        else _infer_episode_ids(terminals)
    )
    if episode_ids.shape != (count,):
        raise ValueError("episode_ids must have shape (N,)")
    ends = _episode_end_lookup(episode_ids, terminals)
    success_indices = np.full(count, -1, dtype=np.int64)
    for episode_id in np.unique(episode_ids):
        episode = np.flatnonzero(episode_ids == episode_id)
        hits = episode[successes[episode]]
        if len(hits):
            success_indices[episode] = int(hits[0])

    need = max(path_horizon, action_chunk_horizon)
    all_indices = np.arange(count, dtype=np.int64)
    valid = all_indices[all_indices + need - 1 <= ends]
    if len(valid) == 0:
        raise ValueError("no episode contains a full path/action-chunk window")
    covered_tasks = np.unique(task_ids[valid])
    if not np.array_equal(covered_tasks, np.arange(1, 6)):
        raise ValueError(
            f"valid transitions do not cover all tasks: {covered_tasks}"
        )
    car_length = _scalar(raw, "car_length", 0.18)
    car_width = _scalar(raw, "car_width", 0.10)
    slot_margin = _scalar(raw, "slot_margin", 0.008)
    observations = reconstruct_states(
        physical, goals, lengths, widths,
        car_length=car_length, car_width=car_width, slot_margin=slot_margin,
    )
    next_observations = reconstruct_states(
        next_physical, goals, lengths, widths,
        car_length=car_length, car_width=car_width, slot_margin=slot_margin,
    )
    return CarParkingDataset(
        physical, actions, next_physical, observations, next_observations,
        terminals, deaths, successes, goals, lengths, widths, episode_ids,
        task_ids, ends, success_indices, valid, int(path_horizon),
        int(action_chunk_horizon), float(goal_relabel_prob),
        int(value_base_horizon), car_length, car_width, slot_margin,
    )


def _sequence_kwargs(base: CarParkingDataset, config: dict) -> dict:
    return {
        "raw_observations": base.raw_observations,
        "actions": base.actions,
        "raw_next_observations": base.raw_next_observations,
        "commanded_goals": base.commanded_goals,
        "slot_lengths": base.slot_lengths,
        "slot_widths": base.slot_widths,
        "success_indices": base.success_indices,
        "episode_ends": base.episode_ends,
        "task_ids": base.task_ids,
        "config": dict(config),
        "car_length": base.car_length,
        "car_width": base.car_width,
        "slot_margin": base.slot_margin,
    }


def load_car_parking_trl_dataset(
    path: str | Path, *, config: dict
) -> CarParkingTRLDataset:
    base = load_car_parking_dataset(
        path, path_horizon=max(1, int(config.get("subgoal_steps", 8)))
    )
    valid = np.flatnonzero(
        np.arange(len(base)) < base.episode_ends
    ).astype(np.int64)
    if len(valid) == 0:
        raise ValueError("TRL dataset has no transitions with a strict future")
    return CarParkingTRLDataset(
        valid_indices=valid, **_sequence_kwargs(base, config)
    )


def load_car_parking_dqc_dataset(
    path: str | Path, *, config: dict
) -> CarParkingDQCDataset:
    horizon = int(config.get("backup_horizon", 8))
    if horizon < 1:
        raise ValueError("backup_horizon must be at least 1")
    base = load_car_parking_dataset(
        path, path_horizon=horizon, action_chunk_horizon=horizon
    )
    valid = np.flatnonzero(
        np.arange(len(base)) + horizon - 1 <= base.episode_ends
    ).astype(np.int64)
    if len(valid) == 0:
        raise ValueError("DQC dataset has no full action chunks")
    return CarParkingDQCDataset(
        valid_indices=valid, **_sequence_kwargs(base, config)
    )
