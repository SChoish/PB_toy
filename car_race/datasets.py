"""Lap-aware offline GCRL dataset loader for CarRace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

TaskView = Literal[
    "navigation",
    "lap_1p",
    "lap_2p",
    "lap_3p",
    "lap_4p",
    "lap_5p",
    "lap_6p",
    "lap_7p",
    "lap_8p",
]
LAP_RING_COUNTS = {f"lap_{n}p": n + 1 for n in range(1, 9)}


def goal_success(
    achieved: np.ndarray,
    desired: np.ndarray,
    *,
    goal_radius: float = 0.07,
) -> np.ndarray:
    """Check position, monotone progress, and fixed direction."""
    achieved = np.asarray(achieved, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float32)
    spatial = np.linalg.norm(
        achieved[..., :2] - desired[..., :2], axis=-1
    ) <= goal_radius
    progress = achieved[..., 2] + 1e-6 >= desired[..., 2]
    direction = np.isclose(achieved[..., 3], desired[..., 3], atol=1e-6)
    return spatial & progress & direction


@dataclass
class CarRaceDataset:
    """Transition store with reachable goal sampling for navigation and lap."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    commanded_goals: np.ndarray
    episode_ids: np.ndarray
    episode_ends: np.ndarray
    valid_indices: np.ndarray
    task: TaskView = "navigation"
    path_horizon: int = 8
    action_chunk_horizon: int = 5
    goal_relabel_prob: float = 1.0
    checkpoint_goal_prob: float = 0.0
    random_goal_prob: float = 0.0
    goal_radius: float = 0.07
    value_base_horizon: int = 5

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.path_horizon)

    def _future_index(
        self, rng: np.random.Generator, index: int
    ) -> int:
        end = int(self.episode_ends[index])
        if index >= end:
            return index
        return int(rng.integers(index, end + 1))

    def _reachable_random_index(
        self, rng: np.random.Generator, index: int
    ) -> int:
        current = self.observations[index, :4]
        directions = self.next_observations[:, 3]
        progress = self.next_observations[:, 2]
        candidates = np.flatnonzero(
            np.isclose(directions, current[3], atol=1e-6)
            & (progress + 1e-6 >= current[2])
        )
        if len(candidates) == 0:
            return self._future_index(rng, index)
        return int(candidates[rng.integers(0, len(candidates))])

    def _future_checkpoint_index(
        self, rng: np.random.Generator, index: int
    ) -> int:
        """Sample a later transition that actually reaches a checkpoint."""
        end = int(self.episode_ends[index])
        current_progress = float(self.observations[index, 2])
        candidates = np.arange(index, end + 1, dtype=np.int64)
        crossed = (
            self.next_observations[candidates, 2]
            > self.observations[candidates, 2] + 1e-6
        ) & (
            self.next_observations[candidates, 2]
            > current_progress + 1e-6
        )
        candidates = candidates[crossed]
        if len(candidates) == 0:
            return self._future_index(rng, index)
        return int(candidates[rng.integers(0, len(candidates))])

    def _commanded_goal_state(
        self, index: int
    ) -> tuple[np.ndarray, bool]:
        """Resolve the active waypoint and report whether it was reached."""
        end = int(self.episode_ends[index])
        goal = self.commanded_goals[index]
        candidates = np.arange(index, end + 1, dtype=np.int64)
        reached = goal_success(
            self.next_observations[candidates, :4],
            np.broadcast_to(goal, (len(candidates), 4)),
            goal_radius=self.goal_radius,
        )
        hits = candidates[reached]
        target = int(hits[0]) if len(hits) else end
        return self.next_observations[target], bool(len(hits))

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
            path_target_offset = self.path_horizon
            draw = float(rng.random())
            if draw < self.random_goal_prob:
                goal_index = self._reachable_random_index(rng, int(index))
                goals[row] = self.next_observations[goal_index, :4]
                subgoal_value_goals[row] = self.next_observations[goal_index]
            elif draw < self.random_goal_prob + self.checkpoint_goal_prob:
                goal_index = self._future_checkpoint_index(rng, int(index))
                goals[row] = self.next_observations[goal_index, :4]
                subgoal_value_goals[row] = self.next_observations[goal_index]
                path_target_offset = min(
                    self.path_horizon, int(goal_index - index + 1)
                )
            elif draw < (
                self.random_goal_prob
                + self.checkpoint_goal_prob
                + self.goal_relabel_prob
            ):
                goal_index = self._future_index(rng, int(index))
                goals[row] = self.next_observations[goal_index, :4]
                subgoal_value_goals[row] = self.next_observations[goal_index]
                path_target_offset = min(
                    self.path_horizon, int(goal_index - index + 1)
                )
            else:
                resolved_state, reachable = self._commanded_goal_state(int(index))
                subgoal_value_goals[row] = resolved_state
                goals[row] = (
                    self.commanded_goals[index]
                    if reachable
                    else resolved_state[:4]
                )

            if path_target_offset < self.path_horizon:
                paths[row, path_target_offset:] = paths[row, path_target_offset]

            # TRL tuple uses one strictly future same-trajectory full-state goal:
            # i < split < j and V(i,j) = V(i,split) V(split,j).
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
                self.value_base_horizon,
                int(self.episode_ends[index] - index + 1),
            )
            base_offset = int(rng.integers(1, max_base_offset + 1))
            base_offsets[row] = base_offset
            base_states[row] = self._state_at_offset(
                int(index), base_offset
            )

        next_observations = self.next_observations[idxs]
        successes = goal_success(
            next_observations[:, :4], goals, goal_radius=self.goal_radius
        ).astype(np.float32)
        masks = (
            (1.0 - successes)
            * (1.0 - self.terminals[idxs].astype(np.float32))
        ).astype(np.float32)
        path_goal_states = paths[:, -1]

        h_a = max(1, int(self.action_chunk_horizon))
        action_dim = int(self.actions.shape[-1])
        action_chunks = np.zeros((batch_size, h_a, action_dim), dtype=np.float32)
        action_chunk_next = np.empty_like(subgoal_value_goals)
        for row, index in enumerate(idxs):
            end = int(self.episode_ends[index])
            for t in range(h_a):
                src = min(int(index) + t, end)
                action_chunks[row, t] = self.actions[src]
            next_idx = min(int(index) + h_a, end)
            action_chunk_next[row] = self.next_observations[next_idx]

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
            # TR-HIQL low-actor conditions on full-state subgoals; HIQL truncates
            # via goal_dim so path endpoints remain compatible.
            "low_actor_goals": path_goal_states.astype(np.float32),
            "high_actor_goals": goals,
            "high_actor_targets": path_goal_states.astype(np.float32),
            "subgoals": path_goal_states.astype(np.float32),
            "path_observations": paths.astype(np.float32),
            "action_chunk_actions": action_chunks.reshape(batch_size, -1),
            "action_chunk_next_observations": action_chunk_next.astype(np.float32),
        }


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


def _augment_task_view(
    raw_observations: np.ndarray,
    raw_next_observations: np.ndarray,
    episode_ids: np.ndarray,
    task: TaskView,
    *,
    lap_directions: np.ndarray | None = None,
    lap_start_angles: np.ndarray | None = None,
    raw_commanded_goals: np.ndarray | None = None,
    goal_radius: float = 0.07,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build navigation or waypoint-exact lap annotations over raw physics."""
    count = len(raw_observations)
    state_dim = 10 if task == "navigation" else 14
    observations = np.zeros((count, state_dim), dtype=np.float32)
    next_observations = np.zeros((count, state_dim), dtype=np.float32)
    observations[:, :2] = raw_observations[:, :2]
    observations[:, 4:10] = raw_observations[:, 2:]
    next_observations[:, :2] = raw_next_observations[:, :2]
    next_observations[:, 4:10] = raw_next_observations[:, 2:]
    commanded_goals = np.zeros((count, 4), dtype=np.float32)

    if task != "navigation" and (
        lap_directions is None or lap_start_angles is None
    ):
        raise ValueError(
            "lap task views require universal lap metadata; regenerate with "
            "`python -m car_race.generate_dataset --task lap ...`"
        )

    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        if task == "navigation":
            if raw_commanded_goals is not None:
                commanded_goals[indices] = raw_commanded_goals[indices]
            else:
                # Legacy files did not store the active navigation command.
                final_xy = raw_next_observations[indices[-1], :2]
                commanded_goals[indices, :2] = final_xy
            continue

        assert lap_directions is not None and lap_start_angles is not None
        directions = np.asarray(lap_directions[indices], dtype=np.int32)
        start_angles = np.asarray(lap_start_angles[indices], dtype=np.float32)
        direction = int(directions[0])
        start_angle = float(start_angles[0])
        if direction not in (-1, 1):
            raise ValueError(f"episode {episode_id} has invalid lap direction")
        if np.any(directions != direction) or not np.allclose(
            start_angles, start_angle, atol=1e-6
        ):
            raise ValueError(f"episode {episode_id} has inconsistent lap metadata")

        ring_count = LAP_RING_COUNTS[task]
        waypoint_angles = start_angle + direction * np.arange(ring_count) * (
            2.0 * np.pi / ring_count
        )
        waypoints = 0.575 * np.stack(
            [np.cos(waypoint_angles), np.sin(waypoint_angles)], axis=1
        ).astype(np.float32)
        completed = 0
        active_ordinal = 1
        reached_previous = False

        for index in indices:
            active_index = active_ordinal % ring_count
            waypoint = waypoints[active_index]
            target_progress = min((completed + 1) / ring_count, 1.0)
            index_normalized = active_index / max(ring_count - 1, 1)

            observations[index, 2] = completed / ring_count
            observations[index, 3] = direction
            observations[index, 10:] = np.array(
                [index_normalized, float(reached_previous), *waypoint],
                dtype=np.float32,
            )
            commanded_goals[index] = np.array(
                [*waypoint, target_progress, direction], dtype=np.float32
            )

            hit = bool(
                np.linalg.norm(raw_next_observations[index, :2] - waypoint)
                <= goal_radius
            )
            next_completed = min(completed + int(hit), ring_count)
            next_active_ordinal = active_ordinal + int(hit and next_completed < ring_count)
            next_active_index = next_active_ordinal % ring_count
            next_waypoint = waypoints[next_active_index]
            next_index_normalized = next_active_index / max(ring_count - 1, 1)

            next_observations[index, 2] = next_completed / ring_count
            next_observations[index, 3] = direction
            next_observations[index, 10:] = np.array(
                [next_index_normalized, float(hit), *next_waypoint],
                dtype=np.float32,
            )
            completed = next_completed
            active_ordinal = next_active_ordinal
            reached_previous = hit

    return observations, next_observations, commanded_goals


def load_car_race_dataset(
    path: str | Path,
    *,
    task: TaskView = "navigation",
    path_horizon: int = 8,
    action_chunk_horizon: int = 5,
    goal_relabel_prob: float | None = None,
    checkpoint_goal_prob: float | None = None,
    random_goal_prob: float = 0.0,
    value_base_horizon: int = 5,
    goal_radius: float = 0.07,
) -> CarRaceDataset:
    """Create a task view over one shared physical-trajectory NPZ."""
    if task != "navigation" and task not in LAP_RING_COUNTS:
        raise ValueError(f"Unknown task view: {task}")
    default_goal_mix = goal_relabel_prob is None
    if goal_relabel_prob is None:
        goal_relabel_prob = 1.0 if task == "navigation" else 0.30
    if checkpoint_goal_prob is None:
        checkpoint_goal_prob = (
            0.20 if task != "navigation" and default_goal_mix else 0.0
        )
    if path_horizon < 1:
        raise ValueError("path_horizon must be at least 1")
    if action_chunk_horizon < 1:
        raise ValueError("action_chunk_horizon must be at least 1")
    if (
        goal_relabel_prob < 0.0
        or checkpoint_goal_prob < 0.0
        or random_goal_prob < 0.0
    ):
        raise ValueError("goal sampling probabilities must be non-negative")
    if goal_relabel_prob + checkpoint_goal_prob + random_goal_prob > 1.0:
        raise ValueError("goal sampling probabilities must sum to at most 1")

    raw = np.load(Path(path))
    required = {
        "observations",
        "actions",
        "next_observations",
        "terminals",
    }
    missing = required.difference(raw.files)
    if missing:
        raise ValueError(f"dataset is missing fields: {sorted(missing)}")

    raw_observations = np.asarray(raw["observations"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.float32)
    raw_next_observations = np.asarray(
        raw["next_observations"], dtype=np.float32
    )
    terminals = np.asarray(raw["terminals"], dtype=bool)
    if raw_observations.ndim != 2 or raw_observations.shape[1] != 8:
        raise ValueError("shared observations must have shape (N, 8)")
    if raw_next_observations.shape != raw_observations.shape:
        raise ValueError("next_observations must match observations")
    if actions.shape != (len(raw_observations), 2):
        raise ValueError("actions must have shape (N, 2)")
    if np.any(actions < -1.00001) or np.any(actions > 1.00001):
        raise ValueError("actions must already be normalized to [-1, 1]")
    if terminals.shape != (len(raw_observations),):
        raise ValueError("terminals must have shape (N,)")

    episode_ids = (
        np.asarray(raw["episode_ids"], dtype=np.int32)
        if "episode_ids" in raw.files
        else _infer_episode_ids(terminals)
    )
    if episode_ids.shape != terminals.shape:
        raise ValueError("episode_ids must have shape (N,)")
    episode_ends = _episode_end_lookup(episode_ids, terminals)
    lap_directions = (
        np.asarray(raw["lap_directions"], dtype=np.int32)
        if "lap_directions" in raw.files
        else None
    )
    lap_start_angles = (
        np.asarray(raw["lap_start_angles"], dtype=np.float32)
        if "lap_start_angles" in raw.files
        else None
    )
    raw_is_lap = (
        np.asarray(raw["is_lap"], dtype=bool)
        if "is_lap" in raw.files
        else None
    )
    raw_commanded_goals = (
        np.asarray(raw["commanded_goals"], dtype=np.float32)
        if "commanded_goals" in raw.files
        and (raw_is_lap is None or not np.any(raw_is_lap))
        else None
    )
    if raw_commanded_goals is not None and raw_commanded_goals.shape != (
        len(raw_observations),
        4,
    ):
        raise ValueError("commanded_goals must have shape (N, 4)")
    if task != "navigation":
        is_lap = raw_is_lap
        if is_lap is None or is_lap.shape != terminals.shape or not np.all(is_lap):
            raise ValueError(
                "lap task views require a universal lap dataset generated with --task lap"
            )
    observations, next_observations, goals = _augment_task_view(
        raw_observations,
        raw_next_observations,
        episode_ids,
        task,
        lap_directions=lap_directions,
        lap_start_angles=lap_start_angles,
        raw_commanded_goals=raw_commanded_goals,
        goal_radius=goal_radius,
    )

    # A universal dense-checkpoint lap can finish a coarser synthetic task
    # earlier (notably ice lap_1p/lap_2p).  Truncate each task view at its first
    # completion so no post-success states or actions leak into the offline MDP.
    if task != "navigation":
        keep = np.ones(len(observations), dtype=bool)
        synthetic_terminals = terminals.copy()
        for episode_id in np.unique(episode_ids):
            episode_indices = np.flatnonzero(episode_ids == episode_id)
            completed = episode_indices[
                next_observations[episode_indices, 2] >= 1.0 - 1e-6
            ]
            if len(completed) == 0:
                continue
            completion = int(completed[0])
            keep[episode_indices[episode_indices > completion]] = False
            synthetic_terminals[completion] = True
        observations = observations[keep]
        actions = actions[keep]
        next_observations = next_observations[keep]
        synthetic_terminals = synthetic_terminals[keep]
        goals = goals[keep]
        episode_ids = episode_ids[keep]
        terminals = synthetic_terminals
        episode_ends = _episode_end_lookup(episode_ids, terminals)

    indices = np.arange(len(observations), dtype=np.int64)
    need = max(int(path_horizon), int(action_chunk_horizon))
    valid_indices = indices[indices + need - 1 <= episode_ends]
    if len(valid_indices) == 0:
        raise ValueError(
            f"no episode contains a full K={path_horizon} / h_a={action_chunk_horizon} window"
        )
    return CarRaceDataset(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        terminals=terminals,
        commanded_goals=goals,
        episode_ids=episode_ids,
        episode_ends=episode_ends,
        valid_indices=valid_indices,
        task=task,
        path_horizon=int(path_horizon),
        action_chunk_horizon=int(action_chunk_horizon),
        goal_relabel_prob=float(goal_relabel_prob),
        checkpoint_goal_prob=float(checkpoint_goal_prob),
        random_goal_prob=float(random_goal_prob),
        goal_radius=float(goal_radius),
        value_base_horizon=int(value_base_horizon),
    )


@dataclass
class CarRaceGoalSequenceDataset:
    """Episode-aware base used by the TRL and DQC adapters."""

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

    def _reachable_random_indices(
        self, rng: np.random.Generator, indices: np.ndarray
    ) -> np.ndarray:
        result = np.empty_like(indices)
        goal_states = self.next_observations[:, :4]
        for row, index in enumerate(indices):
            current = self.observations[index, :4]
            candidates = np.flatnonzero(
                np.isclose(goal_states[:, 3], current[3], atol=1e-6)
                & (goal_states[:, 2] + 1e-6 >= current[2])
            )
            result[row] = (
                int(candidates[rng.integers(0, len(candidates))])
                if len(candidates)
                else int(index)
            )
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
        random_indices = self._reachable_random_indices(rng, indices)
        return np.where(
            rng.random(len(indices)) < random_probability,
            random_indices,
            future,
        )


@dataclass
class CarRaceTRLDataset(CarRaceGoalSequenceDataset):
    """TRL midpoint batches with 4-D lap-aware goals."""

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        indices = self._sample_indices(rng, batch_size)
        # TRL's midpoint construction requires a strictly future value goal
        # from the same trajectory.
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
class CarRaceDQCDataset(CarRaceGoalSequenceDataset):
    """DQC action-chunk batches with reachable lap goals."""

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
            "high_value_next_observations": self.next_observations[
                next_indices
            ].astype(np.float32),
            "high_value_action_chunks": chunks.astype(np.float32),
            "high_value_backup_horizon": backup.astype(np.float32),
            "high_value_rewards": rewards,
            "high_value_masks": (1.0 - success).astype(np.float32),
            "valids": valids,
        }


def _load_sequence_base(
    path: str | Path, *, task: TaskView
) -> CarRaceDataset:
    return load_car_race_dataset(
        path,
        task=task,
        goal_relabel_prob=1.0,
        checkpoint_goal_prob=0.0,
        random_goal_prob=0.0,
    )


def load_car_race_trl_dataset(
    path: str | Path, *, config: dict, task: TaskView = "navigation"
) -> CarRaceTRLDataset:
    base = _load_sequence_base(path, task=task)
    valid = np.flatnonzero(
        np.arange(len(base)) < base.episode_ends
    ).astype(np.int64)
    if len(valid) == 0:
        raise ValueError("TRL dataset has no nonterminal transitions")
    return CarRaceTRLDataset(
        base.observations,
        base.actions,
        base.next_observations,
        base.episode_ends,
        valid,
        dict(config),
    )


def load_car_race_dqc_dataset(
    path: str | Path, *, config: dict, task: TaskView = "navigation"
) -> CarRaceDQCDataset:
    base = _load_sequence_base(path, task=task)
    horizon = int(config.get("backup_horizon", 8))
    valid = np.flatnonzero(
        np.arange(len(base)) + horizon <= base.episode_ends
    ).astype(np.int64)
    if len(valid) == 0:
        raise ValueError("DQC dataset has no full action chunks")
    return CarRaceDQCDataset(
        base.observations,
        base.actions,
        base.next_observations,
        base.episode_ends,
        valid,
        dict(config),
    )
