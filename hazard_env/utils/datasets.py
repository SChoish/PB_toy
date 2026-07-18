"""Navigate-dataset loader with geometric goal / path relabeling."""

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
    future_value_observations: np.ndarray
    path_observations: np.ndarray  # (N, K+1, D) true bridge supervision path

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.path_observations.shape[1] - 1)

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, np.ndarray]:
        idx = rng.integers(0, len(self), size=batch_size)
        paths = self.path_observations[idx]
        goals = self.future_observations[idx]
        subgoal_value_goals = self.future_value_observations[idx]
        next_xy = self.next_observations[idx, :2]
        goal_xy = goals[:, :2]
        success = (np.linalg.norm(next_xy - goal_xy, axis=-1) <= 0.08).astype(np.float32)
        masks = (1.0 - success) * self.masks[idx]

        k = paths.shape[1] - 1
        if k >= 2:
            split_idx = rng.integers(1, k, size=batch_size)
            tri_valid = np.ones(batch_size, dtype=np.float32)
        else:
            split_idx = np.zeros(batch_size, dtype=np.int64)
            tri_valid = np.zeros(batch_size, dtype=np.float32)
        ar = np.arange(batch_size)
        split_obs = paths[ar, split_idx]
        value_goals = paths[ar, -1]
        base_idx = rng.integers(1, k + 1, size=batch_size)
        base_goals = paths[ar, base_idx]

        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "next_observations": self.next_observations[idx],
            "terminals": self.terminals[idx],
            "rewards": success.astype(np.float32),
            "masks": masks.astype(np.float32),
            "goals": goals.astype(np.float32),
            "actor_goals": goals.astype(np.float32),
            "subgoal_value_goals": subgoal_value_goals.astype(np.float32),
            "value_goals": value_goals.astype(np.float32),
            "value_base_goals": base_goals.astype(np.float32),
            "value_base_offsets": base_idx.astype(np.float32),
            "trans_v_split_observations": split_obs.astype(np.float32),
            "trans_v_left_goals": split_obs.astype(np.float32),
            "trans_v_right_goals": value_goals.astype(np.float32),
            "trans_v_valid_mask": tri_valid,
            "trans_v_split_offsets": split_idx.astype(np.float32),
            "value_offsets": np.full(batch_size, k, dtype=np.float32),
            "low_actor_goals": goals.astype(np.float32),
            "high_actor_goals": goals.astype(np.float32),
            "high_actor_targets": paths[:, -1],
            "subgoals": paths[:, -1],
            "path_observations": paths,
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
    """Load OGBench-style compact navigate npz into transitions + K-step paths."""
    path = Path(path)
    raw = np.load(path)
    obs = np.asarray(raw["observations"], dtype=np.float32)
    actions = normalize_actions(np.asarray(raw["actions"], dtype=np.float32))
    terminals = np.asarray(raw["terminals"], dtype=bool)
    k = int(subgoal_steps)

    bounds = _episode_bounds(terminals)
    valid_parts = [
        np.arange(start, end - k, dtype=np.int64)
        for start, end in bounds
        if end - k > start
    ]
    if not valid_parts:
        raise ValueError(f"no episode contains a full K={k} path")
    idxs = np.concatenate(valid_parts)
    next_idxs = idxs + 1

    observations = obs[idxs]
    next_observations = obs[next_idxs]
    acts = actions[idxs]
    terms = terminals[idxs]
    masks = (1.0 - terms.astype(np.float32)).astype(np.float32)

    rng = np.random.default_rng(seed)
    future = np.empty_like(observations)
    future_value = np.empty_like(observations)
    paths = np.zeros((len(idxs), k + 1, obs.shape[-1]), dtype=np.float32)
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
            for i in range(k + 1):
                paths[r, i] = obs[t + i]
            sub_t = t + k
            if rng.random() < goal_relabel_prob:
                future[r] = obs[sub_t]
                future_value[r] = obs[sub_t]
            elif commanded is not None:
                # Commanded goals specify xy only. Resolve them to a real
                # observation from this episode instead of inventing a
                # zero-velocity "full" state.
                episode_states = obs[start:end]
                nearest = int(
                    np.argmin(
                        np.sum(
                            (
                                episode_states[:, :2]
                                - commanded[r, :2][None, :]
                            )
                            ** 2,
                            axis=1,
                        )
                    )
                )
                future[r] = commanded[r]
                future_value[r] = episode_states[nearest]
            else:
                future[r] = obs[ep_last]
                future_value[r] = obs[ep_last]

    return TransitionDataset(
        observations=observations,
        actions=acts,
        next_observations=next_observations,
        terminals=terms,
        rewards=np.zeros(len(idxs), dtype=np.float32),
        masks=masks,
        future_observations=future.astype(np.float32),
        future_value_observations=future_value.astype(np.float32),
        path_observations=paths,
    )


@dataclass
class HGCNavigateDataset:
    """OGBench-style hierarchical GC dataset over the hazard navigate npz."""

    observations: np.ndarray
    actions: np.ndarray
    terminals: np.ndarray
    terminal_locs: np.ndarray
    valid_idxs: np.ndarray
    config: dict

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def _random_idxs(self, rng: np.random.Generator, batch_size: int) -> np.ndarray:
        return self.valid_idxs[
            rng.integers(0, len(self.valid_idxs), size=batch_size)
        ]

    def _sample_goals(
        self,
        rng: np.random.Generator,
        idxs: np.ndarray,
        *,
        p_curgoal: float,
        p_trajgoal: float,
        p_randomgoal: float,
        geom_sample: bool,
    ) -> np.ndarray:
        del p_randomgoal
        batch_size = len(idxs)
        random_goal_idxs = self._random_idxs(rng, batch_size)
        final_state_idxs = self.terminal_locs[
            np.searchsorted(self.terminal_locs, idxs)
        ]
        discount = float(self.config.get("discount", 0.99))
        if geom_sample:
            offsets = rng.geometric(p=1.0 - discount, size=batch_size)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = rng.random(batch_size)
            traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1.0 - distances)
                )
            ).astype(np.int64)
        if p_curgoal >= 1.0 - 1e-8:
            return idxs.copy()
        goal_idxs = np.where(
            rng.random(batch_size) < p_trajgoal / max(1.0 - p_curgoal, 1e-8),
            traj_goal_idxs,
            random_goal_idxs,
        )
        return np.where(rng.random(batch_size) < p_curgoal, idxs, goal_idxs)

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, np.ndarray]:
        cfg = self.config
        idxs = self._random_idxs(rng, batch_size)
        next_idxs = np.minimum(idxs + 1, len(self) - 1)
        obs = self.observations
        final_state_idxs = self.terminal_locs[
            np.searchsorted(self.terminal_locs, idxs)
        ]

        value_goal_idxs = self._sample_goals(
            rng,
            idxs,
            p_curgoal=float(cfg.get("value_p_curgoal", 0.2)),
            p_trajgoal=float(cfg.get("value_p_trajgoal", 0.5)),
            p_randomgoal=float(cfg.get("value_p_randomgoal", 0.3)),
            geom_sample=bool(cfg.get("value_geom_sample", True)),
        )
        successes = (idxs == value_goal_idxs).astype(np.float32)
        masks = 1.0 - successes
        rewards = successes - (1.0 if cfg.get("gc_negative", True) else 0.0)

        low_goal_idxs = np.minimum(
            idxs + int(cfg.get("subgoal_steps", 8)), final_state_idxs
        )

        if bool(cfg.get("actor_geom_sample", False)):
            offsets = rng.geometric(
                p=1.0 - float(cfg.get("discount", 0.99)), size=batch_size
            )
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = rng.random(batch_size)
            high_traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1.0 - distances)
                )
            ).astype(np.int64)
        high_traj_target_idxs = np.minimum(
            idxs + int(cfg.get("subgoal_steps", 8)), high_traj_goal_idxs
        )
        high_random_goal_idxs = self._random_idxs(rng, batch_size)
        high_random_target_idxs = np.minimum(
            idxs + int(cfg.get("subgoal_steps", 8)), final_state_idxs
        )
        pick_random = rng.random(batch_size) < float(
            cfg.get("actor_p_randomgoal", 0.0)
        )
        high_goal_idxs = np.where(
            pick_random, high_random_goal_idxs, high_traj_goal_idxs
        )
        high_target_idxs = np.where(
            pick_random, high_random_target_idxs, high_traj_target_idxs
        )

        return {
            "observations": obs[idxs].astype(np.float32),
            "actions": self.actions[idxs].astype(np.float32),
            "next_observations": obs[next_idxs].astype(np.float32),
            "terminals": self.terminals[idxs].astype(np.float32),
            "rewards": rewards.astype(np.float32),
            "masks": masks.astype(np.float32),
            "value_goals": obs[value_goal_idxs].astype(np.float32),
            "low_actor_goals": obs[low_goal_idxs].astype(np.float32),
            "high_actor_goals": obs[high_goal_idxs].astype(np.float32),
            "high_actor_targets": obs[high_target_idxs].astype(np.float32),
            "actor_goals": obs[high_goal_idxs].astype(np.float32),
            "goals": obs[high_goal_idxs].astype(np.float32),
        }


def load_hgc_navigate_dataset(
    path: str | Path,
    *,
    config: dict | None = None,
    seed: int = 0,
) -> HGCNavigateDataset:
    """Load navigate npz as an OGBench-style hierarchical GC dataset."""
    del seed
    path = Path(path)
    raw = np.load(path)
    obs = np.asarray(raw["observations"], dtype=np.float32)
    actions = normalize_actions(np.asarray(raw["actions"], dtype=np.float32))
    raw_terminals = np.asarray(raw["terminals"], dtype=np.float32)
    if raw_terminals[-1] < 0.5:
        raw_terminals = raw_terminals.copy()
        raw_terminals[-1] = 1.0
    valid_idxs = np.flatnonzero(raw_terminals < 0.5)
    next_terminals = np.concatenate([raw_terminals[1:], [1.0]])
    terminals = np.minimum(raw_terminals + next_terminals, 1.0).astype(np.float32)
    terminal_locs = np.flatnonzero(terminals > 0.5)
    cfg = {
        "discount": 0.99,
        "subgoal_steps": 8,
        "gc_negative": True,
        "value_p_curgoal": 0.2,
        "value_p_trajgoal": 0.5,
        "value_p_randomgoal": 0.3,
        "value_geom_sample": True,
        "actor_p_curgoal": 0.0,
        "actor_p_trajgoal": 1.0,
        "actor_p_randomgoal": 0.0,
        "actor_geom_sample": False,
    }
    if config:
        cfg.update(config)
    return HGCNavigateDataset(
        observations=obs,
        actions=actions,
        terminals=terminals,
        terminal_locs=terminal_locs.astype(np.int64),
        valid_idxs=valid_idxs.astype(np.int64),
        config=cfg,
    )


@dataclass
class GoalSequenceDataset:
    """Episode-aware base for official TRL and DQC relabeling."""

    observations: np.ndarray
    actions: np.ndarray
    terminals: np.ndarray
    terminal_locs: np.ndarray
    initial_locs: np.ndarray
    valid_idxs: np.ndarray
    config: dict

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def _random_idxs(
        self, rng: np.random.Generator, batch_size: int
    ) -> np.ndarray:
        return self.valid_idxs[
            rng.integers(0, len(self.valid_idxs), size=batch_size)
        ]

    def _final_idxs(self, idxs: np.ndarray) -> np.ndarray:
        return self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

    def _sample_goals(
        self,
        rng: np.random.Generator,
        idxs: np.ndarray,
        *,
        prefix: str,
    ) -> np.ndarray:
        p_cur = float(self.config.get(f"{prefix}_p_curgoal", 0.0))
        p_traj = float(self.config.get(f"{prefix}_p_trajgoal", 1.0))
        p_random = float(self.config.get(f"{prefix}_p_randomgoal", 0.0))
        if not np.isclose(p_cur + p_traj + p_random, 1.0):
            raise ValueError(f"{prefix} goal probabilities must sum to one")
        random_idxs = self._random_idxs(rng, len(idxs))
        final_idxs = self._final_idxs(idxs)
        if bool(self.config.get(f"{prefix}_geom_sample", True)):
            offsets = rng.geometric(
                p=1.0 - float(self.config.get("discount", 0.99)),
                size=len(idxs),
            )
            traj_idxs = np.minimum(idxs + offsets, final_idxs)
        else:
            distances = rng.random(len(idxs))
            traj_idxs = np.round(
                np.minimum(idxs + 1, final_idxs) * distances
                + final_idxs * (1.0 - distances)
            ).astype(np.int64)
        if p_cur >= 1.0:
            return idxs.copy()
        goals = np.where(
            rng.random(len(idxs)) < p_traj / max(1.0 - p_cur, 1e-8),
            traj_idxs,
            random_idxs,
        )
        return np.where(rng.random(len(idxs)) < p_cur, idxs, goals)


@dataclass
class TRLNavigateDataset(GoalSequenceDataset):
    """Official TRL midpoint relabeling over compact hazard trajectories."""

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        idxs = self._random_idxs(rng, batch_size)
        value_goal_idxs = self._sample_goals(rng, idxs, prefix="value")
        actor_goal_idxs = self._sample_goals(rng, idxs, prefix="actor")
        if np.any(value_goal_idxs <= idxs):
            raise RuntimeError("TRL value goals must be strictly future states")
        midpoint_idxs = np.asarray(
            [
                rng.integers(int(start), int(goal))
                for start, goal in zip(idxs, value_goal_idxs, strict=False)
            ],
            dtype=np.int64,
        )
        obs = self.observations
        return {
            "observations": obs[idxs].astype(np.float32),
            "actions": self.actions[idxs].astype(np.float32),
            "next_observations": obs[idxs + 1].astype(np.float32),
            "value_goals": obs[value_goal_idxs].astype(np.float32),
            "actor_goals": obs[actor_goal_idxs].astype(np.float32),
            "value_offsets": (value_goal_idxs - idxs).astype(np.float32),
            "value_midpoint_offsets": (midpoint_idxs - idxs).astype(np.float32),
            "value_midpoint_observations": obs[midpoint_idxs].astype(np.float32),
            "value_midpoint_goals": obs[midpoint_idxs].astype(np.float32),
            "value_midpoint_actions": self.actions[midpoint_idxs].astype(
                np.float32
            ),
        }


@dataclass
class DQCNavigateDataset(GoalSequenceDataset):
    """Official DQC chunk relabeling over compact hazard trajectories."""

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> dict[str, np.ndarray]:
        idxs = self._random_idxs(rng, batch_size)
        goals = self._sample_goals(rng, idxs, prefix="value")
        final_idxs = self._final_idxs(idxs)
        horizon = int(self.config.get("backup_horizon", 8))
        backup = np.minimum(horizon, final_idxs - idxs)
        goal_offsets = goals - idxs
        backup = np.where(
            (goal_offsets >= 0) & (goal_offsets < backup), goal_offsets, backup
        ).astype(np.int64)
        next_idxs = idxs + backup
        chunk_offsets = np.arange(horizon, dtype=np.int64)
        chunk_idxs = idxs[:, None] + chunk_offsets[None]
        chunks = self.actions[chunk_idxs].reshape(batch_size, -1)
        valids = (chunk_idxs < final_idxs[:, None]).astype(np.float32)
        success = (backup < horizon).astype(np.float32)
        discount = float(self.config.get("discount", 0.99))
        rewards = np.power(discount, backup).astype(np.float32) * success
        obs = self.observations
        return {
            "observations": obs[idxs].astype(np.float32),
            "actions": self.actions[idxs].astype(np.float32),
            "next_observations": obs[idxs + 1].astype(np.float32),
            "high_value_goals": obs[goals].astype(np.float32),
            "high_value_next_observations": obs[next_idxs].astype(np.float32),
            "high_value_action_chunks": chunks.astype(np.float32),
            "high_value_backup_horizon": backup.astype(np.float32),
            "high_value_rewards": rewards,
            "high_value_masks": (1.0 - success).astype(np.float32),
            "valids": valids,
        }


def _load_goal_sequence_arrays(path: str | Path):
    raw = np.load(Path(path))
    observations = np.asarray(raw["observations"], dtype=np.float32)
    actions = normalize_actions(np.asarray(raw["actions"], dtype=np.float32))
    terminals = np.asarray(raw["terminals"], dtype=np.float32)
    if terminals[-1] < 0.5:
        terminals = terminals.copy()
        terminals[-1] = 1.0
    terminal_locs = np.flatnonzero(terminals > 0.5).astype(np.int64)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1]).astype(
        np.int64
    )
    return observations, actions, terminals, terminal_locs, initial_locs


def load_trl_navigate_dataset(
    path: str | Path, *, config: dict
) -> TRLNavigateDataset:
    arrays = _load_goal_sequence_arrays(path)
    observations, actions, terminals, terminal_locs, initial_locs = arrays
    valid_parts = [
        np.arange(start, end, dtype=np.int64)
        for start, end in zip(initial_locs, terminal_locs, strict=False)
        if end > start
    ]
    return TRLNavigateDataset(
        observations,
        actions,
        terminals,
        terminal_locs,
        initial_locs,
        np.concatenate(valid_parts),
        dict(config),
    )


def load_dqc_navigate_dataset(
    path: str | Path, *, config: dict
) -> DQCNavigateDataset:
    arrays = _load_goal_sequence_arrays(path)
    observations, actions, terminals, terminal_locs, initial_locs = arrays
    horizon = int(config.get("backup_horizon", 8))
    valid_parts = [
        np.arange(start, end + 1 - horizon, dtype=np.int64)
        for start, end in zip(initial_locs, terminal_locs, strict=False)
        if end + 1 - horizon > start
    ]
    return DQCNavigateDataset(
        observations,
        actions,
        terminals,
        terminal_locs,
        initial_locs,
        np.concatenate(valid_parts),
        dict(config),
    )
