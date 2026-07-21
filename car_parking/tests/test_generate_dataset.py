from __future__ import annotations

import pathlib

import numpy as np
import pytest

from car_parking.generate_dataset import (
    ACTION_ENCODING,
    DATASET_SCHEMA,
    _mixture_schedule,
    _savez_compressed_deterministic,
    _validate_arrays,
    dataset_metadata,
    validate_dataset_file,
)


def _minimal_episode() -> dict[str, np.ndarray]:
    episode_length = 2
    episodes = 5
    count = episode_length * episodes
    terminals = np.tile(np.asarray([False, True]), episodes)
    successes = terminals.copy()
    goals = np.zeros((count, 5), dtype=np.float32)
    goals[:, 2] = 1.0
    goals[:, 4] = 1.0
    return {
        "observations": np.zeros((count, 8), dtype=np.float32),
        "actions": np.zeros((count, 2), dtype=np.float32),
        "next_observations": np.zeros((count, 8), dtype=np.float32),
        "commanded_goals": goals,
        "terminals": terminals,
        "successes": successes,
        "collisions": np.zeros(count, dtype=bool),
        "deaths": np.zeros(count, dtype=bool),
        "health_losses": np.zeros(count, dtype=np.float32),
        "impact_impulses": np.zeros(count, dtype=np.float32),
        "timeouts": np.zeros(count, dtype=bool),
        "episode_ids": np.repeat(np.arange(episodes), episode_length),
        "task_ids": np.repeat(np.arange(1, 6), episode_length),
        "behavior_ids": np.zeros(count, dtype=np.int8),
        "maneuver_ids": np.zeros(count, dtype=np.int8),
        "layout_variants": np.ones(count, dtype=np.int16),
        "slot_shifts": np.zeros(count, dtype=np.float32),
        "slot_lengths": np.full(count, 0.28, dtype=np.float32),
        "slot_widths": np.full(count, 0.14, dtype=np.float32),
    }


def test_mixture_schedule_is_exact_and_shuffled() -> None:
    schedule = _mixture_schedule(np.random.default_rng(4))
    assert len(schedule) == 20
    assert schedule.count("expert") == 14
    assert schedule.count("noisy") == 5
    assert schedule.count("recovery") == 1


def test_metadata_and_atomic_save_validate(tmp_path: pathlib.Path) -> None:
    arrays = _minimal_episode()
    arrays.update(
        dataset_metadata(
            policy="expert",
            size="1k",
            seed=0,
            max_episode_steps=400,
            noise=0.015,
            jitter_position=0.005,
            jitter_heading_deg=1.0,
            slot_shift_bounds=(-0.005, 0.005),
        )
    )
    output = tmp_path / "parking.npz"
    _savez_compressed_deterministic(output, arrays)

    result = validate_dataset_file(
        output, minimum_steps=10, require_schema=True
    )
    assert result == {"steps": 10, "schema": DATASET_SCHEMA}
    with np.load(output, allow_pickle=False) as raw:
        assert str(raw["action_encoding"].item()) == ACTION_ENCODING
    assert not list(tmp_path.glob(".*.tmp"))


def test_validation_rejects_short_or_corrupt_dataset(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "short.npz"
    _savez_compressed_deterministic(output, _minimal_episode())
    with pytest.raises(ValueError, match="transitions < minimum"):
        validate_dataset_file(output, minimum_steps=11)

    corrupt = _minimal_episode()
    corrupt["terminals"][0] = True
    _savez_compressed_deterministic(output, corrupt)
    with pytest.raises(RuntimeError, match="episode boundaries"):
        validate_dataset_file(output)


def test_random_dataset_allows_complete_failed_episodes() -> None:
    arrays = _minimal_episode()
    arrays["successes"][:] = False
    _validate_arrays(arrays, require_success=False)
    with pytest.raises(RuntimeError, match="terminate successfully"):
        _validate_arrays(arrays, require_success=True)
