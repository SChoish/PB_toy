"""Generate balanced, goal-independent CarParking offline datasets.

Expert/noisy files retain complete successful trajectories, while the default
mixture also preserves complete recovery failures.  Every transition comes
from ``CarParkingEnv.step``; Hybrid A* path points are never stored as
environment transitions.  Named sizes are minimum transition counts, so the
saved split may be slightly larger in order to finish a balanced five-task
round.

Example:
    python -m car_parking.generate_dataset --size 1k --policy mixture
"""

from __future__ import annotations

import argparse
import io
import os
import pathlib
import zipfile
from collections import defaultdict
from typing import Literal

import numpy as np

from .env import (
    MANEUVERS,
    NUM_FIXED_TASKS,
    CarParkingConfig,
    CarParkingEnv,
    fixed_task_options,
)
from .hybrid_astar import PlannerConfig
from .parking_policy import ParkingExpertPolicy

PolicyName = Literal["expert", "noisy", "mixture", "random"]
BehaviorName = Literal["expert", "noisy", "recovery", "random"]
SizeName = Literal["1k", "10k", "100k"]

POLICIES: tuple[PolicyName, ...] = ("expert", "noisy", "mixture", "random")
SIZES: tuple[SizeName, ...] = ("1k", "10k", "100k")
SIZE_STEPS: dict[SizeName, tuple[int, int]] = {
    "1k": (1_000, 100),
    "10k": (10_000, 1_000),
    "100k": (100_000, 10_000),
}
MANEUVER_TO_ID = {name: index for index, name in enumerate(MANEUVERS)}
# Variants already covered by the canonical expert acceptance suite.  Parallel
# has validated mirrored layouts; the other maneuver experts currently have
# one validated orientation each.
VALIDATED_VARIANTS: dict[str, tuple[int, ...]] = {
    "parallel": (1, 2),
    "t_forward": (2,),
    "t_reverse": (1,),
    "angled": (1,),
}
BEHAVIOR_TO_ID = {"expert": 0, "noisy": 1, "recovery": 2, "random": 3}
DATASET_SCHEMA = "parking_v2"
ACTION_ENCODING = "normalized_steering_throttle"
DATASET_PLANNER_CONFIG = PlannerConfig(max_expansions=5_000)


def physical_observation(env: CarParkingEnv) -> np.ndarray:
    """Return the eight goal-independent Markov features stored in raw NPZ."""
    return np.array(
        [
            env.position[0],
            env.position[1],
            np.cos(env.heading),
            np.sin(env.heading),
            env.speed,
            env.steering / env.config.max_steer_angle,
            env.health,
            env.dwell_progress,
        ],
        dtype=np.float32,
    )


def _empty_store() -> defaultdict[str, list]:
    return defaultdict(list)


def _append_transition(
    store: defaultdict[str, list],
    *,
    observation: np.ndarray,
    action: np.ndarray,
    next_observation: np.ndarray,
    goal: np.ndarray,
    terminal: bool,
    terminated: bool,
    truncated: bool,
    info: dict,
    episode_id: int,
    task_id: int,
    behavior: BehaviorName,
    maneuver: str,
    slot_shift_x: float,
    variant: int,
    slot_length: float,
    slot_width: float,
) -> None:
    store["observations"].append(observation)
    store["actions"].append(action)
    store["next_observations"].append(next_observation)
    store["commanded_goals"].append(goal)
    store["terminals"].append(terminal)
    store["successes"].append(bool(info.get("success")))
    store["collisions"].append(bool(info.get("collision")))
    store["deaths"].append(bool(info.get("dead")))
    store["health_losses"].append(float(info.get("health_loss", 0.0)))
    store["impact_impulses"].append(float(info.get("step_impulse", 0.0)))
    store["timeouts"].append(bool(truncated and not terminated))
    store["episode_ids"].append(episode_id)
    store["task_ids"].append(task_id)
    store["behavior_ids"].append(BEHAVIOR_TO_ID[behavior])
    store["maneuver_ids"].append(MANEUVER_TO_ID[maneuver])
    store["slot_shifts"].append(slot_shift_x)
    store["layout_variants"].append(variant)
    store["slot_lengths"].append(slot_length)
    store["slot_widths"].append(slot_width)


def _as_arrays(store: defaultdict[str, list]) -> dict[str, np.ndarray]:
    bool_keys = {
        "terminals",
        "successes",
        "collisions",
        "deaths",
        "timeouts",
    }
    int32_keys = {"episode_ids"}
    int16_keys = {"layout_variants"}
    int8_keys = {"maneuver_ids", "task_ids", "behavior_ids"}
    arrays: dict[str, np.ndarray] = {}
    for key, values in store.items():
        if key in bool_keys:
            dtype = bool
        elif key in int32_keys:
            dtype = np.int32
        elif key in int16_keys:
            dtype = np.int16
        elif key in int8_keys:
            dtype = np.int8
        else:
            dtype = np.float32
        arrays[key] = np.asarray(values, dtype=dtype)
    return arrays


def _noisy_action(
    expert_action: np.ndarray,
    noise_state: np.ndarray,
    rng: np.random.Generator,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply small temporally correlated behavior noise."""
    noise_state = 0.88 * noise_state + rng.normal(0.0, scale, size=2)
    # Throttle noise is kept smaller to preserve safe low-speed approaches.
    noise_state[1] *= 0.65
    action = np.clip(expert_action + noise_state, -1.0, 1.0)
    return action.astype(np.float32), noise_state


def _mixture_schedule(rng: np.random.Generator) -> list[BehaviorName]:
    """Return a shuffled 20-episode block with exact 70/25/5 counts."""
    schedule: list[BehaviorName] = (
        ["expert"] * 14 + ["noisy"] * 5 + ["recovery"]
    )
    rng.shuffle(schedule)
    return schedule


def _rollout_attempt(
    *,
    env: CarParkingEnv,
    task_id: int,
    maneuver: str,
    variant: int,
    slot_shift_x: float,
    seed: int,
    episode_id: int,
    policy_name: BehaviorName,
    rng: np.random.Generator,
    noise: float,
    jitter_position: float,
    jitter_heading_deg: float,
    reuse_nominal_path: bool,
) -> defaultdict[str, list] | None:
    options: dict[str, object] = {
        "maneuver": maneuver,
        "variant": variant,
        "slot_shift_x": slot_shift_x,
    }
    env.reset(seed=seed, options=options)
    expert = None
    try:
        if policy_name != "random":
            expert = ParkingExpertPolicy(
                env,
                planner_config=DATASET_PLANNER_CONFIG,
                allow_replan=policy_name != "recovery",
            )
            if reuse_nominal_path:
                expert.reset()
        if jitter_position > 0.0 or jitter_heading_deg > 0.0:
            options["position"] = (
                np.asarray(env.layout.start)
                + rng.uniform(-jitter_position, jitter_position, size=2)
            )
            options["heading"] = env.layout.start_heading + np.deg2rad(
                rng.uniform(-jitter_heading_deg, jitter_heading_deg)
            )
        env.reset(seed=seed, options=options)
        if expert is not None:
            if reuse_nominal_path:
                # Millimetre-scale train jitter tests tracker recovery without
                # forcing a fresh search for every rejected rollout.
                expert.reset(path=expert.path)
            else:
                # Validation plans from the held-out start pose, so it also
                # checks that the shifted scenario remains feasible.
                expert.reset()
    except (RuntimeError, ValueError):
        return None

    episode = _empty_store()
    observation = physical_observation(env)
    goal = env.desired_goal.copy()
    slot_length = float(env.layout.slot.length)
    slot_width = float(env.layout.slot.width)
    noise_state = np.zeros(2, dtype=np.float32)
    done = False
    final_info: dict = {}
    perturb_steps = (
        int(rng.integers(12, 31)) if policy_name == "recovery" else 0
    )
    rollout_step = 0
    try:
        while not done:
            if policy_name == "recovery" and rollout_step == perturb_steps:
                # Replan from the perturbed physical state, then let the same
                # low-speed controller recover through real env steps.
                candidate = ParkingExpertPolicy(
                    env,
                    planner_config=DATASET_PLANNER_CONFIG,
                    allow_replan=False,
                )
                try:
                    candidate.reset()
                except (RuntimeError, ValueError):
                    # A failed recovery plan is trajectory data, not a reason
                    # to reject and resample the whole episode.
                    pass
                else:
                    expert = candidate
            if policy_name == "random":
                # Temporally correlated random controls produce meaningful
                # driving trajectories while remaining policy-independent.
                noise_state = (
                    0.90 * noise_state
                    + rng.normal(0.0, (0.35, 0.25), size=2)
                )
                action = np.clip(noise_state, -1.0, 1.0).astype(np.float32)
            else:
                assert expert is not None
                action = expert.action()
            add_noise = policy_name == "noisy" or (
                policy_name == "recovery" and rollout_step < perturb_steps
            )
            if add_noise:
                scale = noise if policy_name == "noisy" else max(0.04, 2.5 * noise)
                action, noise_state = _noisy_action(action, noise_state, rng, scale)
            _, _, terminated, truncated, final_info = env.step(action)
            next_observation = physical_observation(env)
            done = bool(terminated or truncated)
            _append_transition(
                episode,
                observation=observation,
                action=action,
                next_observation=next_observation,
                goal=goal,
                terminal=done,
                terminated=terminated,
                truncated=truncated,
                info=final_info,
                episode_id=episode_id,
                task_id=task_id,
                behavior=policy_name,
                maneuver=maneuver,
                slot_shift_x=slot_shift_x,
                variant=variant,
                slot_length=slot_length,
                slot_width=slot_width,
            )
            observation = next_observation
            rollout_step += 1
    except (RuntimeError, ValueError):
        return None

    return episode


def _validate_arrays(
    arrays: dict[str, np.ndarray], *, require_success: bool = True
) -> None:
    count = len(arrays["actions"])
    if count == 0:
        raise RuntimeError("dataset collection produced no transitions")
    if any(len(value) != count for value in arrays.values()):
        raise RuntimeError("dataset arrays have inconsistent lengths")
    terminals = arrays["terminals"]
    episode_ids = arrays["episode_ids"]
    expected_terminals = np.r_[
        episode_ids[1:] != episode_ids[:-1],
        True,
    ]
    if not np.array_equal(terminals, expected_terminals):
        raise RuntimeError("terminals must be true only at episode boundaries")
    if np.any(arrays["deaths"] & arrays["timeouts"]):
        raise RuntimeError("death and timeout cannot label the same transition")
    if require_success and not np.all(arrays["successes"][terminals]):
        raise RuntimeError("retained episodes must terminate successfully")

    if "task_ids" in arrays:
        starts = np.r_[True, episode_ids[1:] != episode_ids[:-1]]
        task_ids = arrays["task_ids"][starts]
        counts = np.bincount(
            task_ids.astype(np.int64),
            minlength=NUM_FIXED_TASKS + 1,
        )[1:]
        if np.any(counts == 0) or int(counts.max() - counts.min()) > 1:
            raise RuntimeError(
                f"fixed task episode counts are not balanced: {counts.tolist()}"
            )


def collect_split(
    *,
    policy: PolicyName,
    minimum_steps: int,
    seed: int,
    max_episode_steps: int = 400,
    noise: float = 0.015,
    jitter_position: float = 0.005,
    jitter_heading_deg: float = 1.0,
    slot_shift_bounds: tuple[float, float] = (-0.005, 0.005),
    max_attempts_per_episode: int = 80,
    reuse_nominal_paths: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Collect complete episodes in balanced five-task rounds."""
    env = CarParkingEnv(
        CarParkingConfig(maneuver="mixed", max_episode_steps=max_episode_steps)
    )
    shift_low, shift_high = map(float, slot_shift_bounds)
    if shift_low > shift_high or max(abs(shift_low), abs(shift_high)) > 0.08:
        raise ValueError(
            "slot_shift_bounds must be ordered and contained in [-0.08, 0.08]"
        )
    rng = np.random.default_rng(seed)
    behavior_rng = np.random.default_rng(seed + 2_000_003)
    store = _empty_store()
    episode_id = 0
    attempt_id = 0
    rejected = 0
    noisy_episodes = 0
    successful_episodes = 0
    behavior_schedule: list[BehaviorName] = []

    while len(store["actions"]) < minimum_steps:
        for task_id in range(1, NUM_FIXED_TASKS + 1):
            task = fixed_task_options(task_id)
            maneuver = str(task["maneuver"])
            variant = int(task["variant"])
            if shift_low >= 0.0:
                magnitude = float(rng.uniform(shift_low, shift_high))
                slot_shift_x = magnitude * (-1.0 if rng.random() < 0.5 else 1.0)
            elif rng.random() < 0.20:
                slot_shift_x = 0.0
            else:
                slot_shift_x = float(rng.uniform(shift_low, shift_high))
            if policy == "mixture":
                if not behavior_schedule:
                    behavior_schedule = _mixture_schedule(behavior_rng)
                behavior = behavior_schedule.pop(0)
            else:
                behavior = policy
            episode = None
            require_success = policy in ("expert", "noisy") or (
                policy == "mixture" and behavior != "recovery"
            )
            for _ in range(max_attempts_per_episode):
                attempt_seed = seed + attempt_id + 1
                attempt_id += 1
                episode = _rollout_attempt(
                    env=env,
                    task_id=task_id,
                    maneuver=maneuver,
                    variant=variant,
                    slot_shift_x=slot_shift_x,
                    seed=attempt_seed,
                    episode_id=episode_id,
                    policy_name=behavior,
                    rng=np.random.default_rng(attempt_seed),
                    noise=noise,
                    jitter_position=jitter_position,
                    jitter_heading_deg=jitter_heading_deg,
                    reuse_nominal_path=reuse_nominal_paths,
                )
                if episode is None:
                    rejected += 1
                    continue
                accepted = not require_success or bool(
                    episode["successes"][-1]
                )
                if accepted:
                    break
                episode = None
                rejected += 1
            if episode is None:
                env.close()
                raise RuntimeError(
                    f"failed to collect task {task_id} ({maneuver}) after "
                    f"{max_attempts_per_episode} attempts"
                )
            for key, values in episode.items():
                store[key].extend(values)
            noisy_episodes += int(behavior == "noisy")
            successful_episodes += int(
                bool(episode["successes"][-1])
            )
            episode_id += 1

    env.close()
    arrays = _as_arrays(store)
    _validate_arrays(arrays, require_success=policy in ("expert", "noisy"))
    episode_starts = np.r_[True, arrays["episode_ids"][1:] != arrays["episode_ids"][:-1]]
    episode_behaviors = arrays["behavior_ids"][episode_starts]
    stats = {
        "steps": float(len(arrays["actions"])),
        "episodes": float(episode_id),
        "success_rate": float(successful_episodes / max(episode_id, 1)),
        "collision_fraction": float(arrays["collisions"].mean()),
        "death_rate": float(arrays["deaths"].sum() / max(episode_id, 1)),
        "noisy_episode_fraction": float(noisy_episodes / max(episode_id, 1)),
        "recovery_episode_fraction": float(
            np.mean(episode_behaviors == BEHAVIOR_TO_ID["recovery"])
        ),
        "rejected_attempts": float(rejected),
    }
    return arrays, stats


def _savez_compressed_deterministic(
    path: pathlib.Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Atomically write an NPZ with deterministic ZIP metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for key in sorted(arrays):
                buffer = io.BytesIO()
                np.save(buffer, arrays[key], allow_pickle=False)
                entry = zipfile.ZipInfo(
                    f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o600 << 16
                archive.writestr(entry, buffer.getvalue())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def dataset_metadata(
    *,
    policy: PolicyName,
    size: SizeName,
    seed: int,
    max_episode_steps: int,
    noise: float,
    jitter_position: float,
    jitter_heading_deg: float,
    slot_shift_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Return scalar metadata needed to reject stale or incompatible files."""
    config = CarParkingConfig()
    return {
        "dataset_schema": np.asarray(DATASET_SCHEMA),
        "action_encoding": np.asarray(ACTION_ENCODING),
        "policy": np.asarray(policy),
        "named_size": np.asarray(size),
        "generation_seed": np.asarray(seed, dtype=np.int64),
        "max_episode_steps": np.asarray(max_episode_steps, dtype=np.int32),
        "noise_scale": np.asarray(noise, dtype=np.float32),
        "jitter_position": np.asarray(jitter_position, dtype=np.float32),
        "jitter_heading_deg": np.asarray(jitter_heading_deg, dtype=np.float32),
        "slot_shift_bounds": np.asarray(slot_shift_bounds, dtype=np.float32),
        "dwell_steps": np.asarray(config.dwell_steps, dtype=np.int32),
        "orientation_tolerance": np.asarray(
            config.orientation_tolerance, dtype=np.float32
        ),
        "parked_speed_tolerance": np.asarray(
            config.parked_speed_tolerance, dtype=np.float32
        ),
        "car_length": np.asarray(config.car_length, dtype=np.float32),
        "car_width": np.asarray(config.car_width, dtype=np.float32),
        "slot_margin": np.asarray(config.slot_margin, dtype=np.float32),
        "max_steer_angle": np.asarray(
            config.max_steer_angle, dtype=np.float32
        ),
    }


def validate_dataset_file(
    path: str | pathlib.Path,
    *,
    minimum_steps: int = 1,
    require_schema: bool = False,
) -> dict[str, int | str]:
    """Validate a saved split before a launcher treats it as complete."""
    path = pathlib.Path(path)
    with np.load(path, allow_pickle=False) as raw:
        required = {
            "observations",
            "actions",
            "next_observations",
            "commanded_goals",
            "terminals",
            "successes",
            "deaths",
            "timeouts",
            "episode_ids",
            "task_ids",
            "behavior_ids",
            "slot_shifts",
            "slot_lengths",
            "slot_widths",
        }
        missing = required.difference(raw.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        arrays = {key: np.asarray(raw[key]) for key in required}
        count = len(arrays["actions"])
        if count < minimum_steps:
            raise ValueError(
                f"{path}: {count} transitions < minimum {minimum_steps}"
            )
        if arrays["observations"].shape != (count, 8):
            raise ValueError(f"{path}: observations must have shape (N, 8)")
        if arrays["next_observations"].shape != (count, 8):
            raise ValueError(
                f"{path}: next_observations must have shape (N, 8)"
            )
        if arrays["actions"].shape != (count, 2):
            raise ValueError(f"{path}: actions must have shape (N, 2)")
        if arrays["commanded_goals"].shape != (count, 5):
            raise ValueError(
                f"{path}: commanded_goals must have shape (N, 5)"
            )
        for key in (
            "terminals",
            "successes",
            "deaths",
            "timeouts",
            "episode_ids",
            "task_ids",
            "behavior_ids",
            "slot_shifts",
            "slot_lengths",
            "slot_widths",
        ):
            if arrays[key].shape != (count,):
                raise ValueError(f"{path}: {key} must have shape (N,)")
        if np.any(np.abs(arrays["actions"]) > 1.00001):
            raise ValueError(f"{path}: actions are not normalized")
        policy = (
            str(np.asarray(raw["policy"]).item())
            if "policy" in raw.files
            else ""
        )
        _validate_arrays(
            {
                key: np.asarray(raw[key])
                for key in raw.files
                if np.asarray(raw[key]).ndim > 0
                and len(np.asarray(raw[key])) == count
            },
            require_success=policy in ("expert", "noisy"),
        )
        schema = (
            str(np.asarray(raw["dataset_schema"]).item())
            if "dataset_schema" in raw.files
            else "legacy"
        )
        if require_schema and schema != DATASET_SCHEMA:
            raise ValueError(
                f"{path}: schema {schema!r} != {DATASET_SCHEMA!r}"
            )
    return {"steps": count, "schema": schema}


def collect_dataset(
    *,
    policy: PolicyName = "mixture",
    size: SizeName = "1k",
    seed: int = 0,
    max_episode_steps: int = 400,
    noise: float = 0.015,
    jitter_position: float = 0.005,
    jitter_heading_deg: float = 1.0,
    slot_shift: float = 0.005,
    save_path: pathlib.Path | None = None,
) -> dict[str, dict[str, float]]:
    if not 0.0 <= slot_shift <= 0.04:
        raise ValueError("slot_shift must be in [0, 0.04]")
    if policy not in POLICIES or size not in SIZES:
        raise ValueError(f"unknown dataset combination: {policy}/{size}")
    train_steps, val_steps = SIZE_STEPS[size]
    if save_path is None:
        save_path = (
            pathlib.Path(__file__).resolve().parent
            / "datasets"
            / f"car_parking_{policy}_{size}.npz"
        )
    save_path = pathlib.Path(save_path)
    val_path = save_path.with_name(f"{save_path.stem}_val.npz")

    train, train_stats = collect_split(
        policy=policy,
        minimum_steps=train_steps,
        seed=seed,
        max_episode_steps=max_episode_steps,
        noise=noise,
        jitter_position=jitter_position,
        jitter_heading_deg=jitter_heading_deg,
        slot_shift_bounds=(-slot_shift, slot_shift),
    )
    val, val_stats = collect_split(
        policy=policy,
        minimum_steps=val_steps,
        seed=seed + 1_000_000,
        max_episode_steps=max_episode_steps,
        noise=noise,
        # Disjoint validation jitter band.
        jitter_position=1.5 * jitter_position,
        jitter_heading_deg=1.5 * jitter_heading_deg,
        slot_shift_bounds=(1.5 * slot_shift, 2.0 * slot_shift),
        reuse_nominal_paths=False,
    )
    metadata = dataset_metadata(
        policy=policy,
        size=size,
        seed=seed,
        max_episode_steps=max_episode_steps,
        noise=noise,
        jitter_position=jitter_position,
        jitter_heading_deg=jitter_heading_deg,
        slot_shift_bounds=(-slot_shift, slot_shift),
    )
    train.update(metadata)
    val.update(metadata)
    val["jitter_position"] = np.asarray(
        1.5 * jitter_position, dtype=np.float32
    )
    val["jitter_heading_deg"] = np.asarray(
        1.5 * jitter_heading_deg, dtype=np.float32
    )
    val["slot_shift_bounds"] = np.asarray(
        (1.5 * slot_shift, 2.0 * slot_shift), dtype=np.float32
    )
    val["generation_seed"] = np.asarray(
        seed + 1_000_000, dtype=np.int64
    )
    _savez_compressed_deterministic(save_path, train)
    _savez_compressed_deterministic(val_path, val)
    validate_dataset_file(save_path, minimum_steps=train_steps, require_schema=True)
    validate_dataset_file(val_path, minimum_steps=val_steps, require_schema=True)
    print(
        f"saved {save_path} steps={len(train['actions'])} "
        f"episodes={int(train_stats['episodes'])} "
        f"contact={train_stats['collision_fraction']:.3f} "
        f"rejected={int(train_stats['rejected_attempts'])}"
    )
    print(
        f"saved {val_path} steps={len(val['actions'])} "
        f"episodes={int(val_stats['episodes'])}"
    )
    return {"train": train_stats, "val": val_stats}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, default="mixture")
    parser.add_argument("--size", choices=SIZES, default="1k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--slot-shift", type=float, default=0.005)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--jitter-position", type=float, default=0.005)
    parser.add_argument("--jitter-heading-deg", type=float, default=1.0)
    parser.add_argument("--save-path", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_dataset(
        policy=args.policy,
        size=args.size,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        noise=args.noise,
        jitter_position=args.jitter_position,
        jitter_heading_deg=args.jitter_heading_deg,
        slot_shift=args.slot_shift,
        save_path=args.save_path,
    )
    print("=== CAR_PARKING_DATASET_GENERATION_DONE ===", flush=True)


if __name__ == "__main__":
    main()
