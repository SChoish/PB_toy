"""Generate balanced, goal-independent CarParking offline datasets.

Only complete successful expert trajectories are retained.  Every transition
comes from ``CarParkingEnv.step``; Hybrid A* path points are never stored as
environment transitions.  Named sizes are minimum transition counts, so the
saved split may be slightly larger in order to finish a balanced maneuver
round.

Example:
    python -m car_parking.generate_dataset --size 1k --policy expert
"""

from __future__ import annotations

import argparse
import io
import pathlib
import zipfile
from collections import defaultdict
from typing import Literal

import numpy as np

from .env import MANEUVERS, CarParkingConfig, CarParkingEnv
from .parking_policy import ParkingExpertPolicy

PolicyName = Literal["expert", "noisy", "mixture"]
SizeName = Literal["1k", "10k", "100k"]

POLICIES: tuple[PolicyName, ...] = ("expert", "noisy", "mixture")
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


def physical_observation(env: CarParkingEnv) -> np.ndarray:
    """Return the seven goal-independent Markov features stored in raw NPZ."""
    return np.array(
        [
            env.position[0],
            env.position[1],
            np.cos(env.heading),
            np.sin(env.heading),
            env.speed,
            env.steering / env.config.max_steer_angle,
            env.health,
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
    maneuver: str,
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
    store["maneuver_ids"].append(MANEUVER_TO_ID[maneuver])
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
    int8_keys = {"maneuver_ids"}
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


def _episode_policy(
    policy: PolicyName,
    rng: np.random.Generator,
) -> Literal["expert", "noisy"]:
    if policy == "mixture":
        return "expert" if rng.random() < 0.7 else "noisy"
    return policy


def _rollout_attempt(
    *,
    env: CarParkingEnv,
    maneuver: str,
    variant: int,
    seed: int,
    episode_id: int,
    policy_name: Literal["expert", "noisy"],
    rng: np.random.Generator,
    noise: float,
    jitter_position: float,
    jitter_heading_deg: float,
) -> defaultdict[str, list] | None:
    options: dict[str, object] = {"maneuver": maneuver, "variant": variant}
    env.reset(seed=seed, options=options)
    if jitter_position > 0.0 or jitter_heading_deg > 0.0:
        options["position"] = (
            np.asarray(env.layout.start)
            + rng.uniform(-jitter_position, jitter_position, size=2)
        )
        options["heading"] = env.layout.start_heading + np.deg2rad(
            rng.uniform(-jitter_heading_deg, jitter_heading_deg)
        )
    try:
        env.reset(seed=seed, options=options)
        expert = ParkingExpertPolicy(env)
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
    try:
        while not done:
            action = expert.action()
            if policy_name == "noisy":
                action, noise_state = _noisy_action(action, noise_state, rng, noise)
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
                maneuver=maneuver,
                variant=variant,
                slot_length=slot_length,
                slot_width=slot_width,
            )
            observation = next_observation
    except (RuntimeError, ValueError):
        return None

    # The benchmark dataset is success-centered.  Reject planner/controller
    # failures as a whole rather than leaving partial episode fragments.
    if not bool(final_info.get("success")):
        return None
    return episode


def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
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
    if not np.all(arrays["successes"][terminals]):
        raise RuntimeError("retained episodes must terminate successfully")


def collect_split(
    *,
    policy: PolicyName,
    minimum_steps: int,
    seed: int,
    max_episode_steps: int = 400,
    noise: float = 0.015,
    jitter_position: float = 0.008,
    jitter_heading_deg: float = 2.0,
    max_attempts_per_episode: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Collect complete episodes in balanced four-maneuver rounds."""
    env = CarParkingEnv(
        CarParkingConfig(maneuver="mixed", max_episode_steps=max_episode_steps)
    )
    rng = np.random.default_rng(seed)
    store = _empty_store()
    episode_id = 0
    attempt_id = 0
    rejected = 0
    noisy_episodes = 0

    while len(store["actions"]) < minimum_steps:
        for maneuver in MANEUVERS:
            variants = VALIDATED_VARIANTS[maneuver]
            maneuver_round = episode_id // len(MANEUVERS)
            variant = variants[maneuver_round % len(variants)]
            behavior = _episode_policy(policy, rng)
            episode = None
            for _ in range(max_attempts_per_episode):
                attempt_seed = seed + attempt_id + 1
                attempt_id += 1
                episode = _rollout_attempt(
                    env=env,
                    maneuver=maneuver,
                    variant=variant,
                    seed=attempt_seed,
                    episode_id=episode_id,
                    policy_name=behavior,
                    rng=rng,
                    noise=noise,
                    jitter_position=jitter_position,
                    jitter_heading_deg=jitter_heading_deg,
                )
                if episode is not None:
                    break
                rejected += 1
            if episode is None:
                env.close()
                raise RuntimeError(
                    f"failed to collect {maneuver} after "
                    f"{max_attempts_per_episode} attempts"
                )
            for key, values in episode.items():
                store[key].extend(values)
            noisy_episodes += int(behavior == "noisy")
            episode_id += 1

    env.close()
    arrays = _as_arrays(store)
    _validate_arrays(arrays)
    stats = {
        "steps": float(len(arrays["actions"])),
        "episodes": float(episode_id),
        "success_rate": 1.0,
        "collision_fraction": float(arrays["collisions"].mean()),
        "death_rate": float(arrays["deaths"].sum() / max(episode_id, 1)),
        "noisy_episode_fraction": float(noisy_episodes / max(episode_id, 1)),
        "rejected_attempts": float(rejected),
    }
    return arrays, stats


def _savez_compressed_deterministic(
    path: pathlib.Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Write an NPZ with fixed ZIP metadata for byte-identical regeneration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[key], allow_pickle=False)
            entry = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue())


def collect_dataset(
    *,
    policy: PolicyName = "expert",
    size: SizeName = "1k",
    seed: int = 0,
    max_episode_steps: int = 400,
    noise: float = 0.015,
    jitter_position: float = 0.008,
    jitter_heading_deg: float = 2.0,
    save_path: pathlib.Path | None = None,
) -> dict[str, dict[str, float]]:
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
    )
    _savez_compressed_deterministic(save_path, train)
    _savez_compressed_deterministic(val_path, val)
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
    parser.add_argument("--policy", choices=POLICIES, default="expert")
    parser.add_argument("--size", choices=SIZES, default="1k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--jitter-position", type=float, default=0.008)
    parser.add_argument("--jitter-heading-deg", type=float, default=2.0)
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
        save_path=args.save_path,
    )
    print("=== CAR_PARKING_DATASET_GENERATION_DONE ===", flush=True)


if __name__ == "__main__":
    main()
