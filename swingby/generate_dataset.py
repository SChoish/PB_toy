"""Generate offline OrbitalSwingBy physical trajectory datasets.

Named sizes are minimum transition counts. Collection always keeps whole
episodes, so saved arrays may be slightly larger than the requested budget.
Train and validation splits use disjoint seeds.

NPZ schema
----------
observations        [T, 5]  # x, y, vx, vy, fuel_fraction
actions             [T, 2]  # thrust angle, throttle
next_observations   [T, 5]
terminals           [T]
commanded_goals     [T, 4]  # gx, gy, gvx, gvy
episode_ids         [T]
successes           [T]
deaths              [T]
escapes             [T]
"""

from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict
from typing import Literal

import numpy as np

try:
    from .config import black_hole_config, planet_config
    from .env import OrbitalSwingByEnv
    from .policies import (
        PolicyName,
        PolicyState,
        behavior_action,
        commanded_goal,
        physical_observation,
    )
except ImportError:  # script-style: `cd swingby && python generate_dataset.py`
    from config import black_hole_config, planet_config
    from env import OrbitalSwingByEnv
    from policies import (
        PolicyName,
        PolicyState,
        behavior_action,
        commanded_goal,
        physical_observation,
    )

EnvName = Literal["swingby_planet", "swingby_blackhole"]
SizeName = Literal["1k", "10k", "100k"]

ENVS: tuple[EnvName, ...] = ("swingby_planet", "swingby_blackhole")
POLICIES: tuple[PolicyName, ...] = ("expert", "noisy", "random")
SIZES: tuple[SizeName, ...] = ("1k", "10k", "100k")
SIZE_STEPS: dict[SizeName, tuple[int, int]] = {
    "1k": (1_000, 100),
    "10k": (10_000, 1_000),
    "100k": (100_000, 10_000),
}


def make_env_config(env_name: EnvName, *, max_episode_steps: int):
    if env_name == "swingby_blackhole":
        return black_hole_config(
            task_mode="swingby",
            reward_mode="dense",
            max_episode_steps=max_episode_steps,
            show_ballistic_prediction=False,
        )
    return planet_config(
        task_mode="swingby",
        reward_mode="dense",
        max_episode_steps=max_episode_steps,
        show_ballistic_prediction=False,
    )


def _empty_store() -> dict[str, list]:
    return defaultdict(list)


def _append_transition(
    store: dict[str, list],
    *,
    observation: np.ndarray,
    action: np.ndarray,
    next_observation: np.ndarray,
    terminal: bool,
    goal: np.ndarray,
    info: dict,
    episode_id: int,
) -> None:
    store["observations"].append(np.asarray(observation, dtype=np.float32))
    store["actions"].append(np.asarray(action, dtype=np.float32))
    store["next_observations"].append(
        np.asarray(next_observation, dtype=np.float32)
    )
    store["terminals"].append(bool(terminal))
    store["commanded_goals"].append(np.asarray(goal, dtype=np.float32))
    store["episode_ids"].append(int(episode_id))
    store["successes"].append(bool(info.get("is_success", False)))
    store["deaths"].append(bool(info.get("dead", False)))
    store["escapes"].append(bool(info.get("escaped", False)))
    store["fuels"].append(float(info.get("fuel_fraction", 0.0)))


def _as_arrays(store: dict[str, list]) -> dict[str, np.ndarray]:
    bool_keys = {"terminals", "successes", "deaths", "escapes"}
    int_keys = {"episode_ids"}
    arrays: dict[str, np.ndarray] = {}
    for key, values in store.items():
        if key in bool_keys:
            dtype = bool
        elif key in int_keys:
            dtype = np.int32
        else:
            dtype = np.float32
        arrays[key] = np.asarray(values, dtype=dtype)
    return arrays


def _retarget_goal(env: OrbitalSwingByEnv) -> bool:
    """Chain another reachable ballistic goal without ending the episode."""
    sample = env.sample_ballistic_goal()
    if sample is None:
        return False
    goal, goal_velocity = sample
    try:
        env.set_goal(goal, goal_velocity)
    except ValueError:
        return False
    return True


def collect_split(
    *,
    env_name: EnvName,
    policy: PolicyName,
    minimum_steps: int,
    seed: int,
    max_episode_steps: int,
    noise: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    config = make_env_config(env_name, max_episode_steps=max_episode_steps)
    env = OrbitalSwingByEnv(
        config=config,
        observation_mode="state",
        terminate_at_goal=False,
    )
    rng = np.random.default_rng(seed)
    store = _empty_store()
    episode_id = 0
    deaths = 0
    escapes = 0
    goals_reached = 0
    random_actions = 0
    aggressive_episodes = 0

    while len(store["actions"]) < minimum_steps:
        env.reset(seed=seed + episode_id + 1)
        observation = physical_observation(env)
        goal = commanded_goal(env)
        policy_state = PolicyState()
        aggressive = policy != "random" and rng.random() < 0.18
        aggressive_episodes += int(aggressive)
        done = False
        info: dict = {}

        while not done:
            action, used_random = behavior_action(
                env,
                policy,
                rng,
                policy_state,
                aggressive=aggressive,
                noise=noise,
            )
            random_actions += int(used_random)
            _, _, terminated, truncated, info = env.step(action)
            next_observation = physical_observation(env)
            done = bool(terminated or truncated)
            _append_transition(
                store,
                observation=observation,
                action=action,
                next_observation=next_observation,
                terminal=done,
                goal=goal,
                info=info,
                episode_id=episode_id,
            )
            observation = next_observation

            if info.get("is_success", False) and not done:
                goals_reached += 1
                if _retarget_goal(env):
                    goal = commanded_goal(env)
                    policy_state.passed_periapsis = False
                    policy_state.min_radius = float(
                        np.linalg.norm(env.position - env.body_center)
                    )
                else:
                    # No reachable follow-up goal; end the episode early.
                    done = True
                    store["terminals"][-1] = True

        deaths += int(info.get("dead", False))
        escapes += int(info.get("escaped", False))
        episode_id += 1

    env.close()
    arrays = _as_arrays(store)
    count = len(arrays["actions"])
    stats = {
        "steps": float(count),
        "episodes": float(episode_id),
        "goals_per_episode": float(goals_reached / max(episode_id, 1)),
        "death_rate": float(deaths / max(episode_id, 1)),
        "escape_rate": float(escapes / max(episode_id, 1)),
        "success_transition_frac": float(arrays["successes"].mean()),
        "random_action_frac": float(random_actions / max(count, 1)),
        "aggressive_episode_frac": float(
            aggressive_episodes / max(episode_id, 1)
        ),
        "mean_fuel_fraction": float(arrays["fuels"].mean()),
    }
    return arrays, stats


def dataset_stem(
    env_name: EnvName,
    policy: PolicyName,
    size: SizeName,
) -> str:
    return f"{env_name}_{policy}_{size}"


def collect_dataset(
    *,
    env_name: EnvName = "swingby_planet",
    policy: PolicyName = "expert",
    size: SizeName = "1k",
    seed: int = 0,
    max_episode_steps: int = 650,
    noise: float = 0.10,
    save_path: pathlib.Path | None = None,
) -> dict[str, dict[str, float]]:
    if env_name not in ENVS or policy not in POLICIES or size not in SIZES:
        raise ValueError(
            f"Unknown dataset combination: {env_name}/{policy}/{size}"
        )
    train_steps, val_steps = SIZE_STEPS[size]
    if save_path is None:
        save_path = (
            pathlib.Path(__file__).resolve().parent
            / "datasets"
            / f"{dataset_stem(env_name, policy, size)}.npz"
        )
    save_path = pathlib.Path(save_path)
    val_path = save_path.with_name(save_path.stem + "_val.npz")

    train, train_stats = collect_split(
        env_name=env_name,
        policy=policy,
        minimum_steps=train_steps,
        seed=seed,
        max_episode_steps=max_episode_steps,
        noise=noise,
    )
    val, val_stats = collect_split(
        env_name=env_name,
        policy=policy,
        minimum_steps=val_steps,
        seed=seed + 1_000_000,
        max_episode_steps=max_episode_steps,
        noise=noise,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **train)
    np.savez_compressed(val_path, **val)
    print(
        f"saved {save_path} steps={len(train['actions'])} "
        f"episodes={int(train_stats['episodes'])} "
        f"goals/ep={train_stats['goals_per_episode']:.2f} "
        f"death={train_stats['death_rate']:.3f} "
        f"escape={train_stats['escape_rate']:.3f}"
    )
    print(f"saved {val_path} steps={len(val['actions'])}")
    return {"train": train_stats, "val": val_stats}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVS, default="swingby_planet")
    parser.add_argument("--policy", choices=POLICIES, default="expert")
    parser.add_argument("--size", choices=SIZES, default="1k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=650)
    parser.add_argument("--noise", type=float, default=0.10)
    parser.add_argument("--save-path", type=pathlib.Path, default=None)
    parser.add_argument("--generate-all", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose train npz already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = (
        [
            (env_name, policy, size)
            for env_name in ENVS
            for policy in POLICIES
            for size in SIZES
        ]
        if args.generate_all
        else [(args.env, args.policy, args.size)]
    )
    for env_name, policy, size in jobs:
        stem = dataset_stem(env_name, policy, size)
        out = (
            pathlib.Path(__file__).resolve().parent
            / "datasets"
            / f"{stem}.npz"
        )
        if args.skip_existing and out.exists():
            print(f"skip existing {out}", flush=True)
            continue
        collect_dataset(
            env_name=env_name,
            policy=policy,
            size=size,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            noise=args.noise,
            save_path=args.save_path if not args.generate_all else None,
        )
    print("=== SWINGBY_DATASET_GENERATION_DONE ===", flush=True)


if __name__ == "__main__":
    main()
