"""Generate task-agnostic offline CarRace physical trajectory datasets.

The named sizes are minimum transition counts.  Collection always keeps whole
episodes, so saved arrays may be slightly larger than the requested budget.
Train and validation splits are collected independently with disjoint seeds.
Navigation and universal lap files share the same physical schema.  One
``--task lap`` NPZ is re-annotated for every task from lap_1p through lap_8p.
"""

from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .env import (
    GRAVITY_STRENGTHS,
    CarRaceConfig,
    CarRaceEnv,
    _wrap_angle,
    mode_config_kwargs,
)

EnvName = Literal[
    "car_race_plain",
    "car_race_grav",
    "car_race_anti_grav",
    "car_race_ice",
]
PolicyName = Literal["expert", "noisy", "random"]
DatasetTask = Literal["navigation", "lap"]
SizeName = Literal["1k", "10k", "100k"]

ENVS: tuple[EnvName, ...] = tuple(GRAVITY_STRENGTHS)
POLICIES: tuple[PolicyName, ...] = ("expert", "noisy", "random")
SIZES: tuple[SizeName, ...] = ("1k", "10k", "100k")
SIZE_STEPS: dict[SizeName, tuple[int, int]] = {
    "1k": (1_000, 100),
    "10k": (10_000, 1_000),
    "100k": (100_000, 10_000),
}


def physical_observation(env: CarRaceEnv) -> np.ndarray:
    """Task-agnostic physical state stored by both dataset families."""
    return np.array(
        [
            env.position[0],
            env.position[1],
            np.cos(env.heading),
            np.sin(env.heading),
            env.speed,
            env.health,
            env.external_velocity[0],
            env.external_velocity[1],
        ],
        dtype=np.float32,
    )


@dataclass
class PolicyState:
    held_action: np.ndarray | None = None
    hold_steps: int = 0


def _segment_is_safe(env: CarRaceEnv, start: np.ndarray, end: np.ndarray) -> bool:
    fractions = np.linspace(0.0, 1.0, 41, dtype=np.float32)[:, None]
    points = start[None] + fractions * (end - start)[None]
    radii = np.linalg.norm(points, axis=1)
    margin = env.config.collision_radius + 0.025
    return bool(
        np.all(radii > env.config.inner_hazard_radius + margin)
        and np.all(radii < env.config.outer_hazard_radius - margin)
    )


def navigation_waypoint(env: CarRaceEnv, *, aggressive: bool = False) -> np.ndarray:
    """Choose a direct target or a short safe arc waypoint."""
    position = env.position
    goal = env.goal
    if aggressive or _segment_is_safe(env, position, goal):
        return goal.copy()

    current_angle = float(np.arctan2(position[1], position[0]))
    goal_angle = float(np.arctan2(goal[1], goal[0]))
    angular_error = _wrap_angle(goal_angle - current_angle)
    if abs(angular_error) < 0.22:
        return goal.copy()

    lookahead = float(np.clip(angular_error, -0.32, 0.32))
    waypoint_angle = current_angle + lookahead
    radial_error = env.config.track_radius - float(np.linalg.norm(position))
    waypoint_radius = float(
        np.clip(
            np.linalg.norm(position) + 0.45 * radial_error,
            env.config.inner_hazard_radius
            + env.config.collision_radius
            + 0.05,
            env.config.outer_hazard_radius
            - env.config.collision_radius
            - 0.05,
        )
    )
    return (
        waypoint_radius
        * np.array([np.cos(waypoint_angle), np.sin(waypoint_angle)])
    ).astype(np.float32)


def lap_navigation_waypoint(env: CarRaceEnv) -> np.ndarray:
    """Return a short ring lookahead that passes through every real waypoint."""
    current_angle = float(np.arctan2(env.position[1], env.position[0]))
    direction = float(env._lap_direction)
    radius = float(np.linalg.norm(env.position))
    # The signed field bias counters steady radial displacement. Ice has no
    # field and is stabilized by the radial feedback term alone.
    target_radius = float(
        np.clip(
            env.config.track_radius
            + env.config.gravity_strength
            + 1.2 * (env.config.track_radius - radius),
            env.config.inner_hazard_radius
            + env.config.collision_radius
            + 0.065,
            env.config.outer_hazard_radius
            - env.config.collision_radius
            - 0.065,
        )
    )
    target_angle = current_angle + direction * 0.28
    return (
        target_radius
        * np.array([np.cos(target_angle), np.sin(target_angle)])
    ).astype(np.float32)


def lap_expert_action(env: CarRaceEnv) -> np.ndarray:
    """Direction-conditioned circle tracker shared by every lap density."""
    target = lap_navigation_waypoint(env)
    delta = target - env.position
    distance = float(np.linalg.norm(delta))
    desired_heading = float(np.arctan2(delta[1], delta[0]))
    heading_error = _wrap_angle(desired_heading - env.heading)
    steering = float(
        np.clip(1.8 * heading_error / env.config.max_steer_angle, -1.0, 1.0)
    )
    target_direction = delta / max(distance, 1e-8)
    target_speed = 0.40 if env.config.cornering_grip < 0.5 else 0.50
    target_drive_speed = float(
        np.clip(
            target_speed - np.dot(env.external_velocity, target_direction),
            0.0,
            env.config.max_speed,
        )
    )
    throttle = float(
        np.clip(1.5 * (target_drive_speed - env.speed), -1.0, 1.0)
    )
    return np.array([steering, throttle], dtype=np.float32)


def expert_action(
    env: CarRaceEnv,
    *,
    aggressive: bool = False,
) -> np.ndarray:
    """Pure-pursuit-like steering and speed control."""
    if env.config.task_mode == "lap":
        return lap_expert_action(env)
    target = navigation_waypoint(env, aggressive=aggressive)
    delta = target - env.position
    desired_heading = float(
        np.arctan2(delta[1], delta[0])
    )
    heading_error = _wrap_angle(desired_heading - env.heading)
    steering = float(
        np.clip(1.8 * heading_error / env.config.max_steer_angle, -1.0, 1.0)
    )

    # Closed-loop speed control makes sharp turns and goal approaches use real
    # braking instead of remaining pinned at max speed. Aggressive episodes
    # keep a higher cornering speed to preserve risky trajectories.
    if aggressive:
        target_speed = 0.78 - 0.30 * abs(steering)
    else:
        target_speed = 0.72 - 0.42 * abs(steering)
    distance = float(np.linalg.norm(delta))
    if distance < 0.14:
        target_speed = min(target_speed, 0.30 if aggressive else 0.20)
    # Compensate only the external velocity component along the target. This
    # prevents the car from settling just outside a waypoint when the field is
    # stronger than the desired low-speed approach.
    target_direction = delta / max(distance, 1e-8)
    external_along_target = float(
        np.dot(env.external_velocity, target_direction)
    )
    target_drive_speed = float(
        np.clip(
            target_speed - external_along_target,
            0.0,
            env.config.max_speed,
        )
    )
    throttle = float(
        np.clip(1.5 * (target_drive_speed - env.speed), -1.0, 1.0)
    )
    return np.array([steering, throttle], dtype=np.float32)


def _held_random_action(
    rng: np.random.Generator,
    state: PolicyState,
    *,
    min_hold: int,
    max_hold: int,
) -> np.ndarray:
    if state.hold_steps <= 0 or state.held_action is None:
        state.held_action = np.array(
            [rng.uniform(-1.0, 1.0), rng.uniform(-0.45, 1.0)],
            dtype=np.float32,
        )
        state.hold_steps = int(rng.integers(min_hold, max_hold + 1))
    state.hold_steps -= 1
    return state.held_action.copy()


def behavior_action(
    env: CarRaceEnv,
    policy: PolicyName,
    rng: np.random.Generator,
    state: PolicyState,
    *,
    aggressive: bool,
    noise: float,
) -> tuple[np.ndarray, bool]:
    """Return action and whether it came from a random burst."""
    if policy == "random":
        return _held_random_action(rng, state, min_hold=5, max_hold=20), True

    if policy == "noisy":
        if state.hold_steps <= 0 and rng.random() < 0.03:
            state.held_action = np.array(
                [rng.uniform(-1.0, 1.0), rng.uniform(-0.4, 1.0)],
                dtype=np.float32,
            )
            state.hold_steps = int(rng.integers(5, 13))
        if state.hold_steps > 0:
            state.hold_steps -= 1
            assert state.held_action is not None
            return state.held_action.copy(), True

    action = expert_action(env, aggressive=aggressive)
    if policy == "noisy" and noise > 0.0:
        action = action + rng.normal(0.0, noise, size=2).astype(np.float32)
        action[0] += float(rng.normal(0.0, noise * 0.6))
    return np.clip(action, -1.0, 1.0).astype(np.float32), False


def _empty_store() -> dict[str, list]:
    return defaultdict(list)


def _append_transition(
    store: dict[str, list],
    *,
    observation: np.ndarray,
    action: np.ndarray,
    next_observation: np.ndarray,
    terminal: bool,
    info: dict,
    episode_id: int,
    is_lap: bool,
    lap_direction: int,
    lap_start_angle: float,
    commanded_goal: np.ndarray,
) -> None:
    store["observations"].append(np.asarray(observation, dtype=np.float32))
    store["actions"].append(np.asarray(action, dtype=np.float32))
    store["next_observations"].append(
        np.asarray(next_observation, dtype=np.float32)
    )
    store["terminals"].append(bool(terminal))
    store["healths"].append(float(info["health"]))
    store["step_impulses"].append(float(info["step_impulse"]))
    store["hazard_contacts"].append(bool(info["hazard_contact"]))
    store["episode_ids"].append(int(episode_id))
    store["commanded_goals"].append(
        np.asarray(commanded_goal, dtype=np.float32)
    )
    store["is_lap"].append(bool(is_lap))
    store["lap_directions"].append(int(lap_direction))
    store["lap_start_angles"].append(float(lap_start_angle))


def _as_arrays(store: dict[str, list]) -> dict[str, np.ndarray]:
    bool_keys = {"terminals", "hazard_contacts", "is_lap"}
    int_keys = {"episode_ids", "lap_directions"}
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


def collect_split(
    *,
    policy: PolicyName,
    minimum_steps: int,
    seed: int,
    max_episode_steps: int,
    noise: float,
    env_name: EnvName = "car_race_plain",
    task: DatasetTask = "navigation",
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Collect navigation data or universal ring-following lap data."""
    if task not in ("navigation", "lap"):
        raise ValueError(f"Unknown dataset task: {task}")
    config = CarRaceConfig(
        task_mode=task,
        checkpoint_count=9,
        max_episode_steps=max_episode_steps,
        **mode_config_kwargs(env_name),
    )
    env = CarRaceEnv(
        config=config,
        observation_mode="state",
        # Keep one commanded goal per episode so terminal boundaries and goals
        # remain aligned for offline relabeling.
        terminate_on_success=True,
    )
    rng = np.random.default_rng(seed)
    store = _empty_store()
    episode_id = 0
    deaths = 0
    goals_reached = 0
    laps_completed = 0
    random_actions = 0
    aggressive_episodes = 0
    rejected_expert_episodes = 0
    attempt_id = 0

    while len(store["actions"]) < minimum_steps:
        env.reset(seed=seed + attempt_id + 1)
        attempt_id += 1
        observation = physical_observation(env)
        episode_store = _empty_store()
        lap_direction = int(env._lap_direction) if task == "lap" else 0
        lap_start_angle = (
            float(np.arctan2(env.position[1], env.position[0]))
            if task == "lap"
            else 0.0
        )
        policy_state = PolicyState()
        aggressive = policy != "random" and rng.random() < 0.15
        episode_random_actions = 0
        done = False
        info: dict = {}

        while not done:
            commanded_goal = env.desired_goal.copy()
            action, used_random = behavior_action(
                env,
                policy,
                rng,
                policy_state,
                aggressive=aggressive,
                noise=noise,
            )
            episode_random_actions += int(used_random)
            _, _, terminated, truncated, info = env.step(action)
            next_observation = physical_observation(env)
            done = bool(terminated or truncated)
            _append_transition(
                episode_store,
                observation=observation,
                action=action,
                next_observation=next_observation,
                terminal=done,
                info=info,
                episode_id=episode_id,
                is_lap=(task == "lap"),
                lap_direction=lap_direction,
                lap_start_angle=lap_start_angle,
                commanded_goal=commanded_goal,
            )
            observation = next_observation

        succeeded = bool(info.get("is_success", False))
        if policy == "expert" and not succeeded:
            rejected_expert_episodes += 1
            continue
        for key, values in episode_store.items():
            store[key].extend(values)
        aggressive_episodes += int(aggressive)
        random_actions += episode_random_actions
        deaths += int(info.get("dead", False))
        goals_reached += int(task == "navigation" and succeeded)
        laps_completed += int(task == "lap" and succeeded)
        episode_id += 1

    env.close()
    arrays = _as_arrays(store)
    count = len(arrays["actions"])
    stats = {
        "steps": float(count),
        "episodes": float(episode_id),
        "goals_per_episode": float(goals_reached / max(episode_id, 1)),
        "lap_success_rate": float(laps_completed / max(episode_id, 1)),
        "death_rate": float(deaths / max(episode_id, 1)),
        "hazard_contact_frac": float(arrays["hazard_contacts"].mean()),
        "random_action_frac": float(random_actions / max(count, 1)),
        "aggressive_episode_frac": float(aggressive_episodes / max(episode_id, 1)),
        "rejected_expert_episodes": float(rejected_expert_episodes),
    }
    return arrays, stats


def dataset_stem(
    env_name: EnvName,
    policy: PolicyName,
    size: SizeName,
    task: DatasetTask = "navigation",
) -> str:
    middle = "_lap" if task == "lap" else ""
    return f"{env_name}{middle}_{policy}_{size}"


def collect_dataset(
    *,
    env_name: EnvName = "car_race_plain",
    policy: PolicyName,
    size: SizeName,
    seed: int = 0,
    max_episode_steps: int = 500,
    noise: float = 0.08,
    save_path: pathlib.Path | None = None,
    task: DatasetTask = "navigation",
) -> dict[str, dict[str, float]]:
    if (
        env_name not in ENVS
        or policy not in POLICIES
        or size not in SIZES
    ):
        raise ValueError(f"Unknown dataset combination: {env_name}/{policy}/{size}")
    train_steps, val_steps = SIZE_STEPS[size]
    if save_path is None:
        save_path = (
            pathlib.Path(__file__).resolve().parent
            / "datasets"
            / f"{dataset_stem(env_name, policy, size, task)}.npz"
        )
    save_path = pathlib.Path(save_path)
    val_path = save_path.with_name(save_path.stem + "_val.npz")

    train, train_stats = collect_split(
        policy=policy,
        minimum_steps=train_steps,
        seed=seed,
        max_episode_steps=max_episode_steps,
        noise=noise,
        env_name=env_name,
        task=task,
    )
    val, val_stats = collect_split(
        policy=policy,
        minimum_steps=val_steps,
        seed=seed + 1_000_000,
        max_episode_steps=max_episode_steps,
        noise=noise,
        env_name=env_name,
        task=task,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **train)
    np.savez_compressed(val_path, **val)
    task_metric = (
        f"lap_success={train_stats['lap_success_rate']:.3f}"
        if task == "lap"
        else f"goals/ep={train_stats['goals_per_episode']:.2f}"
    )
    print(
        f"saved {save_path} steps={len(train['actions'])} "
        f"episodes={int(train_stats['episodes'])} "
        f"{task_metric} death={train_stats['death_rate']:.3f}"
    )
    print(f"saved {val_path} steps={len(val['actions'])}")
    return {"train": train_stats, "val": val_stats}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVS, default="car_race_plain")
    parser.add_argument("--policy", choices=POLICIES, default="expert")
    parser.add_argument("--task", choices=("navigation", "lap"), default="navigation")
    parser.add_argument("--size", choices=SIZES, default="1k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.08)
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
        stem = dataset_stem(env_name, policy, size, args.task)
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
            task=args.task,
            save_path=args.save_path
            if not args.generate_all
            else None,
        )
    print("=== CAR_RACE_DATASET_GENERATION_DONE ===", flush=True)


if __name__ == "__main__":
    main()
