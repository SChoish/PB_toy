"""Collect an OGBench-style *navigate* dataset for ContinuousHazard2DEnv.

One episode keeps going after each goal reach: a new safe goal is sampled and
the scripted repulsion oracle continues until hazard death or the time limit.

Example::

    PYTHONPATH=/path/to/toy_examples \\
      python -m hazard_env.generate_navigate \\
        --num-episodes 200 --seed 0
"""

from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict
from typing import Literal

import numpy as np

from hazard_env.env import ContinuousHazard2DEnv, Hazard2DConfig

PreferSide = Literal["north", "south"]


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _clip_to_arena(env: ContinuousHazard2DEnv, point: np.ndarray) -> np.ndarray:
    margin = env.config.agent_radius + env.config.spawn_clearance
    low = env.config.arena_low + margin
    high = env.config.arena_high - margin
    return np.clip(point, low, high).astype(np.float32)


def choose_prefer_side(
    env: ContinuousHazard2DEnv,
    *,
    clearance: float = 0.34,
) -> PreferSide:
    """Pick the shorter safe via (north vs south) for the current chord."""
    pos = env.position.astype(np.float32)
    goal = env.goal.astype(np.float32)
    hazard = env.hazard_center.astype(np.float32)
    lethal = env.hazard_radius + env.config.agent_radius
    inflated = lethal + 0.10
    if not ContinuousHazard2DEnv._segment_hits_circle(pos, goal, hazard, inflated):
        return "north" if pos[1] >= hazard[1] else "south"

    scores: dict[PreferSide, float] = {}
    for side in ("north", "south"):
        y_sign = 1.0 if side == "north" else -1.0
        via = hazard + np.array(
            [0.0, y_sign * (lethal + clearance)], dtype=np.float32
        )
        via[0] = float(np.clip(0.55 * pos[0] + 0.45 * goal[0], -0.85, 0.85))
        via = _clip_to_arena(env, via)
        # Penalize vias that still graze the hazard on either leg.
        hit1 = ContinuousHazard2DEnv._segment_hits_circle(
            pos, via, hazard, lethal + 0.02
        )
        hit2 = ContinuousHazard2DEnv._segment_hits_circle(
            via, goal, hazard, lethal + 0.02
        )
        path_len = float(np.linalg.norm(via - pos) + np.linalg.norm(goal - via))
        scores[side] = path_len + (2.5 if hit1 or hit2 else 0.0)
    return "north" if scores["north"] <= scores["south"] else "south"


def navigation_subgoal(
    env: ContinuousHazard2DEnv,
    *,
    prefer_side: PreferSide,
    clearance: float = 0.34,
) -> np.ndarray:
    """Return a via point around the hazard when the straight chord is lethal."""
    pos = env.position.astype(np.float32)
    goal = env.goal.astype(np.float32)
    hazard = env.hazard_center.astype(np.float32)
    lethal = env.hazard_radius + env.config.agent_radius
    inflated = lethal + 0.10

    if not ContinuousHazard2DEnv._segment_hits_circle(pos, goal, hazard, inflated):
        return goal

    y_sign = 1.0 if prefer_side == "north" else -1.0
    # Two-stage via: approach the preferred side, then cross past the hazard.
    side = hazard + np.array([0.0, y_sign * (lethal + clearance)], dtype=np.float32)
    # Keep the via between start and goal in x so we progress across the arena.
    side[0] = float(np.clip(0.55 * pos[0] + 0.45 * goal[0], -0.85, 0.85))
    side = _clip_to_arena(env, side)

    # If we are already on the preferred side with clearance, aim slightly past
    # the hazard toward the goal so we do not stall on the via point.
    on_side = (pos[1] - hazard[1]) * y_sign > (lethal + 0.14)
    past_hazard_x = (pos[0] - hazard[0]) * (goal[0] - hazard[0]) > 0.0 and abs(
        pos[0] - hazard[0]
    ) >= abs(goal[0] - hazard[0]) * 0.15
    if on_side and past_hazard_x:
        return goal
    if float(np.linalg.norm(pos - side)) < 0.12 and on_side:
        return goal
    return side


def oracle_action(
    env: ContinuousHazard2DEnv,
    *,
    prefer_side: PreferSide | None = None,
    repulsion_scale: float = 1.15,
    repulsion_margin: float = 0.42,
) -> np.ndarray:
    """PD-style thrust toward a hazard-aware subgoal."""
    if prefer_side is None:
        prefer_side = choose_prefer_side(env)

    pos = env.position.astype(np.float32)
    vel = env.velocity.astype(np.float32)
    goal = env.goal.astype(np.float32)
    subgoal = navigation_subgoal(env, prefer_side=prefer_side)

    hazard = env.hazard_center.astype(np.float32)
    from_hazard = pos - hazard
    dist_h = float(np.linalg.norm(from_hazard))
    lethal = env.hazard_radius + env.config.agent_radius
    safe_band = lethal + repulsion_margin

    to_sub = subgoal - pos
    dist_sub = float(np.linalg.norm(to_sub))
    direction = _unit(to_sub)

    # Soft speed limit near the hazard and near the final goal.
    dist_goal = float(np.linalg.norm(goal - pos))
    speed_cap = float(env.config.max_speed)
    if dist_h < safe_band:
        speed_cap = min(
            speed_cap,
            0.28 + 0.5 * max(dist_h - lethal, 0.0) / max(repulsion_margin, 1e-6),
        )
        speed_cap = max(0.18, speed_cap)
    if dist_goal < env.config.goal_radius * 2.5:
        speed_cap = min(speed_cap, 0.32)

    desired_speed = min(speed_cap, 0.18 + 0.85 * min(dist_sub / 0.45, 1.0))
    desired_vel = direction * desired_speed

    # Approximate inverse dynamics for the linear-drag point mass.
    dt = float(env.config.dt)
    drag = float(env.config.linear_drag)
    accel = (desired_vel - vel) / max(dt, 1e-3) + drag * vel

    # Explicit repulsion acceleration when inside the soft band.
    if dist_h < safe_band:
        strength = repulsion_scale * (safe_band - dist_h) / max(repulsion_margin, 1e-6)
        into = float(np.dot(vel, -_unit(from_hazard)))
        if into > 0.0:
            strength += 1.2 * into
        accel = accel + strength * _unit(from_hazard) * float(env.config.max_acceleration)
        # Cancel inward velocity with an outward accel kick.
        if into > 0.05:
            accel = accel + into * _unit(from_hazard) * float(env.config.max_acceleration)

    accel_norm = float(np.linalg.norm(accel))
    if accel_norm < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)

    angle = float(np.arctan2(accel[1], accel[0]))
    thrust = float(
        np.clip(accel_norm / float(env.config.max_acceleration), 0.0, 1.0)
    )
    return np.array([angle, thrust], dtype=np.float32)


def _sample_new_goal(env: ContinuousHazard2DEnv) -> np.ndarray:
    """Sample a goal away from the current position (retry on coincidence)."""
    min_dist = max(env.config.goal_radius * 2.5, 0.35)
    for _ in range(200):
        candidate = env.sample_safe_point(
            min_distance_from=env.position,
            minimum_distance=min_dist,
        )
        if np.linalg.norm(candidate - env.position) >= env.config.goal_radius:
            return candidate
    return env.sample_safe_point(
        min_distance_from=env.position,
        minimum_distance=env.config.goal_radius + 1e-3,
    )


def collect_navigate_dataset(
    *,
    num_train_episodes: int,
    num_val_episodes: int,
    seed: int,
    max_episode_steps: int,
    noise: float,
    save_path: pathlib.Path,
    preview_path: pathlib.Path | None,
    preview_episodes: int,
) -> dict[str, float]:
    config = Hazard2DConfig(
        max_episode_steps=max_episode_steps,
        task_mode="random",
        min_start_goal_distance=0.7,
    )
    env = ContinuousHazard2DEnv(
        config=config,
        observation_mode="state",
        terminate_at_goal=False,
    )

    dataset: dict[str, list] = defaultdict(list)
    rng = np.random.default_rng(seed)

    total_steps = 0
    total_train_steps = 0
    n_deaths = 0
    n_goal_reaches = 0
    preview_trajs: list[np.ndarray] = []

    total_episodes = num_train_episodes + num_val_episodes
    for ep_idx in range(total_episodes):
        ep_seed = int(seed + ep_idx + 1)
        ob, info = env.reset(seed=ep_seed, options={"task_mode": "random"})

        done = False
        ep_goals = 0
        ep_positions: list[np.ndarray] = [env.position.copy()]

        while not done:
            action = oracle_action(env)
            # Angle noise is more dangerous near the hazard; keep it modest.
            noise_vec = rng.normal(0.0, noise, size=action.shape).astype(np.float32)
            noise_vec[0] *= 0.7
            action = action + noise_vec
            action[0] = float(np.clip(action[0], -np.pi, np.pi))
            action[1] = float(np.clip(action[1], 0.0, 1.0))

            goal_before = env.goal.copy()
            next_ob, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            success = bool(info.get("is_success", False))

            if success:
                ep_goals += 1
                n_goal_reaches += 1
                # Bleed momentum before the next leg; high speed into a new
                # lethal chord is the main failure mode for this point-mass.
                env.velocity = (env.velocity * 0.25).astype(env.velocity.dtype, copy=False)
                try:
                    env.set_goal(_sample_new_goal(env))
                except ValueError:
                    # Extremely rare; keep current goal and continue.
                    pass

            dataset["observations"].append(np.asarray(ob, dtype=np.float32))
            dataset["actions"].append(action.astype(np.float32))
            dataset["terminals"].append(done)
            dataset["goals"].append(goal_before.astype(np.float32))
            dataset["successes"].append(success)

            ep_positions.append(env.position.copy())
            ob = next_ob

        if info.get("dead", False):
            n_deaths += 1

        total_steps += env.elapsed_steps
        if ep_idx < num_train_episodes:
            total_train_steps += env.elapsed_steps

        if preview_path is not None and len(preview_trajs) < preview_episodes:
            preview_trajs.append(np.stack(ep_positions, axis=0))

    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = pathlib.Path(str(save_path).replace(".npz", "-val.npz"))
    if val_path == save_path:
        val_path = save_path.with_name(save_path.stem + "-val.npz")

    train_dataset: dict[str, np.ndarray] = {}
    val_dataset: dict[str, np.ndarray] = {}
    for key, values in dataset.items():
        if key == "terminals" or key == "successes":
            dtype = bool
        else:
            dtype = np.float32
        arr = np.asarray(values, dtype=dtype)
        train_dataset[key] = arr[:total_train_steps]
        val_dataset[key] = arr[total_train_steps:]

    np.savez_compressed(save_path, **train_dataset)
    np.savez_compressed(val_path, **val_dataset)

    stats = {
        "total_steps": float(total_steps),
        "train_steps": float(total_train_steps),
        "val_steps": float(total_steps - total_train_steps),
        "death_rate": float(n_deaths / max(total_episodes, 1)),
        "goals_per_episode": float(n_goal_reaches / max(total_episodes, 1)),
        "num_deaths": float(n_deaths),
        "num_goal_reaches": float(n_goal_reaches),
    }

    print(f"Saved train: {save_path}  ({int(stats['train_steps'])} steps)")
    print(f"Saved val:   {val_path}  ({int(stats['val_steps'])} steps)")
    print(
        f"death_rate={stats['death_rate']:.3f}  "
        f"goals/ep={stats['goals_per_episode']:.2f}  "
        f"deaths={int(stats['num_deaths'])}/{total_episodes}"
    )

    if preview_path is not None and preview_trajs:
        _write_preview(
            preview_path,
            trajs=preview_trajs,
            hazard_center=env.hazard_center,
            hazard_radius=env.hazard_radius,
            arena_low=env.config.arena_low,
            arena_high=env.config.arena_high,
        )
        print(f"Wrote preview: {preview_path}")

    env.close()
    return stats


def _write_preview(
    path: pathlib.Path,
    *,
    trajs: list[np.ndarray],
    hazard_center: np.ndarray,
    hazard_radius: float,
    arena_low: float,
    arena_high: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.set_aspect("equal")
    ax.set_xlim(arena_low, arena_high)
    ax.set_ylim(arena_low, arena_high)
    ax.add_patch(
        Circle(
            hazard_center,
            hazard_radius,
            facecolor="#c0392b",
            edgecolor="#7b241c",
            alpha=0.85,
            zorder=2,
        )
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(trajs)))
    for traj, color in zip(trajs, colors, strict=False):
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.2, alpha=0.9, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], s=18, color=color, zorder=4)
    ax.set_title("navigate preview")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_out = (
        pathlib.Path(__file__).resolve().parent / "datasets" / "hazard2d-navigate.npz"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument(
        "--num-val-episodes",
        type=int,
        default=None,
        help="Defaults to num_episodes // 10.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.10)
    parser.add_argument("--save-path", type=pathlib.Path, default=default_out)
    parser.add_argument(
        "--preview-path",
        type=pathlib.Path,
        default=None,
        help="Optional trajectory scatter PNG path.",
    )
    parser.add_argument("--preview-episodes", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_val = (
        args.num_val_episodes
        if args.num_val_episodes is not None
        else max(1, args.num_episodes // 10)
    )
    preview = args.preview_path
    if preview is None:
        preview = args.save_path.parent / "navigate_preview.png"

    collect_navigate_dataset(
        num_train_episodes=args.num_episodes,
        num_val_episodes=num_val,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        noise=args.noise,
        save_path=args.save_path,
        preview_path=preview,
        preview_episodes=args.preview_episodes,
    )


if __name__ == "__main__":
    main()
