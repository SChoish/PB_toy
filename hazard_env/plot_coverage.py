"""Plot navigate-dataset spatial coverage for ContinuousHazard2DEnv."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

from hazard_env.env import (
    GRAVITY_STRENGTHS,
    ContinuousHazard2DEnv,
    Hazard2DConfig,
)
from hazard_env.generate_navigate import POLICIES, SIZES, dataset_stem


def _episode_xy(obs: np.ndarray, terminals: np.ndarray) -> list[np.ndarray]:
    ends = np.flatnonzero(terminals)
    trajs: list[np.ndarray] = []
    start = 0
    for end in ends:
        trajs.append(obs[start : int(end) + 1, :2])
        start = int(end) + 1
    if start < len(obs):
        trajs.append(obs[start:, :2])
    return trajs


def _soft_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "coverage",
        [
            "#f7f4ef",
            "#d9e2ec",
            "#9ebcda",
            "#5b8db8",
            "#2b6a99",
            "#134e7a",
        ],
    )


def plot_coverage(
    dataset_path: pathlib.Path | None = None,
    save_path: pathlib.Path | None = None,
    *,
    env_name: str = "hazard_plain",
    policy: str = "navigate",
    size: str = "100k",
    n_sample_trajs: int = 24,
    seed: int = 0,
) -> pathlib.Path:
    if env_name not in GRAVITY_STRENGTHS:
        raise ValueError(f"Unknown env_name={env_name!r}")
    root = pathlib.Path(__file__).resolve().parent
    stem = dataset_stem(env_name, policy, size)
    if dataset_path is None:
        dataset_path = root / "datasets" / f"{stem}.npz"
    if save_path is None:
        save_path = root / "datasets" / f"{stem}_coverage.png"
    dataset_path = pathlib.Path(dataset_path)
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    raw = np.load(dataset_path)
    obs = np.asarray(raw["observations"], dtype=np.float32)
    terminals = np.asarray(raw["terminals"], dtype=bool)
    goals = np.asarray(raw["goals"], dtype=np.float32)
    successes = np.asarray(raw["successes"], dtype=bool)
    xy = obs[:, :2]
    trajs = _episode_xy(obs, terminals)

    env = ContinuousHazard2DEnv(
        config=Hazard2DConfig(
            gravity_strength=GRAVITY_STRENGTHS[env_name]
        ),
        observation_mode="state",
    )
    low = env.config.arena_low
    high = env.config.arena_high
    lethal = env.config.hazard_radius + env.config.agent_radius
    dist_h = np.linalg.norm(xy - env.hazard_center, axis=1)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(trajs), size=min(n_sample_trajs, len(trajs)), replace=False)

    cmap = _soft_cmap()
    task_colors = ["#1b9e77", "#66a61e", "#e6ab02", "#d95f02", "#e7298a"]

    fig = plt.figure(figsize=(11.2, 5.4), facecolor="#fbfaf8")
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.35, 1.0, 0.92],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.32,
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.10,
    )
    ax_main = fig.add_subplot(gs[:, 0])
    ax_goal = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[1, 1])
    ax_rad = fig.add_subplot(gs[0, 2])
    ax_stats = fig.add_subplot(gs[1, 2])

    def _draw_arena(ax, *, show_lethal: bool = False) -> None:
        ax.set_aspect("equal")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_facecolor("#fbfaf8")
        for spine in ax.spines.values():
            spine.set_color("#c8c2b8")
            spine.set_linewidth(0.8)
        ax.tick_params(colors="#6b645c", labelsize=8)
        ax.set_xlabel("x", color="#4a453e", fontsize=9)
        ax.set_ylabel("y", color="#4a453e", fontsize=9)
        ax.add_patch(
            Circle(
                env.hazard_center,
                env.hazard_radius,
                facecolor="#c0392b",
                edgecolor="#7b241c",
                alpha=0.92,
                zorder=5,
                linewidth=0.9,
            )
        )
        if show_lethal:
            ax.add_patch(
                Circle(
                    env.hazard_center,
                    lethal,
                    facecolor="none",
                    edgecolor="#7b241c",
                    linestyle=(0, (2.5, 2.0)),
                    linewidth=0.9,
                    alpha=0.75,
                    zorder=5,
                )
            )

    # --- Main: position occupancy ---
    _draw_arena(ax_main, show_lethal=True)
    hb = ax_main.hexbin(
        xy[:, 0],
        xy[:, 1],
        gridsize=48,
        extent=(low, high, low, high),
        cmap=cmap,
        mincnt=1,
        linewidths=0.0,
        zorder=2,
    )
    cb = fig.colorbar(hb, ax=ax_main, fraction=0.046, pad=0.03)
    cb.set_label("visit count / hex", fontsize=8, color="#4a453e")
    cb.ax.tick_params(labelsize=7, colors="#6b645c")
    cb.outline.set_edgecolor("#c8c2b8")

    for info, color in zip(env.task_infos, task_colors, strict=True):
        s = np.asarray(info["init_xy"], dtype=np.float32)
        g = np.asarray(info["goal_xy"], dtype=np.float32)
        hits = ContinuousHazard2DEnv._segment_hits_circle(
            s, g, env.hazard_center, lethal
        )
        ax_main.plot(
            [s[0], g[0]],
            [s[1], g[1]],
            color=color,
            lw=1.6,
            ls="--" if hits else "-",
            alpha=0.95,
            zorder=6,
        )
        ax_main.scatter(
            s[0],
            s[1],
            s=36,
            facecolors="#fbfaf8",
            edgecolors=color,
            linewidths=1.4,
            zorder=7,
        )
        ax_main.scatter(g[0], g[1], s=70, marker="*", color=color, zorder=7)

    ax_main.set_title("State occupancy + eval tasks", fontsize=11, color="#2f2b26", pad=8)

    # --- Goal coverage ---
    _draw_arena(ax_goal)
    ax_goal.hexbin(
        goals[:, 0],
        goals[:, 1],
        gridsize=36,
        extent=(low, high, low, high),
        cmap=cmap,
        mincnt=1,
        linewidths=0.0,
        zorder=2,
    )
    ax_goal.set_title("Commanded goals", fontsize=10, color="#2f2b26", pad=6)

    # --- Sample trajectories ---
    _draw_arena(ax_traj)
    traj_cmap = plt.cm.magma(np.linspace(0.25, 0.85, len(sample_idx)))
    for i, tid in enumerate(sample_idx):
        tr = trajs[int(tid)]
        ax_traj.plot(
            tr[:, 0],
            tr[:, 1],
            color=traj_cmap[i],
            lw=0.85,
            alpha=0.72,
            zorder=3,
        )
        ax_traj.scatter(
            tr[0, 0],
            tr[0, 1],
            s=10,
            color=traj_cmap[i],
            zorder=4,
            linewidths=0,
        )
    ax_traj.set_title(f"Sample trajectories (n={len(sample_idx)})", fontsize=10, color="#2f2b26", pad=6)

    # --- Radial coverage ---
    ax_rad.set_facecolor("#fbfaf8")
    for spine in ax_rad.spines.values():
        spine.set_color("#c8c2b8")
        spine.set_linewidth(0.8)
    ax_rad.tick_params(colors="#6b645c", labelsize=8)
    bins = np.linspace(0.0, np.sqrt(2.0) * (high - low) / 2 + 0.2, 36)
    ax_rad.hist(
        dist_h,
        bins=bins,
        color="#5b8db8",
        edgecolor="#fbfaf8",
        linewidth=0.4,
        alpha=0.92,
    )
    ax_rad.axvline(
        env.hazard_radius,
        color="#c0392b",
        lw=1.4,
        label=f"hazard r={env.hazard_radius:.2f}",
    )
    ax_rad.axvline(
        lethal,
        color="#7b241c",
        lw=1.2,
        ls="--",
        label=f"lethal r={lethal:.2f}",
    )
    ax_rad.set_xlabel("‖xy − hazard‖", fontsize=9, color="#4a453e")
    ax_rad.set_ylabel("count", fontsize=9, color="#4a453e")
    ax_rad.set_title("Distance to hazard", fontsize=10, color="#2f2b26", pad=6)
    ax_rad.legend(fontsize=7, frameon=False, loc="upper right")

    # --- Stats panel ---
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")
    n_ep = len(trajs)
    n_success = int(successes.sum())
    goals_reached = n_success
    frac_near_hazard = float((dist_h < lethal + 0.15).mean())
    frac_north = float((xy[:, 1] > env.hazard_center[1]).mean())

    card = FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.92,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#f3f0ea",
        edgecolor="#d6d0c6",
        linewidth=0.9,
        transform=ax_stats.transAxes,
        clip_on=False,
    )
    ax_stats.add_patch(card)

    lines = [
        ("Dataset", dataset_path.name),
        ("Transitions", f"{len(obs):,}"),
        ("Episodes", f"{n_ep:,}"),
        ("Goal reaches", f"{goals_reached:,}"),
        ("Success / step", f"{successes.mean():.3%}"),
        ("Near hazard", f"{frac_near_hazard:.1%} (≤ lethal+0.15)"),
        ("North of hazard", f"{frac_north:.1%} of states"),
        ("xy span", f"[{xy[:, 0].min():.2f}, {xy[:, 0].max():.2f}] × [{xy[:, 1].min():.2f}, {xy[:, 1].max():.2f}]"),
    ]
    y = 0.90
    ax_stats.text(
        0.08,
        y,
        "Coverage summary",
        transform=ax_stats.transAxes,
        fontsize=11,
        fontweight="600",
        color="#2f2b26",
        va="top",
    )
    y -= 0.12
    for label, value in lines:
        ax_stats.text(
            0.08,
            y,
            label,
            transform=ax_stats.transAxes,
            fontsize=8,
            color="#7a736a",
            va="top",
        )
        ax_stats.text(
            0.92,
            y,
            value,
            transform=ax_stats.transAxes,
            fontsize=8,
            color="#2f2b26",
            ha="right",
            va="top",
            fontfamily="monospace",
        )
        y -= 0.085

    legend_handles = [
        Line2D([0], [0], color="#c0392b", lw=6, label="hazard", solid_capstyle="round"),
        Line2D([0], [0], color="#7b241c", lw=1.2, ls="--", label="lethal margin"),
        Line2D([0], [0], color="#5b8db8", lw=3, label="state density"),
    ]
    for i, (info, color) in enumerate(zip(env.task_infos, task_colors, strict=True), start=1):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=1.6,
                label=f"T{i} {info['difficulty']}",
            )
        )
    ax_main.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7,
        framealpha=0.92,
        facecolor="#fbfaf8",
        edgecolor="#d6d0c6",
        borderpad=0.5,
    )

    fig.suptitle(
        f"Hazard2D ({env_name}) navigate — dataset coverage",
        fontsize=14,
        fontweight="600",
        color="#2f2b26",
        y=0.97,
    )
    fig.text(
        0.06,
        0.02,
        f"Source: {dataset_path.name} · {len(obs):,} transitions · {n_ep} episodes",
        fontsize=8,
        color="#8a8278",
    )

    fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    env.close()
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", choices=tuple(GRAVITY_STRENGTHS), default="hazard_plain"
    )
    parser.add_argument("--policy", choices=POLICIES, default="navigate")
    parser.add_argument("--size", choices=SIZES, default="100k")
    parser.add_argument("--dataset", type=pathlib.Path, default=None)
    parser.add_argument("--save-path", type=pathlib.Path, default=None)
    parser.add_argument("--sample-trajectories", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = plot_coverage(
        args.dataset,
        args.save_path,
        env_name=args.env,
        policy=args.policy,
        size=args.size,
        n_sample_trajs=args.sample_trajectories,
        seed=args.seed,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
