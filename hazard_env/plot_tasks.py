"""Plot fixed evaluation tasks 1–5 for ContinuousHazard2DEnv."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from hazard_env.env import (
    GRAVITY_STRENGTHS,
    ContinuousHazard2DEnv,
    Hazard2DConfig,
)


def plot_tasks(
    save_path: pathlib.Path | None = None,
    *,
    env_name: str = "hazard_plain",
) -> pathlib.Path:
    if env_name not in GRAVITY_STRENGTHS:
        raise ValueError(f"Unknown env_name={env_name!r}")
    env = ContinuousHazard2DEnv(
        config=Hazard2DConfig(
            gravity_strength=GRAVITY_STRENGTHS[env_name]
        ),
        observation_mode="state",
    )
    if save_path is None:
        save_path = pathlib.Path(__file__).resolve().parent / "datasets" / (
            f"{env_name}_tasks_preview.png"
        )
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lethal = env.config.hazard_radius + env.config.agent_radius
    colors = ["#1b9e77", "#66a61e", "#e6ab02", "#d95f02", "#e7298a"]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_aspect("equal")
    ax.set_xlim(env.config.arena_low, env.config.arena_high)
    ax.set_ylim(env.config.arena_low, env.config.arena_high)
    ax.add_patch(
        Circle(
            env.hazard_center,
            env.hazard_radius,
            facecolor="#c0392b",
            edgecolor="#7b241c",
            alpha=0.88,
            zorder=2,
            label="hazard",
        )
    )

    for task_id, (info, color) in enumerate(
        zip(env.task_infos, colors, strict=True), start=1
    ):
        start = np.asarray(info["init_xy"], dtype=np.float32)
        goal = np.asarray(info["goal_xy"], dtype=np.float32)
        hits = ContinuousHazard2DEnv._segment_hits_circle(
            start, goal, env.hazard_center, lethal
        )
        ls = "--" if hits else "-"
        label = f"T{task_id} {info['difficulty']}"
        ax.plot(
            [start[0], goal[0]],
            [start[1], goal[1]],
            color=color,
            lw=1.8,
            ls=ls,
            zorder=3,
            label=label,
        )
        ax.scatter(
            start[0],
            start[1],
            s=55,
            facecolors="white",
            edgecolors=color,
            linewidths=1.6,
            zorder=4,
        )
        ax.scatter(
            goal[0],
            goal[1],
            s=90,
            marker="*",
            color=color,
            zorder=4,
        )
        ax.text(
            start[0],
            start[1] + 0.06,
            f"S{task_id}",
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_title(f"Hazard2D ({env_name}) evaluation tasks 1–5")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    env.close()
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", choices=tuple(GRAVITY_STRENGTHS), default="hazard_plain"
    )
    parser.add_argument("--save-path", type=pathlib.Path, default=None)
    args = parser.parse_args()
    path = plot_tasks(args.save_path, env_name=args.env)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
