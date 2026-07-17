"""Plot success learning curves for one env × dataset set."""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_TAGS = (
    "hiql_gap0",
    "hiql_gap5",
    "tr_hiql_gap0",
    "tr_hiql_gap5",
    "pbg_gap0",
    "pbg_gap5",
    "pbf_gap0",
    "pbf_gap5",
    "trl",
    "dqc",
)

# Distinct colors without the common purple-gradient look.
COLORS = {
    "hiql_gap0": "#1b9e77",
    "hiql_gap5": "#66c2a5",
    "tr_hiql_gap0": "#d95f02",
    "tr_hiql_gap5": "#fc8d62",
    "pbg_gap0": "#2c7fb8",
    "pbg_gap5": "#7fcdbb",
    "pbf_gap0": "#e7298a",
    "pbf_gap5": "#e78ac3",
    "trl": "#a6761d",
    "dqc": "#444444",
}


def _env_short(env_name: str) -> str:
    return {
        "hazard_plain": "plain",
        "hazard_grav": "grav",
        "hazard_anti_grav": "anti_grav",
    }.get(env_name, env_name)


def load_curve(
    checkpoint_dir: pathlib.Path,
) -> tuple[list[int], list[float], list[float]]:
    steps: list[int] = []
    means: list[float] = []
    stds: list[float] = []
    for path in sorted(checkpoint_dir.glob("step_*.json")):
        match = re.fullmatch(r"step_(\d+)\.json", path.name)
        if match is None:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics") or {}
        if "mean_success" not in metrics:
            continue
        steps.append(int(match.group(1)))
        means.append(float(metrics["mean_success"]))
        stds.append(float(metrics.get("mean_success_std", 0.0)))
    return steps, means, stds


def set_is_complete(
    env_name: str,
    policy: str,
    *,
    size: str = "1k",
    tags: tuple[str, ...] = DEFAULT_TAGS,
    root: pathlib.Path = ROOT,
) -> bool:
    ckpt_root = root / "checkpoints" / env_name / f"{policy}_{size}"
    return all((ckpt_root / tag / "step_50000.json").exists() for tag in tags)


def plot_set_learning_curves(
    env_name: str,
    policy: str,
    *,
    size: str = "1k",
    tags: tuple[str, ...] = DEFAULT_TAGS,
    root: pathlib.Path = ROOT,
    save_path: pathlib.Path | None = None,
) -> pathlib.Path | None:
    """Plot mean±std success vs step for all algos in one env×data set."""
    ckpt_root = root / "checkpoints" / env_name / f"{policy}_{size}"
    if not ckpt_root.exists():
        return None

    if save_path is None:
        save_path = (
            root
            / "plots"
            / "learning_curves"
            / f"{env_name}_{policy}_{size}.png"
        )
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    curves: list[tuple[str, list[float], list[float], list[float], str | None]] = []
    for tag in tags:
        steps, means, stds = load_curve(ckpt_root / tag)
        if not steps:
            continue
        xs = [step / 1000.0 for step in steps]
        curves.append((tag, xs, means, stds, COLORS.get(tag)))

    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for tag, xs, means, stds, color in curves:
        ax.plot(
            xs,
            means,
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            label=tag,
            color=color,
        )

    all_xs = sorted({x for _, xs, _, _, _ in curves for x in xs})
    ax.set_xticks(all_xs)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("train step (k)")
    ax.set_ylabel("success")
    ax.set_title(f"{_env_short(env_name)} · {policy} · {size}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--size", default="1k")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    path = plot_set_learning_curves(
        args.env, args.policy, size=args.size, save_path=args.out
    )
    print(path)


if __name__ == "__main__":
    main()
