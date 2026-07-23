#!/usr/bin/env python3
"""Plot noisy-100k learning curves → PB_toy_learning_curves_noisy100k_csh.png.

Sources: local checkpoints (preferred) + PB_logs csh_server + dgx only if
the run has step200000.json. No source markers. Agents: hiql/tr_hiql/pbg/pbf
(trl excluded). Markers every 20k; x-axis in k steps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PB_LOGS = Path("/home/ext_csh/PB_logs/completed/pb_toy")
OUT = ROOT / "PB_toy_learning_curves_noisy100k_csh.png"

AGENTS = ["hiql", "tr_hiql", "pbg", "pbf"]
COLORS = {
    "hiql": "#4C78A8",
    "tr_hiql": "#F58518",
    "pbg": "#54A24B",
    "pbf": "#E45756",
}
ORDER = [
    "ice/lap_1p",
    "ice/lap_2p",
    "ice/lap_4p",
    "grav/lap_1p",
    "grav/lap_2p",
    "grav/lap_4p",
    "anti_grav/lap_1p",
    "anti_grav/lap_2p",
    "anti_grav/lap_4p",
    "planet/swingby",
    "blackhole/swingby",
    "car_parking",
]
DONE_STEP = 200000
EVAL_STRIDE = 20000


def parse_tag(tag: str) -> tuple[str | None, str]:
    m = re.search(r"_(hiql|tr_hiql|pbg|pbf)_noisy", tag)
    if not m:
        return None, "?"
    agent = m.group(1)
    if "car_parking" in tag:
        return "car_parking", agent
    if "swingby" in tag:
        env = "planet" if "planet" in tag else "blackhole"
        return f"{env}/swingby", agent
    if "anti_grav" in tag:
        env = "anti_grav"
    elif "car_race_grav" in tag or "_grav_lap" in tag:
        env = "grav"
    else:
        env = "ice"
    tm = re.search(r"lap_(\d+p)", tag)
    if not tm:
        return None, agent
    return f"{env}/lap_{tm.group(1)}", agent


def mean_success(metrics) -> float | None:
    if not isinstance(metrics, dict):
        return None
    for k in ("mean_success", "t0_mean_success", "t1_mean_success"):
        if isinstance(metrics.get(k), (int, float)):
            return float(metrics[k])
    return None


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def put_curve(
    store: dict[tuple[str, str], dict[int, tuple[float, int]]],
    key: str,
    agent: str,
    step: int,
    score: float,
    rank: int,
) -> None:
    if step <= 0 or step % EVAL_STRIDE != 0:
        return
    if step > DONE_STEP:
        return
    cur = store.setdefault((key, agent), {})
    if step not in cur or rank <= cur[step][1]:
        cur[step] = (score, rank)


def main() -> None:
    # (key, agent) -> step -> (pct, rank)  rank 0=local, 1=csh, 2=dgx
    store: dict[tuple[str, str], dict[int, tuple[float, int]]] = {}

    ckpt_root = ROOT / "checkpoints"
    if ckpt_root.is_dir():
        for root in ckpt_root.iterdir():
            if not root.is_dir():
                continue
            for ckpt in root.iterdir():
                if (
                    not ckpt.is_dir()
                    or "noisy" not in ckpt.name
                    or not ckpt.name.endswith("_100k")
                ):
                    continue
                key, agent = parse_tag(ckpt.name)
                if key is None or agent not in AGENTS:
                    continue
                for p in ckpt.glob("step_*.json"):
                    m = re.match(r"step_(\d+)\.json$", p.name)
                    if not m:
                        continue
                    step = int(m.group(1))
                    d = load_json(p)
                    if not d:
                        continue
                    s = mean_success(d.get("metrics") or {})
                    if s is None:
                        continue
                    put_curve(store, key, agent, step, s * 100.0, 0)

    if PB_LOGS.exists():
        for p in PB_LOGS.rglob("eval/step*.json"):
            name = p.parent.parent.name
            if "noisy" not in name or "100k" not in name:
                continue
            m = re.search(r"step[_]?(\d+)\.json$", p.name)
            if not m:
                continue
            step = int(m.group(1))
            if "csh_server" in name:
                rank = 1
                tag = name.replace("csh_server_pb_toy_", "")
            elif "ext_csv-box" in name:
                if not (p.parent / "step200000.json").exists():
                    continue
                rank = 2
                tag = name.replace("ext_csv-box_pb_toy_", "")
            else:
                continue
            key, agent = parse_tag(tag)
            if key is None or agent not in AGENTS:
                continue
            d = load_json(p)
            if not d:
                continue
            metrics = d.get("metrics", d)
            s = mean_success(metrics if isinstance(metrics, dict) else {})
            if s is None:
                continue
            put_curve(store, key, agent, step, s * 100.0, rank)

    panels = [k for k in ORDER if any((k, a) in store for a in AGENTS)]
    if not panels:
        raise SystemExit("no curve data")

    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), sharex=True, sharey=False
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, key in zip(axes, panels):
        for agent in AGENTS:
            pts = store.get((key, agent))
            if not pts:
                continue
            xs = sorted(pts)
            ys = [pts[x][0] for x in xs]
            ax.plot(
                [x / 1000.0 for x in xs],
                ys,
                color=COLORS[agent],
                label=agent,
                linewidth=1.7,
                marker="o",
                markersize=3.5,
            )
        ax.axvline(100, color="#888", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(DONE_STEP / 1000.0, color="#bbb", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_title(key, fontsize=11)
        ax.set_ylim(-2, 105)
        ax.set_xlim(0, DONE_STEP / 1000.0)
        ax.set_xticks([0, 40, 80, 120, 160, 200])
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    for ax in axes[len(panels) :]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(AGENTS),
            fontsize=9,
            framealpha=0.95,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(
        "PB_toy noisy 100k · learning curves (hiql / tr_hiql / pbg / pbf)",
        fontsize=13,
        y=1.06,
    )
    fig.supxlabel("train step (k)", fontsize=10)
    fig.supylabel("mean success %", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(
        f"wrote {OUT} panels={len(panels)} "
        f"series={sum(1 for k in panels for a in AGENTS if (k, a) in store)}"
    )


if __name__ == "__main__":
    main()
