#!/usr/bin/env python3
"""Write scores.md and learning-curve PNGs/curves.md from checkpoints + logs."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "checkpoints"
LOG_DIR = ROOT / "nohup_logs" / "matrix"
DEFAULT_OUT = ROOT / "scores.md"
DEFAULT_CURVES_DIR = ROOT / "curves"
DEFAULT_CURVES_INDEX = ROOT / "curves.md"

# Only report the active recipe (ignore older K/h_a runs & stale log tails).
CURRENT_K = 25
CURRENT_H_A = 5

EVAL_RE = re.compile(
    r"\[(?P<agent>\w+)\]\s+eval@(?P<step>\d+)\s+n=(?P<n>\d+)\s+(?P<body>.+)"
)
TEMP_RE = re.compile(
    r"T=(?P<temp>[0-9.]+)\s+success=(?P<mean>[0-9.]+)±(?P<std>[0-9.]+)"
    r"(?:\s+(?P<tasks>(?:t\d+=[0-9.]+\s*)+))?"
)
STEP_RE = re.compile(r"\[(?P<agent>\w+)\]\s+step=(?P<step>\d+)\b")
LOADED_RE = re.compile(r"^Loaded \w+ dataset ", re.MULTILINE)


@dataclass
class TempScore:
    mean: float
    std: float
    tasks: dict[int, float] = field(default_factory=dict)


@dataclass
class RunRow:
    tag: str
    kind: str  # car_race | swingby
    agent: str
    env: str
    task: str
    seed: str
    policy: str = "expert"  # expert | noisy | random
    latest_step: int = 0
    train_step: int = 0
    done: bool = False
    running: bool = False
    temps: dict[float, TempScore] = field(default_factory=dict)
    best_primary: float | None = None
    best_primary_step: int | None = None
    source: str = ""


_POLICY_ORDER = {"expert": 0, "noisy": 1, "random": 2}

# Only the leakage-free swingby benchmark belongs in current learning curves.
ACTIVE_SWINGBY_SUFFIX = "_swingby"


def _parse_tag(tag: str, kind: str) -> tuple[str, str, str, str, str]:
    # car_race_grav_navigation_tr_hiql_s0
    # car_race_grav_navigation_pbg_noisy_s0 / swingby_planet_pbf_random_s0
    m = re.match(r"^(?P<body>.+)_s(?P<seed>\d+)$", tag)
    seed = m.group("seed") if m else "?"
    body = m.group("body") if m else tag
    # Optional dataset policy suffix (expert omitted from legacy tags).
    policy = "expert"
    for pol in ("noisy", "random", "expert"):
        suf = f"_{pol}"
        if body.endswith(suf):
            body = body[: -len(suf)]
            policy = pol
            break
    agents = ("tr_hiql", "pbg", "pbf", "trl", "dqc", "hiql", "gcbc", "gciql")
    agent = "?"
    for a in sorted(agents, key=len, reverse=True):
        suf = f"_{a}"
        if body.endswith(suf):
            agent = a
            body = body[: -len(suf)]
            break
    if kind == "swingby":
        return agent, body, "swingby", seed, policy
    # car_race_<env>_<task>; env is car_race_{ice,grav,anti_grav,plain}
    parts = body.split("_")
    if len(parts) >= 3 and parts[0] == "car" and parts[1] == "race":
        if parts[2] == "anti" and len(parts) >= 4 and parts[3] == "grav":
            env = "car_race_anti_grav"
            task = "_".join(parts[4:]) or "?"
        else:
            env = f"{parts[0]}_{parts[1]}_{parts[2]}"
            task = "_".join(parts[3:]) or "?"
        return agent, env, task, seed, policy
    return agent, body, "?", seed, policy


def _tasks_from_metrics(metrics: dict, prefix: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for tid in range(1, 6):
        key = f"{prefix}_task{tid}_success"
        if key in metrics:
            out[tid] = float(metrics[key])
    return out


def _scores_from_metrics(metrics: dict) -> dict[float, TempScore]:
    temps: dict[float, TempScore] = {}
    # Prefer prefixed t0_/t1_ keys (PathBridger dual-temp).
    for prefix, temp in (("t0", 0.0), ("t1", 1.0)):
        mean_key = f"{prefix}_mean_success"
        if mean_key not in metrics:
            continue
        temps[temp] = TempScore(
            mean=float(metrics[mean_key]),
            std=float(metrics.get(f"{prefix}_mean_success_std", 0.0)),
            tasks=_tasks_from_metrics(metrics, prefix),
        )
    if temps:
        return temps
    if "mean_success" in metrics:
        tasks = {
            tid: float(metrics[f"task{tid}_success"])
            for tid in range(1, 6)
            if f"task{tid}_success" in metrics
        }
        temps[float(metrics.get("eval_temperature", 0.0))] = TempScore(
            mean=float(metrics["mean_success"]),
            std=float(metrics.get("mean_success_std", 0.0)),
            tasks=tasks,
        )
    return temps


def _primary_temp(agent: str, temps: dict[float, TempScore]) -> float | None:
    if not temps:
        return None
    if agent in ("pbg", "pbf") and 1.0 in temps:
        return 1.0
    if 0.0 in temps:
        return 0.0
    return sorted(temps)[0]


def _fmt_score(s: TempScore | None) -> str:
    if s is None:
        return "—"
    return f"{s.mean:.2f}±{s.std:.2f}"


def _fmt_k(step: int | None) -> str:
    """Format step counts as k units (5000 -> 5k, 12500 -> 12.5k)."""
    if step is None or step <= 0:
        return "—"
    if step % 1000 == 0:
        return f"{step // 1000}k"
    return f"{step / 1000:g}k"


def _fmt_tasks(s: TempScore | None) -> str:
    if s is None or not s.tasks:
        return "—"
    return " ".join(f"t{k}={s.tasks[k]:.2f}" for k in sorted(s.tasks))


def _is_current_config(agent: str, config: dict | None) -> bool:
    """Drop sidecars from older recipes (e.g. K=8 / h_a=1)."""
    if not config:
        # No config → only allow agents that do not pin K/h_a.
        return agent in ("trl", "dqc", "bc", "gcbc")
    if agent in ("pbg", "pbf"):
        return (
            int(config.get("dynamics_N", -1)) == CURRENT_K
            and int(config.get("subgoal_steps", -1)) == CURRENT_K
            and int(config.get("action_chunk_horizon", -1)) == CURRENT_H_A
        )
    if agent in ("tr_hiql", "hiql"):
        return int(config.get("subgoal_steps", -1)) == CURRENT_K
    return True


def _session_log_text(text: str) -> str:
    """Keep only the latest train invocation inside an appended log file."""
    matches = list(LOADED_RE.finditer(text))
    if not matches:
        return ""
    return text[matches[-1].start() :]


def _parse_eval_body(body: str) -> dict[float, TempScore]:
    temps: dict[float, TempScore] = {}
    for tm in TEMP_RE.finditer(body):
        tasks: dict[int, float] = {}
        raw_tasks = tm.group("tasks") or ""
        for tbit in re.finditer(r"t(\d+)=([0-9.]+)", raw_tasks):
            tasks[int(tbit.group(1))] = float(tbit.group(2))
        temps[float(tm.group("temp"))] = TempScore(
            mean=float(tm.group("mean")),
            std=float(tm.group("std")),
            tasks=tasks,
        )
    return temps


def _empty_row(tag: str, kind: str) -> RunRow:
    agent, env, task, seed, policy = _parse_tag(tag, kind)
    return RunRow(
        tag=tag,
        kind=kind,
        agent=agent,
        env=env,
        task=task,
        seed=seed,
        policy=policy,
    )


def load_from_checkpoints() -> dict[str, RunRow]:
    rows: dict[str, RunRow] = {}
    for kind in ("car_race", "swingby"):
        base = CKPT_ROOT / kind
        if not base.is_dir():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue
            tag = run_dir.name
            agent, env, task, seed, policy = _parse_tag(tag, kind)
            row = RunRow(
                tag=tag,
                kind=kind,
                agent=agent,
                env=env,
                task=task,
                seed=seed,
                policy=policy,
            )
            best_primary = None
            best_step = None
            latest_step = 0
            latest_temps: dict[float, TempScore] = {}
            saw_current = False
            for js in sorted(run_dir.glob("step_*.json")):
                try:
                    step = int(js.stem.split("_", 1)[1])
                except ValueError:
                    continue
                try:
                    payload = json.loads(js.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if not _is_current_config(agent, payload.get("config")):
                    continue
                saw_current = True
                metrics = payload.get("metrics") or {}
                temps = _scores_from_metrics(metrics)
                if not temps:
                    continue
                if step >= latest_step:
                    latest_step = step
                    latest_temps = temps
                ptemp = _primary_temp(agent, temps)
                if ptemp is None:
                    continue
                val = temps[ptemp].mean
                if best_primary is None or val > best_primary:
                    best_primary = val
                    best_step = step
            if not saw_current:
                # Msgpack-only / old-recipe dirs: skip unless we later mark running.
                continue
            row.latest_step = latest_step
            row.temps = latest_temps
            row.best_primary = best_primary
            row.best_primary_step = best_step
            row.done = (run_dir / "step_50000.msgpack").is_file()
            row.source = "ckpt"
            rows[tag] = row
    return rows


def enrich_from_logs(rows: dict[str, RunRow]) -> None:
    """Update train/eval from the *latest* session only; never invent orphan rows."""
    if not LOG_DIR.is_dir():
        return
    for log in sorted(LOG_DIR.glob("*.log")):
        tag = log.stem
        if tag not in rows:
            continue
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        session = _session_log_text(text)
        if not session:
            continue
        row = rows[tag]
        train_step = 0
        for m in STEP_RE.finditer(session):
            train_step = max(train_step, int(m.group("step")))
        if train_step:
            row.train_step = max(row.train_step, train_step)

        session_best = None
        session_best_step = None
        last_eval = None
        for m in EVAL_RE.finditer(session):
            last_eval = m
            step = int(m.group("step"))
            temps = _parse_eval_body(m.group("body"))
            if not temps:
                continue
            ptemp = _primary_temp(row.agent, temps)
            if ptemp is None:
                continue
            val = temps[ptemp].mean
            if session_best is None or val > session_best:
                session_best = val
                session_best_step = step
        if last_eval is not None:
            step = int(last_eval.group("step"))
            temps = _parse_eval_body(last_eval.group("body"))
            if temps and step >= row.latest_step:
                row.latest_step = step
                row.temps = temps
            if session_best is not None and (
                row.best_primary is None or session_best > row.best_primary
            ):
                row.best_primary = session_best
                row.best_primary_step = session_best_step


def mark_running(rows: dict[str, RunRow]) -> None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", r"python -m (car_race|swingby)\.train"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return
    for line in out.splitlines():
        m = re.search(r"--checkpoint-dir\s+(\S+)", line)
        if not m:
            continue
        ckpt = pathlib.Path(m.group(1))
        tag = ckpt.name
        if tag not in rows:
            kind = "swingby" if "swingby" in ckpt.parts else "car_race"
            rows[tag] = _empty_row(tag, kind)
            rows[tag].source = "live"
        rows[tag].running = True


def render_md(rows: dict[str, RunRow]) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S %z")
    lines: list[str] = [
        "# PB_toy scoreboard",
        "",
        f"_Updated: {now}_",
        "",
        f"Active recipe only: **K={CURRENT_K}**, **h_a={CURRENT_H_A}** "
        "(older checkpoint/log scores are ignored).",
        "",
        "Scores show **T=0** and **T=1** side by side when both were evaluated "
        "(PathBridger). Primary for `pbg`/`pbf` is **T=1**, else **T=0**. "
        "Dataset policy: **expert** / **noisy** / **random** "
        "(legacy tags without a suffix are expert).",
        "",
        "## Task breakdown (latest eval)",
        "",
    ]
    ordered = sorted(
        (
            r
            for r in rows.values()
            if r.running
            or r.done
            or (
                # Current-recipe PB/HiQL partials only (skip idle old trl/dqc resumes).
                r.agent in ("pbg", "pbf", "tr_hiql", "hiql")
                and (r.latest_step or r.train_step or r.temps)
            )
        ),
        key=lambda r: (
            r.env,
            r.task,
            _POLICY_ORDER.get(r.policy, 9),
            r.agent,
            r.tag,
        ),
    )

    by_group: dict[tuple[str, str], list[RunRow]] = {}
    for r in ordered:
        by_group.setdefault((r.env, r.task), []).append(r)
    for (env, task), group in sorted(by_group.items()):
        lines.append(f"### {env} / {task}")
        lines.append("")
        lines.append(
            "| policy | agent | status | step | train | T=0 | T=0 tasks | "
            "T=1 | T=1 tasks | best |"
        )
        lines.append(
            "|---|---|---|---:|---:|---:|---|---:|---|---:|"
        )
        for r in group:
            if r.running:
                st = "RUN"
            elif r.done:
                st = "DONE"
            elif r.latest_step or r.train_step:
                st = "PARTIAL"
            else:
                st = "—"
            t0 = r.temps.get(0.0)
            t1 = r.temps.get(1.0)
            best = (
                f"{r.best_primary:.2f}@{_fmt_k(r.best_primary_step)}"
                if r.best_primary is not None
                else "—"
            )
            lines.append(
                f"| `{r.policy}` | `{r.agent}` | {st} | {_fmt_k(r.latest_step)} | "
                f"{_fmt_k(r.train_step)} | {_fmt_score(t0)} | {_fmt_tasks(t0)} | "
                f"{_fmt_score(t1)} | {_fmt_tasks(t1)} | {best} |"
            )
        lines.append("")

    n_run = sum(1 for r in ordered if r.running)
    n_done = sum(1 for r in ordered if r.done)
    by_policy = {
        pol: sum(1 for r in ordered if r.policy == pol)
        for pol in ("expert", "noisy", "random")
    }
    lines += [
        "## Counts",
        "",
        f"- rows shown: **{len(ordered)}** (scanned {len(rows)})",
        f"- running: **{n_run}**",
        f"- done (50k ckpt): **{n_done}**",
        f"- by policy: expert **{by_policy['expert']}**, "
        f"noisy **{by_policy['noisy']}**, random **{by_policy['random']}**",
        "",
        "Learning curves: [`curves.md`](curves.md).",
        "",
    ]
    return "\n".join(lines)


# --- learning curves ---------------------------------------------------------

_CURVE_AGENTS = ("tr_hiql", "hiql", "pbg", "pbf", "trl")
_AGENT_COLORS = {
    "tr_hiql": "#1f77b4",
    "hiql": "#ff7f0e",
    "pbg": "#2ca02c",
    "pbf": "#d62728",
    "trl": "#9467bd",
}


def collect_curves() -> dict[
    tuple[str, str, str], dict[str, list[tuple[int, float]]]
]:
    """(env, task, policy) -> agent -> [(step, primary_success), ...]."""
    groups: dict[
        tuple[str, str, str], dict[str, list[tuple[int, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    for kind in ("car_race", "swingby"):
        base = CKPT_ROOT / kind
        if not base.is_dir():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue
            tag = run_dir.name
            agent, env, task, _seed, policy = _parse_tag(tag, kind)
            if agent not in _CURVE_AGENTS:
                continue
            if kind == "swingby" and not env.endswith(ACTIVE_SWINGBY_SUFFIX):
                continue
            points: list[tuple[int, float]] = []
            for js in sorted(run_dir.glob("step_*.json")):
                try:
                    step = int(js.stem.split("_", 1)[1])
                except ValueError:
                    continue
                try:
                    payload = json.loads(js.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if not _is_current_config(agent, payload.get("config")):
                    continue
                temps = _scores_from_metrics(payload.get("metrics") or {})
                ptemp = _primary_temp(agent, temps)
                if ptemp is None:
                    continue
                points.append((step, float(temps[ptemp].mean)))
            if not points:
                continue
            points.sort(key=lambda x: x[0])
            groups[(env, task, policy)][agent] = points
    return groups


def _plot_group(
    env: str,
    task: str,
    policy: str,
    series: dict[str, list[tuple[int, float]]],
    out_path: pathlib.Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
    for agent in sorted(series.keys()):
        xs = [p[0] / 1000.0 for p in series[agent]]
        ys = [p[1] for p in series[agent]]
        ax.plot(
            xs,
            ys,
            color=_AGENT_COLORS.get(agent, "#333333"),
            linestyle="-",
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=agent,
        )
    ax.set_xlabel("train step (k)")
    ax.set_ylabel("primary eval success")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{env} / {task} / {policy}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _write_curves_index(
    index_path: pathlib.Path,
    figures: list[tuple[str, str, str, pathlib.Path, int]],
) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S %z")
    rel_root = index_path.parent
    lines = [
        "# PB_toy learning curves",
        "",
        f"_Updated: {now}_",
        "",
        f"Primary success vs step (recipe **K={CURRENT_K}**, "
        f"**h_a={CURRENT_H_A}**). "
        "`pbg`/`pbf` use **T=1**; others **T=0**. "
        "One plot per **expert** / **noisy** / **random** "
        f"(agents: {', '.join(_CURVE_AGENTS)}; dqc excluded).",
        "",
    ]
    for env, task, policy, path, n_series in figures:
        rel = path.relative_to(rel_root).as_posix()
        lines += [
            f"## {env} / {task} / {policy}",
            "",
            f"_{n_series} agents_",
            "",
            f"![{env}/{task}/{policy}]({rel})",
            "",
        ]
    if len(figures) > 1:
        lines += ["## All", ""]
        for env, task, policy, path, _n in figures:
            rel = path.relative_to(rel_root).as_posix()
            lines.append(f"- [`{env} / {task} / {policy}`]({rel})")
        lines.append("")
    index_path.write_text("\n".join(lines))


def write_learning_curves(
    *,
    out_dir: pathlib.Path = DEFAULT_CURVES_DIR,
    index: pathlib.Path = DEFAULT_CURVES_INDEX,
) -> int:
    groups = collect_curves()
    figures: list[tuple[str, str, str, pathlib.Path, int]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    keep: set[pathlib.Path] = set()
    for (env, task, policy), series in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            _POLICY_ORDER.get(item[0][2], 9),
        ),
    ):
        safe = f"{env}__{task}__{policy}".replace("/", "_")
        out = out_dir / f"{safe}.png"
        _plot_group(env, task, policy, series, out)
        keep.add(out.resolve())
        figures.append((env, task, policy, out, len(series)))
        print(f"wrote {out} series={len(series)}", flush=True)
    # Drop stale combined / old-layout plots.
    for png in out_dir.glob("*.png"):
        if png.resolve() not in keep:
            png.unlink(missing_ok=True)
            print(f"removed stale {png}", flush=True)
    _write_curves_index(index, figures)
    print(f"wrote {index} groups={len(figures)}", flush=True)
    return len(figures)


def write_scoreboard(*, out: pathlib.Path = DEFAULT_OUT) -> int:
    rows = load_from_checkpoints()
    mark_running(rows)
    enrich_from_logs(rows)
    out.write_text(render_md(rows))
    print(f"wrote {out} runs={len(rows)}", flush=True)
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    p.add_argument("--curves-dir", type=pathlib.Path, default=DEFAULT_CURVES_DIR)
    p.add_argument("--curves-index", type=pathlib.Path, default=DEFAULT_CURVES_INDEX)
    p.add_argument(
        "--scores-only",
        action="store_true",
        help="Skip learning-curve PNGs / curves.md",
    )
    p.add_argument(
        "--curves-only",
        action="store_true",
        help="Skip scores.md",
    )
    args = p.parse_args()
    if args.scores_only and args.curves_only:
        p.error("cannot combine --scores-only and --curves-only")
    if not args.curves_only:
        write_scoreboard(out=args.out)
    if not args.scores_only:
        write_learning_curves(out_dir=args.curves_dir, index=args.curves_index)


if __name__ == "__main__":
    main()
