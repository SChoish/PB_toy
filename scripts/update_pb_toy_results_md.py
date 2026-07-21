#!/usr/bin/env python3
"""Regenerate the live PB toy result table from checkpoints and NT evals."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PB_LOGS = ROOT.parent / "PB_logs" / "completed" / "pb_toy"
OUTPUT = ROOT / "PB_toy_results_20260721.md"

ALGOS = ("hiql", "tr_hiql", "pbg", "pbf", "trl")
POLICIES = ("expert", "noisy", "random")
WEIGHTED_ALGOS = {"tr_hiql", "pbg", "pbf"}
LAP_TASK_PATTERN = r"lap_[1-8]p"

# These checkpoints used w=1 before resuming with w=0.
MIXED_WEIGHT_RUNS = {
    "car_race_anti_grav_navigation_pbg_random_s0_k10_ha2_10k": "w1→0@4k",
    "car_race_anti_grav_navigation_pbf_random_s0_k10_ha2_10k": "w1→0@6k",
}


def parse_tag(tag: str) -> tuple[str, str, str, str] | None:
    match = re.match(
        r"(car_race_[a-z_]+|swingby_[a-z]+)_"
        rf"({LAP_TASK_PATTERN}|navigation|swingby)_"
        r"(tr_hiql|hiql|pbg|pbf|trl|dqc)"
        r"(?:_(noisy|random))?_s0",
        tag,
    )
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), match.group(4) or "expert"


def primary_success(metrics: dict) -> float | None:
    for key in ("t1_mean_success", "t0_mean_success", "mean_success"):
        if key in metrics:
            return float(metrics[key])
    return None


def weight_label(tag: str, algo: str, config: dict) -> str | None:
    if algo not in WEIGHTED_ALGOS:
        return None
    if tag in MIXED_WEIGHT_RUNS:
        return MIXED_WEIGHT_RUNS[tag]
    key = (
        "value_distance_weight_power"
        if algo in {"pbg", "pbf"}
        else "distance_weight_power"
    )
    # Old TR-HIQL checkpoints predate the explicit config key and used default 1.
    return f"w{float(config.get(key, 1.0)):g}"


def load_train_results() -> dict[str, dict[tuple[str, str, str, str], dict]]:
    results: dict[str, dict[tuple[str, str, str, str], dict]] = {
        "1k": {},
        "10k": {},
    }
    for size in results:
        for checkpoint_root in (
            ROOT / "checkpoints" / "car_race",
            ROOT / "checkpoints" / "swingby",
        ):
            for run_dir in checkpoint_root.glob(f"*k10*{size}"):
                meta_path = run_dir / "step_10000.json"
                if not meta_path.is_file():
                    continue
                parsed = parse_tag(run_dir.name)
                if parsed is None:
                    continue
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                success = primary_success(metadata.get("metrics") or {})
                if success is None:
                    continue
                algo = parsed[2]
                results[size][parsed] = {
                    "success": success,
                    "weight": weight_label(
                        run_dir.name, algo, metadata.get("config") or {}
                    ),
                    "tag": run_dir.name,
                }
    return results


def load_nt_best() -> dict[tuple[str, str, str, str], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    if not PB_LOGS.is_dir():
        return {}
    for result_path in PB_LOGS.glob(
        "**/bridge_eval/ckpt10000/n*_t*_attempt0.json"
    ):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        grouped[str(result.get("tag"))].append(result)

    best: dict[tuple[str, str, str, str], dict] = {}
    for tag, rows in grouped.items():
        parsed = parse_tag(tag)
        if parsed is None:
            continue

        def score(row: dict) -> float:
            metrics = row.get("metrics") or {}
            return float(row.get("primary_success") or metrics.get("mean_success") or 0)

        winner = max(rows, key=score)
        best[parsed] = {
            "success": score(winner),
            "N": int(winner.get("subgoal_eval_num_samples") or 0),
            "T": float(winner.get("subgoal_temperature") or 0),
            "coverage": len(rows),
            "tag": tag,
        }
    return best


def env_label(env: str, task: str) -> str:
    return f"{env.removeprefix('car_race_').removeprefix('swingby_')}/{task}"


def format_value(value: float | None, weight: str | None, nt: bool = False) -> str:
    if value is None:
        return "—"
    suffix = "*" if nt else ""
    weight_suffix = f" [{weight}]" if weight else ""
    return f"{value * 100:.1f}{suffix}{weight_suffix}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def render() -> str:
    train = load_train_results()
    nt_best = load_nt_best()
    envs = sorted(
        {
            (key[0], key[1])
            for size_results in train.values()
            for key in size_results
        }
    )
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    lines = [
        f"# PB toy 결과 값 정리 ({now})",
        "",
        "- 값 = mean success (%), train-time final eval (`step_10000`).",
        "- `*` = 1k NT sweep **best 값**으로 대체.",
        "- `[w1]` / `[w0]` = distance-weight power. `[w1→0@4k]`처럼 표시된 run은 중간에 설정 변경.",
        "- lap task는 `lap_1p`~`lap_8p`를 자동 인식. 현재 4p/8p는 10k train 결과만 존재.",
        "- mean은 각 dataset에서 현재 완료된 셀만 산술평균하며 `—`는 제외. 1k의 `*`는 NT best로 반영.",
        "",
        "## Weight 현황",
        "",
        "| dataset | algo | weight별 완료 run |",
        "|---|---|---|",
    ]
    for size in ("1k", "10k"):
        for algo in ("tr_hiql", "pbg", "pbf"):
            counts: dict[str, int] = defaultdict(int)
            for key, row in train[size].items():
                if key[2] == algo:
                    counts[str(row["weight"])] += 1
            summary = ", ".join(
                f"{label}: {count}" for label, count in sorted(counts.items())
            )
            if summary:
                lines.append(f"| {size} | {algo} | {summary} |")

    lines += [
        "",
        "## NT sweep best (1k, 적용된 셀)",
        "",
        "| env/task | algo | policy | NT best | cell (N,T) | coverage | weight |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for key in sorted(nt_best):
        row = nt_best[key]
        train_row = train["1k"].get(key, {})
        lines.append(
            f"| {env_label(key[0], key[1])} | {key[2]} | {key[3]} | "
            f"{row['success'] * 100:.1f} | N{row['N']} T{row['T']:g} | "
            f"{row['coverage']}/24 | {train_row.get('weight') or '—'} |"
        )

    overall: dict[str, dict[str, list[float]]] = {
        size: {algo: [] for algo in ALGOS} for size in ("1k", "10k")
    }
    sections: list[str] = []
    for size in ("1k", "10k"):
        sections += ["", f"## {size}"]
        for policy in POLICIES:
            sections += [
                "",
                f"### {size} · {policy}",
                "",
                "| env/task | " + " | ".join(ALGOS) + " |",
                "|---|" + "|".join(["---:"] * len(ALGOS)) + "|",
            ]
            policy_values = {algo: [] for algo in ALGOS}
            for env, task in envs:
                rendered_cells: list[str] = []
                any_value = False
                for algo in ALGOS:
                    key = (env, task, algo, policy)
                    train_row = train[size].get(key)
                    value = train_row["success"] if train_row else None
                    is_nt = size == "1k" and key in nt_best
                    if is_nt:
                        value = nt_best[key]["success"]
                    weight = train_row["weight"] if train_row else None
                    rendered_cells.append(format_value(value, weight, is_nt))
                    if value is not None:
                        any_value = True
                        policy_values[algo].append(value)
                        overall[size][algo].append(value)
                if any_value:
                    sections.append(
                        f"| {env_label(env, task)} | "
                        + " | ".join(rendered_cells)
                        + " |"
                    )
            mean_cells = [
                f"**{mean(policy_values[algo]) * 100:.1f}**"
                if policy_values[algo]
                else "—"
                for algo in ALGOS
            ]
            sections.append("| **mean** | " + " | ".join(mean_cells) + " |")

    lines += [
        "",
        "## 전체 mean",
        "",
        "| dataset | " + " | ".join(ALGOS) + " |",
        "|---|" + "|".join(["---:"] * len(ALGOS)) + "|",
    ]
    for size in ("1k", "10k"):
        values = [
            f"{mean(overall[size][algo]) * 100:.1f}"
            if overall[size][algo]
            else "—"
            for algo in ALGOS
        ]
        label = "10k mean (완료분)" if size == "10k" else "1k mean"
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend(sections)
    return "\n".join(lines) + "\n"


def main() -> None:
    content = render()
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"updated {OUTPUT} ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
