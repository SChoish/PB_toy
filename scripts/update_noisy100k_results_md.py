#!/usr/bin/env python3
"""Regenerate PB_toy_results_* @100k/@200k table (csh + dgx-200k-complete, no source marks)."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "PB_toy_results_20260723_noisy100k_eval100k_200k_csh.md"
PB_LOGS = Path("/home/ext_csh/PB_logs/completed/pb_toy")
AGENTS = ["hiql", "tr_hiql", "pbg", "pbf"]  # trl excluded
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


def parse_tag(tag: str) -> tuple[str | None, str]:
    m = re.search(r"_(hiql|tr_hiql|pbg|pbf)_noisy", tag)
    agent = m.group(1) if m else "?"
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


def put(store: dict, key: str, agent: str, score: float, rank: int) -> None:
    k = (key, agent)
    if k not in store or rank <= store[k][1]:
        store[k] = (score, rank)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> None:
    best100: dict = {}
    best200: dict = {}

    # local checkpoints (rank 0)
    for root in (ROOT / "checkpoints").iterdir():
        if not root.is_dir():
            continue
        for ckpt in root.iterdir():
            if not ckpt.is_dir() or "noisy" not in ckpt.name or not ckpt.name.endswith("_100k"):
                continue
            key, agent = parse_tag(ckpt.name)
            if key is None:
                continue
            for step, store in ((100000, best100), (200000, best200)):
                p = ckpt / f"step_{step}.json"
                if not p.exists():
                    continue
                d = load_json(p)
                if not d:
                    continue
                s = mean_success(d.get("metrics", {}))
                if s is None:
                    continue
                put(store, key, agent, round(s * 100, 1), 0)

    # PB_logs
    if PB_LOGS.exists():
        for p in PB_LOGS.rglob("eval/step*.json"):
            name = p.parent.parent.name
            if "noisy" not in name or "100k" not in name:
                continue
            m = re.search(r"step[_]?(\d+)\.json$", p.name)
            if not m:
                continue
            step = int(m.group(1))
            if step not in (100000, 200000):
                continue
            if "csh_server" in name:
                src_rank = 1
                tag = name.replace("csh_server_pb_toy_", "")
            elif "ext_csv-box" in name:
                # dgx: only if this run completed 200k
                if not (p.parent / "step200000.json").exists() and p.name != "step200000.json":
                    continue
                if step == 100000 and not (p.parent / "step200000.json").exists():
                    continue
                src_rank = 2
                tag = name.replace("ext_csv-box_pb_toy_", "")
            else:
                continue
            key, agent = parse_tag(tag)
            if key is None:
                continue
            d = load_json(p)
            if not d:
                continue
            s = mean_success(d.get("metrics", d) if isinstance(d.get("metrics", d), dict) else {})
            if s is None:
                continue
            store = best100 if step == 100000 else best200
            put(store, key, agent, round(s * 100, 1), src_rank)

    now = datetime.now(timezone(timedelta(hours=9)))
    header = "| env/task | " + " | ".join(AGENTS) + " |"
    sep = "|---|" + "|".join(["---:"] * len(AGENTS)) + "|"
    lines = [
        f"# PB toy 결과 — noisy 100k · **@100k / @200k** ({now.strftime('%Y-%m-%d %H:%M')} KST)",
        "",
        "- 셀: **`@100k / @200k`** mean success (%). csh·dgx(200k 완료) 병합, 출처 구분 없음.",
        "- 러닝커브: `PB_toy_learning_curves_noisy100k_csh.png`",
        "- 이 파일은 `scripts/sync_pb_toy_to_pblogs_csh.sh` 워처가 주기적으로 갱신.",
        f"- agents: {', '.join(AGENTS)} (trl 제외)",
        "",
        "## @100k / @200k",
        "",
        header,
        sep,
    ]
    vals100: dict[str, list[float]] = defaultdict(list)
    vals200: dict[str, list[float]] = defaultdict(list)
    for key in ORDER:
        cells = []
        anyd = False
        for a in AGENTS:
            a1 = best100.get((key, a))
            a2 = best200.get((key, a))
            if not a1 and not a2:
                cells.append("—")
                continue
            anyd = True
            left = f"{a1[0]:.1f}" if a1 else "—"
            right = f"{a2[0]:.1f}" if a2 else "—"
            cells.append(f"{left} / {right}")
            if a1:
                vals100[a].append(a1[0])
            if a2:
                vals200[a].append(a2[0])
        if anyd:
            lines.append("| " + " | ".join([key] + cells) + " |")

    lines += [
        "",
        "## Agent mean",
        "",
        "| budget | " + " | ".join(AGENTS) + " |",
        "|---|" + "|".join(["---:"] * len(AGENTS)) + "|",
    ]
    for label, store in (("@100k", vals100), ("@200k", vals200)):
        cells = [
            f"{sum(store[a]) / len(store[a]):.1f}" if store[a] else "—" for a in AGENTS
        ]
        lines.append("| " + " | ".join([label] + cells) + " |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
