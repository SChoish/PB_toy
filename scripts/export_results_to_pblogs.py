#!/usr/bin/env python3
"""Export PB_toy eval results into PathBridger/logs (PB_logs) for GitHub backup.

Writes one latest-eval JSON per run under::

    logs/completed/pb_toy/<env>/<run_id>/eval/step{step}.json

Each record includes ``algorithm``, ``dataset_size``, and ``host`` (plus task /
policy / metrics). Also writes a flat ``logs/pb_toy/results.jsonl`` summary
and copies ``scores.md``.

Idempotent: never overwrites an existing record file.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import socket
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "checkpoints"
LOG_DIR = ROOT / "nohup_logs" / "matrix"
DEFAULT_LOGS = pathlib.Path("/home/ext_csv/PathBridger/logs")

CURRENT_K = 25
CURRENT_H_A = 5

LOADED_RE = re.compile(
    r"Loaded \w+ dataset size=(?P<size>\d+)(?:\s+task=\S+)? from (?P<path>\S+)"
)
SIZE_LABEL_RE = re.compile(r"_(?P<label>\d+k)\.npz$")
STEP_RE = re.compile(r"^step_(\d+)\.json$")

AGENTS = ("tr_hiql", "pbg", "pbf", "trl", "dqc", "hiql", "gcbc", "gciql")


def _host() -> str:
    return os.environ.get("PB_LOG_HOST") or socket.gethostname() or "unknown-host"


def _sanitize(tok: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ".-") else "_" for ch in str(tok)) or "x"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_tag(tag: str, kind: str) -> tuple[str, str, str, str, str]:
    m = re.match(r"^(?P<body>.+)_s(?P<seed>\d+)$", tag)
    seed = m.group("seed") if m else "?"
    body = m.group("body") if m else tag
    policy = "expert"
    for pol in ("noisy", "random", "expert"):
        suf = f"_{pol}"
        if body.endswith(suf):
            body = body[: -len(suf)]
            policy = pol
            break
    agent = "?"
    for a in sorted(AGENTS, key=len, reverse=True):
        suf = f"_{a}"
        if body.endswith(suf):
            agent = a
            body = body[: -len(suf)]
            break
    if kind == "swingby":
        return agent, body, "swingby", seed, policy
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


def _is_current_config(agent: str, config: dict | None) -> bool:
    if not config:
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


def _log_candidates(tag: str) -> list[pathlib.Path]:
    """Resolve matrix log names; accept legacy ``_wpv2`` tags if present."""
    names = [tag]
    if "_wpv2_" in tag:
        names.append(tag.replace("_wpv2_", "_", 1))
    elif tag.endswith("_wpv2"):
        names.append(tag[: -len("_wpv2")])
    # Current lap tags omit _wpv2; also probe the legacy log name.
    if "_lap_" in tag and "_wpv2" not in tag:
        names.append(re.sub(r"(_lap_\d+p)_", r"\1_wpv2_", tag, count=1))
    out: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for name in names:
        p = LOG_DIR / f"{name}.log"
        if p.is_file() and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _dataset_from_log(tag: str) -> tuple[int | None, str | None, str | None]:
    for log in _log_candidates(tag):
        text = log.read_text(encoding="utf-8", errors="replace")
        matches = list(LOADED_RE.finditer(text))
        if not matches:
            continue
        m = matches[-1]
        size = int(m.group("size"))
        path = m.group("path")
        label = None
        lm = SIZE_LABEL_RE.search(path)
        if lm:
            label = lm.group("label")
        return size, label, path
    return None, None, None


def _latest_step_json(run_dir: pathlib.Path) -> tuple[int, pathlib.Path] | None:
    best: tuple[int, pathlib.Path] | None = None
    for p in run_dir.glob("step_*.json"):
        m = STEP_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, p)
    return best


def _metrics_compact(metrics: dict) -> dict:
    out: dict = {}
    for prefix in ("t0", "t1"):
        mean_key = f"{prefix}_mean_success"
        if mean_key in metrics:
            out[mean_key] = float(metrics[mean_key])
            out[f"{prefix}_mean_success_std"] = float(
                metrics.get(f"{prefix}_mean_success_std", 0.0)
            )
            for tid in range(1, 6):
                tk = f"{prefix}_task{tid}_success"
                if tk in metrics:
                    out[tk] = float(metrics[tk])
    if "mean_success" in metrics and "t0_mean_success" not in out:
        out["mean_success"] = float(metrics["mean_success"])
        out["mean_success_std"] = float(metrics.get("mean_success_std", 0.0))
    for k in ("num_eval_envs", "episodes_per_task", "total_eval_episodes", "eval_temperature"):
        if k in metrics:
            out[k] = metrics[k]
    return out


def _primary_success(agent: str, metrics: dict) -> float | None:
    if agent in ("pbg", "pbf") and "t1_mean_success" in metrics:
        return float(metrics["t1_mean_success"])
    if "t0_mean_success" in metrics:
        return float(metrics["t0_mean_success"])
    if "mean_success" in metrics:
        return float(metrics["mean_success"])
    if "t1_mean_success" in metrics:
        return float(metrics["t1_mean_success"])
    return None


def export(*, logs_root: pathlib.Path, apply: bool) -> dict:
    host = _host()
    written = 0
    skipped = 0
    skipped_recipe = 0
    rows: list[dict] = []

    for kind in ("car_race", "swingby"):
        kind_root = CKPT_ROOT / kind
        if not kind_root.is_dir():
            continue
        for run_dir in sorted(p for p in kind_root.iterdir() if p.is_dir()):
            tag = run_dir.name
            latest = _latest_step_json(run_dir)
            if latest is None:
                skipped += 1
                continue
            step, step_path = latest
            data = json.loads(step_path.read_text(encoding="utf-8"))
            agent = str(data.get("agent") or "?")
            config = data.get("config") or {}
            if not _is_current_config(agent, config if isinstance(config, dict) else None):
                skipped_recipe += 1
                continue
            parsed_agent, env, task, seed_s, policy = _parse_tag(tag, kind)
            if agent == "?" and parsed_agent != "?":
                agent = parsed_agent
            try:
                seed = int(seed_s)
            except ValueError:
                seed = 0
            ds_size, ds_label, ds_path = _dataset_from_log(tag)
            if ds_label is None:
                ds_label = "100k"  # matrix default
            metrics = data.get("metrics") or {}
            primary = _primary_success(agent, metrics)
            # Stable across re-exports so dest.exists() stays idempotent.
            run_id = f"{_sanitize(host)}_pb_toy_{_sanitize(tag)}"
            rec = {
                "schema_version": "1.0-pb_toy",
                "record_id": f"{run_id}__eval__step{step}",
                "run_id": run_id,
                "host": host,
                "method": "pb_toy",
                "algorithm": agent,
                "phase": "eval",
                "env": env,
                "task": task,
                "policy": policy,
                "seed": seed,
                "status": "completed",
                "created_at": _now_utc(),
                "dataset_size": ds_size,
                "dataset_size_label": ds_label,
                "dataset_path": ds_path,
                "train": {
                    "steps": int(data.get("steps") or step),
                    "K": CURRENT_K if agent in ("pbg", "pbf", "tr_hiql", "hiql") else None,
                    "h_a": CURRENT_H_A if agent in ("pbg", "pbf") else None,
                    "discount": config.get("discount") if isinstance(config, dict) else None,
                },
                "eval": {
                    "checkpoint": int(step),
                    "tag": tag,
                },
                "metrics": _metrics_compact(metrics),
                "primary_success": primary,
                "artifacts": {
                    "checkpoint_json": str(step_path),
                    "stdout_log": str(LOG_DIR / f"{tag}.log"),
                },
            }
            dest_dir = (
                logs_root
                / "completed"
                / "pb_toy"
                / _sanitize(env)
                / run_id
                / "eval"
            )
            dest = dest_dir / f"step{step}.json"
            row = {
                "host": host,
                "algorithm": agent,
                "env": env,
                "task": task,
                "policy": policy,
                "seed": seed,
                "dataset_size": ds_size,
                "dataset_size_label": ds_label,
                "step": step,
                "primary_success": primary,
                "run_id": run_id,
                "record_path": str(dest.relative_to(logs_root)),
            }
            rows.append(row)
            if apply:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest.write_text(
                    json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                written += 1
            else:
                written += 1

    summary = {
        "host": host,
        "created_at": _now_utc(),
        "recipe": {"K": CURRENT_K, "h_a": CURRENT_H_A},
        "n_written": written,
        "n_skipped_existing_or_empty": skipped,
        "n_skipped_old_recipe": skipped_recipe,
        "n_rows": len(rows),
    }

    if apply:
        out_dir = logs_root / "pb_toy"
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl = out_dir / "results.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        (out_dir / "export_meta.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        scores_src = ROOT / "scores.md"
        if scores_src.is_file():
            shutil.copy2(scores_src, out_dir / "scores.md")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-root", type=pathlib.Path, default=DEFAULT_LOGS)
    ap.add_argument("--apply", action="store_true", help="write files (default: dry-run)")
    args = ap.parse_args()
    summary = export(logs_root=args.logs_root, apply=args.apply)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
