#!/usr/bin/env python3
"""Export PB_toy eval + train stdout into PathBridger/logs (PB_logs).

Writes::

    logs/completed/pb_toy/<env>/<run_id>/eval/step{step}.json   # every ckpt eval
    logs/completed/pb_toy/<env>/<run_id>/train/stdout.txt        # train stdout
    logs/pb_toy/queues/<queue>/<tag>.txt                         # raw nohup copies
    logs/pb_toy/results_<size>_k{K}_<host>.jsonl

Idempotent by default (skip existing). Use ``--overwrite`` to refresh.
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
NOHUP_ROOT = ROOT / "nohup_logs"
LOG_DIRS = (
    NOHUP_ROOT / "queue_100k_k10",
    NOHUP_ROOT / "queue_10k_lap248",
    NOHUP_ROOT / "anti_grav_remain_10k_k10",
    NOHUP_ROOT / "train_10k_k10",
    NOHUP_ROOT / "train_1k_k10",
    NOHUP_ROOT / "matrix",
    NOHUP_ROOT / "car_parking_dataset",
    NOHUP_ROOT / "car_parking_dataset_noisy100k",
    NOHUP_ROOT / "car_parking_dataset_random",
)
DEFAULT_LOGS = pathlib.Path(
    os.environ.get("PB_LOGS_ROOT")
    or "/home/ext_csv/PathBridger/logs"
)

LOADED_RE = re.compile(
    r"Loaded \w+ dataset size=(?P<size>\d+)(?:\s+task=\S+)? from (?P<path>\S+)"
)
SIZE_LABEL_RE = re.compile(r"_(?P<label>\d+k)\.npz$")
STEP_RE = re.compile(r"^step_(\d+)\.json$")
RECIPE_TAIL_RE = re.compile(r"_k(?P<k>\d+)(?:_ha(?P<ha>\d+))?_(?P<label>\d+k)$")
MIXED_WEIGHT_RUNS = {
    "car_race_anti_grav_navigation_pbg_random_s0_k10_ha2_10k": "w1→0@4k",
    "car_race_anti_grav_navigation_pbf_random_s0_k10_ha2_10k": "w1→0@6k",
}

AGENTS = ("tr_hiql", "pbg", "pbf", "trl", "dqc", "hiql", "gcbc", "gciql")


def _host() -> str:
    return os.environ.get("PB_LOG_HOST") or socket.gethostname().split(".")[0]


def _sanitize(tok: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ".-") else "_" for ch in str(tok)) or "x"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_tag(tag: str, kind: str) -> tuple[str, str, str, str, str, int | None, int | None, str | None]:
    """Return agent, env, task, seed, policy, K, h_a, size_label."""
    recipe = RECIPE_TAIL_RE.search(tag)
    k = int(recipe.group("k")) if recipe else None
    ha = int(recipe.group("ha")) if recipe and recipe.group("ha") else None
    size_label = recipe.group("label") if recipe else None
    body = tag[: recipe.start()] if recipe else tag

    seed_m = re.match(r"^(?P<body>.+)_s(?P<seed>\d+)$", body)
    seed = seed_m.group("seed") if seed_m else "?"
    body = seed_m.group("body") if seed_m else body

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
        return agent, body, "swingby", seed, policy, k, ha, size_label
    if kind == "car_parking":
        return agent, "car_parking", "parking", seed, policy, k, ha, size_label

    parts = body.split("_")
    if len(parts) >= 3 and parts[0] == "car" and parts[1] == "race":
        if parts[2] == "anti" and len(parts) >= 4 and parts[3] == "grav":
            env = "car_race_anti_grav"
            task = "_".join(parts[4:]) or "?"
        else:
            env = f"{parts[0]}_{parts[1]}_{parts[2]}"
            task = "_".join(parts[3:]) or "?"
        return agent, env, task, seed, policy, k, ha, size_label
    return agent, body, "?", seed, policy, k, ha, size_label


def _is_selected_config(
    agent: str,
    config: dict | None,
    *,
    require_k: int | None,
    require_ha: int | None,
    tag_k: int | None,
    tag_ha: int | None,
) -> bool:
    if require_k is None and require_ha is None:
        return True
    k = None
    ha = None
    if config:
        if agent in ("pbg", "pbf"):
            k = int(config.get("subgoal_steps", config.get("dynamics_N", -1)))
            ha = int(config.get("action_chunk_horizon", -1))
        elif agent in ("tr_hiql", "hiql", "trl"):
            k = int(config.get("subgoal_steps", -1))
    if k is None:
        k = tag_k
    if ha is None:
        ha = tag_ha
    if require_k is not None and k != require_k:
        return False
    if require_ha is not None and agent in ("pbg", "pbf") and ha != require_ha:
        return False
    return True


def _log_candidates(tag: str) -> list[pathlib.Path]:
    names = [tag, f"{tag}_w0", f"{tag}_w1"]
    out: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for log_dir in LOG_DIRS:
        if not log_dir.is_dir():
            continue
        for name in names:
            p = log_dir / f"{name}.log"
            if p.is_symlink():
                p = p.resolve()
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


def _all_step_jsons(run_dir: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    out: list[tuple[int, pathlib.Path]] = []
    for p in run_dir.glob("step_*.json"):
        m = STEP_RE.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


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
    if agent in ("pbg", "pbf", "trl") and "t1_mean_success" in metrics:
        return float(metrics["t1_mean_success"])
    if "t0_mean_success" in metrics:
        return float(metrics["t0_mean_success"])
    if "mean_success" in metrics:
        return float(metrics["mean_success"])
    if "t1_mean_success" in metrics:
        return float(metrics["t1_mean_success"])
    return None


def _weight_label(tag: str, agent: str, config: dict) -> str | None:
    if tag in MIXED_WEIGHT_RUNS:
        return MIXED_WEIGHT_RUNS[tag]
    if agent == "hiql":
        return None
    if agent == "trl":
        return f"w{float(config.get('lam', 0.0)):g}"
    if agent in {"pbg", "pbf"}:
        return f"w{float(config.get('value_distance_weight_power', 1.0)):g}"
    if agent == "tr_hiql":
        return f"w{float(config.get('distance_weight_power', 1.0)):g}"
    return None


def _copy_text(src: pathlib.Path, dest: pathlib.Path, *, overwrite: bool) -> bool:
    if dest.exists() and not overwrite:
        # Refresh if source is newer/larger (running jobs grow).
        try:
            if dest.stat().st_size >= src.stat().st_size and dest.stat().st_mtime >= src.stat().st_mtime:
                return False
        except OSError:
            return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def sync_queue_logs(logs_root: pathlib.Path, *, overwrite: bool) -> int:
    """Copy raw nohup train logs into logs/pb_toy/queues/ (as .txt; *.log gitignored)."""
    n = 0
    host = _host()
    for log_dir in LOG_DIRS:
        if not log_dir.is_dir():
            continue
        qname = log_dir.name
        dest_q = logs_root / "pb_toy" / "queues" / f"{host}_{qname}"
        for src in sorted(log_dir.glob("*.log")):
            dest = dest_q / f"{src.stem}.txt"
            if _copy_text(src, dest, overwrite=overwrite):
                n += 1
        for src in sorted(log_dir.glob("*.out")):
            dest = dest_q / f"{src.stem}.out.txt"
            if _copy_text(src, dest, overwrite=overwrite):
                n += 1
    return n


def export(
    *,
    logs_root: pathlib.Path,
    apply: bool,
    require_k: int | None,
    require_ha: int | None,
    size_label: str | None,
    overwrite: bool,
    all_steps: bool,
) -> dict:
    host = _host()
    written = 0
    skipped = 0
    skipped_recipe = 0
    skipped_existing = 0
    train_logs = 0
    rows: list[dict] = []

    kinds = ("car_race", "swingby", "car_parking")
    for kind in kinds:
        kind_root = CKPT_ROOT / kind
        if not kind_root.is_dir():
            continue
        for run_dir in sorted(p for p in kind_root.iterdir() if p.is_dir()):
            tag = run_dir.name
            steps = _all_step_jsons(run_dir)
            if not steps:
                skipped += 1
                continue
            if not all_steps:
                steps = [steps[-1]]

            parsed = _parse_tag(tag, kind)
            parsed_agent, env, task, seed_s, policy, tag_k, tag_ha, tag_size = parsed
            # Peek config from latest for recipe filter.
            latest_data = json.loads(steps[-1][1].read_text(encoding="utf-8"))
            config = latest_data.get("config") or {}
            agent = str(latest_data.get("agent") or parsed_agent or "?")
            if not _is_selected_config(
                agent,
                config if isinstance(config, dict) else None,
                require_k=require_k,
                require_ha=require_ha,
                tag_k=tag_k,
                tag_ha=tag_ha,
            ):
                skipped_recipe += 1
                continue
            if size_label is not None and tag_size != size_label:
                skipped_recipe += 1
                continue
            try:
                seed = int(seed_s)
            except ValueError:
                seed = 0
            ds_size, ds_label, ds_path = _dataset_from_log(tag)
            if ds_label is None:
                ds_label = tag_size
            weight = _weight_label(tag, agent, config if isinstance(config, dict) else {})
            run_id = f"{_sanitize(host)}_pb_toy_{_sanitize(tag)}"
            stdout_logs = _log_candidates(tag)
            run_root = logs_root / "completed" / "pb_toy" / _sanitize(env) / run_id

            if apply and stdout_logs:
                dest_stdout = run_root / "train" / "stdout.txt"
                if _copy_text(stdout_logs[0], dest_stdout, overwrite=True):
                    train_logs += 1

            for step, step_path in steps:
                data = json.loads(step_path.read_text(encoding="utf-8"))
                metrics = data.get("metrics") or {}
                primary = _primary_success(agent, metrics)
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
                    "status": "completed" if step == steps[-1][0] else "checkpoint",
                    "created_at": _now_utc(),
                    "dataset_size": ds_size,
                    "dataset_size_label": ds_label,
                    "dataset_path": ds_path,
                    "weight": weight,
                    "train": {
                        "steps": int(data.get("steps") or step),
                        "K": tag_k if tag_k is not None else require_k,
                        "h_a": tag_ha if agent in ("pbg", "pbf") else None,
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
                        "stdout_log": str(stdout_logs[0]) if stdout_logs else None,
                    },
                }
                dest_dir = run_root / "eval"
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
                    "weight": weight,
                    "step": step,
                    "primary_success": primary,
                    "run_id": run_id,
                    "record_path": str(dest.relative_to(logs_root)) if apply or True else str(dest),
                }
                rows.append(row)
                if apply:
                    if dest.exists() and not overwrite:
                        skipped_existing += 1
                        continue
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest.write_text(
                        json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    written += 1
                else:
                    written += 1

    queue_copied = 0
    if apply:
        queue_copied = sync_queue_logs(logs_root, overwrite=overwrite)

    summary = {
        "host": host,
        "created_at": _now_utc(),
        "recipe": {"K": require_k, "h_a": require_ha, "size_label": size_label, "all_steps": all_steps},
        "n_written": written,
        "n_skipped_missing": skipped,
        "n_skipped_recipe_or_size": skipped_recipe,
        "n_skipped_existing": skipped_existing,
        "n_train_stdout": train_logs,
        "n_queue_logs": queue_copied,
        "n_rows": len(rows),
    }

    if apply:
        out_dir = logs_root / "pb_toy"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = []
        if size_label:
            suffix.append(size_label)
        if require_k is not None:
            suffix.append(f"k{require_k}")
        name = "results_" + "_".join(suffix) if suffix else "results"
        host_s = _sanitize(host)
        jsonl = out_dir / f"{name}_{host_s}.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        (out_dir / f"export_meta_{name}_{host_s}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-root", type=pathlib.Path, default=DEFAULT_LOGS)
    ap.add_argument("--apply", action="store_true", help="write files (default: dry-run)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--h-a", type=int, default=2)
    ap.add_argument(
        "--size-label",
        default=None,
        help="Filter size label (1k/10k/100k). Default: all sizes.",
    )
    ap.add_argument(
        "--all-steps",
        action="store_true",
        default=True,
        help="Export every step_*.json (default: on)",
    )
    ap.add_argument("--latest-only", action="store_true", help="Only latest checkpoint eval")
    args = ap.parse_args()
    summary = export(
        logs_root=args.logs_root,
        apply=args.apply,
        require_k=args.k,
        require_ha=args.h_a,
        size_label=args.size_label,
        overwrite=args.overwrite,
        all_steps=not args.latest_only,
    )
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
