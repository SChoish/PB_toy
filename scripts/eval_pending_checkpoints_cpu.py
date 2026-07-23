#!/usr/bin/env python3
"""CPU eval watcher: fill metrics for checkpoints saved with eval_status=pending.

Train can run with --eval-every 0 --save-every N on GPU; this process scans
checkpoints/ and evaluates on CPU without blocking training.

Parallelism: each eval is a separate subprocess (JAX-safe). Default --workers
is sized for cgroup pids (~800 threads/eval).

Usage:
  WATCH=1 INTERVAL_SEC=60 python scripts/eval_pending_checkpoints_cpu.py --workers 4
  python scripts/eval_pending_checkpoints_cpu.py --once --workers 4
  python scripts/eval_pending_checkpoints_cpu.py --job checkpoints/.../tag 140000
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

CKPT_ROOT = ROOT / "checkpoints"
AGENTS = ("hiql", "tr_hiql", "pbg", "pbf")  # queue agents (trl excluded)
KST = timezone(timedelta(hours=9))
# ~800 tids per JAX CPU eval on this host; keep cgroup headroom.
DEFAULT_WORKERS = int(os.environ.get("CPU_EVAL_WORKERS", "4"))


def ts() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def mean_success(metrics: dict) -> float | None:
    if not isinstance(metrics, dict):
        return None
    for k in ("mean_success", "t0_mean_success", "t1_mean_success"):
        if isinstance(metrics.get(k), (int, float)):
            return float(metrics[k])
    return None


def needs_eval(meta: dict | None) -> bool:
    if not meta:
        return False
    if meta.get("eval_status") == "pending":
        return True
    if meta.get("eval_status") in ("done", "running"):
        return False
    if "config" not in meta:
        return False
    return mean_success(meta.get("metrics") or {}) is None


def parse_tag(tag: str) -> dict:
    """Infer kind/env/task/agent/seed/size from checkpoint dir name."""
    m = re.search(r"_(hiql|tr_hiql|pbg|pbf|trl|dqc)_", tag)
    agent = m.group(1) if m else "?"
    seed_m = re.search(r"_s(\d+)", tag)
    seed = int(seed_m.group(1)) if seed_m else 0
    size_m = re.search(r"_(1k|10k|100k)$", tag)
    size = size_m.group(1) if size_m else "100k"
    if tag.startswith("car_parking"):
        return {
            "kind": "car_parking",
            "env": "car_parking",
            "task": "car_parking",
            "agent": agent,
            "seed": seed,
            "size": size,
        }
    if "swingby" in tag:
        env = "swingby_planet" if "planet" in tag else "swingby_blackhole"
        return {
            "kind": "swingby",
            "env": env,
            "task": "swingby",
            "agent": agent,
            "seed": seed,
            "size": size,
        }
    if "anti_grav" in tag:
        env = "car_race_anti_grav"
    elif "car_race_grav" in tag or re.search(
        r"(^|_)grav(_|$)", tag.replace("anti_grav", "")
    ):
        env = "car_race_grav"
    else:
        env = "car_race_ice"
    tm = re.search(r"lap_(\d+p)", tag)
    task = f"lap_{tm.group(1)}" if tm else "navigation"
    return {
        "kind": "car_race",
        "env": env,
        "task": task,
        "agent": agent,
        "seed": seed,
        "size": size,
    }


def dataset_for(info: dict, meta: dict) -> pathlib.Path:
    if meta.get("dataset"):
        p = pathlib.Path(meta["dataset"])
        if p.is_file():
            return p
    kind, env, task, size = info["kind"], info["env"], info["task"], info["size"]
    policy = "noisy"
    if kind == "car_race":
        if task == "navigation":
            return ROOT / "car_race" / "datasets" / f"{env}_{policy}_{size}.npz"
        return ROOT / "car_race" / "datasets" / f"{env}_lap_{policy}_{size}.npz"
    if kind == "swingby":
        return ROOT / "swingby" / "datasets" / f"{env}_swingby_{policy}_{size}.npz"
    return ROOT / "car_parking" / "datasets" / f"car_parking_{policy}_{size}.npz"


def lock_path(ckpt_dir: pathlib.Path, step: int) -> pathlib.Path:
    return ckpt_dir / f"step_{step}.eval.lock"


def claim_lock(ckpt_dir: pathlib.Path, step: int, stale_sec: int = 7200) -> bool:
    """Exclusive claim. Clears stale locks older than stale_sec."""
    lp = lock_path(ckpt_dir, step)
    if lp.exists():
        try:
            age = time.time() - lp.stat().st_mtime
            if age < stale_sec:
                return False
            lp.unlink(missing_ok=True)
        except OSError:
            return False
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {ts()}\n".encode())
        os.close(fd)
    except FileExistsError:
        return False
    meta_path = ckpt_dir / f"step_{step}.json"
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("eval_status") == "pending":
                meta["eval_status"] = "running"
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        pass
    return True


def release_lock(ckpt_dir: pathlib.Path, step: int) -> None:
    try:
        lock_path(ckpt_dir, step).unlink(missing_ok=True)
    except OSError:
        pass


def pending_jobs(stride: int | None) -> list[tuple[pathlib.Path, int, dict, dict]]:
    out = []
    for domain in ("car_race", "swingby", "car_parking"):
        root = CKPT_ROOT / domain
        if not root.is_dir():
            continue
        for ckpt_dir in sorted(root.iterdir()):
            if not ckpt_dir.is_dir() or "noisy" not in ckpt_dir.name:
                continue
            for pack in sorted(ckpt_dir.glob("step_*.msgpack")):
                m = re.match(r"step_(\d+)\.msgpack$", pack.name)
                if not m:
                    continue
                step = int(m.group(1))
                if stride and step % stride != 0:
                    continue
                meta_path = ckpt_dir / f"step_{step}.json"
                meta = None
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                    except Exception:
                        meta = None
                if not needs_eval(meta):
                    continue
                lp = lock_path(ckpt_dir, step)
                if lp.exists():
                    try:
                        if time.time() - lp.stat().st_mtime < 7200:
                            continue
                    except OSError:
                        continue
                info = parse_tag(ckpt_dir.name)
                if info["agent"] not in AGENTS:
                    continue
                out.append((ckpt_dir, step, info, meta or {}))
    return out


def eval_one(
    ckpt_dir: pathlib.Path,
    step: int,
    info: dict,
    meta: dict,
    num_eval_envs: int,
) -> None:
    from car_race import train as car_train
    from swingby import train as swing_train
    from car_parking import train as park_train

    agent_name = info["agent"]
    dataset = dataset_for(info, meta)
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset missing: {dataset}")
    seed = int(meta.get("seed", info["seed"]))
    env_name = str(meta.get("env") or info["env"])
    task = str(meta.get("task") or info["task"])

    log(f"EVAL_CPU {ckpt_dir.name} step={step} agent={agent_name}")

    if info["kind"] == "car_race":
        agent, old_meta = car_train.load_checkpoint(
            checkpoint_dir=ckpt_dir,
            agent_name=agent_name,
            dataset_path=dataset,
            task=task if task != "car_parking" else "navigation",
            steps=step,
        )
        config = dict(old_meta.get("config") or {})
        data = car_train._load_dataset(agent_name, dataset, task, config)
        vgr = (
            car_train._make_value_goal_resolver(data.next_observations)
            if agent_name in ("pbg", "pbf")
            else None
        )
        metrics = car_train.evaluate_suite(
            agent,
            seed=seed + step,
            agent_name=agent_name,
            env_name=env_name,
            task=task,
            num_eval_envs=num_eval_envs,
            value_goal_resolver=vgr,
        )
        old_meta["metrics"] = metrics
        old_meta["eval_status"] = "done"
        old_meta["eval_device"] = "cpu"
        (ckpt_dir / f"step_{step}.json").write_text(
            json.dumps(old_meta, indent=2), encoding="utf-8"
        )
        log(f"DONE {ckpt_dir.name}@{step} {car_train.format_eval_metrics(metrics)}")
        return

    if info["kind"] == "swingby":
        agent, old_meta = swing_train.load_checkpoint(
            checkpoint_dir=ckpt_dir,
            agent_name=agent_name,
            dataset_path=dataset,
            steps=step,
        )
        config = dict(old_meta.get("config") or {})
        data = swing_train._load_dataset(agent_name, dataset, config)
        vgr = (
            swing_train._make_value_goal_resolver(data.next_observations)
            if agent_name in ("pbg", "pbf")
            else None
        )
        metrics = swing_train.evaluate_suite(
            agent,
            seed=seed + step,
            agent_name=agent_name,
            env_name=env_name,
            num_eval_envs=num_eval_envs,
            value_goal_resolver=vgr,
        )
        old_meta["metrics"] = metrics
        old_meta["eval_status"] = "done"
        old_meta["eval_device"] = "cpu"
        (ckpt_dir / f"step_{step}.json").write_text(
            json.dumps(old_meta, indent=2), encoding="utf-8"
        )
        log(f"DONE {ckpt_dir.name}@{step} {swing_train.format_eval_metrics(metrics)}")
        return

    agent, old_meta = park_train.load_checkpoint(
        checkpoint_dir=ckpt_dir,
        agent_name=agent_name,
        dataset_path=dataset,
        steps=step,
    )
    config = dict(old_meta.get("config") or {})
    data = park_train._load_dataset(agent_name, dataset, config)
    vgr = (
        park_train._make_value_goal_resolver(data)
        if agent_name in ("pbg", "pbf")
        else None
    )
    metrics = park_train.evaluate_suite(
        agent,
        seed=seed + step,
        agent_name=agent_name,
        num_eval_envs=min(num_eval_envs, 5),
        value_goal_resolver=vgr,
    )
    old_meta["metrics"] = metrics
    old_meta["eval_status"] = "done"
    old_meta["eval_device"] = "cpu"
    (ckpt_dir / f"step_{step}.json").write_text(
        json.dumps(old_meta, indent=2), encoding="utf-8"
    )
    log(f"DONE {ckpt_dir.name}@{step} {park_train.format_eval_metrics(metrics)}")


def run_job_subprocess(
    ckpt_dir: pathlib.Path, step: int, num_eval_envs: int
) -> tuple[str, int, int]:
    """Spawn one CPU eval process. Returns (tag, step, returncode)."""
    if not claim_lock(ckpt_dir, step):
        return (ckpt_dir.name, step, -2)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--job",
        str(ckpt_dir),
        str(step),
        "--num-eval-envs",
        str(num_eval_envs),
    ]
    rc = -1
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT))
        rc = int(proc.returncode)
    except Exception as exc:
        log(f"FAIL_SPAWN {ckpt_dir.name}@{step}: {exc!r}")
        rc = -1
    finally:
        release_lock(ckpt_dir, step)
        meta_path = ckpt_dir / f"step_{step}.json"
        try:
            if meta_path.exists() and rc != 0:
                meta = json.loads(meta_path.read_text())
                if meta.get("eval_status") == "running":
                    meta["eval_status"] = "pending"
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
    return (ckpt_dir.name, step, rc)


def once(
    stride: int | None,
    num_eval_envs: int,
    limit: int | None,
    workers: int,
) -> int:
    jobs = pending_jobs(stride)
    if limit is not None:
        jobs = jobs[:limit]
    log(f"pending={len(jobs)} workers={workers}")
    if not jobs:
        return 0
    n_ok = 0
    # Dispatch only up to a wave of workers*2 to avoid claiming the whole backlog
    # while pids are tight; watch loop will pick the rest.
    wave = jobs[: max(workers * 2, workers)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(run_job_subprocess, ckpt_dir, step, num_eval_envs): (
                ckpt_dir.name,
                step,
            )
            for ckpt_dir, step, _info, _meta in wave
        }
        for fut in as_completed(futs):
            tag, step, rc = fut.result()
            if rc == 0:
                n_ok += 1
            elif rc == -2:
                log(f"SKIP_LOCKED {tag}@{step}")
            else:
                log(f"FAIL {tag}@{step} rc={rc}")
    return n_ok


def run_single_job(ckpt_dir: pathlib.Path, step: int, num_eval_envs: int) -> int:
    meta_path = ckpt_dir / f"step_{step}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    info = parse_tag(ckpt_dir.name)
    try:
        eval_one(ckpt_dir, step, info, meta, num_eval_envs)
        return 0
    except Exception as exc:
        log(f"FAIL {ckpt_dir.name}@{step}: {exc!r}")
        return 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    p.add_argument("--watch", action="store_true", default=False)
    p.add_argument("--interval-sec", type=int, default=30)
    p.add_argument("--stride", type=int, default=20000)
    p.add_argument("--num-eval-envs", type=int, default=25)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument(
        "--job",
        nargs=2,
        metavar=("CKPT_DIR", "STEP"),
        help="Run a single eval job (used by parallel workers)",
    )
    args = p.parse_args()

    if args.job:
        ckpt_dir = pathlib.Path(args.job[0])
        step = int(args.job[1])
        raise SystemExit(run_single_job(ckpt_dir, step, args.num_eval_envs))

    watch = args.watch or (not args.once and os.environ.get("WATCH") == "1")
    interval = int(os.environ.get("INTERVAL_SEC", args.interval_sec))
    workers = int(os.environ.get("CPU_EVAL_WORKERS", args.workers))
    if watch:
        log(f"WATCH every {interval}s stride={args.stride} workers={workers}")
        while True:
            once(args.stride, args.num_eval_envs, args.limit, workers)
            time.sleep(interval)
    else:
        once(args.stride, args.num_eval_envs, args.limit, workers)


if __name__ == "__main__":
    main()
