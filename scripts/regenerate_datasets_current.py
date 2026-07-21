#!/usr/bin/env python3
"""Delete legacy NPZ files and regenerate all current PB_toy datasets.

Generates:
  - CarRace: 4 envs x 2 tasks x 3 policies x 3 sizes
  - SwingBy: 2 envs x 3 policies x 3 sizes

Each generator writes independent train and validation NPZ files. CarParking is
not included because the current repository does not yet implement its dataset
generator.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "nohup_logs" / "dataset_regen_current"
LOG_DIR = RUNTIME / "jobs"
STATUS_PATH = RUNTIME / "status.json"
SUMMARY_PATH = RUNTIME / "summary.json"
MASTER_LOG = RUNTIME / "master.log"
KST = ZoneInfo("Asia/Seoul")

CAR_ENVS = (
    "car_race_plain",
    "car_race_grav",
    "car_race_anti_grav",
    "car_race_ice",
)
SWING_ENVS = ("swingby_planet", "swingby_blackhole")
POLICIES = ("expert", "noisy", "random")
SIZES = ("1k", "10k", "100k")
SIZE_STEPS = {"1k": 1_000, "10k": 10_000, "100k": 100_000}


@dataclass(frozen=True)
class Job:
    family: str
    env: str
    policy: str
    size: str
    task: str = ""

    @property
    def label(self) -> str:
        parts = (self.family, self.env, self.task, self.policy, self.size)
        return "__".join(part for part in parts if part)

    @property
    def command(self) -> list[str]:
        if self.family == "car_race":
            return [
                "-m",
                "car_race.generate_dataset",
                "--env",
                self.env,
                "--task",
                self.task,
                "--policy",
                self.policy,
                "--size",
                self.size,
            ]
        return [
            "-m",
            "swingby.generate_dataset",
            "--env",
            self.env,
            "--policy",
            self.policy,
            "--size",
            self.size,
            "--dataset-mode",
            "swingby",
        ]

    @property
    def output_path(self) -> Path:
        if self.family == "car_race":
            infix = "_lap" if self.task == "lap" else ""
            name = f"{self.env}{infix}_{self.policy}_{self.size}.npz"
            return ROOT / "car_race" / "datasets" / name
        name = f"{self.env}_swingby_{self.policy}_{self.size}.npz"
        return ROOT / "swingby" / "datasets" / name


def now() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z")


def append_master(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with MASTER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def all_jobs() -> list[Job]:
    jobs = [
        Job("car_race", env, policy, size, task)
        for env in CAR_ENVS
        for task in ("navigation", "lap")
        for policy in POLICIES
        for size in SIZES
    ]
    jobs.extend(
        Job("swingby", env, policy, size)
        for env in SWING_ENVS
        for policy in POLICIES
        for size in SIZES
    )
    return jobs


def clean_legacy_files() -> tuple[int, int]:
    removed = 0
    removed_bytes = 0
    for directory in (
        ROOT / "car_race" / "datasets",
        ROOT / "swingby" / "datasets",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.npz"):
            removed_bytes += path.stat().st_size
            path.unlink()
            removed += 1
    return removed, removed_bytes


def write_status(
    *,
    started_at: str,
    total: int,
    states: dict[str, dict],
    lock: threading.Lock,
) -> None:
    with lock:
        counts: dict[str, int] = {}
        for state in states.values():
            key = state["state"]
            counts[key] = counts.get(key, 0) + 1
        payload = {
            "updated_at": now(),
            "started_at": started_at,
            "total": total,
            "counts": counts,
            "jobs": states,
        }
        temp = STATUS_PATH.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(STATUS_PATH)


def run_job(
    job: Job,
    *,
    python: str,
    started_at: str,
    total: int,
    states: dict[str, dict],
    lock: threading.Lock,
) -> tuple[Job, int, float]:
    log_path = LOG_DIR / f"{job.label}.log"
    with lock:
        states[job.label] = {
            **asdict(job),
            "state": "running",
            "started_at": now(),
            "output": str(job.output_path),
            "log": str(log_path),
        }
    write_status(
        started_at=started_at, total=total, states=states, lock=lock
    )

    begin = time.monotonic()
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[{now()}] command: {python} {' '.join(job.command)}\n")
        log.flush()
        result = subprocess.run(
            [python, "-u", *job.command],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - begin
    state = "done" if result.returncode == 0 else "failed"
    with lock:
        states[job.label].update(
            {
                "state": state,
                "finished_at": now(),
                "elapsed_seconds": round(elapsed, 3),
                "returncode": result.returncode,
            }
        )
    write_status(
        started_at=started_at, total=total, states=states, lock=lock
    )
    append_master(
        f"{state.upper()} {job.label} rc={result.returncode} "
        f"elapsed={elapsed / 60:.1f}m"
    )
    return job, result.returncode, elapsed


def validate_npz(path: Path, minimum_steps: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    val_path = path.with_name(path.stem + "_val.npz")
    if not val_path.is_file():
        raise FileNotFoundError(val_path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "observations",
            "actions",
            "next_observations",
            "terminals",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")
        steps = len(data["actions"])
        if steps < minimum_steps:
            raise ValueError(
                f"{path}: {steps} transitions < minimum {minimum_steps}"
            )
        if not (
            len(data["observations"])
            == len(data["next_observations"])
            == len(data["terminals"])
            == steps
        ):
            raise ValueError(f"{path}: transition arrays have unequal lengths")
        terminal_count = int(np.asarray(data["terminals"]).sum())
    with np.load(val_path, allow_pickle=False) as val:
        val_steps = len(val["actions"])
        if val_steps < minimum_steps // 10:
            raise ValueError(
                f"{val_path}: {val_steps} transitions < "
                f"minimum {minimum_steps // 10}"
            )
    return {
        "path": str(path),
        "steps": steps,
        "val_steps": val_steps,
        "terminal_count": terminal_count,
        "bytes": path.stat().st_size + val_path.stat().st_size,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--python",
        default="/home/ext_csv/miniconda3/envs/pb_toy/bin/python",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing NPZ files before regeneration.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_LOG.write_text("", encoding="utf-8")
    started_at = now()

    if not args.no_clean:
        removed, removed_bytes = clean_legacy_files()
        append_master(
            f"CLEAN removed={removed} npz "
            f"bytes={removed_bytes} ({removed_bytes / 2**20:.1f} MiB)"
        )

    jobs = all_jobs()
    states = {
        job.label: {
            **asdict(job),
            "state": "queued",
            "output": str(job.output_path),
        }
        for job in jobs
    }
    lock = threading.Lock()
    write_status(
        started_at=started_at, total=len(jobs), states=states, lock=lock
    )
    append_master(
        f"START total={len(jobs)} workers={args.workers} python={args.python}"
    )

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_job,
                job,
                python=args.python,
                started_at=started_at,
                total=len(jobs),
                states=states,
                lock=lock,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            job, rc, _ = future.result()
            if rc != 0:
                failed.append(job.label)

    validations: list[dict] = []
    validation_errors: list[str] = []
    if not failed:
        append_master("VALIDATE all generated NPZ files")
        for job in jobs:
            try:
                validations.append(
                    validate_npz(job.output_path, SIZE_STEPS[job.size])
                )
            except Exception as exc:  # report all bad outputs together
                validation_errors.append(f"{job.label}: {exc}")

    payload = {
        "started_at": started_at,
        "finished_at": now(),
        "jobs_total": len(jobs),
        "jobs_failed": failed,
        "validation_errors": validation_errors,
        "train_files": len(validations),
        "npz_files_expected": len(jobs) * 2,
        "total_transitions": sum(v["steps"] for v in validations),
        "total_val_transitions": sum(v["val_steps"] for v in validations),
        "total_bytes": sum(v["bytes"] for v in validations),
        "validations": validations,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if failed or validation_errors:
        append_master(
            f"FAILED generation={len(failed)} "
            f"validation={len(validation_errors)}"
        )
        return 1
    append_master(
        f"COMPLETE jobs={len(jobs)} npz={len(jobs) * 2} "
        f"train_transitions={payload['total_transitions']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
