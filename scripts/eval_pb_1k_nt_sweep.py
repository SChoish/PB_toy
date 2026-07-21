#!/usr/bin/env python3
"""Parallel NT sweep eval for final PB_toy 1k PBG/PBF checkpoints.

Mirrors PathBridger NT eval semantics:
  - N = subgoal_eval_num_samples (best-of-N value selection)
  - T = subgoal_temperature (noise scale); T=0 uses subgoal mean

Example:
  python scripts/eval_pb_1k_nt_sweep.py --apply --agents pbg,pbf --workers 4 \\
    --ns 1,2,4,8,16,32 --ts 0,0.25,0.5,1.0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TAG_RE = re.compile(
    r"^(?P<body>.+)_(?P<agent>pbg|pbf)_(?:(?P<policy>noisy|random)_)?s(?P<seed>\d+)_k10_ha\d+_1k$"
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _host() -> str:
    return socket.gethostname() or "unknown-host"


def _t_tag(t: float) -> str:
    if float(t) == 0.0:
        return "0"
    return f"{float(t):g}".replace(".", "p")


def parse_tag(tag: str) -> dict:
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"unrecognized pb 1k tag: {tag}")
    body = m.group("body")
    agent = m.group("agent")
    policy = m.group("policy") or "expert"
    seed = int(m.group("seed"))
    if body.startswith("swingby_"):
        if not body.endswith("_swingby"):
            raise ValueError(f"bad swingby tag body: {body}")
        env = body[: -len("_swingby")]
        return {
            "kind": "swingby",
            "agent": agent,
            "env": env,
            "task": "swingby",
            "policy": policy,
            "seed": seed,
            "tag": tag,
        }
    if body.startswith("car_race_"):
        if body.endswith("_navigation"):
            env = body[: -len("_navigation")]
            task = "navigation"
        else:
            idx = body.rfind("_lap_")
            if idx < 0:
                raise ValueError(f"bad car_race tag body: {body}")
            env = body[:idx]
            task = body[idx + 1 :]
        return {
            "kind": "car_race",
            "agent": agent,
            "env": env,
            "task": task,
            "policy": policy,
            "seed": seed,
            "tag": tag,
        }
    raise ValueError(f"unknown domain for tag: {tag}")


def dataset_for(info: dict) -> pathlib.Path:
    policy = info["policy"]
    if info["kind"] == "car_race":
        root = ROOT / "car_race" / "datasets"
        if info["task"] == "navigation":
            return root / f"{info['env']}_{policy}_1k.npz"
        return root / f"{info['env']}_lap_{policy}_1k.npz"
    return ROOT / "swingby" / "datasets" / f"{info['env']}_swingby_{policy}_1k.npz"


def discover_ckpts(agents: set[str]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for kind in ("car_race", "swingby"):
        kind_root = ROOT / "checkpoints" / kind
        if not kind_root.is_dir():
            continue
        for run_dir in sorted(kind_root.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.endswith("_1k"):
                continue
            if not any(f"_{a}_" in run_dir.name for a in agents):
                continue
            try:
                info = parse_tag(run_dir.name)
            except ValueError:
                continue
            if info["agent"] not in agents:
                continue
            pack = run_dir / "step_10000.msgpack"
            meta = run_dir / "step_10000.json"
            if pack.is_file() and meta.is_file():
                out.append(run_dir)
    return out


def override_nt(agent, *, n: int, t: float):
    cfg = dict(agent.config)
    cfg["subgoal_eval_num_samples"] = int(n)
    cfg["subgoal_temperature"] = float(t)
    dyn_cfg = dict(agent.dynamics.config)
    dyn_cfg["subgoal_eval_num_samples"] = int(n)
    dyn_cfg["subgoal_temperature"] = float(t)
    return agent.replace(
        config=cfg,
        dynamics=agent.dynamics.replace(config=dyn_cfg),
    )


def eval_one(job: dict) -> dict:
    """Worker entry: one (ckpt, N, T) eval. Isolated for ProcessPoolExecutor."""
    run_dir = pathlib.Path(job["run_dir"])
    n = int(job["n"])
    t = float(job["t"])
    episodes = int(job["episodes"])
    out_root = pathlib.Path(job["logs_root"])
    apply = bool(job["apply"])

    info = parse_tag(run_dir.name)
    agent_name = info["agent"]
    dataset = dataset_for(info)
    if not dataset.is_file():
        raise FileNotFoundError(f"missing dataset {dataset} for {run_dir.name}")

    dest_dir = (
        out_root
        / "completed"
        / "pb_toy"
        / info["env"]
        / f"{_host()}_pb_toy_{run_dir.name}"
        / "bridge_eval"
        / "ckpt10000"
    )
    dest = dest_dir / f"n{n}_t{_t_tag(t)}_attempt0.json"
    if dest.is_file():
        return {
            "status": "skip",
            "path": str(dest),
            "tag": run_dir.name,
            "agent": agent_name,
            "n": n,
            "t": t,
        }

    sample_t = 0.0 if float(t) == 0.0 else 1.0
    t0 = time.time()

    if info["kind"] == "car_race":
        from car_race.train import (  # noqa: WPS433
            _load_dataset,
            _make_value_goal_resolver,
            evaluate,
            load_checkpoint,
        )

        agent, meta = load_checkpoint(
            checkpoint_dir=run_dir,
            agent_name=agent_name,
            dataset_path=dataset,
            task=info["task"],
            steps=10_000,
        )
        data = _load_dataset(agent_name, dataset, info["task"], dict(agent.config))
        resolver = _make_value_goal_resolver(data.next_observations)
        agent = override_nt(agent, n=n, t=t)
        metrics = evaluate(
            agent,
            env_name=info["env"],
            task=info["task"],
            episodes_per_task=episodes,
            seed=info["seed"],
            temperature=sample_t,
            value_goal_resolver=resolver,
        )
    else:
        from swingby.train import (  # noqa: WPS433
            _load_dataset,
            _make_value_goal_resolver,
            evaluate,
            load_checkpoint,
        )

        agent, meta = load_checkpoint(
            checkpoint_dir=run_dir,
            agent_name=agent_name,
            dataset_path=dataset,
            steps=10_000,
        )
        data = _load_dataset(agent_name, dataset, dict(agent.config))
        resolver = _make_value_goal_resolver(data.next_observations)
        agent = override_nt(agent, n=n, t=t)
        metrics = evaluate(
            agent,
            env_name=info["env"],
            episodes_per_task=episodes,
            seed=info["seed"],
            temperature=sample_t,
            value_goal_resolver=resolver,
        )

    rec = {
        "schema_version": "1.0-pb_toy-nt",
        "created_at": _now(),
        "host": _host(),
        "method": "pb_toy",
        "algorithm": agent_name,
        "phase": "bridge_eval",
        "env": info["env"],
        "task": info["task"],
        "policy": info["policy"],
        "seed": info["seed"],
        "tag": run_dir.name,
        "dataset_path": str(dataset),
        "checkpoint": 10000,
        "subgoal_eval_num_samples": int(n),
        "subgoal_temperature": float(t),
        "sample_actions_temperature": float(sample_t),
        "episodes_per_task": int(episodes),
        "elapsed_sec": float(time.time() - t0),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "primary_success": float(metrics.get("mean_success", 0.0)),
        "train_meta_steps": int(meta.get("steps", 10000)),
    }
    if apply:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return {
        "status": "wrote" if apply else "dry",
        "path": str(dest),
        "tag": run_dir.name,
        "agent": agent_name,
        "n": n,
        "t": t,
        "success": rec["primary_success"],
        "elapsed_sec": rec["elapsed_sec"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-root", type=pathlib.Path, default=pathlib.Path("/home/svcho/PB_logs"))
    ap.add_argument("--agents", default="pbg,pbf")
    ap.add_argument("--ns", default="1,2,4,8,16,32")
    ap.add_argument("--ts", default="0,0.25,0.5,1.0")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--tags", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    agents = {a.strip() for a in args.agents.split(",") if a.strip()}
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    ts = [float(x) for x in args.ts.split(",") if x.strip()]
    tag_filter = {t.strip() for t in args.tags.split(",") if t.strip()}

    ckpts = discover_ckpts(agents)
    if tag_filter:
        ckpts = [c for c in ckpts if c.name in tag_filter]
    jobs = [
        {
            "run_dir": str(c),
            "n": n,
            "t": t,
            "episodes": args.episodes,
            "logs_root": str(args.logs_root),
            "apply": bool(args.apply),
        }
        for c in ckpts
        for n in ns
        for t in ts
    ]
    if args.limit > 0:
        jobs = jobs[: args.limit]

    workers = max(1, int(args.workers))
    print(
        f"[nt-sweep] host={_host()} agents={sorted(agents)} ckpts={len(ckpts)} "
        f"jobs={len(jobs)} workers={workers} ns={ns} ts={ts} apply={args.apply}",
        flush=True,
    )

    done = skip = fail = 0
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            try:
                res = eval_one(job)
                status = res["status"]
                if status == "skip":
                    skip += 1
                else:
                    done += 1
                print(
                    f"[{i}/{len(jobs)}] {status} {res['agent']} {res['tag']} "
                    f"N={res['n']} T={res['t']} success={res.get('success', float('nan'))} "
                    f"elapsed={res.get('elapsed_sec', 0):.1f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(
                    f"[{i}/{len(jobs)}] FAIL {pathlib.Path(job['run_dir']).name} "
                    f"N={job['n']} T={job['t']}: {exc}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(eval_one, job): job for job in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                job = futs[fut]
                try:
                    res = fut.result()
                    status = res["status"]
                    if status == "skip":
                        skip += 1
                    else:
                        done += 1
                    print(
                        f"[{i}/{len(jobs)}] {status} {res['agent']} {res['tag']} "
                        f"N={res['n']} T={res['t']} success={res.get('success', float('nan'))} "
                        f"elapsed={res.get('elapsed_sec', 0):.1f}s",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    print(
                        f"[{i}/{len(jobs)}] FAIL {pathlib.Path(job['run_dir']).name} "
                        f"N={job['n']} T={job['t']}: {exc}",
                        flush=True,
                    )

    print(f"[nt-sweep] finished done={done} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    # Needed for CUDA/JAX fork safety on some hosts.
    try:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
