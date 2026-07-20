#!/usr/bin/env python3
"""CPU eval of existing SwingBy checkpoints under the swingby (eval_fixed) protocol.

Loads an existing checkpoint (legacy / taskmix_v2), then runs ``evaluate_suite``
which uses ``OrbitalSwingByEnv(task_profile='eval_fixed')`` by default.

Schema check is relaxed for loading legacy checkpoints; action encoding must
still match the template dataset.

Usage:
  # one job
  python scripts/eval_swingby_cpu.py --ckpt-dir checkpoints/swingby/swingby_planet_tmv2_pbf_s0

  # discover + print
  python scripts/eval_swingby_cpu.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import flax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from swingby.datasets import (  # noqa: E402
    LEGACY_ACTION_ENCODING,
    read_dataset_metadata,
)
from swingby.train import (  # noqa: E402
    AGENTS,
    DEFAULT_CONFIGS,
    _load_dataset,
    _make_value_goal_resolver,
    evaluate_suite,
    format_eval_metrics,
)

CKPT_ROOT = ROOT / "checkpoints" / "swingby"
DS_ROOT = ROOT / "swingby" / "datasets"
AGENTS_ORDER = ("tr_hiql", "pbg", "pbf", "trl", "dqc", "hiql", "gcbc", "gciql")


def _parse_tag(tag: str) -> tuple[str, str, str]:
    """Return (env, agent, policy) from checkpoint dir name."""
    m = re.match(r"^(?P<body>.+)_s(?P<seed>\d+)$", tag)
    body = m.group("body") if m else tag
    policy = "expert"
    for pol in ("noisy", "random", "expert"):
        suf = f"_{pol}"
        if body.endswith(suf):
            body = body[: -len(suf)]
            policy = pol
            break
    agent = "?"
    for a in sorted(AGENTS_ORDER, key=len, reverse=True):
        suf = f"_{a}"
        if body.endswith(suf):
            agent = a
            body = body[: -len(suf)]
            break
    # strip the canonical swingby or legacy tmv2 marker from the env body
    env = body
    for mark in ("_swingby", "_tmv2"):
        if env.endswith(mark):
            env = env[: -len(mark)]
            break
    return env, agent, policy


def _dataset_for_ckpt(env: str, policy: str, schema: str | None) -> pathlib.Path:
    if schema == "taskmix_v2":
        return DS_ROOT / f"{env}_taskmix_{policy}_100k.npz"
    if schema == "swingby":
        return DS_ROOT / f"{env}_swingby_{policy}_100k.npz"
    # ballistic / legacy
    return DS_ROOT / f"{env}_{policy}_100k.npz"


def _load_agent_relaxed(
    *,
    checkpoint_dir: pathlib.Path,
    agent_name: str,
    dataset_path: pathlib.Path,
    steps: int,
):
    meta_path = checkpoint_dir / f"step_{steps}.json"
    pack_path = checkpoint_dir / f"step_{steps}.msgpack"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    config = dict(metadata["config"])
    ds_meta = read_dataset_metadata(dataset_path)
    ckpt_enc = str(config.get("action_encoding") or LEGACY_ACTION_ENCODING)
    if ckpt_enc != ds_meta["action_encoding"]:
        raise ValueError(
            f"action encoding mismatch ckpt={ckpt_enc} dataset={ds_meta['action_encoding']}"
        )
    if isinstance(config.get("hidden_dims"), list):
        config["hidden_dims"] = tuple(config["hidden_dims"])
    if agent_name in ("pbg", "pbf"):
        fresh = DEFAULT_CONFIGS[agent_name]()
        config["subgoal_eval_num_samples"] = int(fresh["subgoal_eval_num_samples"])
        config["phi_goal_obs_indices"] = (0, 1, 2, 3)
        config["subgoal_value_goal_representation"] = "full"
        config["env_name"] = str(config.get("env_name") or "swingby")
    data = _load_dataset(agent_name, dataset_path, config)
    if agent_name in ("trl", "dqc"):
        example = {k: jnp.asarray(v) for k, v in data.sample(np.random.default_rng(0), 8).items()}
        template = AGENTS[agent_name].create(0, example, config)
    else:
        template = AGENTS[agent_name].create(
            0, data.observations[:8], data.actions[:8], config
        )
    agent = flax.serialization.from_bytes(template, pack_path.read_bytes())
    return agent, metadata, data


def discover(steps: int = 50_000) -> list[pathlib.Path]:
    out = []
    if not CKPT_ROOT.is_dir():
        return out
    for d in sorted(CKPT_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if (d / f"step_{steps}.msgpack").is_file() and (d / f"step_{steps}.json").is_file():
            out.append(d)
    return out


def eval_one(
    ckpt_dir: pathlib.Path,
    *,
    steps: int,
    num_eval_envs: int,
    force: bool,
) -> dict:
    tag = ckpt_dir.name
    env, agent_name, policy = _parse_tag(tag)
    if agent_name not in AGENTS:
        raise ValueError(f"cannot parse agent from tag={tag}")
    out_dir = ckpt_dir / "eval_swingby"
    out_path = out_dir / f"step_{steps}.json"
    if out_path.is_file() and not force:
        return {"tag": tag, "status": "skip_exists", "path": str(out_path)}

    meta = json.loads((ckpt_dir / f"step_{steps}.json").read_text(encoding="utf-8"))
    schema = meta["config"].get("task_schema")
    ds = _dataset_for_ckpt(env, policy, schema)
    if not ds.is_file():
        # fallback expert
        ds2 = _dataset_for_ckpt(env, "expert", schema)
        if ds2.is_file():
            ds = ds2
        else:
            raise FileNotFoundError(f"dataset missing for {tag}: tried {ds}")

    # Prefer canonical swingby offline states for pbg/pbf value-goal resolver under fixed eval.
    resolver_ds = DS_ROOT / f"{env}_swingby_expert_100k.npz"
    if not resolver_ds.is_file():
        resolver_ds = ds

    agent, metadata, data = _load_agent_relaxed(
        checkpoint_dir=ckpt_dir,
        agent_name=agent_name,
        dataset_path=ds,
        steps=steps,
    )
    value_goal_resolver = (
        _make_value_goal_resolver(
            _load_dataset(agent_name, resolver_ds, dict(agent.config)).next_observations
        )
        if agent_name in ("pbg", "pbf")
        else None
    )
    seed = 0
    m = re.search(r"_s(\d+)$", tag)
    if m:
        seed = int(m.group(1))

    metrics = evaluate_suite(
        agent,
        seed=seed + steps,
        env_name=env,
        agent_name=agent_name,
        num_eval_envs=num_eval_envs,
        value_goal_resolver=value_goal_resolver,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "swingby",
        "task_profile": "eval_fixed",
        "source_tag": tag,
        "env": env,
        "agent": agent_name,
        "policy": policy,
        "seed": seed,
        "steps": steps,
        "ckpt_task_schema": schema,
        "template_dataset": str(ds),
        "resolver_dataset": str(resolver_ds),
        "metrics": metrics,
        "format": format_eval_metrics(metrics),
    }
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[{agent_name}] eval_swingby@{steps} {format_eval_metrics(metrics)} -> {out_path}", flush=True)
    return {"tag": tag, "status": "ok", "path": str(out_path), "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=pathlib.Path, default=None)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--num-eval-envs", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only-tmv2", action="store_true", help="only *tmv2* checkpoint dirs")
    args = ap.parse_args()

    if args.list or args.ckpt_dir is None:
        dirs = discover(args.steps)
        if args.only_tmv2:
            dirs = [d for d in dirs if "_tmv2_" in d.name or d.name.endswith("_tmv2")]
        for d in dirs:
            env, agent, policy = _parse_tag(d.name)
            print(f"{d.name}\tenv={env}\tagent={agent}\tpolicy={policy}")
        if args.list or args.ckpt_dir is None and not dirs:
            return
        if args.ckpt_dir is None:
            # list-only when no ckpt-dir
            if args.list:
                return

    if args.ckpt_dir is not None:
        res = eval_one(
            args.ckpt_dir,
            steps=args.steps,
            num_eval_envs=args.num_eval_envs,
            force=args.force,
        )
        print(json.dumps({k: res[k] for k in res if k != "metrics"}, sort_keys=True))


if __name__ == "__main__":
    main()
