"""Re-evaluate existing checkpoints with 25 episodes/task. No train/render."""

from __future__ import annotations

import json
import pathlib
import re
import time

from hazard_env.agents.train import (
    evaluate_suite,
    format_eval_metrics,
    load_checkpoint,
)
from hazard_env.env import GRAVITY_STRENGTHS
from hazard_env.generate_navigate import POLICIES, dataset_stem
from hazard_env.plot_learning_curves import (
    DEFAULT_TAGS,
    plot_set_learning_curves,
)


ROOT = pathlib.Path(__file__).resolve().parent
STEP_RE = re.compile(r"^step_(\d+)\.msgpack$")


def tag_to_agent(tag: str) -> str:
    for name in ("tr_hiql", "hiql", "pbg", "pbf", "trl", "dqc", "bc"):
        if tag == name or tag.startswith(f"{name}_"):
            return name
    raise ValueError(f"Unknown agent tag: {tag}")


def already_reevaled(meta_path: pathlib.Path) -> bool:
    if not meta_path.exists():
        return False
    metrics = json.loads(meta_path.read_text(encoding="utf-8")).get("metrics") or {}
    return int(metrics.get("episodes_per_task", 0)) == 25


def set_is_reeval_complete(
    env_name: str,
    policy: str,
    *,
    size: str = "1k",
    tags: tuple[str, ...] = DEFAULT_TAGS,
) -> bool:
    """True when every step_*.json for all algos in the set has 25-eps metrics."""
    ckpt_root = ROOT / "checkpoints" / env_name / f"{policy}_{size}"
    for tag in tags:
        metas = sorted((ckpt_root / tag).glob("step_*.json"))
        if not metas:
            return False
        if not all(already_reevaled(path) for path in metas):
            return False
    return True


def maybe_plot_set(env_name: str, policy: str, plotted: set[tuple[str, str]]) -> None:
    key = (env_name, policy)
    if key in plotted:
        return
    if not set_is_reeval_complete(env_name, policy):
        return
    path = plot_set_learning_curves(env_name, policy)
    plotted.add(key)
    print(
        f"######## LEARNING-CURVE {env_name}/{policy}/1k {path} ########",
        flush=True,
    )


def iter_checkpoints(root: pathlib.Path = ROOT):
    for pack in sorted((root / "checkpoints").glob("*/*_1k/*/step_*.msgpack")):
        match = STEP_RE.fullmatch(pack.name)
        if match is None:
            continue
        step = int(match.group(1))
        tag_dir = pack.parent
        policy_size = tag_dir.parent.name  # navigate_1k
        env_name = tag_dir.parent.parent.name
        policy, size = policy_size.rsplit("_", 1)
        tag = tag_dir.name
        yield {
            "pack": pack,
            "meta": tag_dir / f"step_{step}.json",
            "env_name": env_name,
            "policy": policy,
            "size": size,
            "tag": tag,
            "agent_name": tag_to_agent(tag),
            "step": step,
            "dataset": root
            / "datasets"
            / f"{dataset_stem(env_name, policy, size)}.npz",
        }


def update_metrics(meta_path: pathlib.Path, metrics: dict) -> None:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["metrics"] = metrics
    meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    jobs = list(iter_checkpoints())
    todo = [j for j in jobs if not already_reevaled(j["meta"])]
    plotted: set[tuple[str, str]] = set()
    print(
        f"=== REEVAL 25 eps/task: {len(todo)}/{len(jobs)} remaining "
        f"(skip done + render/train; curve per set) ===",
        flush=True,
    )

    # Plot any sets already fully re-evaled before this run.
    for env_name in GRAVITY_STRENGTHS:
        for policy in POLICIES:
            maybe_plot_set(env_name, policy, plotted)

    for idx, job in enumerate(todo, 1):
        label = (
            f"{job['env_name']}/{job['policy']}/{job['size']}/"
            f"{job['tag']}@{job['step']}"
        )
        started = time.time()
        print(f"\n######## [{idx}/{len(todo)}] EVAL {label} ########", flush=True)
        agent, _ = load_checkpoint(
            checkpoint_dir=job["pack"].parent,
            agent_name=job["agent_name"],
            dataset_path=job["dataset"],
            steps=job["step"],
        )
        metrics = evaluate_suite(
            agent,
            seed=job["step"],
            agent_name=job["agent_name"],
            num_eval_envs=25,
            env_name=job["env_name"],
        )
        update_metrics(job["meta"], metrics)
        print(
            f"######## [{idx}/{len(todo)}] EVAL-DONE {label} "
            f"{format_eval_metrics(metrics)} "
            f"seconds={time.time() - started:.1f} ########",
            flush=True,
        )
        maybe_plot_set(job["env_name"], job["policy"], plotted)

    print("=== REEVAL_25_DONE ===", flush=True)


if __name__ == "__main__":
    main()
