"""Resume the 1k hazard sweep, including official TRL and DQC toy ports."""

from __future__ import annotations

import json
import pathlib
import time

from hazard_env.agents.train import (
    _eval_temperature,
    format_eval_metrics,
    load_checkpoint,
    render_agent,
    train,
)
from hazard_env.env import GRAVITY_STRENGTHS
from hazard_env.generate_navigate import POLICIES, dataset_stem
from hazard_env.plot_learning_curves import (
    plot_set_learning_curves,
    set_is_complete,
)


ROOT = pathlib.Path(__file__).resolve().parent
AGENT_SPECS = (
    ("hiql", "gap0", {"low_alpha": 0.0, "high_alpha": 0.0}),
    ("hiql", "gap5", {"low_alpha": 5.0, "high_alpha": 5.0}),
    ("tr_hiql", "gap0", {"low_alpha": 0.0, "high_alpha": 0.0}),
    ("tr_hiql", "gap5", {"low_alpha": 5.0, "high_alpha": 5.0}),
    (
        "pbg",
        "gap0",
        {
            "subgoal_value_gap_scale": 0.0,
            "subgoal_num_candidates": 1,
            "subgoal_include_mean": True,
        },
    ),
    (
        "pbg",
        "gap5",
        {
            "subgoal_value_gap_scale": 5.0,
            "subgoal_num_candidates": 1,
            "subgoal_include_mean": True,
        },
    ),
    (
        "pbf",
        "gap0",
        {
            "subgoal_value_gap_scale": 0.0,
            "subgoal_num_candidates": 8,
            "subgoal_include_mean": True,
        },
    ),
    (
        "pbf",
        "gap5",
        {
            "subgoal_value_gap_scale": 5.0,
            "subgoal_num_candidates": 8,
            "subgoal_include_mean": True,
        },
    ),
    ("trl", "", {}),
    ("dqc", "", {}),
)


def checkpoint_is_compatible(agent_name: str, checkpoint_dir: pathlib.Path) -> bool:
    pack = checkpoint_dir / "step_50000.msgpack"
    meta = checkpoint_dir / "step_50000.json"
    if not pack.exists() or not meta.exists():
        return False
    if agent_name != "hiql":
        return True
    config = json.loads(meta.read_text(encoding="utf-8")).get("config", {})
    return config.get("target_goal_encoder") == "independent"


def main() -> None:
    jobs = []
    for env_name in GRAVITY_STRENGTHS:
        for policy in POLICIES:
            dataset = (
                ROOT
                / "datasets"
                / f"{dataset_stem(env_name, policy, '1k')}.npz"
            )
            for agent_name, suffix, overrides in AGENT_SPECS:
                tag = f"{agent_name}_{suffix}" if suffix else agent_name
                jobs.append(
                    (env_name, policy, dataset, tag, agent_name, overrides)
                )

    print(
        f"\n=== 1k sweep: OGBench HIQL + TRL + DQC ({len(jobs)} jobs) ===",
        flush=True,
    )
    for idx, job in enumerate(jobs, 1):
        env_name, policy, dataset, tag, agent_name, overrides = job
        checkpoint_dir = (
            ROOT / "checkpoints" / env_name / f"{policy}_1k" / tag
        )
        render_dir = ROOT / "renders" / env_name / f"{policy}_1k" / tag
        label = f"{env_name}/{policy}/1k/{tag}"
        started = time.time()

        videos_ready = all(
            (render_dir / f"task{tid}.mp4").exists() for tid in (1, 2, 3, 4, 5)
        )
        if checkpoint_is_compatible(agent_name, checkpoint_dir) and videos_ready:
            print(
                f"\n######## [{idx}/{len(jobs)}] SKIP {label} "
                f"(ckpt+renders ready) ########",
                flush=True,
            )
        elif checkpoint_is_compatible(agent_name, checkpoint_dir):
            print(
                f"\n######## [{idx}/{len(jobs)}] RENDER-ONLY {label} ########",
                flush=True,
            )
            agent, _ = load_checkpoint(
                checkpoint_dir=checkpoint_dir,
                agent_name=agent_name,
                dataset_path=dataset,
                steps=50_000,
            )
            paths = render_agent(
                agent,
                output_dir=render_dir,
                task_ids=[1, 2, 3, 4, 5],
                seed=50_000,
                temperature=_eval_temperature(agent_name),
                env_name=env_name,
                diagnostics=True,
            )
            print(
                f"######## [{idx}/{len(jobs)}] RENDER-DONE {label} "
                f"videos={len(paths)} seconds={time.time() - started:.1f} ########",
                flush=True,
            )
        else:
            print(
                f"\n######## [{idx}/{len(jobs)}] START {label} ########",
                flush=True,
            )
            agent, metrics = train(
                agent_name=agent_name,
                dataset_path=dataset,
                steps=50_000,
                seed=0,
                eval_every=10_000,
                log_every=2_000,
                config_overrides=overrides,
                checkpoint_dir=checkpoint_dir,
                num_eval_envs=25,
                env_name=env_name,
            )
            print(
                f"######## [{idx}/{len(jobs)}] DONE {label} "
                f"{format_eval_metrics(metrics)} ########",
                flush=True,
            )
            paths = render_agent(
                agent,
                output_dir=render_dir,
                task_ids=[1, 2, 3, 4, 5],
                seed=50_000,
                temperature=_eval_temperature(agent_name),
                env_name=env_name,
                diagnostics=True,
            )
            print(
                f"######## [{idx}/{len(jobs)}] RENDER-DONE {label} "
                f"videos={len(paths)} seconds={time.time() - started:.1f} ########",
                flush=True,
            )

        # One learning-curve figure per env × dataset set (after last algo).
        if tag == "dqc" and set_is_complete(env_name, policy):
            curve_path = plot_set_learning_curves(env_name, policy)
            print(
                f"######## LEARNING-CURVE {env_name}/{policy}/1k "
                f"{curve_path} ########",
                flush=True,
            )

    print("=== FULL_SWEEP_1K_90_DONE ===", flush=True)


if __name__ == "__main__":
    main()
