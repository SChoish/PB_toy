"""Train / evaluate simplified hazard_env agents on the navigate dataset."""

from __future__ import annotations

import argparse
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from hazard_env.agents import AGENTS, DEFAULT_CONFIGS
from hazard_env.agents.dataset import denormalize_actions, load_navigate_dataset
from hazard_env.env import ContinuousHazard2DEnv, Hazard2DConfig


def _to_jnp(batch: dict) -> dict:
    return {k: jnp.asarray(v) for k, v in batch.items()}


def evaluate(
    agent,
    *,
    task_ids: list[int],
    episodes_per_task: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    env = ContinuousHazard2DEnv(
        config=Hazard2DConfig(max_episode_steps=300),
        observation_mode="state",
        terminate_at_goal=True,
    )
    rng = np.random.default_rng(seed)
    results: dict[str, float] = {}
    for task_id in task_ids:
        successes = 0
        deaths = 0
        for ep in range(episodes_per_task):
            ob, info = env.reset(seed=int(rng.integers(0, 1_000_000)), options={"task_id": task_id})
            goal = np.concatenate(
                [info["goal"], np.zeros(2, dtype=np.float32)]
            ).astype(np.float32)
            done = False
            while not done:
                obs_j = jnp.asarray(ob)[None]
                goal_j = jnp.asarray(goal)[None]
                key = jax.random.PRNGKey(int(rng.integers(0, 1_000_000)))
                action = np.asarray(agent.sample_actions(obs_j, goal_j, seed=key))[0]
                action = denormalize_actions(action)
                ob, _r, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
            successes += int(info.get("is_success", False))
            deaths += int(info.get("dead", False))
        results[f"task{task_id}_success"] = successes / episodes_per_task
        results[f"task{task_id}_death"] = deaths / episodes_per_task
    env.close()
    results["mean_success"] = float(
        np.mean([results[f"task{t}_success"] for t in task_ids])
    )
    return results


def train(
    *,
    agent_name: str,
    dataset_path: pathlib.Path,
    steps: int,
    seed: int,
    eval_every: int,
    log_every: int,
) -> None:
    if agent_name not in AGENTS:
        raise SystemExit(f"Unknown agent {agent_name}; choose from {list(AGENTS)}")

    config = DEFAULT_CONFIGS[agent_name]()
    data = load_navigate_dataset(
        dataset_path,
        subgoal_steps=int(config.get("subgoal_steps", 8)),
        seed=seed,
    )
    print(f"Loaded {len(data)} transitions from {dataset_path}")

    ex_obs = data.observations[:8]
    ex_act = data.actions[:8]
    agent = AGENTS[agent_name].create(seed, ex_obs, ex_act, config)

    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = _to_jnp(data.sample(rng, config["batch_size"]))
        agent, info = agent.update(batch)
        if step % log_every == 0 or step == 1:
            pretty = {
                k: float(v) for k, v in info.items() if np.ndim(np.asarray(v)) == 0
            }
            print(f"[{agent_name}] step={step} {pretty}")
        if eval_every > 0 and step % eval_every == 0:
            metrics = evaluate(agent, task_ids=[1, 2, 3, 4, 5], seed=seed + step)
            print(f"[{agent_name}] eval@{step} {metrics}")

    metrics = evaluate(agent, task_ids=[1, 2, 3, 4, 5], seed=seed + steps)
    print(f"[{agent_name}] final eval {metrics}  ({time.time() - t0:.1f}s)")


def parse_args() -> argparse.Namespace:
    default_data = (
        pathlib.Path(__file__).resolve().parents[1]
        / "datasets"
        / "hazard2d-navigate.npz"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", choices=sorted(AGENTS), required=True)
    p.add_argument("--dataset", type=pathlib.Path, default=default_data)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        agent_name=args.agent,
        dataset_path=args.dataset,
        steps=args.steps,
        seed=args.seed,
        eval_every=args.eval_every,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
