"""Command-line validation for the CarParking expert."""

from __future__ import annotations

import argparse
from collections import Counter
import math

import numpy as np

from .env import NUM_FIXED_TASKS, CarParkingEnv
from .parking_policy import rollout_expert


def validate(
    *,
    episodes_per_task: int = 1,
    seed: int = 0,
    jitter_position: float = 0.0,
    jitter_heading_deg: float = 0.0,
) -> dict[int, Counter[str]]:
    rng = np.random.default_rng(seed)
    results: dict[int, Counter[str]] = {}
    for task_id in range(1, NUM_FIXED_TASKS + 1):
        counts: Counter[str] = Counter()
        for episode in range(episodes_per_task):
            env = CarParkingEnv()
            options: dict[str, object] = {}
            if jitter_position > 0.0 or jitter_heading_deg > 0.0:
                env.reset(
                    seed=seed + episode,
                    options={"task_id": task_id},
                )
                options["position"] = (
                    np.asarray(env.layout.start)
                    + rng.uniform(
                        -jitter_position, jitter_position, size=2
                    )
                )
                options["heading"] = env.layout.start_heading + math.radians(
                    float(
                        rng.uniform(
                            -jitter_heading_deg, jitter_heading_deg
                        )
                    )
                )
            try:
                result = rollout_expert(
                    env,
                    task_id=task_id,
                    seed=seed + episode,
                    reset_options=options,
                )
            except (RuntimeError, ValueError):
                counts["planning_failure"] += 1
                env.close()
                continue
            counts["success" if result.success else "failure"] += 1
            counts["collision"] += int(result.collision)
            counts["timeout"] += int(result.timeout)
            counts["steps"] += result.steps
            env.close()
        results[task_id] = counts
        total = episodes_per_task
        print(
            f"task {task_id}: success={counts['success']}/{total} "
            f"({counts['success'] / total:.1%}), "
            f"collision={counts['collision'] / total:.1%}, "
            f"timeout={counts['timeout'] / total:.1%}, "
            f"planning_failure={counts['planning_failure'] / total:.1%}"
        )

    aggregate = sum(results.values(), Counter())
    total = episodes_per_task * NUM_FIXED_TASKS
    print(
        f"overall: success={aggregate['success']}/{total} "
        f"({aggregate['success'] / total:.1%}), "
        f"collision={aggregate['collision'] / total:.1%}, "
        f"timeout={aggregate['timeout'] / total:.1%}, "
        f"mean_steps={aggregate['steps'] / max(aggregate['success'] + aggregate['failure'], 1):.1f}"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter-position", type=float, default=0.0)
    parser.add_argument("--jitter-heading-deg", type=float, default=0.0)
    args = parser.parse_args()
    if args.episodes_per_task < 1:
        parser.error("--episodes-per-task must be positive")
    validate(
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        jitter_position=args.jitter_position,
        jitter_heading_deg=args.jitter_heading_deg,
    )


if __name__ == "__main__":
    main()

