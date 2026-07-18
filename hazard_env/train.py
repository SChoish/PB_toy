"""Train / evaluate simplified hazard_env agents on the navigate dataset."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections.abc import Callable

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from agents import AGENTS, DEFAULT_CONFIGS
from hazard_env.env import (
    GRAVITY_STRENGTHS,
    ContinuousHazard2DEnv,
    Hazard2DConfig,
)
from hazard_env.utils.datasets import (
    denormalize_actions,
    load_dqc_navigate_dataset,
    load_hgc_navigate_dataset,
    load_navigate_dataset,
    load_trl_navigate_dataset,
)
from hazard_env.utils.rendering import (
    collect_agent_diagnostics,
    compose_diagnostic_frame,
)

ENVS = tuple(GRAVITY_STRENGTHS)


def _make_eval_env(
    env_name: str = "hazard_plain",
    *,
    render_mode: str | None = None,
    render_size: int = 256,
):
    """Build the evaluation / rendering environment."""
    if env_name not in GRAVITY_STRENGTHS:
        raise ValueError(f"Unknown env_name={env_name!r}; choose from {ENVS}")
    return ContinuousHazard2DEnv(
        config=Hazard2DConfig(
            max_episode_steps=300,
            gravity_strength=GRAVITY_STRENGTHS[env_name],
        ),
        observation_mode="state",
        render_mode=render_mode,
        render_size=render_size,
        terminate_at_goal=True,
    )


def _default_dataset(
    env_name: str,
    *,
    policy: str = "navigate",
    size: str = "100k",
) -> pathlib.Path:
    from hazard_env.generate_navigate import dataset_stem

    datasets = pathlib.Path(__file__).resolve().parents[1] / "datasets"
    return datasets / f"{dataset_stem(env_name, policy, size)}.npz"

def _to_jnp(batch: dict) -> dict:
    return {k: jnp.asarray(v) for k, v in batch.items()}


def _make_value_goal_resolver(
    states: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve a 4-D task goal to a real full state from offline support."""
    states = np.asarray(states, dtype=np.float32)
    scale = np.maximum(np.std(states, axis=0), 1e-3)
    cache: dict[bytes, np.ndarray] = {}

    def resolve(goal: np.ndarray) -> np.ndarray:
        full_goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        if full_goal.shape[0] != states.shape[1]:
            raise ValueError(
                f"expected {states.shape[1]}-D task goal, got {full_goal.shape}"
            )
        key = full_goal.tobytes()
        if key not in cache:
            distance = np.sum(((states - full_goal) / scale) ** 2, axis=1)
            cache[key] = states[int(np.argmin(distance))].copy()
        return cache[key]

    return resolve


def _load_value_goal_support(path: pathlib.Path) -> np.ndarray:
    raw = np.load(path)
    observations = np.asarray(raw["observations"], dtype=np.float32)
    if observations.ndim != 2:
        raise ValueError(
            f"offline observations must be rank 2, got {observations.shape}"
        )
    return observations


def _format_eval_goal(info_goal: np.ndarray, state_dim: int) -> np.ndarray:
    """Eval goals are full states with commanded xy and zero velocity."""
    xy = np.asarray(info_goal, dtype=np.float32).reshape(-1)[:2]
    out = np.zeros(state_dim, dtype=np.float32)
    out[:2] = xy
    return out


# Small start jitter so the 25 eval seeds are not identical under fixed tasks + T=0.
_EVAL_INIT_NOISE = 0.03


def _reset_eval_task(env, *, task_id: int, episode_seed: int):
    """Reset a fixed eval task with a distinct seed (and seed-dependent init noise)."""
    task_info = env.task_infos[int(task_id) - 1]
    goal = np.asarray(task_info["goal_xy"], dtype=np.float32)
    init_xy = np.asarray(task_info["init_xy"], dtype=np.float32)
    reset_seed = int(episode_seed) * 10007 + int(task_id)
    local_rng = np.random.default_rng(reset_seed)
    for _ in range(32):
        position = init_xy + local_rng.normal(0.0, _EVAL_INIT_NOISE, size=2).astype(
            np.float32
        )
        try:
            return env.reset(
                seed=reset_seed,
                options={"position": position, "goal": goal},
            )
        except ValueError:
            continue
    return env.reset(seed=reset_seed, options={"task_id": int(task_id)})


def evaluate(
    agent,
    *,
    task_ids: list[int],
    episodes_per_task: int = 3,
    seed: int = 0,
    temperature: float = 0.0,
    env_name: str = "hazard_plain",
    value_goal_resolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    env = _make_eval_env(env_name)
    results: dict[str, float] = {}
    # Unit e uses episode_seed = seed + e. Score s_e = mean_i success_{i,e}; report
    # mean±std over the n unit scores (not over pooled raw episodes / 5 task means).
    task_successes = {task_id: [] for task_id in task_ids}
    task_deaths = {task_id: [] for task_id in task_ids}
    cross_success: list[float] = []
    cross_death: list[float] = []
    for ep in range(episodes_per_task):
        episode_seed = int(seed) + int(ep)
        action_rng = np.random.default_rng(episode_seed + 17)
        ep_success: list[float] = []
        ep_death: list[float] = []
        for task_id in task_ids:
            ob, info = _reset_eval_task(
                env, task_id=task_id, episode_seed=episode_seed
            )
            goal = _format_eval_goal(info["goal"], ob.shape[-1])
            is_pathbridger = hasattr(agent, "_sample_candidates")
            if is_pathbridger and value_goal_resolver is None:
                raise ValueError(
                    "PathBridger evaluation requires a full-state value-goal resolver"
                )
            value_goal = (
                value_goal_resolver(goal)
                if is_pathbridger and value_goal_resolver is not None
                else None
            )
            done = False
            while not done:
                obs_j = jnp.asarray(ob)[None]
                goal_j = jnp.asarray(goal)[None]
                value_goal_j = (
                    jnp.asarray(value_goal)[None]
                    if value_goal is not None
                    else None
                )
                if temperature == 0.0:
                    if is_pathbridger:
                        action = np.asarray(
                            agent.sample_actions(
                                obs_j,
                                goal_j,
                                value_goals=value_goal_j,
                                seed=None,
                                temperature=0.0,
                            )
                        )[0]
                    else:
                        action = np.asarray(agent.sample_actions(
                            obs_j, goal_j, seed=None, temperature=0.0
                        ))[0]
                else:
                    key = jax.random.PRNGKey(int(action_rng.integers(0, 2**31 - 1)))
                    if is_pathbridger:
                        action = np.asarray(
                            agent.sample_actions(
                                obs_j,
                                goal_j,
                                value_goals=value_goal_j,
                                seed=key,
                                temperature=temperature,
                            )
                        )[0]
                    else:
                        action = np.asarray(agent.sample_actions(
                            obs_j, goal_j, seed=key, temperature=temperature
                        ))[0]
                action = denormalize_actions(action)
                ob, _r, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
            succ = float(info.get("is_success", False))
            dead = float(info.get("dead", False))
            task_successes[task_id].append(succ)
            task_deaths[task_id].append(dead)
            ep_success.append(succ)
            ep_death.append(dead)
        cross_success.append(float(np.mean(ep_success)))
        cross_death.append(float(np.mean(ep_death)))
    env.close()
    for task_id in task_ids:
        succ = np.asarray(task_successes[task_id], dtype=np.float64)
        dead = np.asarray(task_deaths[task_id], dtype=np.float64)
        results[f"task{task_id}_success"] = float(succ.mean())
        results[f"task{task_id}_success_std"] = float(succ.std(ddof=0))
        results[f"task{task_id}_death"] = float(dead.mean())
        results[f"task{task_id}_death_std"] = float(dead.std(ddof=0))
    cross_s = np.asarray(cross_success, dtype=np.float64)
    cross_d = np.asarray(cross_death, dtype=np.float64)
    results["mean_success"] = float(cross_s.mean()) if len(cross_s) else 0.0
    results["mean_success_std"] = float(cross_s.std(ddof=0)) if len(cross_s) else 0.0
    results["mean_death"] = float(cross_d.mean()) if len(cross_d) else 0.0
    results["mean_death_std"] = float(cross_d.std(ddof=0)) if len(cross_d) else 0.0
    results["eval_temperature"] = float(temperature)
    return results


def _eval_temperature(agent_name: str) -> float:
    """BC / HIQL-family use mode (T=0); PathBridger family samples at T=1."""
    if agent_name in ("pbg", "pbf", "trl", "dqc"):
        return 1.0
    return 0.0


def _eval_temperatures(agent_name: str) -> tuple[float, ...]:
    if agent_name in ("pbg", "pbf"):
        return (0.0, 1.0)
    return (_eval_temperature(agent_name),)


def _temp_metric_prefix(temperature: float) -> str:
    return "t0" if float(temperature) == 0.0 else "t1"


def format_eval_metrics(metrics: dict, task_ids: list[int] | None = None) -> str:
    """Compact eval line: per-task means + mean±std over 25 cross-task averages."""
    task_ids = task_ids or [1, 2, 3, 4, 5]
    parts = [
        f"n={int(metrics.get('episodes_per_task', metrics.get('num_eval_envs', 0)))}"
    ]
    temperatures = metrics.get(
        "eval_temperatures",
        [float(metrics.get("eval_temperature", 0.0))],
    )
    for temperature in temperatures:
        prefix = _temp_metric_prefix(float(temperature))
        mean = float(
            metrics.get(f"{prefix}_mean_success", metrics.get("mean_success", 0.0))
        )
        std = float(
            metrics.get(
                f"{prefix}_mean_success_std",
                metrics.get("mean_success_std", 0.0),
            )
        )
        task_bits = " ".join(
            f"t{task_id}={float(metrics.get(f'{prefix}_task{task_id}_success', metrics.get(f'task{task_id}_success', 0.0))):.2f}"
            for task_id in task_ids
        )
        parts.append(
            f"T={float(temperature):g} success={mean:.2f}±{std:.2f} {task_bits}"
        )
    return " ".join(parts)


def evaluate_suite(
    agent,
    *,
    seed: int,
    agent_name: str,
    task_ids: list[int] | None = None,
    num_eval_envs: int = 25,
    env_name: str = "hazard_plain",
    value_goal_resolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    """Evaluate with the agent-family temperature (BC/HIQL: 0, PBG/PBF: 1).

    ``num_eval_envs`` (= logged ``n``) is the number of eval units with distinct
    seeds ``seed .. seed+n-1``. Each unit runs one episode per task (with
    seed-dependent init jitter) and scores ``(∑_i success_i) / n_tasks``. Then
    ``success=mean±std`` is over those ``n`` unit scores.
    """
    task_ids = task_ids or [1, 2, 3, 4, 5]
    episodes_per_task = max(1, int(num_eval_envs))
    temperatures = _eval_temperatures(agent_name)
    primary = _eval_temperature(agent_name)
    out: dict[str, float | int | str | list[float]] = {
        "num_eval_envs": episodes_per_task,
        "episodes_per_task": episodes_per_task,
        "total_eval_episodes": (
            episodes_per_task * len(task_ids) * len(temperatures)
        ),
        "eval_temperature": primary,
        "eval_temperatures": [float(t) for t in temperatures],
        # mean±std over n seeded cross-task unit scores (∑_i t_i / n_tasks).
        "eval_agg": "cross_task_unit_seeded",
    }
    for temperature in temperatures:
        metrics = evaluate(
            agent,
            task_ids=task_ids,
            episodes_per_task=episodes_per_task,
            seed=seed,
            temperature=temperature,
            env_name=env_name,
            value_goal_resolver=value_goal_resolver,
        )
        prefix = _temp_metric_prefix(temperature)
        for key in (
            "mean_success",
            "mean_success_std",
            "mean_death",
            "mean_death_std",
        ):
            out[f"{prefix}_{key}"] = metrics[key]
        for task_id in task_ids:
            for suffix in (
                "success",
                "success_std",
                "death",
                "death_std",
            ):
                out[f"{prefix}_task{task_id}_{suffix}"] = metrics[
                    f"task{task_id}_{suffix}"
                ]
        if float(temperature) == float(primary):
            for key, value in metrics.items():
                if key != "eval_temperature":
                    out[key] = value
    return out


def load_checkpoint(
    *,
    checkpoint_dir: pathlib.Path,
    agent_name: str,
    dataset_path: pathlib.Path,
    steps: int = 50_000,
):
    """Restore a saved agent from msgpack + json metadata."""
    from agents import AGENTS
    from hazard_env.utils.datasets import (
        load_dqc_navigate_dataset,
        load_navigate_dataset,
        load_trl_navigate_dataset,
    )

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    meta_path = checkpoint_dir / f"step_{steps}.json"
    pack_path = checkpoint_dir / f"step_{steps}.msgpack"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    config = metadata["config"]
    if isinstance(config.get("hidden_dims"), list):
        config["hidden_dims"] = tuple(config["hidden_dims"])
    if agent_name == "trl":
        data = load_trl_navigate_dataset(dataset_path, config=config)
        example = _to_jnp(data.sample(np.random.default_rng(0), 8))
        template = AGENTS[agent_name].create(0, example, config)
    elif agent_name == "dqc":
        data = load_dqc_navigate_dataset(dataset_path, config=config)
        example = _to_jnp(data.sample(np.random.default_rng(0), 8))
        template = AGENTS[agent_name].create(0, example, config)
    else:
        data = load_navigate_dataset(dataset_path, seed=0)
        template = AGENTS[agent_name].create(
            0, data.observations[:8], data.actions[:8], config
        )
    return flax.serialization.from_bytes(template, pack_path.read_bytes()), metadata


def save_checkpoint(
    agent,
    *,
    output_dir: pathlib.Path,
    agent_name: str,
    steps: int,
    metrics: dict[str, float],
) -> pathlib.Path:
    """Save a restorable Flax agent state and human-readable metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"step_{steps}.msgpack"
    checkpoint_path.write_bytes(flax.serialization.to_bytes(agent))
    metadata = {
        "agent": agent_name,
        "steps": steps,
        "config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dict(agent.config).items()
        },
        "metrics": metrics,
    }
    (output_dir / f"step_{steps}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return checkpoint_path


def render_agent(
    agent,
    *,
    output_dir: pathlib.Path,
    task_ids: list[int],
    seed: int,
    render_size: int = 256,
    temperature: float = 0.0,
    env_name: str = "hazard_plain",
    diagnostics: bool = True,
    value_goal_resolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[pathlib.Path]:
    """Render evaluation episodes, optionally with model diagnostic overlays."""
    import imageio.v2 as imageio

    output_dir.mkdir(parents=True, exist_ok=True)
    env = _make_eval_env(env_name, render_mode="rgb_array", render_size=render_size)
    rng = np.random.default_rng(seed)
    paths = []
    try:
        for task_id in task_ids:
            path = output_dir / f"task{task_id}.mp4"
            ob, info = env.reset(
                seed=int(rng.integers(0, 1_000_000)),
                options={"task_id": task_id},
            )
            goal = _format_eval_goal(info["goal"], ob.shape[-1])
            is_pathbridger = hasattr(agent, "_sample_candidates")
            if is_pathbridger and value_goal_resolver is None:
                raise ValueError(
                    "PathBridger rendering requires a full-state value-goal resolver"
                )
            value_goal = (
                value_goal_resolver(goal) if is_pathbridger else None
            )
            trail = [np.asarray(ob, dtype=np.float32).copy()]
            subgoal_trail: list[np.ndarray] = []
            cached_value_field = None
            with imageio.get_writer(
                path, fps=env.metadata["render_fps"], codec="libx264"
            ) as writer:
                done = False
                while not done:
                    if temperature == 0.0:
                        key = agent.rng
                        if is_pathbridger:
                            action = np.asarray(
                                agent.sample_actions(
                                    jnp.asarray(ob)[None],
                                    jnp.asarray(goal)[None],
                                    value_goals=jnp.asarray(value_goal)[None],
                                    seed=None,
                                    temperature=0.0,
                                )
                            )[0]
                        else:
                            action = np.asarray(agent.sample_actions(
                                jnp.asarray(ob)[None],
                                jnp.asarray(goal)[None],
                                seed=None,
                                temperature=0.0,
                            ))[0]
                    else:
                        key = jax.random.PRNGKey(int(rng.integers(0, 1_000_000)))
                        if is_pathbridger:
                            action = np.asarray(
                                agent.sample_actions(
                                    jnp.asarray(ob)[None],
                                    jnp.asarray(goal)[None],
                                    value_goals=jnp.asarray(value_goal)[None],
                                    seed=key,
                                    temperature=temperature,
                                )
                            )[0]
                        else:
                            action = np.asarray(agent.sample_actions(
                                jnp.asarray(ob)[None],
                                jnp.asarray(goal)[None],
                                seed=key,
                                temperature=temperature,
                            ))[0]

                    frame = env.render()
                    if diagnostics:
                        diagnostic = collect_agent_diagnostics(
                            agent,
                            ob,
                            goal,
                            seed=key,
                            value_goal=value_goal,
                            compute_value_field=cached_value_field is None,
                            temperature=temperature,
                        )
                        if "value_field" in diagnostic:
                            cached_value_field = diagnostic["value_field"]
                        elif cached_value_field is not None:
                            diagnostic["value_field"] = cached_value_field
                        if "subgoal" in diagnostic:
                            subgoal_trail.append(
                                np.asarray(diagnostic["subgoal"], dtype=np.float32).copy()
                            )
                        frame = compose_diagnostic_frame(
                            frame,
                            diagnostic,
                            arena_low=float(env.config.arena_low),
                            arena_high=float(env.config.arena_high),
                            trail=trail,
                            subgoal_trail=subgoal_trail,
                        )
                    writer.append_data(frame)

                    ob, _reward, terminated, truncated, _info = env.step(
                        denormalize_actions(action)
                    )
                    trail.append(np.asarray(ob, dtype=np.float32).copy())
                    done = bool(terminated or truncated)

                final_frame = env.render()
                if diagnostics:
                    final_diagnostic = collect_agent_diagnostics(
                        agent,
                        ob,
                        goal,
                        seed=key,
                        value_goal=value_goal,
                        compute_value_field=False,
                        temperature=temperature,
                    )
                    if cached_value_field is not None:
                        final_diagnostic["value_field"] = cached_value_field
                    if "subgoal" in final_diagnostic:
                        subgoal_trail.append(
                            np.asarray(
                                final_diagnostic["subgoal"], dtype=np.float32
                            ).copy()
                        )
                    final_frame = compose_diagnostic_frame(
                        final_frame,
                        final_diagnostic,
                        arena_low=float(env.config.arena_low),
                        arena_high=float(env.config.arena_high),
                        trail=trail,
                        subgoal_trail=subgoal_trail,
                    )
                writer.append_data(final_frame)
            paths.append(path)
    finally:
        env.close()
    return paths


def train(
    *,
    agent_name: str,
    dataset_path: pathlib.Path,
    steps: int,
    seed: int,
    eval_every: int,
    log_every: int,
    config_overrides: dict | None = None,
    checkpoint_dir: pathlib.Path | None = None,
    num_eval_envs: int = 25,
    env_name: str = "hazard_plain",
) -> tuple[object, dict[str, float]]:
    if agent_name not in AGENTS:
        raise SystemExit(f"Unknown agent {agent_name}; choose from {list(AGENTS)}")

    config = DEFAULT_CONFIGS[agent_name]()
    if config_overrides:
        config.update(config_overrides)
    config["goal_dim"] = 4

    if agent_name == "hiql":
        data = load_hgc_navigate_dataset(dataset_path, config=config, seed=seed)
        print(f"Loaded HGC dataset size={len(data)} from {dataset_path}")
    elif agent_name == "trl":
        data = load_trl_navigate_dataset(dataset_path, config=config)
        print(f"Loaded TRL dataset size={len(data)} from {dataset_path}")
    elif agent_name == "dqc":
        data = load_dqc_navigate_dataset(dataset_path, config=config)
        print(f"Loaded DQC dataset size={len(data)} from {dataset_path}")
    else:
        data = load_navigate_dataset(
            dataset_path,
            subgoal_steps=int(config.get("subgoal_steps", 8)),
            seed=seed,
        )
        print(f"Loaded {len(data)} transitions from {dataset_path}")
    value_goal_resolver = (
        _make_value_goal_resolver(_load_value_goal_support(dataset_path))
        if agent_name in ("pbg", "pbf")
        else None
    )

    rng = np.random.default_rng(seed)
    if agent_name in ("trl", "dqc"):
        example = _to_jnp(data.sample(rng, 8))
        agent = AGENTS[agent_name].create(seed, example, config)
    else:
        ex_obs = data.observations[:8]
        ex_act = data.actions[:8]
        agent = AGENTS[agent_name].create(seed, ex_obs, ex_act, config)

    t0 = time.time()
    metrics: dict[str, float] = {}
    for step in range(1, steps + 1):
        batch = _to_jnp(data.sample(rng, config["batch_size"]))
        agent, info = agent.update(batch)
        if step % log_every == 0 or step == 1:
            pretty = {
                k: float(v) for k, v in info.items() if np.ndim(np.asarray(v)) == 0
            }
            print(f"[{agent_name}] step={step} {pretty}")
        if eval_every > 0 and step % eval_every == 0:
            metrics = evaluate_suite(
                agent,
                seed=seed + step,
                agent_name=agent_name,
                num_eval_envs=num_eval_envs,
                env_name=env_name,
                value_goal_resolver=value_goal_resolver,
            )
            print(f"[{agent_name}] eval@{step} {format_eval_metrics(metrics)}")
            if checkpoint_dir is not None:
                path = save_checkpoint(
                    agent,
                    output_dir=checkpoint_dir,
                    agent_name=agent_name,
                    steps=step,
                    metrics=metrics,
                )
                print(f"[{agent_name}] checkpoint@{step} {path}")

    if not metrics or (eval_every <= 0 or steps % eval_every != 0):
        metrics = evaluate_suite(
            agent,
            seed=seed + steps,
            agent_name=agent_name,
            num_eval_envs=num_eval_envs,
            env_name=env_name,
            value_goal_resolver=value_goal_resolver,
        )
        if checkpoint_dir is not None:
            save_checkpoint(
                agent,
                output_dir=checkpoint_dir,
                agent_name=agent_name,
                steps=steps,
                metrics=metrics,
            )
    print(
        f"[{agent_name}] final eval {format_eval_metrics(metrics)}  "
        f"({time.time() - t0:.1f}s)"
    )
    return agent, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", choices=sorted(AGENTS), required=True)
    p.add_argument("--env", choices=list(ENVS), default="hazard_plain")
    p.add_argument(
        "--dataset-policy",
        choices=("navigate", "noisy", "random"),
        default="navigate",
    )
    p.add_argument("--dataset-size", choices=("1k", "10k", "100k"), default="100k")
    p.add_argument("--dataset", type=pathlib.Path, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument(
        "--subgoal-alpha",
        type=float,
        default=None,
        help="Override PBG/PBF value-gap exponent scale.",
    )
    p.add_argument(
        "--subgoal-weight-max",
        type=float,
        default=None,
        help="Override PBG/PBF value-gap weight cap.",
    )
    p.add_argument("--checkpoint-dir", type=pathlib.Path, default=None)
    p.add_argument("--render-dir", type=pathlib.Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.subgoal_alpha is not None:
        overrides["subgoal_value_gap_scale"] = args.subgoal_alpha
    if args.subgoal_weight_max is not None:
        overrides["subgoal_value_weight_max"] = args.subgoal_weight_max
    dataset = args.dataset or _default_dataset(
        args.env, policy=args.dataset_policy, size=args.dataset_size
    )
    agent, _metrics = train(
        agent_name=args.agent,
        dataset_path=dataset,
        steps=args.steps,
        seed=args.seed,
        eval_every=args.eval_every,
        log_every=args.log_every,
        config_overrides=overrides,
        checkpoint_dir=args.checkpoint_dir,
        env_name=args.env,
    )
    if args.render_dir is not None:
        value_goal_resolver = None
        if args.agent in ("pbg", "pbf"):
            value_goal_resolver = _make_value_goal_resolver(
                _load_value_goal_support(dataset)
            )
        paths = render_agent(
            agent,
            output_dir=args.render_dir,
            task_ids=[1, 2, 3, 4, 5],
            seed=args.seed + args.steps,
            temperature=_eval_temperature(args.agent),
            env_name=args.env,
            value_goal_resolver=value_goal_resolver,
        )
        print(f"Rendered {len(paths)} videos to {args.render_dir}")


if __name__ == "__main__":
    main()
