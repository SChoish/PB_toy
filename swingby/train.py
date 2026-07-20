"""Train / evaluate shared agents on OrbitalSwingBy offline datasets."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time
from collections.abc import Callable

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from agents import AGENTS, DEFAULT_CONFIGS

from .config import black_hole_config, planet_config
from .datasets import (
    LEGACY_ACTION_ENCODING,
    denormalize_actions,
    load_swingby_dataset,
    load_swingby_dqc_dataset,
    load_swingby_trl_dataset,
    read_dataset_metadata,
)
from .env import OrbitalSwingByEnv, swingby_eval_rotation
from .generate_dataset import ENVS, dataset_stem

TASK_IDS = (1, 2, 3, 4, 5)


def _make_eval_env(
    env_name: str,
    *,
    render_mode: str | None = None,
    render_size: int = 256,
) -> OrbitalSwingByEnv:
    if env_name not in ENVS:
        raise ValueError(f"Unknown env_name={env_name!r}; choose from {ENVS}")
    if env_name == "swingby_blackhole":
        config = black_hole_config(
            task_mode="swingby",
            max_episode_steps=650,
            show_ballistic_prediction=False,
        )
    else:
        config = planet_config(
            task_mode="swingby",
            max_episode_steps=650,
            show_ballistic_prediction=False,
        )
    return OrbitalSwingByEnv(
        config=config,
        observation_mode="state",
        render_mode=render_mode,
        render_size=render_size,
        terminate_at_goal=True,
    )


def _default_dataset(env_name: str, *, size: str = "100k") -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parent / "datasets"
    return root / f"{dataset_stem(env_name, 'expert', size)}.npz"  # type: ignore[arg-type]


def _to_jnp(batch: dict) -> dict:
    return {k: jnp.asarray(v) for k, v in batch.items()}


def _make_value_goal_resolver(
    states: np.ndarray,
    *,
    goal_dim: int = 4,
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve a task goal to a real full state from the offline support."""
    states = np.asarray(states, dtype=np.float32)
    features = states[:, :goal_dim]
    scale = np.maximum(np.std(features, axis=0), 1e-3)
    cache: dict[bytes, np.ndarray] = {}

    def resolve(goal: np.ndarray) -> np.ndarray:
        task_goal = np.asarray(goal, dtype=np.float32).reshape(-1)[:goal_dim]
        key = task_goal.tobytes()
        if key not in cache:
            distance = np.sum(((features - task_goal) / scale) ** 2, axis=1)
            resolved = states[int(np.argmin(distance))].copy()
            resolved[:goal_dim] = task_goal
            cache[key] = resolved
        return cache[key]

    return resolve


def _eval_temperature(agent_name: str) -> float:
    if agent_name in ("pbg", "pbf", "trl", "dqc"):
        return 1.0
    return 0.0


def _eval_temperatures(agent_name: str) -> tuple[float, ...]:
    if agent_name in ("pbg", "pbf"):
        return (0.0, 1.0)
    return (_eval_temperature(agent_name),)


def _temp_metric_prefix(temperature: float) -> str:
    if float(temperature) == 0.0:
        return "t0"
    if float(temperature) == 1.0:
        return "t1"
    return f"t{float(temperature):g}"


def _format_eval_goal(info: dict) -> np.ndarray:
    goal = np.asarray(info["goal"], dtype=np.float32).reshape(-1)
    if goal.size >= 4:
        return goal[:4].astype(np.float32, copy=True)
    return np.concatenate(
        [
            goal[:2],
            np.asarray(info["goal_velocity"], dtype=np.float32).reshape(-1)[:2],
        ]
    ).astype(np.float32)


def _action_chunk_horizon(agent) -> int:
    config = dict(getattr(agent, "config", {}) or {})
    return max(1, int(config.get("action_chunk_horizon", 1)))


def _uses_action_chunks(agent) -> bool:
    return hasattr(agent, "sample_action_chunk")


def _action_encoding(agent) -> str:
    config = dict(getattr(agent, "config", {}) or {})
    return str(config.get("action_encoding", LEGACY_ACTION_ENCODING))


def _select_action(
    agent,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    value_goal: np.ndarray | None,
    rng: np.random.Generator,
    temperature: float,
    action_chunk: np.ndarray | None,
    chunk_index: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    obs_j = jnp.asarray(observation)[None]
    goal_j = jnp.asarray(goal)[None]
    if _uses_action_chunks(agent):
        if value_goal is None:
            raise ValueError("PathBridger requires a resolved full-state value goal")
        value_goal_j = jnp.asarray(value_goal)[None]
        need_replan = action_chunk is None or chunk_index >= int(action_chunk.shape[0])
        if need_replan:
            if temperature == 0.0:
                chunk = np.asarray(
                    agent.sample_action_chunk(
                        obs_j,
                        goal_j,
                        value_goals=value_goal_j,
                        seed=None,
                        temperature=0.0,
                    )
                )[0]
            else:
                key = jax.random.PRNGKey(int(rng.integers(0, 1_000_000)))
                chunk = np.asarray(
                    agent.sample_action_chunk(
                        obs_j,
                        goal_j,
                        value_goals=value_goal_j,
                        seed=key,
                        temperature=temperature,
                    )
                )[0]
            action_chunk = np.clip(chunk.astype(np.float32), -1.0, 1.0)
            chunk_index = 0
        assert action_chunk is not None
        action = action_chunk[chunk_index]
        return action, action_chunk, chunk_index + 1

    if temperature == 0.0:
        action = np.asarray(
            agent.sample_actions(obs_j, goal_j, seed=None, temperature=0.0)
        )[0]
    else:
        key = jax.random.PRNGKey(int(rng.integers(0, 1_000_000)))
        action = np.asarray(
            agent.sample_actions(
                obs_j, goal_j, seed=key, temperature=temperature
            )
        )[0]
    return (
        np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0),
        None,
        0,
    )


def _load_dataset(agent_name: str, dataset_path: pathlib.Path, config: dict):
    if agent_name == "trl":
        return load_swingby_trl_dataset(dataset_path, config=config)
    if agent_name == "dqc":
        return load_swingby_dqc_dataset(dataset_path, config=config)
    return load_swingby_dataset(
        dataset_path,
        path_horizon=int(config.get("subgoal_steps", 8)),
        action_chunk_horizon=int(config.get("action_chunk_horizon", 5)),
        value_base_horizon=int(config.get("value_base_horizon", 5)),
    )


def format_eval_metrics(metrics: dict, task_ids: list[int] | None = None) -> str:
    task_ids = task_ids or list(TASK_IDS)
    n = int(metrics.get("episodes_per_task", metrics.get("num_eval_envs", 0)))
    temps = metrics.get("eval_temperatures")
    if temps is None:
        temps = (float(metrics.get("eval_temperature", 0.0)),)
    else:
        temps = tuple(float(t) for t in temps)

    chunks: list[str] = [f"n={n}"]
    for temperature in temps:
        prefix = _temp_metric_prefix(temperature)
        mean_key = f"{prefix}_mean_success"
        if mean_key in metrics:
            mean = float(metrics[mean_key])
            std = float(metrics.get(f"{prefix}_mean_success_std", 0.0))
            task_bits = " ".join(
                f"t{tid}={float(metrics.get(f'{prefix}_task{tid}_success', 0.0)):.2f}"
                for tid in task_ids
            )
        else:
            mean = float(metrics.get("mean_success", 0.0))
            std = float(metrics.get("mean_success_std", 0.0))
            task_bits = " ".join(
                f"t{tid}={float(metrics.get(f'task{tid}_success', 0.0)):.2f}"
                for tid in task_ids
            )
        chunks.append(f"T={temperature:g} success={mean:.2f}±{std:.2f} {task_bits}")
    return " ".join(chunks)


def evaluate(
    agent,
    *,
    env_name: str,
    task_ids: list[int] | None = None,
    episodes_per_task: int = 25,
    seed: int = 0,
    temperature: float = 0.0,
    value_goal_resolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    task_ids = task_ids or list(TASK_IDS)
    env = _make_eval_env(env_name)
    results: dict[str, float] = {}
    task_successes = {task_id: [] for task_id in task_ids}
    task_deaths = {task_id: [] for task_id in task_ids}
    cross_success: list[float] = []
    cross_death: list[float] = []
    try:
        for ep in range(episodes_per_task):
            episode_seed = int(seed) + int(ep)
            action_rng = np.random.default_rng(episode_seed + 17)
            ep_success: list[float] = []
            ep_death: list[float] = []
            for task_id in task_ids:
                reset_seed = episode_seed * 10007 + int(task_id)
                rotation = swingby_eval_rotation(
                    int(task_id), int(ep), int(episodes_per_task)
                )
                ob, info = env.reset(
                    seed=reset_seed,
                    options={
                        "task_id": int(task_id),
                        "task_rotation": rotation,
                    },
                )
                goal = _format_eval_goal(info)
                value_goal = (
                    value_goal_resolver(goal)
                    if _uses_action_chunks(agent)
                    and value_goal_resolver is not None
                    else None
                )
                done = False
                action_chunk = None
                chunk_index = 0
                while not done:
                    action, action_chunk, chunk_index = _select_action(
                        agent,
                        ob,
                        goal,
                        value_goal=value_goal,
                        rng=action_rng,
                        temperature=temperature,
                        action_chunk=action_chunk,
                        chunk_index=chunk_index,
                    )
                    ob, _r, terminated, truncated, info = env.step(
                        denormalize_actions(
                            action, encoding=_action_encoding(agent)
                        )
                    )
                    done = bool(terminated or truncated)
                succ = float(info.get("is_success", False))
                dead = float(info.get("dead", False))
                task_successes[task_id].append(succ)
                task_deaths[task_id].append(dead)
                ep_success.append(succ)
                ep_death.append(dead)
            cross_success.append(float(np.mean(ep_success)))
            cross_death.append(float(np.mean(ep_death)))
    finally:
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


def evaluate_suite(
    agent,
    *,
    seed: int,
    agent_name: str,
    env_name: str,
    task_ids: list[int] | None = None,
    num_eval_envs: int = 25,
    value_goal_resolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    task_ids = task_ids or list(TASK_IDS)
    episodes_per_task = max(1, int(num_eval_envs))
    temperatures = _eval_temperatures(agent_name)
    primary = _eval_temperature(agent_name)
    out: dict[str, float | str | list[float]] = {
        "num_eval_envs": float(episodes_per_task),
        "episodes_per_task": float(episodes_per_task),
        "total_eval_episodes": float(
            episodes_per_task * len(task_ids) * len(temperatures)
        ),
        "eval_temperature": float(primary),
        "eval_temperatures": [float(t) for t in temperatures],
        "eval_agg": "cross_task_unit_seeded",
    }
    for temperature in temperatures:
        metrics = evaluate(
            agent,
            env_name=env_name,
            task_ids=task_ids,
            episodes_per_task=episodes_per_task,
            seed=seed,
            temperature=float(temperature),
            value_goal_resolver=value_goal_resolver,
        )
        prefix = _temp_metric_prefix(temperature)
        out[f"{prefix}_mean_success"] = metrics["mean_success"]
        out[f"{prefix}_mean_success_std"] = metrics["mean_success_std"]
        out[f"{prefix}_mean_death"] = metrics["mean_death"]
        out[f"{prefix}_mean_death_std"] = metrics["mean_death_std"]
        for task_id in task_ids:
            out[f"{prefix}_task{task_id}_success"] = metrics[f"task{task_id}_success"]
            out[f"{prefix}_task{task_id}_success_std"] = metrics[
                f"task{task_id}_success_std"
            ]
            out[f"{prefix}_task{task_id}_death"] = metrics[f"task{task_id}_death"]
            out[f"{prefix}_task{task_id}_death_std"] = metrics[
                f"task{task_id}_death_std"
            ]
        if float(temperature) == float(primary):
            out["mean_success"] = metrics["mean_success"]
            out["mean_success_std"] = metrics["mean_success_std"]
            out["mean_death"] = metrics["mean_death"]
            out["mean_death_std"] = metrics["mean_death_std"]
            for task_id in task_ids:
                out[f"task{task_id}_success"] = metrics[f"task{task_id}_success"]
                out[f"task{task_id}_success_std"] = metrics[
                    f"task{task_id}_success_std"
                ]
                out[f"task{task_id}_death"] = metrics[f"task{task_id}_death"]
                out[f"task{task_id}_death_std"] = metrics[f"task{task_id}_death_std"]
    return out


def save_checkpoint(
    agent,
    *,
    output_dir: pathlib.Path,
    agent_name: str,
    steps: int,
    metrics: dict[str, float],
) -> pathlib.Path:
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


def load_checkpoint(
    *,
    checkpoint_dir: pathlib.Path,
    agent_name: str,
    dataset_path: pathlib.Path,
    steps: int = 50_000,
):
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    meta_path = checkpoint_dir / f"step_{steps}.json"
    pack_path = checkpoint_dir / f"step_{steps}.msgpack"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    config = metadata["config"]
    dataset_meta = read_dataset_metadata(dataset_path)
    if config.get("task_schema", "ballistic_v1") != dataset_meta["dataset_schema"]:
        raise ValueError(
            "SwingBy checkpoint/data schema mismatch. Retrain the canonical "
            "swingby dataset in a matrix-provided _swingby checkpoint directory."
        )
    if config.get("action_encoding", LEGACY_ACTION_ENCODING) != dataset_meta[
        "action_encoding"
    ]:
        raise ValueError("SwingBy checkpoint/data action encoding mismatch")
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
        example = _to_jnp(data.sample(np.random.default_rng(0), 8))
        template = AGENTS[agent_name].create(0, example, config)
    else:
        template = AGENTS[agent_name].create(
            0, data.observations[:8], data.actions[:8], config
        )
    return flax.serialization.from_bytes(template, pack_path.read_bytes()), metadata


def latest_checkpoint_step(checkpoint_dir: pathlib.Path | None) -> int | None:
    if checkpoint_dir is None:
        return None
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None
    found: list[int] = []
    for pack in checkpoint_dir.glob("step_*.msgpack"):
        try:
            step = int(pack.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (checkpoint_dir / f"step_{step}.json").exists():
            found.append(step)
    return max(found) if found else None


def train(
    *,
    agent_name: str,
    dataset_path: pathlib.Path,
    env_name: str,
    steps: int,
    seed: int,
    eval_every: int,
    log_every: int,
    config_overrides: dict | None = None,
    checkpoint_dir: pathlib.Path | None = None,
    num_eval_envs: int = 25,
    resume: bool = True,
) -> tuple[object, dict[str, float]]:
    if agent_name not in AGENTS:
        raise SystemExit(f"Unknown agent {agent_name}; choose from {list(AGENTS)}")
    if env_name not in ENVS:
        raise SystemExit(f"Unknown env {env_name}; choose from {list(ENVS)}")

    config = DEFAULT_CONFIGS[agent_name]()
    if config_overrides:
        config.update(config_overrides)
    config["goal_dim"] = 4
    config["env_name"] = env_name
    dataset_meta = read_dataset_metadata(dataset_path)
    config["task_schema"] = dataset_meta["dataset_schema"]
    config["action_encoding"] = dataset_meta["action_encoding"]
    if agent_name in ("pbg", "pbf"):
        config["phi_goal_obs_indices"] = (0, 1, 2, 3)
        config["subgoal_value_goal_representation"] = "full"

    data = _load_dataset(agent_name, dataset_path, config)
    value_goal_resolver = (
        _make_value_goal_resolver(data.next_observations)
        if agent_name in ("pbg", "pbf")
        else None
    )
    print(
        f"Loaded {agent_name} dataset size={len(data)} from {dataset_path}",
        flush=True,
    )

    start_step = 0
    metrics: dict[str, float] = {}
    latest = latest_checkpoint_step(checkpoint_dir) if resume else None
    if latest is not None and latest >= int(steps):
        print(
            f"[{agent_name}] checkpoint already complete at step={latest}; loading",
            flush=True,
        )
        agent, meta = load_checkpoint(
            checkpoint_dir=pathlib.Path(checkpoint_dir),
            agent_name=agent_name,
            dataset_path=dataset_path,
            steps=int(latest),
        )
        return agent, dict(meta.get("metrics") or {})
    if latest is not None and latest > 0:
        print(
            f"[{agent_name}] resuming from step={latest} → {steps}",
            flush=True,
        )
        agent, meta = load_checkpoint(
            checkpoint_dir=pathlib.Path(checkpoint_dir),
            agent_name=agent_name,
            dataset_path=dataset_path,
            steps=int(latest),
        )
        start_step = int(latest)
        metrics = dict(meta.get("metrics") or {})
        config = dict(meta.get("config") or config)
        if isinstance(config.get("hidden_dims"), list):
            config["hidden_dims"] = tuple(config["hidden_dims"])
    else:
        rng_init = np.random.default_rng(seed)
        if agent_name in ("trl", "dqc"):
            example = _to_jnp(data.sample(rng_init, 8))
            agent = AGENTS[agent_name].create(seed, example, config)
        else:
            agent = AGENTS[agent_name].create(
                seed, data.observations[:8], data.actions[:8], config
            )

    rng = np.random.default_rng(seed + start_step)
    t0 = time.time()
    for step in range(start_step + 1, steps + 1):
        batch = _to_jnp(data.sample(rng, config["batch_size"]))
        agent, info = agent.update(batch)
        if step % log_every == 0 or step == start_step + 1:
            pretty = {
                k: float(v) for k, v in info.items() if np.ndim(np.asarray(v)) == 0
            }
            print(f"[{agent_name}] step={step} {pretty}", flush=True)
        if eval_every > 0 and step % eval_every == 0:
            metrics = evaluate_suite(
                agent,
                seed=seed + step,
                agent_name=agent_name,
                env_name=env_name,
                num_eval_envs=num_eval_envs,
                value_goal_resolver=value_goal_resolver,
            )
            print(
                f"[{agent_name}] eval@{step} {format_eval_metrics(metrics)}",
                flush=True,
            )
            if checkpoint_dir is not None:
                path = save_checkpoint(
                    agent,
                    output_dir=checkpoint_dir,
                    agent_name=agent_name,
                    steps=step,
                    metrics=metrics,
                )
                print(f"[{agent_name}] checkpoint@{step} {path}", flush=True)

    if not metrics or (eval_every <= 0 or steps % eval_every != 0):
        metrics = evaluate_suite(
            agent,
            seed=seed + steps,
            agent_name=agent_name,
            env_name=env_name,
            num_eval_envs=num_eval_envs,
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
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    return agent, metrics


RENDER_KINDS = ("env", "overlay")


def render_videos_ready(
    output_dir: pathlib.Path, task_ids: list[int] | tuple[int, ...] | None = None
) -> bool:
    task_ids = list(task_ids or TASK_IDS)
    return all(
        (pathlib.Path(output_dir) / kind / f"task{tid}.mp4").exists()
        for kind in RENDER_KINDS
        for tid in task_ids
    )


def render_agent(
    agent,
    *,
    output_dir: pathlib.Path,
    env_name: str,
    task_ids: list[int] | None = None,
    seed: int = 0,
    render_size: int = 256,
    temperature: float = 0.0,
    value_goal_resolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[pathlib.Path]:
    """Render each fixed eval task into ``env/`` and ``overlay/`` videos."""
    import imageio.v2 as imageio

    from agents.rendering import (
        collect_agent_diagnostics,
        compose_diagnostic_frame,
    )

    task_ids = task_ids or list(TASK_IDS)
    output_dir = pathlib.Path(output_dir)
    env_dir = output_dir / "env"
    overlay_dir = output_dir / "overlay"
    env_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    env = _make_eval_env(
        env_name, render_mode="rgb_array", render_size=render_size
    )
    rng = np.random.default_rng(seed)
    paths: list[pathlib.Path] = []
    try:
        for task_id in task_ids:
            env_path = env_dir / f"task{task_id}.mp4"
            overlay_path = overlay_dir / f"task{task_id}.mp4"
            reset_seed = int(rng.integers(0, 1_000_000))
            ob, info = env.reset(
                seed=reset_seed,
                options={
                    "task_id": int(task_id),
                    "task_rotation": swingby_eval_rotation(
                        int(task_id), 1, 25
                    ),
                },
            )
            goal = _format_eval_goal(info)
            is_pathbridger = hasattr(agent, "_sample_candidates") and hasattr(
                agent, "_plan"
            )
            value_goal = (
                value_goal_resolver(goal)
                if is_pathbridger and value_goal_resolver is not None
                else None
            )
            trail = [np.asarray(ob, dtype=np.float32).copy()]
            subgoal_trail: list[np.ndarray] = []
            cached_value_field = None
            cached_policy_diagnostic: dict[str, np.ndarray] = {}
            # Refresh V(x,y|·) when fuel changes meaningfully.
            cached_fuel: float | None = None
            fps = env.metadata["render_fps"]
            with imageio.get_writer(
                env_path, fps=fps, codec="libx264"
            ) as env_writer, imageio.get_writer(
                overlay_path, fps=fps, codec="libx264"
            ) as overlay_writer:
                done = False
                action_chunk = None
                chunk_index = 0
                overlay_seed = agent.rng
                while not done:
                    if temperature == 0.0:
                        overlay_seed = agent.rng
                    else:
                        overlay_seed = jax.random.PRNGKey(
                            int(rng.integers(0, 1_000_000))
                        )
                    replanned = is_pathbridger and (
                        action_chunk is None
                        or chunk_index >= int(action_chunk.shape[0])
                    )
                    action, action_chunk, chunk_index = _select_action(
                        agent,
                        ob,
                        goal,
                        value_goal=value_goal,
                        rng=rng,
                        temperature=temperature,
                        action_chunk=action_chunk,
                        chunk_index=chunk_index,
                    )
                    frame = env.render()
                    env_writer.append_data(frame)
                    fuel = float(ob[4]) if np.asarray(ob).shape[-1] > 4 else None
                    refresh_field = cached_value_field is None or (
                        fuel is not None
                        and (
                            cached_fuel is None
                            or abs(fuel - cached_fuel) > 1e-3
                        )
                    )
                    diagnostic = collect_agent_diagnostics(
                        agent,
                        ob,
                        goal,
                        seed=overlay_seed,
                        value_goal=value_goal,
                        compute_value_field=refresh_field,
                        compute_policy_diagnostics=(
                            not is_pathbridger or replanned
                        ),
                        temperature=temperature,
                    )
                    if "value_field" in diagnostic:
                        cached_value_field = diagnostic["value_field"]
                        cached_fuel = fuel
                    elif cached_value_field is not None:
                        diagnostic["value_field"] = cached_value_field
                    if is_pathbridger:
                        if replanned:
                            cached_policy_diagnostic = {
                                key: value
                                for key, value in diagnostic.items()
                                if key != "value_field"
                            }
                        else:
                            diagnostic.update(cached_policy_diagnostic)
                    if "subgoal" in diagnostic and (
                        not is_pathbridger or replanned
                    ):
                        subgoal_trail.append(
                            np.asarray(
                                diagnostic["subgoal"], dtype=np.float32
                            ).copy()
                        )
                    overlay_writer.append_data(
                        compose_diagnostic_frame(
                            frame,
                            diagnostic,
                            arena_low=float(env.config.arena_low),
                            arena_high=float(env.config.arena_high),
                            trail=trail,
                            subgoal_trail=subgoal_trail,
                        )
                    )
                    ob, _reward, terminated, truncated, _info = env.step(
                        denormalize_actions(
                            action, encoding=_action_encoding(agent)
                        )
                    )
                    trail.append(np.asarray(ob, dtype=np.float32).copy())
                    done = bool(terminated or truncated)

                final_frame = env.render()
                env_writer.append_data(final_frame)
                fuel = float(ob[4]) if np.asarray(ob).shape[-1] > 4 else None
                refresh_field = cached_value_field is None or (
                    fuel is not None
                    and (
                        cached_fuel is None or abs(fuel - cached_fuel) > 1e-3
                    )
                )
                final_diagnostic = collect_agent_diagnostics(
                    agent,
                    ob,
                    goal,
                    seed=overlay_seed,
                    value_goal=value_goal,
                    compute_value_field=refresh_field,
                    compute_policy_diagnostics=not is_pathbridger,
                    temperature=temperature,
                )
                if "value_field" in final_diagnostic:
                    cached_value_field = final_diagnostic["value_field"]
                elif cached_value_field is not None:
                    final_diagnostic["value_field"] = cached_value_field
                if is_pathbridger:
                    final_diagnostic.update(cached_policy_diagnostic)
                if "subgoal" in final_diagnostic and not is_pathbridger:
                    subgoal_trail.append(
                        np.asarray(
                            final_diagnostic["subgoal"], dtype=np.float32
                        ).copy()
                    )
                overlay_writer.append_data(
                    compose_diagnostic_frame(
                        final_frame,
                        final_diagnostic,
                        arena_low=float(env.config.arena_low),
                        arena_high=float(env.config.arena_high),
                        trail=trail,
                        subgoal_trail=subgoal_trail,
                    )
                )
            paths.extend([env_path, overlay_path])
    finally:
        env.close()
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", choices=sorted(AGENTS), required=True)
    p.add_argument("--env", choices=list(ENVS), default="swingby_planet")
    p.add_argument("--dataset-size", choices=("1k", "10k", "100k"), default="100k")
    p.add_argument("--dataset", type=pathlib.Path, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--checkpoint-dir", type=pathlib.Path, default=None)
    p.add_argument("--render-dir", type=pathlib.Path, default=None)
    p.add_argument("--num-eval-envs", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset or _default_dataset(args.env, size=args.dataset_size)
    agent, _metrics = train(
        agent_name=args.agent,
        dataset_path=dataset,
        env_name=args.env,
        steps=args.steps,
        seed=args.seed,
        eval_every=args.eval_every,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        num_eval_envs=args.num_eval_envs,
    )
    if args.render_dir is not None:
        render_config = dict(getattr(agent, "config", {}) or {})
        render_data = _load_dataset(args.agent, dataset, render_config)
        value_goal_resolver = (
            _make_value_goal_resolver(render_data.next_observations)
            if args.agent in ("pbg", "pbf")
            else None
        )
        paths = render_agent(
            agent,
            output_dir=args.render_dir,
            env_name=args.env,
            seed=args.seed + args.steps,
            temperature=_eval_temperature(args.agent),
            value_goal_resolver=value_goal_resolver,
        )
        print(f"Rendered {len(paths)} videos to {args.render_dir}", flush=True)
        _collect_renders_gallery()


def _collect_renders_gallery() -> None:
    """Symlink new checkpoint videos into the flat renders/ gallery."""
    root = pathlib.Path(__file__).resolve().parents[1]
    script = root / "scripts" / "collect_renders.sh"
    if not script.is_file():
        return
    try:
        out = subprocess.run(
            ["bash", str(script)],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
        )
        if out.stdout.strip():
            print(out.stdout.rstrip(), flush=True)
        if out.returncode != 0 and out.stderr.strip():
            print(out.stderr.rstrip(), flush=True)
    except OSError as exc:
        print(f"collect_renders failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
