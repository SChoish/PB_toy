"""Train and evaluate goal-conditioned agents on CarParking datasets."""

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

from .env import CarParkingConfig, CarParkingEnv

ENV_NAME = "car_parking"
TASK_SCHEMA = "parking_v2"
SUPPORTED_AGENTS = ("hiql", "tr_hiql", "pbg", "pbf", "trl", "dqc")
TASK_IDS = (1, 2, 3, 4, 5)
STATE_DIM = 11
GOAL_DIM = 5


def _make_eval_env(
    *,
    render_mode: str | None = None,
    render_size: int = 256,
) -> CarParkingEnv:
    return CarParkingEnv(
        CarParkingConfig(maneuver="mixed", max_episode_steps=400),
        observation_mode="state",
        render_mode=render_mode,
        render_size=render_size,
    )


def _default_dataset(*, size: str = "100k") -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parent
        / "datasets"
        / f"car_parking_mixture_{size}.npz"
    )


def _to_jnp(batch: dict) -> dict:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _load_dataset(agent_name: str, dataset_path: pathlib.Path, config: dict):
    # Kept local so this training entry point remains importable while the
    # car_parking dataset adapters are being landed independently.
    try:
        from .datasets import (
            load_car_parking_dataset,
            load_car_parking_dqc_dataset,
            load_car_parking_trl_dataset,
        )
    except ImportError as exc:
        raise ImportError(
            "car_parking.datasets is required for training; expected "
            "load_car_parking_dataset, load_car_parking_trl_dataset, and "
            "load_car_parking_dqc_dataset"
        ) from exc

    if agent_name == "trl":
        return load_car_parking_trl_dataset(dataset_path, config=config)
    if agent_name == "dqc":
        return load_car_parking_dqc_dataset(dataset_path, config=config)
    return load_car_parking_dataset(
        dataset_path,
        path_horizon=int(config.get("subgoal_steps", 8)),
        action_chunk_horizon=int(config.get("action_chunk_horizon", 5)),
    )


def _validate_dataset(data, agent_name: str) -> None:
    if agent_name in ("trl", "dqc"):
        return
    observations = np.asarray(data.observations)
    actions = np.asarray(data.actions)
    if observations.ndim != 2 or observations.shape[1] != STATE_DIM:
        raise ValueError(
            f"CarParking observations must be [N, {STATE_DIM}], got "
            f"{observations.shape}"
        )
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"CarParking actions must be [N, 2], got {actions.shape}")


def _make_value_goal_resolver(
    data,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a full goal state without reusing stale goal-relative features."""
    from .datasets import achieved_goals, reconstruct_states

    physical_states = np.asarray(data.raw_next_observations, dtype=np.float32)
    support = np.unique(
        np.asarray(data.success_indices, dtype=np.int64)[
            np.asarray(data.success_indices) >= 0
        ]
    )
    if len(support) == 0:
        support = np.arange(len(physical_states), dtype=np.int64)
    features = achieved_goals(physical_states[support])
    scale = np.maximum(np.std(features, axis=0), 1e-3)
    cache: dict[bytes, np.ndarray] = {}

    def resolve(goal: np.ndarray) -> np.ndarray:
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal.size != GOAL_DIM:
            raise ValueError(f"Expected a {GOAL_DIM}-D goal, got {goal.shape}")
        key = goal.tobytes()
        if key not in cache:
            distance = np.sum(((features - goal) / scale) ** 2, axis=1)
            index = int(support[int(np.argmin(distance))])
            physical = physical_states[index].copy()
            physical[:4] = goal[:4]
            physical[7] = goal[4]
            cache[key] = reconstruct_states(
                physical,
                goal,
                data.slot_lengths[index],
                data.slot_widths[index],
                car_length=float(data.car_length),
                car_width=float(data.car_width),
                slot_margin=float(data.slot_margin),
            )
        return cache[key]

    return resolve


def _uses_action_chunks(agent) -> bool:
    return hasattr(agent, "sample_action_chunk")


def _eval_temperature(agent_name: str) -> float:
    return 1.0 if agent_name in ("pbg", "pbf", "trl", "dqc") else 0.0


def _eval_temperatures(agent_name: str) -> tuple[float, ...]:
    return (0.0, 1.0) if agent_name in ("pbg", "pbf") else (
        _eval_temperature(agent_name),
    )


def _temp_metric_prefix(temperature: float) -> str:
    if float(temperature) == 0.0:
        return "t0"
    if float(temperature) == 1.0:
        return "t1"
    return f"t{float(temperature):g}"


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
    observation_jnp = jnp.asarray(observation)[None]
    goal_jnp = jnp.asarray(goal)[None]
    if _uses_action_chunks(agent):
        if value_goal is None:
            raise ValueError("PathBridger requires a resolved full-state value goal")
        if action_chunk is None or chunk_index >= len(action_chunk):
            key = (
                None
                if temperature == 0.0
                else jax.random.PRNGKey(int(rng.integers(0, 1_000_000)))
            )
            action_chunk = np.asarray(
                agent.sample_action_chunk(
                    observation_jnp,
                    goal_jnp,
                    value_goals=jnp.asarray(value_goal)[None],
                    seed=key,
                    temperature=temperature,
                )
            )[0].astype(np.float32)
            action_chunk = np.clip(action_chunk, -1.0, 1.0)
            chunk_index = 0
        action = action_chunk[chunk_index]
        return action, action_chunk, chunk_index + 1

    key = (
        None
        if temperature == 0.0
        else jax.random.PRNGKey(int(rng.integers(0, 1_000_000)))
    )
    action = np.asarray(
        agent.sample_actions(
            observation_jnp,
            goal_jnp,
            seed=key,
            temperature=temperature,
        )
    )[0]
    return np.clip(action.astype(np.float32), -1.0, 1.0), None, 0


def evaluate(
    agent,
    *,
    task_ids: list[int] | None = None,
    episodes_per_task: int = 5,
    seed: int = 0,
    temperature: float = 0.0,
    value_goal_resolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    task_ids = task_ids or list(TASK_IDS)
    env = _make_eval_env()
    task_successes = {task_id: [] for task_id in task_ids}
    task_deaths = {task_id: [] for task_id in task_ids}
    cross_successes: list[float] = []
    cross_deaths: list[float] = []
    try:
        for episode in range(max(1, int(episodes_per_task))):
            episode_successes: list[float] = []
            episode_deaths: list[float] = []
            for task_id in task_ids:
                episode_seed = int(seed) + episode
                reset_seed = episode_seed * 10007 + int(task_id)
                rng = np.random.default_rng(reset_seed + 17)
                observation, info = env.reset(
                    seed=reset_seed,
                    options={"task_id": int(task_id)},
                )
                goal = np.asarray(info["goal"], dtype=np.float32)
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
                        observation,
                        goal,
                        value_goal=value_goal,
                        rng=rng,
                        temperature=temperature,
                        action_chunk=action_chunk,
                        chunk_index=chunk_index,
                    )
                    observation, _, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                success = float(info.get("is_success", False))
                death = float(info.get("dead", False))
                task_successes[task_id].append(success)
                task_deaths[task_id].append(death)
                episode_successes.append(success)
                episode_deaths.append(death)
            cross_successes.append(float(np.mean(episode_successes)))
            cross_deaths.append(float(np.mean(episode_deaths)))
    finally:
        env.close()

    results: dict[str, float] = {}
    for task_id in task_ids:
        success = np.asarray(task_successes[task_id], dtype=np.float64)
        death = np.asarray(task_deaths[task_id], dtype=np.float64)
        results[f"task{task_id}_success"] = float(success.mean())
        results[f"task{task_id}_success_std"] = float(success.std(ddof=0))
        results[f"task{task_id}_death"] = float(death.mean())
        results[f"task{task_id}_death_std"] = float(death.std(ddof=0))
    success = np.asarray(cross_successes, dtype=np.float64)
    death = np.asarray(cross_deaths, dtype=np.float64)
    results["mean_success"] = float(success.mean())
    results["mean_success_std"] = float(success.std(ddof=0))
    results["mean_death"] = float(death.mean())
    results["mean_death_std"] = float(death.std(ddof=0))
    results["eval_temperature"] = float(temperature)
    return results


def evaluate_suite(
    agent,
    *,
    seed: int,
    agent_name: str,
    num_eval_envs: int = 5,
    value_goal_resolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    episodes_per_task = max(1, int(num_eval_envs))
    temperatures = _eval_temperatures(agent_name)
    primary_temperature = _eval_temperature(agent_name)
    output: dict = {
        "num_eval_envs": episodes_per_task,
        "episodes_per_task": episodes_per_task,
        "total_eval_episodes": (
            episodes_per_task * len(TASK_IDS) * len(temperatures)
        ),
        "eval_temperature": primary_temperature,
        "eval_temperatures": list(temperatures),
        "eval_agg": "cross_task_unit_seeded",
    }
    for temperature in temperatures:
        metrics = evaluate(
            agent,
            episodes_per_task=episodes_per_task,
            seed=seed,
            temperature=temperature,
            value_goal_resolver=value_goal_resolver,
        )
        prefix = _temp_metric_prefix(temperature)
        for key, value in metrics.items():
            if key != "eval_temperature":
                output[f"{prefix}_{key}"] = value
        if float(temperature) == float(primary_temperature):
            output.update(metrics)
    return output


def format_eval_metrics(metrics: dict) -> str:
    count = int(metrics.get("episodes_per_task", 0))
    pieces = [f"n={count}"]
    for temperature in metrics.get(
        "eval_temperatures",
        [metrics.get("eval_temperature", 0.0)],
    ):
        prefix = _temp_metric_prefix(float(temperature))
        mean = float(
            metrics.get(
                f"{prefix}_mean_success",
                metrics.get("mean_success", 0.0),
            )
        )
        std = float(
            metrics.get(
                f"{prefix}_mean_success_std",
                metrics.get("mean_success_std", 0.0),
            )
        )
        tasks = " ".join(
            f"t{task_id}="
            f"{float(metrics.get(f'{prefix}_task{task_id}_success', metrics.get(f'task{task_id}_success', 0.0))):.2f}"
            for task_id in TASK_IDS
        )
        pieces.append(f"T={float(temperature):g} success={mean:.2f}±{std:.2f} {tasks}")
    return " ".join(pieces)


def save_checkpoint(
    agent,
    *,
    output_dir: pathlib.Path,
    agent_name: str,
    steps: int,
    metrics: dict,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"step_{steps}.msgpack"
    checkpoint_path.write_bytes(flax.serialization.to_bytes(agent))
    metadata = {
        "agent": agent_name,
        "env": ENV_NAME,
        "steps": int(steps),
        "config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in dict(agent.config).items()
        },
        "metrics": metrics,
    }
    (output_dir / f"step_{steps}.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path


def load_checkpoint(
    *,
    checkpoint_dir: pathlib.Path,
    agent_name: str,
    dataset_path: pathlib.Path,
    steps: int,
):
    metadata = json.loads(
        (checkpoint_dir / f"step_{steps}.json").read_text(encoding="utf-8")
    )
    if metadata.get("agent") not in (None, agent_name):
        raise ValueError(
            f"Checkpoint agent {metadata.get('agent')!r} does not match "
            f"{agent_name!r}"
        )
    config = dict(metadata["config"])
    if isinstance(config.get("hidden_dims"), list):
        config["hidden_dims"] = tuple(config["hidden_dims"])
    config["goal_dim"] = GOAL_DIM
    config["env_name"] = ENV_NAME
    config["task_schema"] = TASK_SCHEMA
    if agent_name in ("pbg", "pbf"):
        fresh = DEFAULT_CONFIGS[agent_name]()
        config["subgoal_eval_num_samples"] = int(
            fresh["subgoal_eval_num_samples"]
        )
        config["phi_goal_obs_indices"] = (0, 1, 2, 3, 10)
        config["subgoal_value_goal_representation"] = "full"
        config["value_distance_weight_power"] = 0.0
        config["env_name"] = ENV_NAME
    if agent_name == "tr_hiql":
        config["distance_weight_power"] = 0.0
    data = _load_dataset(agent_name, dataset_path, config)
    _validate_dataset(data, agent_name)
    if agent_name in ("trl", "dqc"):
        example = _to_jnp(data.sample(np.random.default_rng(0), 8))
        template = AGENTS[agent_name].create(0, example, config)
    else:
        template = AGENTS[agent_name].create(
            0,
            data.observations[:8],
            data.actions[:8],
            config,
        )
    checkpoint = (checkpoint_dir / f"step_{steps}.msgpack").read_bytes()
    return flax.serialization.from_bytes(template, checkpoint), metadata


def latest_checkpoint_step(checkpoint_dir: pathlib.Path | None) -> int | None:
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return None
    steps: list[int] = []
    for checkpoint in checkpoint_dir.glob("step_*.msgpack"):
        try:
            step = int(checkpoint.stem.removeprefix("step_"))
        except ValueError:
            continue
        if (checkpoint_dir / f"step_{step}.json").is_file():
            steps.append(step)
    return max(steps) if steps else None


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
    num_eval_envs: int = 5,
    resume: bool = True,
) -> tuple[object, dict]:
    if agent_name not in SUPPORTED_AGENTS:
        raise SystemExit(
            f"Unsupported agent {agent_name!r}; choose from {SUPPORTED_AGENTS}"
        )
    if env_name != ENV_NAME:
        raise SystemExit(f"Unsupported env {env_name!r}; expected {ENV_NAME!r}")
    if steps < 0 or log_every < 1 or eval_every < 0 or num_eval_envs < 1:
        raise SystemExit(
            "--steps and --eval-every must be non-negative; "
            "--log-every and --num-eval-envs must be positive"
        )

    config = DEFAULT_CONFIGS[agent_name]()
    if config_overrides:
        config.update(config_overrides)
    config["goal_dim"] = GOAL_DIM
    config["env_name"] = ENV_NAME
    config["task_schema"] = TASK_SCHEMA
    if agent_name in ("pbg", "pbf"):
        config["phi_goal_obs_indices"] = (0, 1, 2, 3, 10)
        config["subgoal_value_goal_representation"] = "full"
        config["value_distance_weight_power"] = 0.0
    if agent_name == "tr_hiql":
        config["distance_weight_power"] = 0.0

    data = _load_dataset(agent_name, dataset_path, config)
    _validate_dataset(data, agent_name)
    value_goal_resolver = (
        _make_value_goal_resolver(data)
        if agent_name in ("pbg", "pbf")
        else None
    )
    print(
        f"Loaded {agent_name} CarParking dataset size={len(data)} "
        f"from {dataset_path}",
        flush=True,
    )

    start_step = 0
    metrics: dict = {}
    latest = latest_checkpoint_step(checkpoint_dir) if resume else None
    if latest is not None:
        agent, metadata = load_checkpoint(
            checkpoint_dir=pathlib.Path(checkpoint_dir),
            agent_name=agent_name,
            dataset_path=dataset_path,
            steps=latest,
        )
        metrics = dict(metadata.get("metrics") or {})
        if latest >= steps:
            print(
                f"[{agent_name}] checkpoint already complete at step={latest}",
                flush=True,
            )
            return agent, metrics
        start_step = latest
        config = dict(metadata.get("config") or config)
        if isinstance(config.get("hidden_dims"), list):
            config["hidden_dims"] = tuple(config["hidden_dims"])
        print(f"[{agent_name}] resuming from step={latest} → {steps}", flush=True)
    else:
        init_rng = np.random.default_rng(seed)
        if agent_name in ("trl", "dqc"):
            example = _to_jnp(data.sample(init_rng, 8))
            agent = AGENTS[agent_name].create(seed, example, config)
        else:
            agent = AGENTS[agent_name].create(
                seed,
                data.observations[:8],
                data.actions[:8],
                config,
            )

    rng = np.random.default_rng(seed + start_step)
    started = time.time()
    for step in range(start_step + 1, steps + 1):
        batch = _to_jnp(data.sample(rng, int(config["batch_size"])))
        agent, info = agent.update(batch)
        if step == start_step + 1 or step % log_every == 0:
            scalar_info = {
                key: float(value)
                for key, value in info.items()
                if np.ndim(np.asarray(value)) == 0
            }
            print(f"[{agent_name}] step={step} {scalar_info}", flush=True)
        if eval_every > 0 and step % eval_every == 0:
            metrics = evaluate_suite(
                agent,
                seed=seed + step,
                agent_name=agent_name,
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

    if not metrics or eval_every <= 0 or steps % eval_every != 0:
        metrics = evaluate_suite(
            agent,
            seed=seed + steps,
            agent_name=agent_name,
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
        f"[{agent_name}] final eval {format_eval_metrics(metrics)} "
        f"({time.time() - started:.1f}s)",
        flush=True,
    )
    return agent, metrics


def render_agent(
    agent,
    *,
    output_dir: pathlib.Path,
    seed: int = 0,
    render_size: int = 256,
    temperature: float = 0.0,
    value_goal_resolver: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[pathlib.Path]:
    """Best-effort RGB-array rendering; unavailable video support is non-fatal."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        print(f"Rendering skipped: imageio is unavailable ({exc})", flush=True)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []
    env = _make_eval_env(render_mode="rgb_array", render_size=render_size)
    try:
        for task_id in TASK_IDS:
            path = output_dir / f"task{task_id}.mp4"
            try:
                observation, info = env.reset(
                    seed=seed * 10007 + task_id,
                    options={"task_id": task_id},
                )
                goal = np.asarray(info["goal"], dtype=np.float32)
                value_goal = (
                    value_goal_resolver(goal)
                    if _uses_action_chunks(agent)
                    and value_goal_resolver is not None
                    else None
                )
                rng = np.random.default_rng(seed + task_id)
                action_chunk = None
                chunk_index = 0
                with imageio.get_writer(
                    path,
                    fps=int(env.metadata.get("render_fps", 20)),
                    codec="libx264",
                ) as writer:
                    done = False
                    while not done:
                        frame = env.render()
                        if frame is not None:
                            writer.append_data(frame)
                        action, action_chunk, chunk_index = _select_action(
                            agent,
                            observation,
                            goal,
                            value_goal=value_goal,
                            rng=rng,
                            temperature=temperature,
                            action_chunk=action_chunk,
                            chunk_index=chunk_index,
                        )
                        observation, _, terminated, truncated, _ = env.step(action)
                        done = bool(terminated or truncated)
                    frame = env.render()
                    if frame is not None:
                        writer.append_data(frame)
                paths.append(path)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                path.unlink(missing_ok=True)
                print(f"Rendering task {task_id} skipped: {exc}", flush=True)
    finally:
        env.close()
    return paths


def _horizon_overrides(
    agent_name: str,
    *,
    subgoal_steps: int | None,
    action_chunk_horizon: int | None,
) -> dict:
    overrides: dict = {}
    if subgoal_steps is not None:
        if subgoal_steps < 1:
            raise SystemExit("--subgoal-steps must be >= 1")
        overrides["subgoal_steps"] = subgoal_steps
        if agent_name in ("pbg", "pbf"):
            overrides["dynamics_N"] = subgoal_steps
            overrides["full_chunk_horizon"] = subgoal_steps
    if action_chunk_horizon is not None:
        if action_chunk_horizon < 1:
            raise SystemExit("--action-chunk-horizon must be >= 1")
        if agent_name in ("pbg", "pbf"):
            overrides["action_chunk_horizon"] = action_chunk_horizon
            overrides["forward_bridge_path_loss_horizon"] = (
                action_chunk_horizon
            )
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=(ENV_NAME,), default=ENV_NAME)
    parser.add_argument("--agent", choices=SUPPORTED_AGENTS, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, default=None)
    parser.add_argument(
        "--dataset-size",
        choices=("1k", "10k", "100k"),
        default="100k",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--num-eval-envs", type=int, default=5)
    parser.add_argument("--subgoal-steps", type=int, default=None)
    parser.add_argument("--action-chunk-horizon", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=None)
    parser.add_argument("--render-dir", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or _default_dataset(size=args.dataset_size)
    overrides = _horizon_overrides(
        args.agent,
        subgoal_steps=args.subgoal_steps,
        action_chunk_horizon=args.action_chunk_horizon,
    )
    agent, _ = train(
        agent_name=args.agent,
        dataset_path=dataset_path,
        env_name=args.env,
        steps=args.steps,
        seed=args.seed,
        eval_every=args.eval_every,
        log_every=args.log_every,
        config_overrides=overrides or None,
        checkpoint_dir=args.checkpoint_dir,
        num_eval_envs=args.num_eval_envs,
    )
    if args.render_dir is not None:
        render_config = dict(getattr(agent, "config", {}) or {})
        render_data = _load_dataset(args.agent, dataset_path, render_config)
        value_goal_resolver = (
            _make_value_goal_resolver(render_data)
            if args.agent in ("pbg", "pbf")
            else None
        )
        paths = render_agent(
            agent,
            output_dir=args.render_dir,
            seed=args.seed + args.steps,
            temperature=_eval_temperature(args.agent),
            value_goal_resolver=value_goal_resolver,
        )
        print(f"Rendered {len(paths)} videos to {args.render_dir}", flush=True)


if __name__ == "__main__":
    main()
