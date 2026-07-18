"""Agent diagnostic overlays for rendered Hazard2D frames.

This module does not modify or subclass the environment.  It post-processes
the RGB array returned by ``env.render()`` with model-side diagnostics.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from agents.critic import pick_best_candidates, score_transitive_ratio


def _has_module(agent: Any, name: str) -> bool:
    modules = getattr(getattr(agent.network, "model_def", None), "modules", None)
    return isinstance(modules, dict) and name in modules


def _value_field(
    agent: Any,
    goal: np.ndarray,
    grid_size: int,
    *,
    state_dim: int,
    condition_state: np.ndarray | None = None,
) -> np.ndarray | None:
    """Evaluate V on a world-space (x, y) grid.

    Non-(x, y) coordinates default to 0. When ``condition_state`` is given
    (e.g. car_race with health), those coordinates are copied from it so the
    slice is V(x, y | o_{¬xy}, g) rather than V(x, y, 0, ...).
    """
    xs = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    ys = np.linspace(1.0, -1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    states = np.zeros((grid_size * grid_size, int(state_dim)), dtype=np.float32)
    if condition_state is not None:
        cond = np.asarray(condition_state, dtype=np.float32).reshape(-1)
        if cond.shape[0] == int(state_dim):
            states[:] = cond[None, :]
    states[:, 0] = xx.reshape(-1)
    states[:, 1] = yy.reshape(-1)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    goals = np.broadcast_to(goal[None], (states.shape[0], goal.shape[0]))
    states_j = jnp.asarray(states)
    goals_j = jnp.asarray(goals)

    config = dict(agent.config)
    if "dynamics_N" in config:
        values = jax.nn.sigmoid(
            agent.network.select("value")(states_j, goals_j)
        )
    elif hasattr(agent, "_sigmoid_v"):
        values = agent._sigmoid_v(states_j, goals_j)
    elif hasattr(agent, "_value"):
        heads = agent._value(states_j, goals_j)
        values = jnp.mean(heads, axis=0)
    else:
        return None
    return np.asarray(values, dtype=np.float32).reshape(grid_size, grid_size)


def _decode_hiql_subgoal(
    agent: Any,
    observation: np.ndarray,
    predicted_rep: np.ndarray,
    *,
    grid_size: int = 33,
) -> np.ndarray:
    """Map HIQL latent high-actor output to a physical xy via nearest goal_rep."""
    xs = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    goal_dim = int(agent.config.get("goal_dim", observation.shape[-1]))
    candidates = np.zeros((grid_size * grid_size, goal_dim), dtype=np.float32)
    candidates[:, 0] = xx.reshape(-1)
    candidates[:, 1] = yy.reshape(-1)
    obs = np.broadcast_to(
        np.asarray(observation, dtype=np.float32),
        (candidates.shape[0], observation.shape[-1]),
    )
    reps = np.asarray(
        agent._goal_rep(jnp.asarray(obs), jnp.asarray(candidates)),
        dtype=np.float32,
    )
    pred = np.asarray(predicted_rep, dtype=np.float32).reshape(-1)
    idx = int(np.argmin(np.sum((reps - pred[None]) ** 2, axis=-1)))
    out = np.zeros(observation.shape[-1], dtype=np.float32)
    out[:goal_dim] = candidates[idx]
    return out


def collect_agent_diagnostics(
    agent: Any,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    seed: jax.Array,
    grid_size: int = 48,
    compute_value_field: bool = True,
    compute_policy_diagnostics: bool = True,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """Collect value field and agent-specific subgoal/path predictions."""
    diagnostics: dict[str, np.ndarray] = {}
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    if compute_value_field:
        # Condition on current non-xy features when the state carries them
        # (car_race health/heading/...). Hazard 4D keeps the zero-velocity slice.
        state_dim = int(observation.shape[-1])
        condition = observation if state_dim > 4 else None
        field = _value_field(
            agent,
            goal,
            grid_size,
            state_dim=state_dim,
            condition_state=condition,
        )
        if field is not None:
            diagnostics["value_field"] = field

    if not compute_policy_diagnostics:
        return diagnostics

    obs_j = jnp.asarray(observation, dtype=jnp.float32)[None]
    goal_j = jnp.asarray(goal, dtype=jnp.float32)[None]
    config = dict(agent.config)
    # Prefer method presence over config keys: sweeps may inject horizons
    # (e.g. dynamics_N) into non-PathBridger agents.
    if hasattr(agent, "_sample_candidates") and hasattr(agent, "_plan"):
        candidates, _ = agent._sample_candidates(
            obs_j, goal_j, seed, temperature=float(temperature)
        )
        scores = score_transitive_ratio(
            agent.network, obs_j, candidates, goal_j
        )
        selected = pick_best_candidates(candidates, scores)
        planned = agent._plan(obs_j, selected)
        diagnostics["subgoal"] = np.asarray(selected[0])
        diagnostics["planned_path"] = np.asarray(planned[0])
        # PBF (flow) only shows the chosen subgoal; PBG keeps candidates.
        if str(config.get("subgoal_distribution", "")).lower() != "flow":
            diagnostics["candidates"] = np.asarray(candidates[0])
            diagnostics["candidate_scores"] = np.asarray(scores[0])
    elif _has_module(agent, "high_actor") and hasattr(agent, "_sigmoid_v"):
        # TR-HIQL predicts a physical state as its high-level subgoal.
        high = agent.network.select("high_actor")(obs_j, goal_j)
        diagnostics["subgoal"] = np.asarray(high.mode()[0])
    elif (
        _has_module(agent, "high_actor")
        and hasattr(agent, "_goal_rep")
        and "rep_dim" in config
    ):
        # HIQL high actor predicts a latent; decode to nearest physical xy.
        high = agent.network.select("high_actor")(obs_j, goal_j)
        rep = high.mode()[0]
        rep = rep / jnp.linalg.norm(rep) * jnp.sqrt(rep.shape[-1])
        diagnostics["subgoal"] = _decode_hiql_subgoal(
            agent, observation, np.asarray(rep), grid_size=33
        )

    return diagnostics


def _normalize_field(field: np.ndarray) -> np.ndarray:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return np.zeros_like(field)
    lo, hi = np.percentile(finite, [5.0, 95.0])
    if hi - lo < 1e-6:
        return np.zeros_like(field)
    return np.clip((field - lo) / (hi - lo), 0.0, 1.0)


def _value_colormap(values: np.ndarray) -> np.ndarray:
    """Dark blue -> cyan -> yellow, matching the concept illustration."""
    t = _normalize_field(values)[..., None]
    blue = np.array([25.0, 75.0, 135.0], dtype=np.float32)
    cyan = np.array([85.0, 175.0, 195.0], dtype=np.float32)
    yellow = np.array([250.0, 185.0, 45.0], dtype=np.float32)
    lower = blue + np.minimum(t * 2.0, 1.0) * (cyan - blue)
    upper = cyan + np.maximum(t * 2.0 - 1.0, 0.0) * (yellow - cyan)
    return np.where(t <= 0.5, lower, upper).astype(np.uint8)


def compose_diagnostic_frame(
    frame: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    *,
    arena_low: float,
    arena_high: float,
    trail: list[np.ndarray] | None = None,
    subgoal_trail: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Blend diagnostics onto an environment RGB frame."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    width, height = image.size

    field = diagnostics.get("value_field")
    if field is not None:
        heat = Image.fromarray(_value_colormap(field)).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        image = Image.blend(image, heat, alpha=0.42)

    draw = ImageDraw.Draw(image)
    span = float(arena_high - arena_low)

    def pixel(state: np.ndarray) -> tuple[int, int]:
        x, y = np.asarray(state)[:2]
        col = int(np.clip(round((x - arena_low) / span * (width - 1)), 0, width - 1))
        row = int(np.clip(round((arena_high - y) / span * (height - 1)), 0, height - 1))
        return col, row

    # Agent execution trail (purple).
    if trail and len(trail) >= 2:
        draw.line([pixel(p) for p in trail], fill=(86, 25, 150), width=4)

    # History of estimated subgoals (orange) — distinct from agent / bridge.
    if subgoal_trail and len(subgoal_trail) >= 2:
        draw.line([pixel(p) for p in subgoal_trail], fill=(255, 120, 35), width=4)

    planned = diagnostics.get("planned_path")
    if planned is not None and len(planned) >= 2:
        draw.line([pixel(p) for p in planned], fill=(20, 20, 20), width=4)

    candidates = diagnostics.get("candidates")
    if candidates is not None:
        scores = diagnostics.get("candidate_scores")
        best = int(np.argmax(scores)) if scores is not None else 0
        for index, candidate in enumerate(candidates):
            x, y = pixel(candidate)
            radius = 4 if index != best else 6
            fill = (190, 55, 210) if index != best else (255, 255, 255)
            outline = (125, 25, 170)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=fill,
                outline=outline,
                width=2,
            )

    subgoal = diagnostics.get("subgoal")
    if subgoal is not None:
        # Highlight current subgoal and a short agent→subgoal segment.
        if trail:
            draw.line(
                [pixel(trail[-1]), pixel(subgoal)],
                fill=(255, 120, 35),
                width=3,
            )
        x, y = pixel(subgoal)
        radius = 8
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255),
            outline=(255, 120, 35),
            width=4,
        )

    return np.asarray(image, dtype=np.uint8)
