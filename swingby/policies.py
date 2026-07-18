"""Behavior policies for offline orbital swing-by dataset collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    from .env import OrbitalSwingByEnv
except ImportError:  # script-style: `cd swingby && python ...`
    from env import OrbitalSwingByEnv

PolicyName = Literal["expert", "noisy", "random"]
Array = np.ndarray


@dataclass
class PolicyState:
    held_action: Array | None = None
    hold_steps: int = 0
    passed_periapsis: bool = False
    min_radius: float = np.inf


def physical_observation(env: OrbitalSwingByEnv) -> Array:
    """Task-agnostic physical state: [x, y, vx, vy, fuel_fraction]."""
    return env.state.astype(np.float32, copy=True)


def commanded_goal(env: OrbitalSwingByEnv) -> Array:
    return env.desired_goal.astype(np.float32, copy=True)


def _coast_action() -> Array:
    return np.array([0.0, 0.0], dtype=np.float32)


def _update_periapsis(env: OrbitalSwingByEnv, state: PolicyState) -> None:
    radius = float(np.linalg.norm(env.position - env.body_center))
    state.min_radius = min(state.min_radius, radius)
    relative = env.position - env.body_center
    radial_speed = float(np.dot(relative, env.velocity) / max(radius, 1e-8))
    # Leave periapsis once radius starts increasing after a close approach.
    if (
        env.elapsed_steps > 10
        and radius < state.min_radius + 0.08
        and radial_speed > 0.01
        and env.position[0] > -0.35
    ):
        state.passed_periapsis = True


def expert_action(
    env: OrbitalSwingByEnv,
    state: PolicyState,
    *,
    aggressive: bool = False,
) -> Array:
    """Coast through periapsis, then phase-space PD correction burns."""
    _update_periapsis(env, state)

    position_error = env.goal - env.position
    velocity_error = env.goal_velocity - env.velocity
    distance = float(np.linalg.norm(position_error))
    speed_error = float(np.linalg.norm(velocity_error))

    # Pure coast while inbound if the ballistic goal is already close enough.
    if not state.passed_periapsis and not aggressive:
        return _coast_action()

    # Near goal: prioritize velocity matching with a small position pull.
    position_gain = 1.15 if aggressive else 0.75
    velocity_gain = 1.55 if aggressive else 1.25
    if distance < env.config.goal_radius * 2.5:
        position_gain *= 1.35
        velocity_gain *= 1.45

    desired_acceleration = (
        position_gain * position_error
        + velocity_gain * velocity_error
        - env.gravity_acceleration()
    )
    magnitude = float(np.linalg.norm(desired_acceleration))

    # Save fuel when already on a good coasting intercept.
    if (
        not aggressive
        and distance < env.config.goal_radius * 1.8
        and speed_error < env.config.goal_velocity_tolerance * 0.85
    ):
        return _coast_action()
    if magnitude <= 1e-8 or env.fuel <= 0.0:
        return _coast_action()

    angle = float(np.arctan2(desired_acceleration[1], desired_acceleration[0]))
    throttle = float(
        np.clip(
            magnitude * env.mass / env.config.max_thrust_force,
            0.0,
            # Normal episodes should make correction burns without overpowering
            # the gravity field. Aggressive episodes still exercise full thrust.
            1.0 if aggressive else 0.65,
        )
    )
    # Soft gate: avoid tiny thruster chatter.
    if throttle < 0.04:
        return _coast_action()
    return np.array([angle, throttle], dtype=np.float32)


def _held_random_action(
    rng: np.random.Generator,
    state: PolicyState,
    *,
    min_hold: int,
    max_hold: int,
) -> Array:
    if state.hold_steps <= 0 or state.held_action is None:
        # Bias toward coasting; pure random thrust escapes too often.
        if rng.random() < 0.55:
            state.held_action = _coast_action()
        else:
            state.held_action = np.array(
                [rng.uniform(-np.pi, np.pi), rng.uniform(0.0, 0.65)],
                dtype=np.float32,
            )
        state.hold_steps = int(rng.integers(min_hold, max_hold + 1))
    state.hold_steps -= 1
    return state.held_action.copy()


def behavior_action(
    env: OrbitalSwingByEnv,
    policy: PolicyName,
    rng: np.random.Generator,
    state: PolicyState,
    *,
    aggressive: bool,
    noise: float,
) -> tuple[Array, bool]:
    """Return action and whether it came from a random burst."""
    if policy == "random":
        return _held_random_action(rng, state, min_hold=4, max_hold=18), True

    if policy == "noisy":
        if state.hold_steps <= 0 and rng.random() < 0.04:
            state.held_action = np.array(
                [rng.uniform(-np.pi, np.pi), rng.uniform(0.0, 0.7)],
                dtype=np.float32,
            )
            state.hold_steps = int(rng.integers(4, 12))
        if state.hold_steps > 0:
            state.hold_steps -= 1
            assert state.held_action is not None
            return state.held_action.copy(), True

    action = expert_action(env, state, aggressive=aggressive)
    if policy == "noisy" and noise > 0.0:
        action = action.copy()
        action[0] = float(action[0] + rng.normal(0.0, noise * 0.9))
        action[1] = float(action[1] + rng.normal(0.0, noise * 0.35))
        # Wrap angle into [-pi, pi].
        action[0] = float((action[0] + np.pi) % (2.0 * np.pi) - np.pi)
        action[1] = float(np.clip(action[1], 0.0, 1.0))
    return action.astype(np.float32), False
