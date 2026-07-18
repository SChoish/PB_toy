"""Backward-compatible facade for the split orbital swing-by modules.

Prefer importing from this module or the package root::

    from orbital_swingby_env import OrbitalSwingByEnv, planet_config
    # or
    from swingby import OrbitalSwingByEnv
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .config import (
        GRAVITY_MODEL_ALIASES,
        BodyKind,
        GravityModel,
        ObservationMode,
        OrbitalSwingByConfig,
        OrbitalSwingbyConfig,
        RewardMode,
        TaskMode,
        black_hole_config,
        planet_config,
    )
    from .env import (
        OrbitalSwingByEnv,
        OrbitalSwingbyEnv,
        register_environment,
        register_environments,
    )
except ImportError:
    from config import (
        GRAVITY_MODEL_ALIASES,
        BodyKind,
        GravityModel,
        ObservationMode,
        OrbitalSwingByConfig,
        OrbitalSwingbyConfig,
        RewardMode,
        TaskMode,
        black_hole_config,
        planet_config,
    )
    from env import (
        OrbitalSwingByEnv,
        OrbitalSwingbyEnv,
        register_environment,
        register_environments,
    )

__all__ = [
    "GRAVITY_MODEL_ALIASES",
    "BodyKind",
    "GravityModel",
    "ObservationMode",
    "OrbitalSwingByConfig",
    "OrbitalSwingByEnv",
    "OrbitalSwingbyConfig",
    "OrbitalSwingbyEnv",
    "RewardMode",
    "TaskMode",
    "black_hole_config",
    "planet_config",
    "register_environment",
    "register_environments",
]


def _smoke_test() -> None:
    from gymnasium.utils.env_checker import check_env

    env = OrbitalSwingByEnv(
        config=planet_config(max_episode_steps=120),
        observation_mode="goal_dict",
    )
    check_env(env, skip_render_check=True)

    env.reset(
        seed=0,
        options={
            "position": np.array([-0.82, 0.55], dtype=np.float32),
            "velocity": np.array([0.35, 0.0], dtype=np.float32),
            "goal": np.array([0.75, 0.55], dtype=np.float32),
            "goal_velocity": np.array([0.35, 0.0], dtype=np.float32),
        },
    )
    fuel0 = env.fuel
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert np.isclose(env.fuel, fuel0)
    env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert env.fuel < fuel0

    collision = OrbitalSwingByEnv(
        config=planet_config(max_episode_steps=80),
        observation_mode="state",
    )
    collision.reset(
        seed=1,
        options={
            "position": np.array([-0.34, 0.0], dtype=np.float32),
            "velocity": np.array([0.65, 0.0], dtype=np.float32),
            "goal": np.array([0.70, 0.70], dtype=np.float32),
            "goal_velocity": np.zeros(2, dtype=np.float32),
        },
    )
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = collision.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
    assert info["dead"] and info["termination_reason"] == "body_collision"

    black_hole = OrbitalSwingByEnv(
        config=black_hole_config(max_episode_steps=80),
        observation_mode="state_goal",
    )
    black_hole.reset(
        seed=2,
        options={
            "position": np.array([-0.34, 0.0], dtype=np.float32),
            "velocity": np.array([0.52, 0.0], dtype=np.float32),
            "goal": np.array([0.70, 0.60], dtype=np.float32),
            "goal_velocity": np.zeros(2, dtype=np.float32),
        },
    )
    accel = black_hole.gravity_acceleration()
    assert accel[0] > 0.0 and abs(accel[1]) < 1e-6
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = black_hole.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
    assert info["dead"] and info["termination_reason"] == "event_horizon"

    # Alias gravity models and documentation-facing class names.
    aliased = OrbitalSwingbyEnv(
        config=OrbitalSwingbyConfig(
            body_kind="black_hole",
            gravity_model="pseudo_schwarzschild",
            body_radius=0.115,
            schwarzschild_radius=0.060,
            gravitational_parameter=0.048,
        )
    )
    assert aliased.config.gravity_model == "paczynski_wiita"

    preview = OrbitalSwingByEnv(
        config=planet_config(show_ballistic_prediction=True),
        render_mode="rgb_array",
    )
    preview.reset(seed=3, options={"task_id": 2})
    frame = preview.render()
    assert frame is not None and frame.shape == (640, 640, 3)
    assert frame.dtype == np.uint8

    env.close()
    collision.close()
    black_hole.close()
    aliased.close()
    preview.close()
    print("OrbitalSwingByEnv smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
