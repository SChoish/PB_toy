"""Public package exports for the orbital swing-by environment."""

from __future__ import annotations

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
