"""Continuous 2-D point-mass environment with a lethal circular hazard."""

from .env import (
    ContinuousHazard2DEnv,
    Hazard2DConfig,
    register_environment,
)

__all__ = [
    "ContinuousHazard2DEnv",
    "Hazard2DConfig",
    "register_environment",
]
