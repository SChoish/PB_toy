"""Geometry, gravity, fuel, task, and reward configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

ObservationMode = Literal["state", "state_goal", "goal_dict"]
RewardMode = Literal["sparse", "dense"]
TaskMode = Literal["random", "swingby", "orbit_transfer"]
BodyKind = Literal["planet", "black_hole"]
GravityModel = Literal[
    "newtonian",
    "paczynski_wiita",
    # Aliases kept for readability / external docs.
    "inverse_square",
    "pseudo_schwarzschild",
]

GRAVITY_MODEL_ALIASES: dict[str, str] = {
    "newtonian": "newtonian",
    "inverse_square": "newtonian",
    "paczynski_wiita": "paczynski_wiita",
    "pseudo_schwarzschild": "paczynski_wiita",
}


@dataclass(frozen=True)
class OrbitalSwingByConfig:
    """Geometry, orbital dynamics, fuel, task, and reward parameters."""

    # Geometry
    arena_low: float = -1.0
    arena_high: float = 1.0
    body_center: tuple[float, float] = (0.0, 0.0)
    body_kind: BodyKind = "planet"
    body_radius: float = 0.115
    satellite_radius: float = 0.018
    goal_radius: float = 0.075
    goal_velocity_tolerance: float = 0.35
    goal_requires_velocity_match: bool = True
    spawn_clearance: float = 0.07

    # Gravity and integration
    gravity_model: GravityModel = "newtonian"
    gravitational_parameter: float = 0.085  # normalized GM
    schwarzschild_radius: float = 0.060
    gravity_softening: float = 0.012
    max_gravity_acceleration: float = 12.0  # numerical safety cap
    dt: float = 0.04
    physics_substeps: int = 8
    max_speed: float = 3.0  # terminal numerical safety bound, not a clip

    # Thruster and fuel
    dry_mass: float = 1.0
    fuel_capacity: float = 1.0
    initial_fuel: float = 1.0
    fuel_mass_scale: float = 0.30
    max_thrust_force: float = 0.15
    fuel_burn_rate: float = 0.115  # fuel units / second at full throttle

    # Episode and task distribution
    max_episode_steps: int = 700
    min_start_goal_distance: float = 1.10
    task_mode: TaskMode = "swingby"
    initial_speed_low: float = 0.32
    initial_speed_high: float = 0.50
    target_speed_low: float = 0.20
    target_speed_high: float = 0.52

    # Reward
    reward_mode: RewardMode = "dense"
    step_penalty: float = 0.002
    fuel_penalty_scale: float = 0.10
    progress_scale: float = 1.0
    velocity_progress_scale: float = 0.08
    success_reward: float = 2.0
    remaining_fuel_bonus: float = 1.0
    body_collision_penalty: float = -2.0
    escape_penalty: float = -1.5

    # Rendering options
    show_ballistic_prediction: bool = False

    @property
    def canonical_gravity_model(self) -> str:
        return GRAVITY_MODEL_ALIASES.get(self.gravity_model, self.gravity_model)

    def normalized(self) -> OrbitalSwingByConfig:
        """Return a copy with gravity-model aliases resolved."""
        canonical = self.canonical_gravity_model
        if canonical == self.gravity_model:
            return self
        return replace(self, gravity_model=canonical)  # type: ignore[arg-type]

    def validate(self) -> None:
        if not self.arena_low < self.arena_high:
            raise ValueError("arena_low must be smaller than arena_high")
        if self.body_kind not in ("planet", "black_hole"):
            raise ValueError(f"Unknown body_kind: {self.body_kind}")
        if self.gravity_model not in GRAVITY_MODEL_ALIASES:
            raise ValueError(f"Unknown gravity_model: {self.gravity_model}")
        if self.body_radius <= 0.0:
            raise ValueError("body_radius must be positive")
        if self.satellite_radius <= 0.0:
            raise ValueError("satellite_radius must be positive")
        if self.goal_radius <= 0.0:
            raise ValueError("goal_radius must be positive")
        if self.goal_velocity_tolerance <= 0.0:
            raise ValueError("goal_velocity_tolerance must be positive")
        if self.spawn_clearance < 0.0:
            raise ValueError("spawn_clearance must be non-negative")
        if self.gravitational_parameter < 0.0:
            raise ValueError("gravitational_parameter must be non-negative")
        if self.schwarzschild_radius <= 0.0:
            raise ValueError("schwarzschild_radius must be positive")
        if self.canonical_gravity_model == "paczynski_wiita" and not (
            self.schwarzschild_radius < self.body_radius
        ):
            raise ValueError(
                "paczynski_wiita requires schwarzschild_radius < body_radius "
                "so the episode terminates before the potential singularity"
            )
        if self.gravity_softening <= 0.0:
            raise ValueError("gravity_softening must be positive")
        if self.max_gravity_acceleration <= 0.0:
            raise ValueError("max_gravity_acceleration must be positive")
        if self.dt <= 0.0 or self.physics_substeps < 1:
            raise ValueError("dt must be positive and physics_substeps >= 1")
        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive")
        if self.dry_mass <= 0.0:
            raise ValueError("dry_mass must be positive")
        if self.fuel_capacity <= 0.0:
            raise ValueError("fuel_capacity must be positive")
        if not 0.0 <= self.initial_fuel <= self.fuel_capacity:
            raise ValueError("initial_fuel must be in [0, fuel_capacity]")
        if self.fuel_mass_scale < 0.0:
            raise ValueError("fuel_mass_scale must be non-negative")
        if self.max_thrust_force <= 0.0:
            raise ValueError("max_thrust_force must be positive")
        if self.fuel_burn_rate <= 0.0:
            raise ValueError("fuel_burn_rate must be positive")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if self.min_start_goal_distance < 0.0:
            raise ValueError("min_start_goal_distance must be non-negative")
        if self.task_mode not in ("random", "swingby", "orbit_transfer"):
            raise ValueError(f"Unknown task_mode: {self.task_mode}")
        if not 0.0 <= self.initial_speed_low <= self.initial_speed_high:
            raise ValueError("invalid initial speed interval")
        if not 0.0 <= self.target_speed_low <= self.target_speed_high:
            raise ValueError("invalid target speed interval")
        if self.reward_mode not in ("sparse", "dense"):
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        center = np.asarray(self.body_center, dtype=np.float64)
        if center.shape != (2,):
            raise ValueError("body_center must have two coordinates")
        margin = self.body_radius + self.satellite_radius
        if np.any(center - margin <= self.arena_low) or np.any(
            center + margin >= self.arena_high
        ):
            raise ValueError("central body must fit inside the arena")


# Public alias matching external documentation naming.
OrbitalSwingbyConfig = OrbitalSwingByConfig


def planet_config(**overrides: Any) -> OrbitalSwingByConfig:
    """Convenience factory for a Newtonian planet task."""
    base = OrbitalSwingByConfig(
        body_kind="planet",
        gravity_model="newtonian",
        body_radius=0.115,
        gravitational_parameter=0.085,
    )
    return replace(base, **overrides).normalized()


def black_hole_config(**overrides: Any) -> OrbitalSwingByConfig:
    """Convenience factory for a pseudo-Schwarzschild black-hole task."""
    base = OrbitalSwingByConfig(
        body_kind="black_hole",
        gravity_model="paczynski_wiita",
        body_radius=0.115,
        schwarzschild_radius=0.060,
        gravitational_parameter=0.075,
        max_gravity_acceleration=14.0,
    )
    return replace(base, **overrides).normalized()
