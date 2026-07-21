"""Low-speed continuous-control car parking environments.

The dynamics and public observation modes intentionally mirror ``car_race``.
Unlike racing, parking success is pose based: the complete oriented vehicle
must be inside the bay, aligned with it, and nearly stationary for a short
dwell period.  Layouts use a lane-consistent approach convention, parked
vehicles have the same footprint as the controllable vehicle, and collisions
cause cumulative impulse-proportional damage instead of an unconditional
one-touch termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Array = np.ndarray
ParkingManeuver = Literal[
    "parallel", "t_forward", "t_reverse", "angled", "mixed"
]
ObservationMode = Literal["state", "state_goal", "goal_dict"]
RewardMode = Literal["sparse", "dense"]

MANEUVERS: tuple[str, ...] = (
    "parallel",
    "t_forward",
    "t_reverse",
    "angled",
)
NUM_FIXED_TASKS = 5


def _wrap_angle(angle: float | Array) -> float | Array:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class OrientedBox:
    """An obstacle or parking bay represented by an oriented rectangle."""

    center: tuple[float, float]
    length: float
    width: float
    heading: float = 0.0
    kind: str = "vehicle"


@dataclass(frozen=True)
class ParkingLayout:
    maneuver: str
    start: tuple[float, float]
    start_heading: float
    slot: OrientedBox
    obstacles: tuple[OrientedBox, ...]


@dataclass(frozen=True)
class CarParkingConfig:
    """Geometry, bicycle dynamics, task, and reward configuration."""

    arena_low: float = -1.0
    arena_high: float = 1.0
    car_length: float = 0.18
    car_width: float = 0.10
    wheelbase: float = 0.13
    collision_margin: float = 0.004
    # Total width of the two-way aisle.  Each lane is 0.24 by default,
    # which is about 2.4 vehicle widths and visually matches the car scale.
    aisle_width: float = 0.48

    dt: float = 0.05
    physics_substeps: int = 5
    max_steer_angle: float = 0.62
    max_steer_rate: float = 2.8
    max_acceleration: float = 0.75
    max_braking: float = 1.15
    max_speed: float = 0.42
    max_reverse_speed: float = 0.32
    rolling_drag: float = 0.75

    maneuver: ParkingManeuver = "mixed"
    max_episode_steps: int = 400
    # Default behavior is cumulative damage.  Set this to True only when an
    # immediate terminal collision is explicitly desired for an ablation.
    terminate_on_collision: bool = False
    orientation_tolerance: float = np.deg2rad(10.0)
    parked_speed_tolerance: float = 0.025
    slot_margin: float = 0.008
    dwell_steps: int = 8

    # Collision damage.  The deliberately small capacity makes the environment
    # unforgiving: a gentle normal contact at 0.05 world-units/s removes about
    # 15% of full health, a 0.10 contact removes about 30%, and a hard impact can
    # destroy the car in one hit.
    initial_health: float = 1.0
    damage_capacity: float = 0.32
    impact_impulse_scale: float = 0.95
    collision_restitution: float = 0.05
    min_effective_impact_speed: float = 0.025
    glancing_impact_fraction: float = 0.35
    health_bar_tau: float = 0.045

    reward_mode: RewardMode = "dense"
    step_penalty: float = 0.002
    control_cost: float = 0.0005
    position_scale: float = 1.0
    orientation_scale: float = 0.25
    containment_bonus: float = 0.08
    success_reward: float = 2.0
    collision_penalty: float = -0.05
    damage_penalty_scale: float = 0.35
    death_penalty: float = -2.0

    def validate(self) -> None:
        if not self.arena_low < self.arena_high:
            raise ValueError("arena_low must be smaller than arena_high")
        if min(self.car_length, self.car_width, self.wheelbase) <= 0.0:
            raise ValueError("car dimensions and wheelbase must be positive")
        if self.car_width >= self.car_length:
            raise ValueError("car_width must be smaller than car_length")
        if not 2.5 * self.car_width <= self.aisle_width <= 1.0:
            raise ValueError("aisle_width must be between 2.5 car widths and 1.0")
        if self.dt <= 0.0 or self.physics_substeps < 1:
            raise ValueError("dt must be positive and physics_substeps >= 1")
        if self.max_steer_angle <= 0.0 or self.max_steer_rate <= 0.0:
            raise ValueError("steering limits must be positive")
        if self.max_acceleration <= 0.0 or self.max_braking <= 0.0:
            raise ValueError("acceleration and braking must be positive")
        if self.max_speed <= 0.0 or self.max_reverse_speed <= 0.0:
            raise ValueError("speed limits must be positive")
        if self.rolling_drag < 0.0:
            raise ValueError("rolling_drag must be non-negative")
        if self.maneuver not in (*MANEUVERS, "mixed"):
            raise ValueError(f"Unknown maneuver: {self.maneuver}")
        if self.max_episode_steps < 1 or self.dwell_steps < 1:
            raise ValueError("episode and dwell steps must be positive")
        if not 0.0 < self.orientation_tolerance < np.pi / 2.0:
            raise ValueError("orientation_tolerance must be in (0, pi/2)")
        if self.parked_speed_tolerance < 0.0 or self.slot_margin < 0.0:
            raise ValueError("parking tolerances must be non-negative")
        if self.initial_health <= 0.0 or self.damage_capacity <= 0.0:
            raise ValueError("initial_health and damage_capacity must be positive")
        if self.impact_impulse_scale < 0.0:
            raise ValueError("impact_impulse_scale must be non-negative")
        if not 0.0 <= self.collision_restitution <= 1.0:
            raise ValueError("collision_restitution must be in [0, 1]")
        if self.min_effective_impact_speed < 0.0:
            raise ValueError("min_effective_impact_speed must be non-negative")
        if not 0.0 <= self.glancing_impact_fraction <= 1.0:
            raise ValueError("glancing_impact_fraction must be in [0, 1]")
        if self.health_bar_tau <= 0.0:
            raise ValueError("health_bar_tau must be positive")
        if self.damage_penalty_scale < 0.0:
            raise ValueError("damage_penalty_scale must be non-negative")
        if self.reward_mode not in ("sparse", "dense"):
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")


def fixed_task_options(task_id: int) -> dict[str, Any]:
    """Return one of five deterministic mixed-maneuver evaluation tasks."""
    task_id = int(task_id)
    if not 1 <= task_id <= NUM_FIXED_TASKS:
        raise ValueError(f"task_id must be in [1, {NUM_FIXED_TASKS}]")
    maneuver, variant = (
        ("parallel", 1),
        ("parallel", 2),
        ("t_reverse", 1),
        ("t_forward", 2),
        ("angled", 1),
    )[task_id - 1]
    return {"maneuver": maneuver, "variant": variant}


def _box_corners(box: OrientedBox, *, margin: float = 0.0) -> Array:
    half_l = box.length / 2.0 + margin
    half_w = box.width / 2.0 + margin
    local = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]],
        dtype=np.float64,
    )
    c, s = np.cos(box.heading), np.sin(box.heading)
    rotation = np.array([[c, -s], [s, c]])
    return local @ rotation.T + np.asarray(box.center)


def _boxes_overlap(first: OrientedBox, second: OrientedBox) -> bool:
    """Separating-axis test for two oriented rectangles."""
    a = _box_corners(first)
    b = _box_corners(second)
    axes: list[Array] = []
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges[:2]:
            normal = np.array([-edge[1], edge[0]], dtype=np.float64)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            axes.append(normal)
    for axis in axes:
        projection_a = a @ axis
        projection_b = b @ axis
        if projection_a.max() < projection_b.min() or projection_b.max() < projection_a.min():
            return False
    return True


def _box_contact_normal(
    moving: OrientedBox,
    obstacle: OrientedBox,
) -> tuple[Array, float] | None:
    """Return an outward contact normal and penetration depth for two boxes.

    The normal points from ``obstacle`` toward ``moving``.  This is a compact
    separating-axis contact estimate; it is used only for impact magnitude and
    does not change the oriented-box collision criterion.
    """
    moving_corners = _box_corners(moving)
    obstacle_corners = _box_corners(obstacle)
    axes: list[Array] = []
    for polygon in (moving_corners, obstacle_corners):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges[:2]:
            normal = np.array([-edge[1], edge[0]], dtype=np.float64)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            axes.append(normal)

    best_axis: Array | None = None
    minimum_overlap = np.inf
    for axis in axes:
        moving_projection = moving_corners @ axis
        obstacle_projection = obstacle_corners @ axis
        overlap = min(moving_projection.max(), obstacle_projection.max()) - max(
            moving_projection.min(), obstacle_projection.min()
        )
        if overlap < 0.0:
            return None
        if overlap < minimum_overlap:
            minimum_overlap = float(overlap)
            best_axis = axis.copy()

    assert best_axis is not None
    center_delta = np.asarray(moving.center, dtype=np.float64) - np.asarray(
        obstacle.center, dtype=np.float64
    )
    if float(np.dot(best_axis, center_delta)) < 0.0:
        best_axis = -best_axis
    return best_axis.astype(np.float32), float(minimum_overlap)


def _opposite_island(
    mirror: float, row_inner_y: float, aisle_width: float
) -> OrientedBox:
    """Build the raised boundary that gives the driving aisle a real width."""
    road_edge = row_inner_y - mirror * aisle_width
    outer_edge = -mirror
    return OrientedBox(
        (0.0, 0.5 * (road_edge + outer_edge)),
        2.0,
        abs(outer_edge - road_edge),
        0.0,
        "island",
    )


def _parking_row_geometry(
    slot: OrientedBox,
    aisle_width: float,
) -> tuple[float, float, float, float, float, float]:
    """Return the row/aisle geometry derived from a target parking bay.

    Returns ``(mirror, row_inner_y, road_edge, road_center,
    approach_lane_center, opposite_lane_center)``.  ``mirror`` is +1 for a
    parking row above the aisle and -1 for a row below it.  The approach lane
    is always the lane adjacent to the parking row.
    """
    mirror = float(np.sign(slot.center[1]) or -1.0)
    vertical_half = (
        abs(np.cos(slot.heading)) * slot.width / 2.0
        + abs(np.sin(slot.heading)) * slot.length / 2.0
    )
    row_inner_y = float(slot.center[1] - mirror * vertical_half)
    road_edge = float(row_inner_y - mirror * aisle_width)
    road_center = 0.5 * (row_inner_y + road_edge)
    approach_lane_center = row_inner_y - mirror * 0.25 * aisle_width
    opposite_lane_center = row_inner_y - mirror * 0.75 * aisle_width
    return (
        mirror,
        row_inner_y,
        road_edge,
        road_center,
        approach_lane_center,
        opposite_lane_center,
    )


def _layout(
    maneuver: str,
    variant: int,
    aisle_width: float = 0.48,
    car_length: float = 0.18,
    car_width: float = 0.10,
) -> ParkingLayout:
    """Construct a compact, lane-consistent parking lot.

    Parked vehicles use exactly the same physical footprint as the ego car.
    The ego starts in the traffic lane adjacent to the target row, and its
    default travel direction is consistent with the target bay orientation.
    """
    variant = int(variant)
    mirror = -1.0 if variant % 2 else 1.0
    lateral_shift = 0.035 * ((variant - 1) % 3 - 1)

    # A simple right-side parking convention:
    # bottom row -> drive east; top row -> drive west.  In both cases the
    # target row is on the vehicle's right-hand side.
    travel_heading = 0.0 if mirror < 0.0 else np.pi
    travel_direction = np.array(
        [np.cos(travel_heading), np.sin(travel_heading)], dtype=np.float64
    )
    row_normal = np.array([0.0, mirror], dtype=np.float64)

    curb = OrientedBox(
        (0.0, 0.965 * mirror),
        2.0,
        0.07,
        0.0,
        "curb",
    )
    curb_inner_magnitude = 0.93
    curb_gap = 0.018

    def row_center_y(length: float, width: float, heading: float) -> float:
        vertical_half = (
            abs(np.cos(heading)) * width / 2.0
            + abs(np.sin(heading)) * length / 2.0
        )
        return mirror * (curb_inner_magnitude - curb_gap - vertical_half)

    def start_pose(slot: OrientedBox) -> tuple[tuple[float, float], float]:
        (
            _,
            _,
            _,
            _,
            approach_lane_center,
            _,
        ) = _parking_row_geometry(slot, aisle_width)
        start_x = -0.74 if travel_heading == 0.0 else 0.74
        return (start_x, approach_lane_center), travel_heading

    parked_length = float(car_length)
    parked_width = float(car_width)

    if maneuver == "parallel":
        slot_length = max(car_length + 0.10, 1.55 * car_length)
        slot_width = max(car_width + 0.045, 1.40 * car_width)
        heading = travel_heading
        y = row_center_y(slot_length, slot_width, heading)
        slot = OrientedBox(
            (lateral_shift, y), slot_length, slot_width, heading, "slot"
        )

        bay_pitch = slot_length + 0.030
        obstacles = (
            OrientedBox(
                (lateral_shift - bay_pitch, y),
                parked_length,
                parked_width,
                heading,
            ),
            OrientedBox(
                (lateral_shift + bay_pitch, y),
                parked_length,
                parked_width,
                heading,
            ),
            curb,
            _opposite_island(
                mirror,
                _parking_row_geometry(slot, aisle_width)[1],
                aisle_width,
            ),
        )
        start, start_heading = start_pose(slot)
        return ParkingLayout(maneuver, start, start_heading, slot, obstacles)

    if maneuver in ("t_forward", "t_reverse"):
        slot_length = max(car_length + 0.080, 1.40 * car_length)
        slot_width = max(car_width + 0.045, 1.40 * car_width)
        goal_heading = (
            mirror * np.pi / 2.0
            if maneuver == "t_forward"
            else -mirror * np.pi / 2.0
        )
        y = row_center_y(slot_length, slot_width, goal_heading)
        slot = OrientedBox(
            (lateral_shift, y),
            slot_length,
            slot_width,
            goal_heading,
            "slot",
        )

        bay_pitch = slot_width + 0.035
        obstacles = (
            OrientedBox(
                (lateral_shift - bay_pitch, y),
                parked_length,
                parked_width,
                goal_heading,
            ),
            OrientedBox(
                (lateral_shift + bay_pitch, y),
                parked_length,
                parked_width,
                goal_heading,
            ),
            curb,
            _opposite_island(
                mirror,
                _parking_row_geometry(slot, aisle_width)[1],
                aisle_width,
            ),
        )
        start, start_heading = start_pose(slot)
        return ParkingLayout(maneuver, start, start_heading, slot, obstacles)

    if maneuver == "angled":
        slot_length = max(car_length + 0.085, 1.43 * car_length)
        slot_width = max(car_width + 0.045, 1.40 * car_width)

        # The bay points halfway between the approach direction and the row.
        # This makes both mirrored variants enterable from the adjacent lane.
        angled_direction = travel_direction + row_normal
        angled_direction /= max(float(np.linalg.norm(angled_direction)), 1e-12)
        heading = float(np.arctan2(angled_direction[1], angled_direction[0]))
        y = row_center_y(slot_length, slot_width, heading)
        slot = OrientedBox(
            (lateral_shift, y), slot_length, slot_width, heading, "slot"
        )

        # Adjacent angled bays repeat along the curb/road direction, not
        # along the car's lateral axis.  This keeps every bay in the same
        # parking row and avoids placing one parked car inside the aisle.
        horizontal_half_extent = (
            abs(np.cos(heading)) * slot_length / 2.0
            + abs(np.sin(heading)) * slot_width / 2.0
        )
        bay_pitch = 2.0 * horizontal_half_extent + 0.020
        centers = (
            np.asarray(slot.center) + np.array([-bay_pitch, 0.0]),
            np.asarray(slot.center) + np.array([bay_pitch, 0.0]),
        )
        obstacles = (
            OrientedBox(
                tuple(centers[0]),
                parked_length,
                parked_width,
                heading,
            ),
            OrientedBox(
                tuple(centers[1]),
                parked_length,
                parked_width,
                heading,
            ),
            curb,
            _opposite_island(
                mirror,
                _parking_row_geometry(slot, aisle_width)[1],
                aisle_width,
            ),
        )
        start, start_heading = start_pose(slot)
        return ParkingLayout(maneuver, start, start_heading, slot, obstacles)

    raise ValueError(f"Unknown maneuver: {maneuver}")


class CarParkingEnv(gym.Env):
    """Parallel, T-bay, and angled parking with bicycle dynamics.

    Actions are normalized ``[steering, throttle/brake]`` in ``[-1, 1]^2``.
    The state is ``[x, y, cos(yaw), sin(yaw), speed, normalized_steering,
    distance_to_slot, normalized_yaw_error, inside_slot, health]``.
    The first four elements are also the achieved-goal representation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    num_tasks = NUM_FIXED_TASKS

    def __init__(
        self,
        config: CarParkingConfig | None = None,
        observation_mode: ObservationMode = "state_goal",
        render_mode: str | None = None,
        render_size: int = 512,
    ) -> None:
        super().__init__()
        self.config = config or CarParkingConfig()
        self.config.validate()
        if observation_mode not in ("state", "state_goal", "goal_dict"):
            raise ValueError(f"Unknown observation_mode: {observation_mode}")
        if render_mode not in (*self.metadata["render_modes"], None):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if render_size < 64:
            raise ValueError("render_size must be at least 64")

        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.render_size = int(render_size)
        self._dtype = np.float32
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=self._dtype)

        state_low = np.array(
            [
                self.config.arena_low,
                self.config.arena_low,
                -1.0,
                -1.0,
                -self.config.max_reverse_speed,
                -1.0,
                0.0,
                -1.0,
                0.0,
                0.0,
            ],
            dtype=self._dtype,
        )
        state_high = np.array(
            [
                self.config.arena_high,
                self.config.arena_high,
                1.0,
                1.0,
                self.config.max_speed,
                1.0,
                3.0,
                1.0,
                1.0,
                self.config.initial_health,
            ],
            dtype=self._dtype,
        )
        goal_low = np.array(
            [self.config.arena_low, self.config.arena_low, -1.0, -1.0],
            dtype=self._dtype,
        )
        goal_high = np.array(
            [self.config.arena_high, self.config.arena_high, 1.0, 1.0],
            dtype=self._dtype,
        )
        if observation_mode == "state":
            self.observation_space = spaces.Box(state_low, state_high, dtype=self._dtype)
        elif observation_mode == "state_goal":
            self.observation_space = spaces.Box(
                np.concatenate([state_low, goal_low]),
                np.concatenate([state_high, goal_high]),
                dtype=self._dtype,
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(state_low, state_high, dtype=self._dtype),
                    "achieved_goal": spaces.Box(goal_low, goal_high, dtype=self._dtype),
                    "desired_goal": spaces.Box(goal_low, goal_high, dtype=self._dtype),
                }
            )

        self.position = np.zeros(2, dtype=self._dtype)
        self.heading = 0.0
        self.speed = 0.0
        self.steering = 0.0
        self.elapsed_steps = 0
        self.dwell_count = 0
        self.collision = False
        self.dead = False
        self.success = False
        self.health = float(self.config.initial_health)
        self._display_health = float(self.config.initial_health)
        self.step_impulse = 0.0
        self.total_impulse = 0.0
        self.health_loss = 0.0
        self._containment_awarded = False
        self.cur_task_id: int | None = None
        self.layout = _layout(
            "parallel",
            1,
            self.config.aisle_width,
            self.config.car_length,
            self.config.car_width,
        )
        self._human_figure: Any = None
        self._human_image: Any = None

    @property
    def vehicle_box(self) -> OrientedBox:
        return OrientedBox(
            tuple(float(x) for x in self.position),
            self.config.car_length,
            self.config.car_width,
            self.heading,
        )

    @property
    def desired_goal(self) -> Array:
        slot = self.layout.slot
        return np.array(
            [slot.center[0], slot.center[1], np.cos(slot.heading), np.sin(slot.heading)],
            dtype=self._dtype,
        )

    @property
    def achieved_goal(self) -> Array:
        return np.array(
            [self.position[0], self.position[1], np.cos(self.heading), np.sin(self.heading)],
            dtype=self._dtype,
        )

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.position - np.asarray(self.layout.slot.center)))

    @property
    def heading_error(self) -> float:
        return float(_wrap_angle(self.heading - self.layout.slot.heading))

    @property
    def fully_inside_slot(self) -> bool:
        corners = _box_corners(self.vehicle_box)
        center = np.asarray(self.layout.slot.center)
        c, s = np.cos(self.layout.slot.heading), np.sin(self.layout.slot.heading)
        rotation_inverse = np.array([[c, s], [-s, c]])
        local = (corners - center) @ rotation_inverse.T
        half_l = self.layout.slot.length / 2.0 - self.config.slot_margin
        half_w = self.layout.slot.width / 2.0 - self.config.slot_margin
        return bool(
            np.all(np.abs(local[:, 0]) <= half_l + 1e-8)
            and np.all(np.abs(local[:, 1]) <= half_w + 1e-8)
        )

    @property
    def parked_pose(self) -> bool:
        return bool(
            self.fully_inside_slot
            and abs(self.heading_error) <= self.config.orientation_tolerance
            and abs(self.speed) <= self.config.parked_speed_tolerance
        )

    @property
    def state(self) -> Array:
        return np.array(
            [
                self.position[0],
                self.position[1],
                np.cos(self.heading),
                np.sin(self.heading),
                self.speed,
                self.steering / self.config.max_steer_angle,
                self.distance_to_goal,
                self.heading_error / np.pi,
                float(self.fully_inside_slot),
                float(self.health),
            ],
            dtype=self._dtype,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        options = dict(options or {})
        task_id = options.pop("task_id", None)
        if task_id is not None:
            if "maneuver" in options or "variant" in options:
                raise ValueError("task_id cannot be combined with maneuver or variant")
            self.cur_task_id = int(task_id)
            options.update(fixed_task_options(self.cur_task_id))
        else:
            self.cur_task_id = None

        configured = self.config.maneuver
        requested = options.pop("maneuver", None)
        if configured == "mixed":
            maneuver = requested or str(self.np_random.choice(MANEUVERS))
        else:
            if requested is not None and requested != configured:
                raise ValueError(
                    f"maneuver {requested!r} conflicts with configured {configured!r}"
                )
            maneuver = configured
        if maneuver not in MANEUVERS:
            raise ValueError(f"Unknown maneuver: {maneuver}")
        variant = int(options.pop("variant", self.np_random.integers(1, 6)))
        self.layout = _layout(
            maneuver,
            variant,
            self.config.aisle_width,
            self.config.car_length,
            self.config.car_width,
        )

        default_position = np.asarray(self.layout.start, dtype=self._dtype)
        position = np.asarray(options.pop("position", default_position), dtype=self._dtype)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite vector with shape (2,)")
        self.position = position.copy()
        self.heading = float(
            _wrap_angle(options.pop("heading", self.layout.start_heading))
        )
        self.speed = float(options.pop("speed", 0.0))
        self.steering = float(options.pop("steering", 0.0))
        health = float(options.pop("health", self.config.initial_health))
        if not 0.0 < health <= self.config.initial_health:
            raise ValueError("health must be in (0, initial_health]")
        if options:
            raise ValueError(f"Unknown reset options: {tuple(options)}")
        if not -self.config.max_reverse_speed <= self.speed <= self.config.max_speed:
            raise ValueError("initial speed is outside configured limits")
        if abs(self.steering) > self.config.max_steer_angle:
            raise ValueError("initial steering is outside configured limits")
        if self._collides(self.vehicle_box):
            raise ValueError("initial vehicle pose collides with the layout")

        self.elapsed_steps = 0
        self.dwell_count = 0
        self.collision = False
        self.dead = False
        self.success = False
        self.health = health
        self._display_health = health
        self.step_impulse = 0.0
        self.total_impulse = 0.0
        self.health_loss = 0.0
        self._containment_awarded = False
        observation = self._get_observation()
        info = self._get_info(termination_reason=None)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: Array) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.dead or self.success or self.elapsed_steps >= self.config.max_episode_steps:
            raise RuntimeError("step() called after episode end; call reset() first")
        action = np.asarray(action, dtype=self._dtype)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector with shape (2,)")
        action = np.clip(action, -1.0, 1.0)
        previous_cost = self._pose_cost()

        target_steering = float(action[0]) * self.config.max_steer_angle
        throttle = float(action[1])
        collided = False
        step_impulse = 0.0
        health_before = float(self.health)
        sub_dt = self.config.dt / self.config.physics_substeps

        for _ in range(self.config.physics_substeps):
            steer_delta = np.clip(
                target_steering - self.steering,
                -self.config.max_steer_rate * sub_dt,
                self.config.max_steer_rate * sub_dt,
            )
            self.steering = float(
                np.clip(
                    self.steering + steer_delta,
                    -self.config.max_steer_angle,
                    self.config.max_steer_angle,
                )
            )
            acceleration_limit = (
                self.config.max_acceleration
                if throttle * self.speed >= 0.0
                else self.config.max_braking
            )
            acceleration = (
                acceleration_limit * throttle
                - self.config.rolling_drag * self.speed
            )
            candidate_speed = float(
                np.clip(
                    self.speed + acceleration * sub_dt,
                    -self.config.max_reverse_speed,
                    self.config.max_speed,
                )
            )
            yaw_rate = (
                candidate_speed
                * np.tan(self.steering)
                / self.config.wheelbase
            )
            candidate_heading = float(
                _wrap_angle(self.heading + yaw_rate * sub_dt)
            )
            average_heading = self.heading + 0.5 * float(
                _wrap_angle(candidate_heading - self.heading)
            )
            candidate_position = self.position + candidate_speed * sub_dt * np.array(
                [np.cos(average_heading), np.sin(average_heading)],
                dtype=self._dtype,
            )
            candidate_box = OrientedBox(
                tuple(float(x) for x in candidate_position),
                self.config.car_length,
                self.config.car_width,
                candidate_heading,
            )

            contacts = self._collision_contacts(candidate_box)
            if contacts:
                # The kinematic model has one signed longitudinal speed.  The
                # SAT normal supplies the true normal component, while the
                # glancing term intentionally makes side brushes costly too.
                forward = np.array(
                    [np.cos(candidate_heading), np.sin(candidate_heading)],
                    dtype=np.float32,
                )
                velocity = candidate_speed * forward
                normal_speed = max(
                    max(0.0, -float(np.dot(velocity, normal)))
                    for normal in contacts
                )
                effective_speed = max(
                    normal_speed,
                    self.config.glancing_impact_fraction
                    * abs(candidate_speed),
                    self.config.min_effective_impact_speed,
                )
                impulse = float(
                    self.config.impact_impulse_scale
                    * (1.0 + self.config.collision_restitution)
                    * effective_speed
                )
                step_impulse += impulse
                self.health = max(
                    0.0,
                    self.health - impulse / self.config.damage_capacity,
                )

                # Reject the penetrating pose and apply a small scalar rebound.
                # This keeps the collision geometry exact and avoids tunnelling.
                self.speed = float(
                    np.clip(
                        -candidate_speed * self.config.collision_restitution,
                        -self.config.max_reverse_speed,
                        self.config.max_speed,
                    )
                )
                collided = True
                break

            self.position = candidate_position.astype(self._dtype)
            self.heading = candidate_heading
            self.speed = candidate_speed

        self.elapsed_steps += 1
        self.collision = bool(collided)
        self.step_impulse = float(step_impulse)
        self.total_impulse += float(step_impulse)
        self.health_loss = max(0.0, health_before - float(self.health))
        self.dead = bool(
            self.health <= 0.0
            or (collided and self.config.terminate_on_collision)
        )
        self._update_display_health(self.config.dt)

        if not self.dead and self.parked_pose:
            self.dwell_count += 1
        else:
            self.dwell_count = 0
        self.success = bool(
            not self.dead and self.dwell_count >= self.config.dwell_steps
        )

        terminated = bool(self.success or self.dead)
        truncated = bool(
            not terminated
            and self.elapsed_steps >= self.config.max_episode_steps
        )
        if self.success:
            reason = "success"
        elif self.dead and collided and self.config.terminate_on_collision:
            reason = "collision"
        elif self.dead:
            reason = "health_depleted"
        elif truncated:
            reason = "time_limit"
        else:
            reason = None

        containment_entry = bool(
            self.fully_inside_slot and not self._containment_awarded
        )
        if containment_entry:
            self._containment_awarded = True
        reward = self._step_reward(
            action,
            previous_cost,
            collided=collided,
            health_loss=self.health_loss,
            containment_entry=containment_entry,
        )
        observation = self._get_observation()
        info = self._get_info(termination_reason=reason)
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _collision_contacts(self, vehicle: OrientedBox) -> list[Array]:
        """Return all inward-safe contact normals for a candidate vehicle pose."""
        expanded = OrientedBox(
            vehicle.center,
            vehicle.length + 2.0 * self.config.collision_margin,
            vehicle.width + 2.0 * self.config.collision_margin,
            vehicle.heading,
        )
        corners = _box_corners(expanded)
        contacts: list[Array] = []

        if float(corners[:, 0].min()) < self.config.arena_low:
            contacts.append(np.array([1.0, 0.0], dtype=np.float32))
        if float(corners[:, 0].max()) > self.config.arena_high:
            contacts.append(np.array([-1.0, 0.0], dtype=np.float32))
        if float(corners[:, 1].min()) < self.config.arena_low:
            contacts.append(np.array([0.0, 1.0], dtype=np.float32))
        if float(corners[:, 1].max()) > self.config.arena_high:
            contacts.append(np.array([0.0, -1.0], dtype=np.float32))

        for obstacle in self.layout.obstacles:
            contact = _box_contact_normal(expanded, obstacle)
            if contact is not None:
                normal, _ = contact
                contacts.append(normal)
        return contacts

    def _collides(self, vehicle: OrientedBox) -> bool:
        return bool(self._collision_contacts(vehicle))

    def _update_display_health(self, dt: float) -> None:
        alpha = 1.0 - float(np.exp(-dt / self.config.health_bar_tau))
        self._display_health += alpha * (self.health - self._display_health)
        if self.health <= 0.0 and self._display_health < 0.01:
            self._display_health = 0.0

    def _pose_cost(self) -> float:
        return float(
            self.config.position_scale * self.distance_to_goal
            + self.config.orientation_scale * abs(self.heading_error)
        )

    def _step_reward(
        self,
        action: Array,
        previous_cost: float,
        *,
        collided: bool,
        health_loss: float,
        containment_entry: bool,
    ) -> float:
        if self.success:
            return float(self.config.success_reward)
        if self.dead:
            return float(self.config.death_penalty)

        reward = (
            -self.config.step_penalty
            - self.config.control_cost * float(np.dot(action, action))
        )
        if collided:
            reward += self.config.collision_penalty
            reward -= self.config.damage_penalty_scale * float(health_loss)
        if self.config.reward_mode == "dense":
            reward += previous_cost - self._pose_cost()
            if containment_entry:
                reward += self.config.containment_bonus
        return float(reward)

    def compute_reward(
        self, achieved_goal: Array, desired_goal: Array, info: Any | None = None
    ) -> Array | float:
        """Goal-conditioned sparse reward for scalar or batched poses."""
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        position_ok = np.linalg.norm(achieved[..., :2] - desired[..., :2], axis=-1) <= (
            min(self.layout.slot.length - self.config.car_length, self.layout.slot.width - self.config.car_width) / 2.0
        )
        dot = np.sum(achieved[..., 2:4] * desired[..., 2:4], axis=-1)
        angle_ok = dot >= np.cos(self.config.orientation_tolerance)
        result = np.where(position_ok & angle_ok, self.config.success_reward, -self.config.step_penalty).astype(np.float32)
        return float(result) if result.ndim == 0 else result

    def _get_observation(self) -> Any:
        state = self.state
        goal = self.desired_goal
        if self.observation_mode == "state":
            return state
        if self.observation_mode == "state_goal":
            return np.concatenate([state, goal]).astype(self._dtype)
        return {
            "observation": state,
            "achieved_goal": self.achieved_goal,
            "desired_goal": goal,
        }

    def _get_info(self, termination_reason: str | None) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "is_success": bool(self.success),
            "collision": bool(self.collision),
            "collision_contact": bool(self.collision),
            "dead": bool(self.dead),
            "health": float(self.health),
            "health_fraction": float(
                self.health / self.config.initial_health
            ),
            "health_loss": float(self.health_loss),
            "step_impulse": float(self.step_impulse),
            "total_impulse": float(self.total_impulse),
            "parked_pose": bool(self.parked_pose),
            "fully_inside_slot": bool(self.fully_inside_slot),
            "dwell_count": int(self.dwell_count),
            "distance_to_goal": self.distance_to_goal,
            "heading_error": self.heading_error,
            "maneuver": self.layout.maneuver,
            "aisle_width": float(self.config.aisle_width),
            "goal": self.desired_goal.copy(),
            "task_id": self.cur_task_id,
            "termination_reason": termination_reason,
        }

    def render(self) -> Array | None:
        frame = self._render_rgb()
        if self.render_mode == "human":
            self._render_human(frame)
            return None
        return frame

    def close(self) -> None:
        if self._human_figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._human_figure)
        self._human_figure = None
        self._human_image = None

    def _render_rgb(self) -> Array:
        """Render a compact parking scene with only a narrow tree backdrop.

        The camera is cropped to the active aisle, parking row, sidewalk, and
        one tree line.  It deliberately omits the previous oversized lawn,
        multi-meter HUD, maneuver badge, direction arrows, and decorative
        start stencil.
        """
        width = self.render_size
        x_low, x_high, y_low, y_high, height = self._render_view()
        span = x_high - x_low
        xs = np.linspace(x_low, x_high, width)
        ys = np.linspace(y_high, y_low, height)
        xx, yy = np.meshgrid(xs, ys)

        slot = self.layout.slot
        (
            mirror,
            row_inner_y,
            road_edge,
            road_center,
            approach_lane_center,
            opposite_lane_center,
        ) = _parking_row_geometry(slot, self.config.aisle_width)

        # Asphalt base.
        aggregate = (
            2.4 * np.sin(83.0 * xx + 41.0 * yy)
            + 1.5 * np.cos(137.0 * xx - 67.0 * yy)
            + 0.9 * np.sin(211.0 * xx + 19.0 * yy)
        )
        asphalt = np.clip(58.0 + aggregate, 48.0, 68.0)
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.clip(asphalt * 0.95, 0, 255).astype(np.uint8)
        frame[..., 1] = np.clip(asphalt, 0, 255).astype(np.uint8)
        frame[..., 2] = np.clip(asphalt * 1.05, 0, 255).astype(np.uint8)

        road_low = min(row_inner_y, road_edge)
        road_high = max(row_inner_y, road_edge)
        road_mask = (yy >= road_low) & (yy <= road_high)
        approach_lane = road_mask & (mirror * (yy - road_center) >= 0.0)
        self._alpha_blend_mask(
            frame,
            approach_lane,
            np.array([77, 85, 89], dtype=np.uint8),
            0.07,
        )

        # Subtle wheel tracks only; no large directional overlays.
        tire_offset = min(0.034, 0.32 * self.config.car_width)
        rubber = np.zeros_like(xx, dtype=np.float64)
        for lane_center, strength in (
            (approach_lane_center, 3.8),
            (opposite_lane_center, 2.3),
        ):
            for offset in (-tire_offset, tire_offset):
                rubber += strength * np.exp(
                    -((yy - (lane_center + offset)) / 0.030) ** 2
                )
        rubber *= road_mask
        for channel, multiplier in ((0, 1.0), (1, 1.0), (2, 0.72)):
            frame[..., channel] = np.clip(
                frame[..., channel].astype(np.float32)
                - multiplier * rubber,
                0,
                255,
            ).astype(np.uint8)

        # The opposite side ends after a compact sidewalk and one tree row.
        island_depth = -mirror * (yy - road_edge)
        island = island_depth >= 0.0
        sidewalk = island & (island_depth <= 0.105)
        grass = island & (island_depth > 0.105)

        tile_x = np.floor((xx - x_low) / 0.095).astype(np.int32)
        tile_y = np.floor((yy - y_low) / 0.095).astype(np.int32)
        tile = (tile_x + tile_y) % 2
        paver = 154.0 + 6.0 * tile + 1.4 * np.sin(41.0 * xx - 23.0 * yy)
        frame[..., 0][sidewalk] = np.clip(
            paver[sidewalk] * 1.03, 0, 255
        ).astype(np.uint8)
        frame[..., 1][sidewalk] = np.clip(paver[sidewalk], 0, 255).astype(
            np.uint8
        )
        frame[..., 2][sidewalk] = np.clip(
            paver[sidewalk] * 0.95, 0, 255
        ).astype(np.uint8)

        grass_texture = (
            80.0
            + 4.0 * np.sin(25.0 * xx + 13.0 * yy)
            + 2.5 * np.cos(39.0 * xx - 17.0 * yy)
        )
        frame[..., 0][grass] = np.clip(
            0.48 * grass_texture[grass], 0, 255
        ).astype(np.uint8)
        frame[..., 1][grass] = np.clip(
            1.14 * grass_texture[grass], 0, 255
        ).astype(np.uint8)
        frame[..., 2][grass] = np.clip(
            0.58 * grass_texture[grass], 0, 255
        ).astype(np.uint8)

        curb_width_world = max(0.010, 2.0 * span / width)
        island_curb = np.abs(yy - road_edge) <= curb_width_world
        frame[island_curb] = np.array([211, 207, 193], dtype=np.uint8)

        tree_y = road_edge - mirror * 0.205
        for tree_x in np.linspace(-0.72, 0.72, 5):
            shadow_center = np.array(
                [tree_x + 0.018, tree_y - 0.018], dtype=np.float32
            )
            canopy_center = np.array([tree_x, tree_y], dtype=np.float32)
            self._blend_disk_world(
                frame,
                shadow_center,
                0.061,
                np.array([3, 8, 6], dtype=np.uint8),
                0.38,
            )
            self._draw_disk_world(
                frame,
                canopy_center,
                0.054,
                np.array([37, 105, 59], dtype=np.uint8),
            )
            self._blend_disk_world(
                frame,
                canopy_center + np.array([-0.016, 0.015], dtype=np.float32),
                0.028,
                np.array([98, 164, 91], dtype=np.uint8),
                0.58,
            )

        # Parking-row curb: simple concrete, not decorative yellow/black tape.
        for obstacle in self.layout.obstacles:
            if obstacle.kind != "curb":
                continue
            mask = self._box_mask(xx, yy, obstacle)
            frame[mask] = np.array([194, 193, 185], dtype=np.uint8)
            curb_inner = obstacle.center[1] - mirror * obstacle.width / 2.0
            curb_shadow = np.abs(yy - (curb_inner - mirror * 0.012)) <= 0.010
            self._alpha_blend_mask(
                frame,
                curb_shadow,
                np.array([5, 7, 10], dtype=np.uint8),
                0.30,
            )

        # Restrained lane lines.
        edge_width = max(0.0045, 1.2 * span / width)
        self._alpha_blend_mask(
            frame,
            (np.abs(yy - row_inner_y) <= edge_width)
            | (np.abs(yy - road_edge) <= edge_width),
            np.array([225, 227, 218], dtype=np.uint8),
            0.88,
        )
        dash = np.mod(xx - x_low, 0.27) < 0.135
        center_line = (
            np.abs(yy - road_center) <= max(0.0045, 1.15 * span / width)
        ) & dash
        self._alpha_blend_mask(
            frame,
            center_line,
            np.array([224, 184, 66], dtype=np.uint8),
            0.86,
        )

        # Neighboring bay outlines use the target bay dimensions.
        bay_outline = np.array([182, 186, 181], dtype=np.uint8)
        for obstacle in self.layout.obstacles:
            if obstacle.kind != "vehicle":
                continue
            bay = OrientedBox(
                obstacle.center,
                slot.length,
                slot.width,
                obstacle.heading,
                "bay",
            )
            self._draw_oriented_outline(
                frame, bay, bay_outline, max(1, width // 320)
            )

        # Target bay and target-pose ghost.
        slot_mask = self._box_mask(xx, yy, slot)
        guide_color = np.array(
            [61, 226, 133] if self.success else [48, 205, 180],
            dtype=np.uint8,
        )
        self._alpha_blend_mask(frame, slot_mask, guide_color, 0.10)
        self._draw_oriented_outline(
            frame,
            slot,
            np.array([237, 240, 229], dtype=np.uint8),
            max(2, width // 230),
        )
        target_vehicle = OrientedBox(
            slot.center,
            self.config.car_length,
            self.config.car_width,
            slot.heading,
            "target_vehicle",
        )
        self._draw_target_vehicle(
            frame, xx, yy, target_vehicle, guide_color
        )

        # Parked cars and ego car share the same physical dimensions.
        parked_colors = (
            np.array([197, 66, 61], dtype=np.uint8),
            np.array([217, 151, 47], dtype=np.uint8),
            np.array([107, 115, 128], dtype=np.uint8),
        )
        vehicle_index = 0
        for obstacle in self.layout.obstacles:
            if obstacle.kind != "vehicle":
                continue
            self._draw_vehicle(
                frame,
                xx,
                yy,
                obstacle,
                parked_colors[vehicle_index % len(parked_colors)],
                player=False,
            )
            vehicle_index += 1

        if self.dead:
            ego_color = np.array([92, 94, 98], dtype=np.uint8)
        elif self.collision:
            ego_color = np.array([231, 72, 48], dtype=np.uint8)
        elif self.success:
            ego_color = np.array([49, 210, 119], dtype=np.uint8)
        else:
            ego_color = np.array([35, 133, 233], dtype=np.uint8)
        self._draw_vehicle(
            frame, xx, yy, self.vehicle_box, ego_color, player=True
        )

        # Health bar only.  No panel, P badge, maneuver pips, or extra meters.
        health_fraction = float(
            np.clip(
                self._display_health / self.config.initial_health,
                0.0,
                1.0,
            )
        )
        margin = max(8, width // 55)
        bar_width = max(92, width // 4)
        bar_height = max(9, width // 45)
        bar_y = margin if mirror < 0.0 else height - margin - bar_height
        bar_x = margin
        frame[
            bar_y : bar_y + bar_height,
            bar_x : bar_x + bar_width,
        ] = np.array([25, 29, 33], dtype=np.uint8)
        inner = max(2, width // 260)
        fill_width = int((bar_width - 2 * inner) * health_fraction)
        if fill_width > 0:
            healthy = np.array([54, 196, 98], dtype=np.float32)
            damaged = np.array([230, 52, 48], dtype=np.float32)
            health_color = (
                health_fraction * healthy
                + (1.0 - health_fraction) * damaged
            ).astype(np.uint8)
            frame[
                bar_y + inner : bar_y + bar_height - inner,
                bar_x + inner : bar_x + inner + fill_width,
            ] = health_color

        border = max(2, width // 220)
        frame[:border] = np.array([17, 20, 24], dtype=np.uint8)
        frame[-border:] = np.array([17, 20, 24], dtype=np.uint8)
        frame[:, :border] = np.array([17, 20, 24], dtype=np.uint8)
        frame[:, -border:] = np.array([17, 20, 24], dtype=np.uint8)
        return frame

    def _render_view(self) -> tuple[float, float, float, float, int]:
        """Return a fixed-aspect crop ending just beyond the tree line."""
        mirror, _, road_edge, _, _, _ = _parking_row_geometry(
            self.layout.slot, self.config.aisle_width
        )
        tree_y = road_edge - mirror * 0.205
        far_tree_edge = tree_y - mirror * 0.095
        parking_outer_edge = mirror * 1.0

        x_low, x_high = -0.92, 0.92
        world_height = 1.22
        center_y = 0.5 * (far_tree_edge + parking_outer_edge)
        y_low = center_y - 0.5 * world_height
        y_high = center_y + 0.5 * world_height
        height = max(
            1,
            int(round(self.render_size * world_height / (x_high - x_low))),
        )
        return x_low, x_high, y_low, y_high, height

    @staticmethod
    def _box_mask(xx: Array, yy: Array, box: OrientedBox) -> Array:
        dx = xx - box.center[0]
        dy = yy - box.center[1]
        c, s = np.cos(box.heading), np.sin(box.heading)
        longitudinal = c * dx + s * dy
        lateral = -s * dx + c * dy
        return (
            (np.abs(longitudinal) <= box.length / 2.0)
            & (np.abs(lateral) <= box.width / 2.0)
        )

    @staticmethod
    def _alpha_blend_mask(
        frame: Array, mask: Array, color: Array, alpha: float
    ) -> None:
        if not np.any(mask):
            return
        blended = (
            (1.0 - alpha) * frame[mask].astype(np.float32)
            + alpha * color.astype(np.float32)
        )
        frame[mask] = np.clip(blended, 0, 255).astype(np.uint8)

    def _draw_world_arrow(
        self,
        frame: Array,
        start: Array,
        end: Array,
        color: Array,
        thickness: int,
    ) -> None:
        """Draw a compact arrow specified in world coordinates."""
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        displacement = end - start
        length = float(np.linalg.norm(displacement))
        if length <= 1e-10:
            return
        direction = displacement / length
        perpendicular = np.array([-direction[1], direction[0]])
        self._draw_line(
            frame,
            self._world_to_pixel(start),
            self._world_to_pixel(end),
            color,
            thickness,
        )
        head_length = min(0.027, 0.42 * length)
        head_width = 0.62 * head_length
        for side in (-1.0, 1.0):
            wing = (
                end
                - head_length * direction
                + side * head_width * perpendicular
            )
            self._draw_line(
                frame,
                self._world_to_pixel(wing),
                self._world_to_pixel(end),
                color,
                thickness,
            )

    def _draw_target_vehicle(
        self,
        frame: Array,
        xx: Array,
        yy: Array,
        box: OrientedBox,
        color: Array,
    ) -> None:
        """Draw a translucent target-pose vehicle with the ego footprint."""
        c, s = np.cos(box.heading), np.sin(box.heading)
        dx = xx - box.center[0]
        dy = yy - box.center[1]
        longitudinal = c * dx + s * dy
        lateral = -s * dx + c * dy
        half_l = box.length / 2.0
        half_w = box.width / 2.0
        body = (
            (np.abs(longitudinal / (0.94 * half_l + 1e-8)) ** 6)
            + (np.abs(lateral / (0.90 * half_w + 1e-8)) ** 6)
            <= 1.0
        )
        self._alpha_blend_mask(frame, body, color, 0.14)
        roof = (
            (np.abs(longitudinal / (0.48 * half_l + 1e-8)) ** 4)
            + (np.abs(lateral / (0.63 * half_w + 1e-8)) ** 4)
            <= 1.0
        )
        self._alpha_blend_mask(
            frame,
            roof,
            np.array([213, 255, 244], dtype=np.uint8),
            0.13,
        )
        self._draw_oriented_outline(
            frame,
            box,
            color,
            max(1, self.render_size // 280),
        )
        forward = np.array([c, s], dtype=np.float64)
        center = np.asarray(box.center, dtype=np.float64)
        self._draw_world_arrow(
            frame,
            center - 0.12 * box.length * forward,
            center + 0.29 * box.length * forward,
            np.array([220, 255, 244], dtype=np.uint8),
            max(1, self.render_size // 300),
        )

    def _draw_vehicle(
        self,
        frame: Array,
        xx: Array,
        yy: Array,
        box: OrientedBox,
        body_color: Array,
        *,
        player: bool,
    ) -> None:
        """Draw a compact sedan with shadow, tires, glazing, and lights."""
        c, s = np.cos(box.heading), np.sin(box.heading)
        dx = xx - box.center[0]
        dy = yy - box.center[1]
        longitudinal = c * dx + s * dy
        lateral = -s * dx + c * dy
        half_l, half_w = box.length / 2.0, box.width / 2.0

        shadow_dx = xx - (box.center[0] + 0.014)
        shadow_dy = yy - (box.center[1] - 0.016)
        shadow_long = c * shadow_dx + s * shadow_dy
        shadow_lat = -s * shadow_dx + c * shadow_dy
        shadow = (
            (np.abs(shadow_long) <= 1.05 * half_l)
            & (np.abs(shadow_lat) <= 1.13 * half_w)
        )
        self._alpha_blend_mask(frame, shadow, np.array([0, 0, 0], dtype=np.uint8), 0.32)

        wheel_l = max(0.010, box.length * 0.12)
        wheel_w = max(0.005, box.width * 0.10)
        for wheel_x in (-0.29 * box.length, 0.29 * box.length):
            for wheel_y in (-0.54 * box.width, 0.54 * box.width):
                wheel = (
                    (np.abs(longitudinal - wheel_x) <= wheel_l)
                    & (np.abs(lateral - wheel_y) <= wheel_w)
                )
                frame[wheel] = np.array([10, 13, 17], dtype=np.uint8)

        outline = (
            (np.abs(longitudinal / (1.01 * half_l + 1e-8)) ** 6)
            + (np.abs(lateral / (1.06 * half_w + 1e-8)) ** 6)
            <= 1.0
        )
        frame[outline] = np.array([15, 22, 31], dtype=np.uint8)
        body = (
            (np.abs(longitudinal / (0.95 * half_l + 1e-8)) ** 6)
            + (np.abs(lateral / (0.91 * half_w + 1e-8)) ** 6)
            <= 1.0
        )
        frame[body] = body_color

        roof = (
            (np.abs((longitudinal + 0.03 * half_l) / (0.52 * half_l + 1e-8)) ** 4)
            + (np.abs(lateral / (0.68 * half_w + 1e-8)) ** 4)
            <= 1.0
        )
        frame[roof] = np.array([25, 48, 66], dtype=np.uint8)
        windshield = roof & (longitudinal > 0.07 * half_l)
        self._alpha_blend_mask(
            frame,
            windshield,
            np.array([105, 190, 216], dtype=np.uint8),
            0.34,
        )
        center_glint = body & (np.abs(lateral) < 0.10 * half_w)
        self._alpha_blend_mask(
            frame,
            center_glint,
            np.array([236, 245, 247], dtype=np.uint8),
            0.18 if player else 0.11,
        )
        # Small side mirrors and bumper accents make equal-sized cars easier
        # to compare at a glance.
        mirror_l = 0.10 * box.length
        mirror_w = 0.06 * box.width
        for mirror_x in (0.08 * box.length,):
            for mirror_y in (-0.56 * box.width, 0.56 * box.width):
                side_mirror = (
                    (np.abs(longitudinal - mirror_x) <= mirror_l)
                    & (np.abs(lateral - mirror_y) <= mirror_w)
                )
                frame[side_mirror] = np.array([17, 24, 30], dtype=np.uint8)

        front_bumper = body & (
            (longitudinal >= 0.83 * half_l)
            & (np.abs(lateral) <= 0.58 * half_w)
        )
        rear_bumper = body & (
            (longitudinal <= -0.83 * half_l)
            & (np.abs(lateral) <= 0.58 * half_w)
        )
        self._alpha_blend_mask(
            frame,
            front_bumper | rear_bumper,
            np.array([234, 239, 240], dtype=np.uint8),
            0.20 if player else 0.12,
        )

        headlights = (
            (longitudinal >= 0.76 * half_l)
            & (longitudinal <= 0.98 * half_l)
            & (np.abs(lateral) >= 0.42 * half_w)
            & (np.abs(lateral) <= 0.80 * half_w)
        )
        frame[headlights] = np.array([255, 239, 151], dtype=np.uint8)
        taillights = (
            (longitudinal <= -0.76 * half_l)
            & (longitudinal >= -0.98 * half_l)
            & (np.abs(lateral) >= 0.40 * half_w)
            & (np.abs(lateral) <= 0.82 * half_w)
        )
        frame[taillights] = np.array([236, 43, 49], dtype=np.uint8)

    def _draw_oriented_outline(
        self, frame: Array, box: OrientedBox, color: Array, width: int
    ) -> None:
        corners = _box_corners(box)
        for index in range(4):
            self._draw_line(
                frame,
                self._world_to_pixel(corners[index]),
                self._world_to_pixel(corners[(index + 1) % 4]),
                color,
                width,
            )

    @staticmethod
    def _draw_pixel_disk(
        frame: Array, center: tuple[int, int], radius: int, color: Array
    ) -> None:
        x0 = max(0, center[0] - radius)
        x1 = min(frame.shape[1], center[0] + radius + 1)
        y0 = max(0, center[1] - radius)
        y1 = min(frame.shape[0], center[1] + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
        frame[y0:y1, x0:x1][mask] = color

    def _draw_disk_world(
        self,
        frame: Array,
        center: Array,
        radius: float,
        color: Array,
    ) -> None:
        x_low, x_high, y_low, y_high, _ = self._render_view()
        pixels_per_world = frame.shape[1] / max(x_high - x_low, 1e-8)
        pixel_radius = max(1, int(round(radius * pixels_per_world)))
        self._draw_pixel_disk(
            frame, self._world_to_pixel(center), pixel_radius, color
        )

    def _blend_disk_world(
        self,
        frame: Array,
        center: Array,
        radius: float,
        color: Array,
        alpha: float,
    ) -> None:
        cx, cy = self._world_to_pixel(center)
        x_low, x_high, _, _, _ = self._render_view()
        pixels_per_world = frame.shape[1] / max(x_high - x_low, 1e-8)
        pixel_radius = max(1, int(round(radius * pixels_per_world)))
        x0 = max(0, cx - pixel_radius)
        x1 = min(frame.shape[1], cx + pixel_radius + 1)
        y0 = max(0, cy - pixel_radius)
        y1 = min(frame.shape[0], cy + pixel_radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= pixel_radius**2
        region = frame[y0:y1, x0:x1]
        blended = (
            (1.0 - alpha) * region[mask].astype(np.float32)
            + alpha * color.astype(np.float32)
        )
        region[mask] = np.clip(blended, 0, 255).astype(np.uint8)

    def _render_human(self, frame: Array) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("human rendering requires matplotlib") from exc
        if self._human_figure is None:
            aspect = frame.shape[1] / max(frame.shape[0], 1)
            self._human_figure, axis = plt.subplots(
                figsize=(7.0, 7.0 / aspect)
            )
            axis.axis("off")
            self._human_image = axis.imshow(frame)
            self._human_figure.tight_layout(pad=0)
            plt.show(block=False)
        else:
            self._human_image.set_data(frame)
        self._human_figure.canvas.draw_idle()
        self._human_figure.canvas.flush_events()

    def _world_to_pixel(self, point: Array) -> tuple[int, int]:
        x_low, x_high, y_low, y_high, height = self._render_view()
        width = self.render_size
        x = int(
            round(
                (float(point[0]) - x_low)
                / max(x_high - x_low, 1e-8)
                * (width - 1)
            )
        )
        y = int(
            round(
                (y_high - float(point[1]))
                / max(y_high - y_low, 1e-8)
                * (height - 1)
            )
        )
        return (
            int(np.clip(x, 0, width - 1)),
            int(np.clip(y, 0, height - 1)),
        )

    @staticmethod
    def _draw_line(
        frame: Array,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        length = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
        xs = np.rint(np.linspace(start[0], end[0], length + 1)).astype(int)
        ys = np.rint(np.linspace(start[1], end[1], length + 1)).astype(int)
        radius = max(0, width // 2)
        for x, y in zip(xs, ys):
            frame[
                max(0, y - radius) : min(frame.shape[0], y + radius + 1),
                max(0, x - radius) : min(frame.shape[1], x + radius + 1),
            ] = color


def register_environment() -> None:
    """Register all parking variants with Gymnasium (idempotently)."""
    registrations = {
        "CarParking-v0": "mixed",
        "CarParkingParallel-v0": "parallel",
        "CarParkingTForward-v0": "t_forward",
        "CarParkingTReverse-v0": "t_reverse",
        "CarParkingAngled-v0": "angled",
    }
    for env_id, maneuver in registrations.items():
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="car_parking.env:CarParkingEnv",
            kwargs={"config": CarParkingConfig(maneuver=maneuver)},
        )
