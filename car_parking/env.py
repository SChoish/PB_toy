"""Low-speed continuous-control car parking environments.

The dynamics and public observation modes intentionally mirror ``car_race``.
Unlike racing, parking success is pose based: the complete oriented vehicle
must be inside the bay, aligned with it, and nearly stationary for a short
dwell period.  Layouts use a lane-consistent approach convention and parked
vehicles have the same physical footprint as the controllable vehicle.
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
    terminate_on_collision: bool = True
    orientation_tolerance: float = np.deg2rad(10.0)
    parked_speed_tolerance: float = 0.025
    slot_margin: float = 0.008
    dwell_steps: int = 8

    reward_mode: RewardMode = "dense"
    step_penalty: float = 0.002
    control_cost: float = 0.0005
    position_scale: float = 1.0
    orientation_scale: float = 0.25
    containment_bonus: float = 0.08
    success_reward: float = 2.0
    collision_penalty: float = -2.0

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
    distance_to_slot, normalized_yaw_error, inside_slot, collision]``.
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
                1.0,
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
        self.success = False
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
                float(self.collision),
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
        self.success = False
        observation = self._get_observation()
        info = self._get_info(termination_reason=None)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: Array) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.collision or self.success or self.elapsed_steps >= self.config.max_episode_steps:
            raise RuntimeError("step() called after episode end; call reset() first")
        action = np.asarray(action, dtype=self._dtype)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector with shape (2,)")
        action = np.clip(action, -1.0, 1.0)
        previous_cost = self._pose_cost()

        target_steering = float(action[0]) * self.config.max_steer_angle
        throttle = float(action[1])
        collided = False
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
                self.config.max_acceleration if throttle * self.speed >= 0.0 else self.config.max_braking
            )
            acceleration = acceleration_limit * throttle - self.config.rolling_drag * self.speed
            candidate_speed = float(
                np.clip(
                    self.speed + acceleration * sub_dt,
                    -self.config.max_reverse_speed,
                    self.config.max_speed,
                )
            )
            yaw_rate = candidate_speed * np.tan(self.steering) / self.config.wheelbase
            candidate_heading = float(_wrap_angle(self.heading + yaw_rate * sub_dt))
            average_heading = self.heading + 0.5 * float(_wrap_angle(candidate_heading - self.heading))
            candidate_position = self.position + candidate_speed * sub_dt * np.array(
                [np.cos(average_heading), np.sin(average_heading)], dtype=self._dtype
            )
            candidate_box = OrientedBox(
                tuple(float(x) for x in candidate_position),
                self.config.car_length,
                self.config.car_width,
                candidate_heading,
            )
            if self._collides(candidate_box):
                self.speed = 0.0
                collided = True
                break
            self.position = candidate_position.astype(self._dtype)
            self.heading = candidate_heading
            self.speed = candidate_speed

        self.elapsed_steps += 1
        self.collision = bool(collided and self.config.terminate_on_collision)
        if self.parked_pose:
            self.dwell_count += 1
        else:
            self.dwell_count = 0
        self.success = self.dwell_count >= self.config.dwell_steps

        terminated = bool(self.success or self.collision)
        truncated = bool(
            not terminated and self.elapsed_steps >= self.config.max_episode_steps
        )
        reason = (
            "success" if self.success else "collision" if self.collision else "time_limit" if truncated else None
        )
        reward = self._step_reward(action, previous_cost, collided)
        observation = self._get_observation()
        info = self._get_info(termination_reason=reason)
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _collides(self, vehicle: OrientedBox) -> bool:
        corners = _box_corners(vehicle, margin=self.config.collision_margin)
        if np.any(corners < self.config.arena_low) or np.any(corners > self.config.arena_high):
            return True
        expanded = OrientedBox(
            vehicle.center,
            vehicle.length + 2.0 * self.config.collision_margin,
            vehicle.width + 2.0 * self.config.collision_margin,
            vehicle.heading,
        )
        return any(_boxes_overlap(expanded, obstacle) for obstacle in self.layout.obstacles)

    def _pose_cost(self) -> float:
        return float(
            self.config.position_scale * self.distance_to_goal
            + self.config.orientation_scale * abs(self.heading_error)
        )

    def _step_reward(self, action: Array, previous_cost: float, collided: bool) -> float:
        if self.success:
            return float(self.config.success_reward)
        if collided:
            return float(self.config.collision_penalty)
        reward = -self.config.step_penalty - self.config.control_cost * float(np.dot(action, action))
        if self.config.reward_mode == "dense":
            reward += previous_cost - self._pose_cost()
            if self.fully_inside_slot:
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
        """Render a coherent two-lane parking lot without changing state.

        The visual geometry is derived from the same oriented boxes used by
        collision checking.  In particular, parked cars and the ego vehicle
        share the same footprint, and the highlighted approach lane is the lane
        adjacent to the requested parking row.
        """
        size = self.render_size
        low, high = self.config.arena_low, self.config.arena_high
        span = high - low
        xs = np.linspace(low, high, size)
        ys = np.linspace(high, low, size)
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

        # --------------------------------------------------------------
        # Asphalt base with deterministic aggregate and two wheel tracks.
        # --------------------------------------------------------------
        aggregate = (
            3.0 * np.sin(83.0 * xx + 41.0 * yy)
            + 1.9 * np.cos(137.0 * xx - 67.0 * yy)
            + 1.2 * np.sin(211.0 * xx + 19.0 * yy)
        )
        asphalt = np.clip(57.0 + aggregate, 45.0, 69.0)
        frame = np.empty((size, size, 3), dtype=np.uint8)
        frame[..., 0] = np.clip(asphalt * 0.94, 0, 255).astype(np.uint8)
        frame[..., 1] = np.clip(asphalt, 0, 255).astype(np.uint8)
        frame[..., 2] = np.clip(asphalt * 1.07, 0, 255).astype(np.uint8)

        lane_half_width = 0.25 * self.config.aisle_width
        road_low = min(row_inner_y, road_edge)
        road_high = max(row_inner_y, road_edge)
        road_mask = (yy >= road_low) & (yy <= road_high)
        approach_lane = road_mask & (
            mirror * (yy - road_center) >= 0.0
        )
        opposite_lane = road_mask & ~approach_lane

        # A very light tint makes the correct approach lane legible without
        # looking like a navigation overlay.
        self._alpha_blend_mask(
            frame,
            approach_lane,
            np.array([72, 85, 91], dtype=np.uint8),
            0.10,
        )
        self._alpha_blend_mask(
            frame,
            opposite_lane,
            np.array([36, 42, 48], dtype=np.uint8),
            0.05,
        )

        tire_offset = min(0.035, 0.32 * self.config.car_width)
        rubber = np.zeros_like(xx, dtype=np.float64)
        for lane_center, lane_strength in (
            (approach_lane_center, 4.8),
            (opposite_lane_center, 3.1),
        ):
            for offset in (-tire_offset, tire_offset):
                rubber += lane_strength * np.exp(
                    -((yy - (lane_center + offset)) / 0.030) ** 2
                )
        rubber *= road_mask
        frame[..., 0] = np.clip(
            frame[..., 0].astype(np.float32) - rubber, 0, 255
        ).astype(np.uint8)
        frame[..., 1] = np.clip(
            frame[..., 1].astype(np.float32) - rubber, 0, 255
        ).astype(np.uint8)
        frame[..., 2] = np.clip(
            frame[..., 2].astype(np.float32) - 0.72 * rubber, 0, 255
        ).astype(np.uint8)

        # --------------------------------------------------------------
        # Opposite-side island, curb, landscaping, and parking-row curb.
        # --------------------------------------------------------------
        for obstacle in self.layout.obstacles:
            mask = self._box_mask(xx, yy, obstacle)
            if obstacle.kind == "island":
                tile_x = np.floor((xx - low) / 0.105).astype(np.int32)
                tile_y = np.floor((yy - low) / 0.105).astype(np.int32)
                tile = (tile_x + tile_y) % 2
                paver = 151 + 7 * tile + 2.0 * np.sin(47.0 * xx - 29.0 * yy)
                frame[..., 0][mask] = np.clip(
                    paver[mask] * 1.04, 0, 255
                ).astype(np.uint8)
                frame[..., 1][mask] = np.clip(paver[mask], 0, 255).astype(
                    np.uint8
                )
                frame[..., 2][mask] = np.clip(
                    paver[mask] * 0.94, 0, 255
                ).astype(np.uint8)

                near_edge = obstacle.center[1] + mirror * obstacle.width / 2.0
                depth = -mirror * (yy - near_edge)
                landscape = mask & (depth >= 0.22)
                grass = (
                    74.0
                    + 5.0 * np.sin(23.0 * xx + 11.0 * yy)
                    + 3.0 * np.cos(41.0 * xx - 17.0 * yy)
                )
                frame[..., 0][landscape] = np.clip(
                    grass[landscape] * 0.48, 0, 255
                ).astype(np.uint8)
                frame[..., 1][landscape] = np.clip(
                    grass[landscape] * 1.18, 0, 255
                ).astype(np.uint8)
                frame[..., 2][landscape] = np.clip(
                    grass[landscape] * 0.58, 0, 255
                ).astype(np.uint8)

                planter_edge = mask & (np.abs(depth - 0.22) <= 0.018)
                frame[planter_edge] = np.array([202, 195, 176], dtype=np.uint8)
                shrub_y = near_edge - mirror * 0.42
                for shrub_x in np.linspace(-0.80, 0.80, 6):
                    shrub_shadow = (
                        (xx - (shrub_x + 0.016)) ** 2
                        + (yy - (shrub_y - 0.020)) ** 2
                        <= 0.070**2
                    )
                    self._alpha_blend_mask(
                        frame,
                        shrub_shadow & landscape,
                        np.array([4, 9, 7], dtype=np.uint8),
                        0.34,
                    )
                    shrub = (
                        (xx - shrub_x) ** 2 + (yy - shrub_y) ** 2 <= 0.059**2
                    )
                    frame[shrub & landscape] = np.array(
                        [37, 105, 59], dtype=np.uint8
                    )
                    highlight = (
                        (xx - (shrub_x - 0.015)) ** 2
                        + (yy - (shrub_y + 0.014)) ** 2
                        <= 0.031**2
                    )
                    self._alpha_blend_mask(
                        frame,
                        highlight & landscape,
                        np.array([92, 164, 89], dtype=np.uint8),
                        0.55,
                    )

                curb = mask & (np.abs(yy - near_edge) <= 0.024)
                frame[curb] = np.array([213, 208, 190], dtype=np.uint8)
                curb_shadow = np.abs(
                    yy - (near_edge + mirror * 0.028)
                ) <= 0.012
                self._alpha_blend_mask(
                    frame,
                    curb_shadow,
                    np.array([4, 7, 10], dtype=np.uint8),
                    0.34,
                )
            elif obstacle.kind == "curb":
                local_x = xx - obstacle.center[0]
                segments = (
                    np.floor((local_x + 1.0) / 0.12).astype(np.int32) % 2
                )
                frame[mask & (segments == 0)] = np.array(
                    [231, 193, 63], dtype=np.uint8
                )
                frame[mask & (segments == 1)] = np.array(
                    [38, 41, 45], dtype=np.uint8
                )

        # --------------------------------------------------------------
        # Lane boundaries, center dashes, and directional markings.
        # --------------------------------------------------------------
        edge_width = max(0.005, 1.25 * span / size)
        row_edge_line = np.abs(yy - row_inner_y) <= edge_width
        island_edge_line = np.abs(yy - road_edge) <= edge_width
        self._alpha_blend_mask(
            frame,
            row_edge_line | island_edge_line,
            np.array([228, 230, 220], dtype=np.uint8),
            0.90,
        )

        dash = np.mod(xx - low, 0.27) < 0.14
        center_line = (
            np.abs(yy - road_center) <= max(0.005, 1.3 * span / size)
        ) & dash
        self._alpha_blend_mask(
            frame,
            center_line,
            np.array([228, 186, 64], dtype=np.uint8),
            0.90,
        )

        travel_direction = np.array(
            [np.cos(self.layout.start_heading), np.sin(self.layout.start_heading)],
            dtype=np.float64,
        )
        arrow_color = np.array([185, 193, 194], dtype=np.uint8)
        for x_position in (-0.48, 0.38):
            start = np.array([x_position, approach_lane_center]) - 0.055 * travel_direction
            end = np.array([x_position, approach_lane_center]) + 0.055 * travel_direction
            self._draw_world_arrow(
                frame,
                start,
                end,
                arrow_color,
                max(1, size // 280),
            )

            reverse_direction = -travel_direction
            start = np.array([x_position, opposite_lane_center]) - 0.050 * reverse_direction
            end = np.array([x_position, opposite_lane_center]) + 0.050 * reverse_direction
            self._draw_world_arrow(
                frame,
                start,
                end,
                np.array([145, 151, 154], dtype=np.uint8),
                max(1, size // 300),
            )

        # A subtle blue start stencil remains visible after the car moves.
        start_box = OrientedBox(
            self.layout.start,
            self.config.car_length * 1.04,
            self.config.car_width * 1.08,
            self.layout.start_heading,
            "start",
        )
        start_mask = self._box_mask(xx, yy, start_box)
        self._alpha_blend_mask(
            frame,
            start_mask,
            np.array([49, 147, 231], dtype=np.uint8),
            0.055,
        )
        self._draw_oriented_outline(
            frame,
            start_box,
            np.array([72, 156, 222], dtype=np.uint8),
            max(1, size // 360),
        )

        # --------------------------------------------------------------
        # All bays use the target bay dimensions.  Parked cars remain exact
        # ego-car size, so visual and collision geometry agree.
        # --------------------------------------------------------------
        bay_outline = np.array([178, 184, 180], dtype=np.uint8)
        bay_boxes: list[OrientedBox] = [slot]
        for obstacle in self.layout.obstacles:
            if obstacle.kind != "vehicle":
                continue
            bay_boxes.append(
                OrientedBox(
                    obstacle.center,
                    slot.length,
                    slot.width,
                    obstacle.heading,
                    "bay",
                )
            )

        for bay in bay_boxes[1:]:
            self._draw_oriented_outline(
                frame, bay, bay_outline, max(1, size // 300)
            )
            if self.layout.maneuver != "parallel":
                self._draw_parking_stop(
                    frame,
                    bay,
                    mirror,
                    np.array([224, 198, 92], dtype=np.uint8),
                )

        slot_mask = self._box_mask(xx, yy, slot)
        guide_color = np.array(
            [62, 232, 139] if self.success else [51, 219, 189],
            dtype=np.uint8,
        )
        pulse = 0.105 + 0.020 * np.sin(0.20 * self.elapsed_steps)
        self._alpha_blend_mask(frame, slot_mask, guide_color, pulse)
        self._draw_oriented_outline(
            frame,
            slot,
            np.array([236, 241, 230], dtype=np.uint8),
            max(2, size // 225),
        )
        inset = OrientedBox(
            slot.center,
            slot.length - 0.020,
            slot.width - 0.020,
            slot.heading,
            "slot_inset",
        )
        self._draw_oriented_outline(
            frame, inset, guide_color, max(1, size // 320)
        )
        if self.layout.maneuver != "parallel":
            self._draw_parking_stop(
                frame,
                slot,
                mirror,
                np.array([247, 218, 103], dtype=np.uint8),
            )

        target_vehicle = OrientedBox(
            slot.center,
            self.config.car_length,
            self.config.car_width,
            slot.heading,
            "target_vehicle",
        )
        self._draw_target_vehicle(
            frame,
            xx,
            yy,
            target_vehicle,
            guide_color,
        )

        # --------------------------------------------------------------
        # Parked cars and ego car.
        # --------------------------------------------------------------
        parked_colors = (
            np.array([202, 70, 64], dtype=np.uint8),
            np.array([221, 158, 48], dtype=np.uint8),
            np.array([108, 116, 132], dtype=np.uint8),
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

        ego_color = np.array(
            [52, 219, 126]
            if self.success
            else [230, 63, 55]
            if self.collision
            else [35, 133, 233],
            dtype=np.uint8,
        )
        self._draw_vehicle(
            frame, xx, yy, self.vehicle_box, ego_color, player=True
        )

        # --------------------------------------------------------------
        # Compact HUD: position, alignment, stop, and dwell quality.
        # --------------------------------------------------------------
        panel_margin = max(8, size // 42)
        panel_width = max(150, size // 3)
        panel_height = max(70, size // 6)
        panel_y = (
            panel_margin
            if mirror < 0.0
            else size - panel_margin - panel_height
        )
        self._blend_rect(
            frame,
            panel_margin,
            panel_y,
            panel_margin + panel_width,
            panel_y + panel_height,
            np.array([8, 14, 20], dtype=np.uint8),
            0.84,
        )
        meter_x0 = panel_margin + max(25, size // 20)
        meter_x1 = panel_margin + panel_width - max(10, size // 68)
        meter_h = max(5, size // 92)
        alignment_window = max(self.config.orientation_tolerance * 5.0, 0.45)
        values = (
            1.0 - np.clip(self.distance_to_goal / 0.72, 0.0, 1.0),
            1.0 - np.clip(abs(self.heading_error) / alignment_window, 0.0, 1.0),
            1.0 - np.clip(
                abs(self.speed)
                / max(self.config.max_speed, self.config.max_reverse_speed),
                0.0,
                1.0,
            ),
            np.clip(self.dwell_count / self.config.dwell_steps, 0.0, 1.0),
        )
        colors = (
            np.array([52, 163, 235], dtype=np.uint8),
            np.array([238, 188, 66], dtype=np.uint8),
            np.array([188, 111, 226], dtype=np.uint8),
            np.array([55, 222, 126], dtype=np.uint8),
        )
        row_gap = max(5, size // 100)
        for row, (value, color) in enumerate(zip(values, colors)):
            y0 = panel_y + max(9, size // 74) + row * (meter_h + row_gap)
            frame[y0 : y0 + meter_h, meter_x0:meter_x1] = np.array(
                [42, 48, 55], dtype=np.uint8
            )
            fill_x = meter_x0 + int(
                (meter_x1 - meter_x0) * float(value)
            )
            frame[y0 : y0 + meter_h, meter_x0:fill_x] = color
            self._draw_pixel_disk(
                frame,
                (panel_margin + max(13, size // 48), y0 + meter_h // 2),
                max(2, size // 150),
                color,
            )

        # Parking badge and maneuver pips.
        badge_x = size - panel_margin - max(24, size // 20)
        badge_y = panel_y + max(10, size // 60)
        badge_w = max(3, size // 130)
        badge_color = np.array([84, 221, 246], dtype=np.uint8)
        self._draw_line(
            frame,
            (badge_x, badge_y),
            (badge_x, badge_y + 28),
            badge_color,
            badge_w,
        )
        self._draw_line(
            frame,
            (badge_x, badge_y),
            (badge_x + 14, badge_y),
            badge_color,
            badge_w,
        )
        self._draw_line(
            frame,
            (badge_x + 14, badge_y),
            (badge_x + 14, badge_y + 13),
            badge_color,
            badge_w,
        )
        self._draw_line(
            frame,
            (badge_x, badge_y + 13),
            (badge_x + 14, badge_y + 13),
            badge_color,
            badge_w,
        )
        mode_index = MANEUVERS.index(self.layout.maneuver)
        for index in range(len(MANEUVERS)):
            color = (
                badge_color
                if index == mode_index
                else np.array([57, 67, 76], dtype=np.uint8)
            )
            self._draw_pixel_disk(
                frame,
                (
                    badge_x - 2 + index * max(8, size // 60),
                    badge_y + 40,
                ),
                max(2, size // 170),
                color,
            )

        border = max(3, size // 150)
        frame[:border] = np.array([14, 18, 23], dtype=np.uint8)
        frame[-border:] = np.array([14, 18, 23], dtype=np.uint8)
        frame[:, :border] = np.array([14, 18, 23], dtype=np.uint8)
        frame[:, -border:] = np.array([14, 18, 23], dtype=np.uint8)
        accent = max(10, size // 17)
        frame[:border, :accent] = np.array([39, 151, 225], dtype=np.uint8)
        frame[:border, accent : 2 * accent] = np.array(
            [233, 190, 64], dtype=np.uint8
        )
        frame[-border:, -accent:] = np.array(
            [53, 220, 128], dtype=np.uint8
        )
        return frame

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

    def _draw_parking_stop(
        self,
        frame: Array,
        bay: OrientedBox,
        mirror: float,
        color: Array,
    ) -> None:
        """Draw a wheel stop at the curb-side longitudinal end of a bay."""
        forward = np.array(
            [np.cos(bay.heading), np.sin(bay.heading)], dtype=np.float64
        )
        lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
        row_normal = np.array([0.0, mirror], dtype=np.float64)
        end_sign = 1.0 if float(np.dot(forward, row_normal)) >= 0.0 else -1.0
        center = (
            np.asarray(bay.center, dtype=np.float64)
            + end_sign * max(0.02, bay.length / 2.0 - 0.032) * forward
        )
        half_width = 0.34 * bay.width
        self._draw_line(
            frame,
            self._world_to_pixel(center - half_width * lateral),
            self._world_to_pixel(center + half_width * lateral),
            color,
            max(2, self.render_size // 240),
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
    def _blend_rect(
        frame: Array,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: Array,
        alpha: float,
    ) -> None:
        region = frame[max(0, y0) : min(frame.shape[0], y1), max(0, x0) : min(frame.shape[1], x1)]
        blended = (1.0 - alpha) * region.astype(np.float32) + alpha * color.astype(np.float32)
        region[:] = np.clip(blended, 0, 255).astype(np.uint8)

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

    def _render_human(self, frame: Array) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("human rendering requires matplotlib") from exc
        if self._human_figure is None:
            self._human_figure, axis = plt.subplots(figsize=(6, 6))
            axis.axis("off")
            self._human_image = axis.imshow(frame)
            plt.show(block=False)
        else:
            self._human_image.set_data(frame)
        self._human_figure.canvas.draw_idle()
        self._human_figure.canvas.flush_events()

    def _world_to_pixel(self, point: Array) -> tuple[int, int]:
        scale = (self.render_size - 1) / (
            self.config.arena_high - self.config.arena_low
        )
        x = int(round((float(point[0]) - self.config.arena_low) * scale))
        y = int(round((self.config.arena_high - float(point[1])) * scale))
        return (
            int(np.clip(x, 0, self.render_size - 1)),
            int(np.clip(y, 0, self.render_size - 1)),
        )

    def _draw_polygon(self, frame: Array, polygon: Array, color: tuple[int, int, int]) -> None:
        pixels = np.array([self._world_to_pixel(point) for point in polygon], dtype=np.float64)
        x0 = max(0, int(np.floor(pixels[:, 0].min())))
        x1 = min(self.render_size - 1, int(np.ceil(pixels[:, 0].max())))
        y0 = max(0, int(np.floor(pixels[:, 1].min())))
        y1 = min(self.render_size - 1, int(np.ceil(pixels[:, 1].max())))
        if x0 > x1 or y0 > y1:
            return
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        inside = np.zeros(xx.shape, dtype=bool)
        j = len(pixels) - 1
        for i in range(len(pixels)):
            xi, yi = pixels[i]
            xj, yj = pixels[j]
            crosses = ((yi > yy) != (yj > yy)) & (
                xx < (xj - xi) * (yy - yi) / (yj - yi + 1e-12) + xi
            )
            inside ^= crosses
            j = i
        frame[y0 : y1 + 1, x0 : x1 + 1][inside] = color

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
