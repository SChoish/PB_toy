"""RGB rendering helpers for OrbitalSwingByEnv."""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray


class OrbitalSwingByRenderer:
    """Mixin providing rgb_array / human rendering for the orbital env."""

    # Attributes provided by the host environment.
    config: Any
    render_size: int
    render_mode: str | None
    metadata: dict[str, Any]
    position: Array
    velocity: Array
    goal: Array
    goal_velocity: Array
    fuel: float
    fuel_fraction: float
    elapsed_steps: int
    dead: bool
    _body_center: Array
    _trail: list[Array]
    _stars: Array
    _star_brightness: Array
    _render_xs: Array
    _render_ys: Array
    _render_xx: Array
    _render_yy: Array
    _last_action_angle: float
    _last_actual_throttle: float
    _human_figure: Any
    _human_axis: Any
    _human_image: Any

    def _init_renderer(self) -> None:
        cfg = self.config
        rng = np.random.default_rng(2027)
        self._stars = rng.uniform(0.02, 0.98, size=(170, 2)).astype(np.float32)
        self._star_brightness = rng.integers(120, 256, size=170, dtype=np.uint8)
        self._render_xs = np.linspace(
            cfg.arena_low, cfg.arena_high, self.render_size, dtype=np.float32
        )
        self._render_ys = np.linspace(
            cfg.arena_high, cfg.arena_low, self.render_size, dtype=np.float32
        )
        self._render_xx, self._render_yy = np.meshgrid(
            self._render_xs, self._render_ys
        )
        self._human_figure = None
        self._human_axis = None
        self._human_image = None

    def render(self) -> Array | None:
        frame = self._render_rgb()
        if self.render_mode == "rgb_array":
            return frame
        if self.render_mode == "human":
            self._render_human(frame)
            return None
        return None

    def close_renderer(self) -> None:
        if self._human_figure is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._human_figure)
            except ImportError:
                pass
        self._human_figure = None
        self._human_axis = None
        self._human_image = None

    def _render_rgb(self) -> Array:
        size = self.render_size
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:] = np.array([4, 7, 17], dtype=np.uint8)

        rows = np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None]
        cols = np.linspace(-1.0, 1.0, size, dtype=np.float32)[None, :]
        nebula = np.exp(-((cols + 0.45) ** 2 / 0.50 + (rows - 0.25) ** 2 / 0.22))
        frame[..., 0] = np.clip(frame[..., 0] + 15 * nebula, 0, 255)
        frame[..., 1] = np.clip(frame[..., 1] + 11 * nebula, 0, 255)
        frame[..., 2] = np.clip(frame[..., 2] + 30 * nebula, 0, 255)

        for (sx, sy), brightness in zip(
            self._stars, self._star_brightness, strict=False
        ):
            x = int(sx * (size - 1))
            y = int(sy * (size - 1))
            radius = 1 if brightness < 225 else 2
            y0, y1 = max(0, y - radius), min(size, y + radius + 1)
            x0, x1 = max(0, x - radius), min(size, x + radius + 1)
            color = np.array(
                [brightness, brightness, min(255, int(brightness) + 16)],
                dtype=np.uint8,
            )
            frame[y0:y1, x0:x1] = color

        self._draw_goal(frame)

        if len(self._trail) >= 2:
            trail = self._trail[-260:]
            for index in range(1, len(trail)):
                alpha = index / len(trail)
                color = np.array(
                    [
                        30 + int(55 * alpha),
                        100 + int(105 * alpha),
                        145 + int(90 * alpha),
                    ],
                    dtype=np.uint8,
                )
                self._draw_line(
                    frame,
                    self._world_to_pixel(trail[index - 1]),
                    self._world_to_pixel(trail[index]),
                    color,
                    max(1, size // 500),
                )

        if self.config.show_ballistic_prediction:
            self._draw_ballistic_prediction(frame)

        if self.config.body_kind == "black_hole":
            self._draw_black_hole(frame)
        else:
            self._draw_planet(frame)

        self._draw_satellite(frame)
        self._draw_fuel_hud(frame)

        border = max(2, size // 260)
        border_color = np.array([60, 73, 94], dtype=np.uint8)
        frame[:border] = border_color
        frame[-border:] = border_color
        frame[:, :border] = border_color
        frame[:, -border:] = border_color
        return frame

    def _draw_ballistic_prediction(self, frame: Array) -> None:
        """Faint coasting trajectory preview (visual only; not in observations)."""
        predict = getattr(self, "predict_ballistic_trajectory", None)
        if predict is None:
            return
        path = predict(horizon_steps=48)
        if path.shape[0] < 2:
            return
        color = np.array([92, 108, 138], dtype=np.uint8)
        thickness = max(1, self.render_size // 480)
        for index in range(1, path.shape[0]):
            self._draw_line(
                frame,
                self._world_to_pixel(path[index - 1]),
                self._world_to_pixel(path[index]),
                color,
                thickness,
            )

    def _draw_goal(self, frame: Array) -> None:
        pulse = 0.5 + 0.5 * np.sin(0.11 * self.elapsed_steps)
        outer = self.config.goal_radius * (1.45 + 0.10 * pulse)
        self._blend_ring(
            frame,
            self.goal,
            outer_radius=outer,
            inner_radius=self.config.goal_radius * 0.92,
            color=np.array([80, 244, 169], dtype=np.uint8),
            alpha=0.45,
        )
        self._blend_ring(
            frame,
            self.goal,
            outer_radius=self.config.goal_radius * 0.92,
            inner_radius=self.config.goal_radius * 0.72,
            color=np.array([190, 255, 225], dtype=np.uint8),
            alpha=0.78,
        )
        velocity_end = self.goal + 0.22 * self.goal_velocity
        self._draw_world_arrow(
            frame,
            self.goal,
            velocity_end,
            np.array([104, 255, 184], dtype=np.uint8),
            max(1, self.render_size // 300),
        )

    def _draw_planet(self, frame: Array) -> None:
        _, _, xx, yy = self._coordinate_grid()
        dx = xx - self._body_center[0]
        dy = yy - self._body_center[1]
        radius = np.sqrt(dx * dx + dy * dy)

        atmosphere = (radius <= self.config.body_radius * 1.15) & (
            radius > self.config.body_radius
        )
        if np.any(atmosphere):
            alpha = np.clip(
                (self.config.body_radius * 1.15 - radius[atmosphere])
                / (self.config.body_radius * 0.15),
                0.0,
                1.0,
            )
            base = frame[atmosphere].astype(np.float32)
            glow = np.array([62, 157, 255], dtype=np.float32)
            frame[atmosphere] = (
                (1.0 - 0.55 * alpha[:, None]) * base
                + (0.55 * alpha[:, None]) * glow
            ).astype(np.uint8)

        disk = radius <= self.config.body_radius
        if not np.any(disk):
            return
        nx = dx[disk] / self.config.body_radius
        ny = dy[disk] / self.config.body_radius
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
        light = np.clip(0.18 + 0.80 * (0.32 * nx - 0.22 * ny + 0.92 * nz), 0.0, 1.0)
        bands = 0.5 + 0.5 * np.sin(19.0 * ny + 2.8 * np.sin(8.0 * nx))
        base = np.stack(
            [
                40 + 38 * bands,
                83 + 65 * bands,
                118 + 84 * bands,
            ],
            axis=1,
        )
        shaded = np.clip(base * (0.45 + 0.72 * light[:, None]), 0, 255)
        frame[disk] = shaded.astype(np.uint8)

        self._blend_ring(
            frame,
            self._body_center,
            outer_radius=self.config.body_radius * 1.025,
            inner_radius=self.config.body_radius * 0.985,
            color=np.array([188, 225, 255], dtype=np.uint8),
            alpha=0.82,
        )

    def _draw_black_hole(self, frame: Array) -> None:
        center = self._body_center
        size = self.render_size
        xs = np.linspace(self.config.arena_low, self.config.arena_high, size)
        ys = np.linspace(self.config.arena_high, self.config.arena_low, size)
        xx, yy = np.meshgrid(xs, ys)
        dx = xx - center[0]
        dy = yy - center[1]
        elliptical_r = np.sqrt((dx / 1.0) ** 2 + (dy / 0.34) ** 2)
        angle = np.arctan2(dy / 0.34, dx)

        disk_inner = self.config.body_radius * 1.12
        disk_outer = self.config.body_radius * 2.25
        disk = (elliptical_r >= disk_inner) & (elliptical_r <= disk_outer)
        if np.any(disk):
            t = np.clip(
                (disk_outer - elliptical_r[disk]) / (disk_outer - disk_inner),
                0.0,
                1.0,
            )
            asymmetry = 0.62 + 0.38 * np.cos(angle[disk] - 0.45)
            color = np.stack(
                [
                    185 + 70 * t,
                    45 + 150 * t,
                    18 + 58 * t,
                ],
                axis=1,
            )
            color *= asymmetry[:, None]
            alpha = (0.30 + 0.58 * t)[:, None]
            base = frame[disk].astype(np.float32)
            frame[disk] = np.clip(
                (1.0 - alpha) * base + alpha * color,
                0,
                255,
            ).astype(np.uint8)

        self._blend_ring(
            frame,
            center,
            outer_radius=self.config.body_radius * 1.17,
            inner_radius=self.config.body_radius * 1.04,
            color=np.array([255, 215, 118], dtype=np.uint8),
            alpha=0.92,
        )
        self._blend_disk(
            frame,
            center,
            self.config.body_radius,
            np.array([0, 0, 0], dtype=np.uint8),
            1.0,
        )
        self._blend_disk(
            frame,
            center + np.array([-0.025, 0.018], dtype=np.float32),
            self.config.body_radius * 0.45,
            np.array([13, 9, 24], dtype=np.uint8),
            0.44,
        )

    def _draw_satellite(self, frame: Array) -> None:
        size = self.render_size
        _, _, xx, yy = self._coordinate_grid()
        dx = xx - self.position[0]
        dy = yy - self.position[1]

        speed = float(np.linalg.norm(self.velocity))
        orientation = (
            float(np.arctan2(self.velocity[1], self.velocity[0]))
            if speed > 0.035
            else self._last_action_angle
        )
        c, s = np.cos(orientation), np.sin(orientation)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy

        scale = self.config.satellite_radius
        body = (np.abs(local_x) <= scale * 0.85) & (np.abs(local_y) <= scale * 0.55)
        left_panel = (local_x >= -scale * 2.65) & (local_x <= -scale * 1.00) & (
            np.abs(local_y) <= scale * 0.72
        )
        right_panel = (local_x <= scale * 2.65) & (local_x >= scale * 1.00) & (
            np.abs(local_y) <= scale * 0.72
        )
        panel_lines = (
            (np.abs(np.mod((local_x / max(scale, 1e-8)) + 8.0, 0.55) - 0.275) < 0.035)
            & (left_panel | right_panel)
        )

        frame[left_panel | right_panel] = np.array([38, 91, 161], dtype=np.uint8)
        frame[panel_lines] = np.array([104, 178, 232], dtype=np.uint8)
        frame[body] = np.array([211, 218, 225], dtype=np.uint8)

        core = (local_x**2 + (local_y / 0.72) ** 2) <= (scale * 0.52) ** 2
        frame[core] = np.array([245, 186, 63], dtype=np.uint8)

        if self._last_actual_throttle > 1e-4 and self.fuel >= 0.0:
            thrust_angle = self._last_action_angle
            ct, st = np.cos(thrust_angle), np.sin(thrust_angle)
            tx = ct * dx + st * dy
            ty = -st * dx + ct * dy
            flame_length = scale * (1.2 + 2.2 * self._last_actual_throttle)
            flame = (
                (tx <= -scale * 0.75)
                & (tx >= -scale * 0.75 - flame_length)
                & (
                    np.abs(ty)
                    <= scale * 0.42 * (1.0 + (tx + scale * 0.75) / flame_length)
                )
            )
            frame[flame] = np.array([255, 112, 42], dtype=np.uint8)

        velocity_end = self.position + 0.20 * self.velocity
        self._draw_world_arrow(
            frame,
            self.position,
            velocity_end,
            np.array([90, 221, 255], dtype=np.uint8),
            max(1, size // 360),
        )

        if self.dead:
            p = self._world_to_pixel(self.position)
            r = max(5, size // 70)
            x_color = np.array([255, 70, 70], dtype=np.uint8)
            thickness = max(2, size // 220)
            self._draw_line(frame, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), x_color, thickness)
            self._draw_line(frame, (p[0] - r, p[1] + r), (p[0] + r, p[1] - r), x_color, thickness)

    def _draw_fuel_hud(self, frame: Array) -> None:
        size = self.render_size
        x0 = max(10, size // 28)
        y0 = max(10, size // 30)
        width = max(90, size // 4)
        height = max(11, size // 42)
        segments = 12
        gap = max(1, size // 400)
        segment_width = (width - gap * (segments - 1)) // segments

        icon_w = max(15, size // 38)
        frame[y0 : y0 + height, x0 : x0 + icon_w] = np.array(
            [126, 139, 155], dtype=np.uint8
        )
        frame[
            y0 + height // 3 : y0 + 2 * height // 3,
            x0 + icon_w : x0 + icon_w + max(3, size // 180),
        ] = np.array([200, 206, 216], dtype=np.uint8)

        bar_x = x0 + icon_w + max(8, size // 80)
        filled = int(np.ceil(self.fuel_fraction * segments - 1e-9))
        for index in range(segments):
            sx0 = bar_x + index * (segment_width + gap)
            sx1 = sx0 + segment_width
            if index < filled:
                frac = index / max(segments - 1, 1)
                color = np.array(
                    [56 + int(178 * (1.0 - frac)), 210 - int(90 * frac), 104],
                    dtype=np.uint8,
                )
            else:
                color = np.array([37, 44, 57], dtype=np.uint8)
            frame[y0 : y0 + height, sx0:sx1] = color

        # Approximate remaining delta-v bar under the fuel tank.
        isp_proxy = self.config.max_thrust_force / max(
            self.config.fuel_burn_rate * self.config.fuel_mass_scale, 1e-8
        )
        delta_v_remaining = isp_proxy * np.log1p(
            self.config.fuel_mass_scale
            * self.fuel
            / max(self.config.dry_mass, 1e-8)
        )
        delta_v_capacity = isp_proxy * np.log1p(
            self.config.fuel_mass_scale
            * self.config.fuel_capacity
            / max(self.config.dry_mass, 1e-8)
        )
        dv_frac = float(
            np.clip(delta_v_remaining / max(delta_v_capacity, 1e-8), 0.0, 1.0)
        )
        dv_y = y0 + height + max(4, size // 120)
        dv_height = max(6, size // 70)
        dv_filled = int(np.ceil(dv_frac * segments - 1e-9))
        for index in range(segments):
            sx0 = bar_x + index * (segment_width + gap)
            sx1 = sx0 + segment_width
            color = (
                np.array([70, 150, 220], dtype=np.uint8)
                if index < dv_filled
                else np.array([28, 34, 46], dtype=np.uint8)
            )
            frame[dv_y : dv_y + dv_height, sx0:sx1] = color

    def _coordinate_grid(self) -> tuple[Array, Array, Array, Array]:
        return (
            self._render_xs,
            self._render_ys,
            self._render_xx,
            self._render_yy,
        )

    def _blend_disk(
        self,
        frame: Array,
        center: Array,
        radius: float,
        color: Array,
        alpha: float,
    ) -> None:
        _, _, xx, yy = self._coordinate_grid()
        mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
        if not np.any(mask):
            return
        base = frame[mask].astype(np.float32)
        frame[mask] = np.clip(
            (1.0 - alpha) * base + alpha * color.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

    def _blend_ring(
        self,
        frame: Array,
        center: Array,
        outer_radius: float,
        inner_radius: float,
        color: Array,
        alpha: float,
    ) -> None:
        _, _, xx, yy = self._coordinate_grid()
        squared = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        mask = (squared <= outer_radius**2) & (squared >= inner_radius**2)
        if not np.any(mask):
            return
        base = frame[mask].astype(np.float32)
        frame[mask] = np.clip(
            (1.0 - alpha) * base + alpha * color.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

    def _draw_world_arrow(
        self,
        frame: Array,
        start: Array,
        end: Array,
        color: Array,
        thickness: int,
    ) -> None:
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        displacement = end - start
        length = float(np.linalg.norm(displacement))
        if length <= 1e-8:
            return
        direction = displacement / length
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
        self._draw_line(
            frame,
            self._world_to_pixel(start),
            self._world_to_pixel(end),
            color,
            thickness,
        )
        head_length = min(0.035, length * 0.38)
        head_width = head_length * 0.58
        left = end - head_length * direction + head_width * perpendicular
        right = end - head_length * direction - head_width * perpendicular
        self._draw_line(
            frame,
            self._world_to_pixel(end),
            self._world_to_pixel(left),
            color,
            thickness,
        )
        self._draw_line(
            frame,
            self._world_to_pixel(end),
            self._world_to_pixel(right),
            color,
            thickness,
        )

    def _render_human(self, frame: Array) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "render_mode='human' requires matplotlib; install it with "
                "`pip install matplotlib`."
            ) from exc

        if self._human_figure is None:
            plt.ion()
            self._human_figure, self._human_axis = plt.subplots(figsize=(7, 7))
            self._human_image = self._human_axis.imshow(frame)
            self._human_axis.set_axis_off()
            self._human_figure.tight_layout(pad=0)
        else:
            self._human_image.set_data(frame)
        self._human_figure.canvas.draw_idle()
        self._human_figure.canvas.flush_events()
        plt.pause(1.0 / self.metadata["render_fps"])

    def _world_to_pixel(self, point: Array) -> tuple[int, int]:
        scale = (self.render_size - 1) / (
            self.config.arena_high - self.config.arena_low
        )
        col = int(round((point[0] - self.config.arena_low) * scale))
        row = int(round((self.config.arena_high - point[1]) * scale))
        col = int(np.clip(col, 0, self.render_size - 1))
        row = int(np.clip(row, 0, self.render_size - 1))
        return col, row

    @staticmethod
    def _draw_line(
        image: Array,
        start: tuple[int, int],
        end: tuple[int, int],
        color: Array,
        thickness: int,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(int)
        ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(int)
        radius = max(0, thickness // 2)
        height, width = image.shape[:2]
        for x, y in zip(xs, ys, strict=False):
            x_lo = max(0, x - radius)
            x_hi = min(width, x + radius + 1)
            y_lo = max(0, y - radius)
            y_hi = min(height, y + radius + 1)
            image[y_lo:y_hi, x_lo:x_hi] = color
