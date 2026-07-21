"""Render the five fixed CarParking tasks as reproducible PNG demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .env import NUM_FIXED_TASKS, CarParkingEnv


def render_demos(output_dir: Path, render_size: int = 512) -> list[Path]:
    """Render every fixed task and a compact overview contact sheet."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("demo rendering requires Pillow") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    env = CarParkingEnv(render_mode="rgb_array", render_size=render_size)
    rendered: list[Path] = []
    frames: list[np.ndarray] = []

    try:
        for task_id in range(1, NUM_FIXED_TASKS + 1):
            _, info = env.reset(options={"task_id": task_id})
            frame = env.render()
            assert frame is not None
            image = Image.fromarray(frame)
            path = output_dir / f"task{task_id}_{info['maneuver']}.png"
            image.save(path)
            rendered.append(path)
            frames.append(frame)
    finally:
        env.close()

    columns = 3
    rows = 2
    frame_height, frame_width = frames[0].shape[:2]
    label_height = max(28, frame_width // 14)
    overview = Image.new(
        "RGB",
        (columns * frame_width, rows * (frame_height + label_height)),
        (12, 16, 20),
    )
    draw = ImageDraw.Draw(overview)
    labels = (
        "Task 1 - Parallel (lower)",
        "Task 2 - Parallel (upper)",
        "Task 3 - T reverse",
        "Task 4 - T forward",
        "Task 5 - Angled",
    )
    for index, (frame, label) in enumerate(zip(frames, labels)):
        column = index % columns
        row = index // columns
        x = column * frame_width
        y = row * (frame_height + label_height)
        overview.paste(Image.fromarray(frame), (x, y))
        draw.text((x + 10, y + frame_height + 7), label, fill=(232, 238, 240))

    overview_path = output_dir / "overview.png"
    overview.save(overview_path)
    rendered.append(overview_path)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("demo"),
    )
    parser.add_argument("--render-size", type=int, default=512)
    args = parser.parse_args()
    for path in render_demos(args.output_dir, args.render_size):
        print(path)


if __name__ == "__main__":
    main()
