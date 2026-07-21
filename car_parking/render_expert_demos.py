"""Render expert rollout videos and keyframe strips for tasks 1-5."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .env import NUM_FIXED_TASKS, CarParkingEnv
from .parking_policy import ParkingExpertPolicy


def rollout_with_frames(task_id: int, seed: int = 0) -> tuple[list[np.ndarray], dict]:
    env = CarParkingEnv(render_mode="rgb_array", render_size=512)
    env.reset(seed=seed, options={"task_id": task_id})
    policy = ParkingExpertPolicy(env)
    policy.reset()

    frames = [env.render()]
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action = policy.action()
        _, _, terminated, truncated, info = env.step(action)
        frames.append(env.render())
    env.close()
    return frames, info


def keyframe_strip(frames: list[np.ndarray], columns: int = 5) -> Image.Image:
    indices = np.linspace(0, len(frames) - 1, columns).round().astype(int)
    size = frames[0].shape[0]
    label_h = 26
    strip = Image.new("RGB", (columns * size, size + label_h), (12, 16, 20))
    draw = ImageDraw.Draw(strip)
    for column, index in enumerate(indices):
        strip.paste(Image.fromarray(frames[index]), (column * size, 0))
        draw.text(
            (column * size + 8, size + 5),
            f"step {index}/{len(frames) - 1}",
            fill=(232, 238, 240),
        )
    return strip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("demo") / "expert",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for task_id in range(1, NUM_FIXED_TASKS + 1):
        frames, info = rollout_with_frames(task_id, seed=args.seed)
        maneuver = info.get("maneuver", "unknown")
        outcome = info.get("termination_reason") or "time_limit"
        stem = f"task{task_id}_{maneuver}_{outcome}"

        video_path = args.output_dir / f"{stem}.mp4"
        with imageio.get_writer(
            video_path, fps=args.fps, codec="libx264", quality=8
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

        strip_path = args.output_dir / f"{stem}_strip.png"
        keyframe_strip(frames).save(strip_path)
        print(f"task {task_id}: {outcome} steps={len(frames) - 1} -> {video_path}")


if __name__ == "__main__":
    main()
