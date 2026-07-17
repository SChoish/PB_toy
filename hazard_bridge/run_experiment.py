from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toy_pathbridger.dataset import generate_dataset, save_dataset, validate_dataset
from toy_pathbridger.env import ToyEnv
from toy_pathbridger.learning import (
    LearnedPathBridger,
    PinnedBridgeRegressor,
    evaluate,
    select_bridge_example,
)
from toy_pathbridger.plot_svg import render_svg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the hazard-aware toy PathBridger dataset and figure")
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    env = ToyEnv()
    episodes = generate_dataset(env, args.episodes, args.seed)
    split = int(0.8 * len(episodes))
    model = LearnedPathBridger.fit(env, episodes[:split])
    pinned = PinnedBridgeRegressor.fit(episodes[:split])
    data_metrics = validate_dataset(env, episodes)
    learned_metrics = evaluate(env, model, episodes[split:])
    metrics = {**data_metrics, **learned_metrics}

    start, endpoint, bridge_path, support_windows = select_bridge_example(env, pinned, episodes[split:])
    direct_unsafe = not env.segment_is_safe(start, endpoint)
    bridge_safe = all(env.segment_is_safe(a, b) for a, b in zip(bridge_path[:-1], bridge_path[1:]))
    metrics.update({
        "concept_direct_path_unsafe": direct_unsafe,
        "concept_pinned_bridge_safe": bridge_safe,
        "bridge_horizon": pinned.horizon,
    })

    save_dataset(args.output / "toy_hazard_dataset.json", env, episodes, args.seed)
    np.savez(
        args.output / "learned_models.npz",
        subgoal_weights=model.subgoal.weights,
        subgoal_mean=model.subgoal.mean,
        subgoal_scale=model.subgoal.scale,
        bridge_weights=model.bridge.weights,
        bridge_mean=model.bridge.mean,
        bridge_scale=model.bridge.scale,
        lookahead=np.asarray(model.lookahead),
        pinned_descriptors=pinned.descriptors,
        pinned_residuals=pinned.residuals,
        pinned_mean=pinned.mean,
        pinned_scale=pinned.scale,
        pinned_horizon=np.asarray(pinned.horizon),
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    svg_path = args.output / "toy_pathbridger.svg"
    render_svg(
        svg_path,
        env,
        episodes,
        start,
        endpoint,
        bridge_path,
        support_windows,
        metrics,
    )
    png_path = args.output / "toy_pathbridger.png"
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2.0)
    except Exception:
        # Keep prior PNG if conversion tools are unavailable.
        pass
    print(json.dumps(metrics, indent=2))
    print(f"saved outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
