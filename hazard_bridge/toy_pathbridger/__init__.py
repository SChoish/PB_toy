"""PathBridger toy package: dataset, env, pinned-bridge regressor, SVG plot."""

from .dataset import Episode, generate_dataset, save_dataset, validate_dataset
from .env import CircleHazard, ToyEnv
from .learning import LearnedPathBridger, PinnedBridgeRegressor, select_bridge_example
from .plot_svg import render_svg

__all__ = [
    "CircleHazard",
    "Episode",
    "LearnedPathBridger",
    "PinnedBridgeRegressor",
    "ToyEnv",
    "generate_dataset",
    "render_svg",
    "save_dataset",
    "select_bridge_example",
    "validate_dataset",
]
