"""PathBridger core (DynamicsAgent + CriticAgent) for PBG/PBF.

Toy-suite training goes through ``agents.dynamics.PathBridgerAgent``, which
wraps these modules with toy defaults (horizon, φ recipe, hidden widths).
"""

from agents.pathbridger.critic import CriticAgent
from agents.pathbridger.critic import get_config as get_critic_config
from agents.pathbridger.dynamics import DynamicsAgent
from agents.pathbridger.dynamics import get_dynamics_config

__all__ = [
    "CriticAgent",
    "DynamicsAgent",
    "get_critic_config",
    "get_dynamics_config",
]
