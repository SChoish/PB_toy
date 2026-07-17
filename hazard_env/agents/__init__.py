"""Hazard-env toy agents: BC, HIQL, PBG, PBF."""

from hazard_env.agents.bc import BCAgent, default_config as bc_config
from hazard_env.agents.hiql import HIQLAgent, default_config as hiql_config
from hazard_env.agents.pbg import PBGAgent, default_config as pbg_config
from hazard_env.agents.pbf import PBFAgent, default_config as pbf_config

AGENTS = {
    "bc": BCAgent,
    "hiql": HIQLAgent,
    "pbg": PBGAgent,
    "pbf": PBFAgent,
}

DEFAULT_CONFIGS = {
    "bc": bc_config,
    "hiql": hiql_config,
    "pbg": pbg_config,
    "pbf": pbf_config,
}

__all__ = [
    "AGENTS",
    "DEFAULT_CONFIGS",
    "BCAgent",
    "HIQLAgent",
    "PBGAgent",
    "PBFAgent",
]
