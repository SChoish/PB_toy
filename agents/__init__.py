"""Hazard-env toy agents: BC, HIQL, TR-HIQL, PathBridger (PBG / PBF)."""

from agents.bc import BCAgent, default_config as bc_config
from agents.dynamics import (
    PathBridgerAgent,
    default_config_pbf as pbf_config,
    default_config_pbg as pbg_config,
)
from agents.dqc import DQCAgent, default_config as dqc_config
from agents.hiql import HIQLAgent, default_config as hiql_config
from agents.trl import TRLAgent, default_config as trl_config
from agents.tr_hiql import TRHIQLAgent, default_config as tr_hiql_config

# Back-compat aliases (PBG/PBF = PathBridgerAgent + different subgoal_distribution).
PBGAgent = PathBridgerAgent
PBFAgent = PathBridgerAgent

AGENTS = {
    "bc": BCAgent,
    "hiql": HIQLAgent,
    "tr_hiql": TRHIQLAgent,
    "trl": TRLAgent,
    "dqc": DQCAgent,
    "pbg": PathBridgerAgent,
    "pbf": PathBridgerAgent,
}

DEFAULT_CONFIGS = {
    "bc": bc_config,
    "hiql": hiql_config,
    "tr_hiql": tr_hiql_config,
    "trl": trl_config,
    "dqc": dqc_config,
    "pbg": pbg_config,
    "pbf": pbf_config,
}

__all__ = [
    "AGENTS",
    "DEFAULT_CONFIGS",
    "BCAgent",
    "HIQLAgent",
    "TRHIQLAgent",
    "TRLAgent",
    "DQCAgent",
    "PathBridgerAgent",
    "PBGAgent",
    "PBFAgent",
]
