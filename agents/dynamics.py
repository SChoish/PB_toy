"""PathBridger PBG/PBF adapter for the toy suite.

Wraps ``agents.pathbridger`` (``DynamicsAgent`` + ``CriticAgent``) while keeping
the toy train/eval API: ``create`` / ``update`` / ``sample_action_chunk``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.pathbridger import (
    CriticAgent,
    DynamicsAgent,
    get_critic_config,
    get_dynamics_config,
)
from agents.pathbridger.utils.flax_utils import nonpytree_field


def _as_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "to_dict"):
        return dict(cfg.to_dict())
    return dict(cfg)


def _merge_dynamics_config(overrides: dict | None = None) -> dict:
    cfg = _as_dict(get_dynamics_config())
    toy_hidden = (256, 256)
    cfg.update(
        {
            "dynamics_N": 25,
            "subgoal_steps": 25,
            "forward_bridge_path_loss_horizon": 5,
            "action_chunk_horizon": 5,
            "batch_size": 256,
            "goal_dim": 4,
            # Actor (subgoal): φ = task-goal prefix (0,1,2,3). Critic/value: full.
            "subgoal_value_goal_representation": "full",
            "phi_goal_obs_indices": (0, 1, 2, 3),
            # Toy suite width (not PathBridger's 512^3).
            "residual_model_hidden_dims": toy_hidden,
            "path_residual_hidden_dims": toy_hidden,
            "subgoal_hidden_dims": toy_hidden,
            "idm_hidden_dims": toy_hidden,
            "subgoal_value_hidden_dims": toy_hidden,
            # env_name filled by train.py (car_race_* / swingby_*).
            "env_name": "car_race",
            "require_matching_horizon": True,
            "path_loss_weight": 1.0,
            "subgoal_include_mean": False,
        }
    )
    if overrides:
        cfg.update(overrides)
    return cfg


def _merge_critic_config(overrides: dict | None = None) -> dict:
    cfg = _as_dict(get_critic_config())
    cfg.update(
        {
            "full_chunk_horizon": 25,
            "action_chunk_horizon": 5,
            "batch_size": 256,
            # Critic / transitive V: full-state goals (not φ).
            "goal_representation": "full",
            "env_name": "car_race",
            "phi_goal_obs_indices": (),
            "discount": 0.99,
            "value_hidden_dims": (256, 256),
        }
    )
    if overrides:
        cfg.update(overrides)
    return cfg


def default_config_pbg() -> dict:
    # PBG: single Gaussian sample at T=1 (mean is T=0 diagnostic).
    return _merge_dynamics_config(
        {
            "subgoal_distribution": "diag_gaussian",
            "subgoal_eval_num_samples": 1,
        }
    )


def default_config_pbf() -> dict:
    # PBF: best-of-8 stochastic flow endpoints at T=1.
    return _merge_dynamics_config(
        {
            "subgoal_distribution": "flow",
            "subgoal_flow_steps": 8,
            "subgoal_flow_t_min": 1e-4,
            "subgoal_eval_num_samples": 8,
        }
    )


def _map_dynamics_batch(batch: dict) -> dict:
    out = dict(batch)
    if "trajectory_segment" not in out and "path_observations" in out:
        out["trajectory_segment"] = out["path_observations"]
    # Subgoal net applies φ(·) on high_actor_goals; value gap needs full state.
    # Prefer full-state subgoal_value_goals so φ extracts (0,1,2,3) and V sees full.
    if "subgoal_value_goals" in out:
        out["high_actor_goals"] = out["subgoal_value_goals"]
    return out


def _map_critic_batch(batch: dict) -> dict:
    out = dict(batch)
    if "action_chunk_actions" not in out:
        raise KeyError("PB critic batch requires action_chunk_actions")
    if "action_chunk_next_observations" not in out:
        raise KeyError("PB critic batch requires action_chunk_next_observations")
    return out


def _extract_critic_value_params(critic: CriticAgent):
    return critic.network.params.get("modules_target_value", None)


class PathBridgerAgent(flax.struct.PyTreeNode):
    """Toy-facing wrapper around PathBridger DynamicsAgent + CriticAgent."""

    dynamics: DynamicsAgent
    critic: CriticAgent
    config: dict = nonpytree_field()

    @property
    def rng(self):
        """Expose dynamics RNG for toy render/eval overlays."""
        return self.dynamics.rng

    @property
    def network(self):
        """Critic TrainState; used by value-field / BoN scoring overlays."""
        return self.critic.network

    def _plan(self, observations, endpoints, goal=None):
        """Duck-typed planner used by ``collect_agent_diagnostics``."""
        return self.dynamics.plan(observations, endpoints, goal=goal)[
            "trajectory"
        ]

    def update(self, batch: dict):
        critic_batch = _map_critic_batch(batch)
        new_critic, critic_info = self.critic.update(critic_batch)
        dyn_batch = _map_dynamics_batch(batch)
        value_params = _extract_critic_value_params(new_critic)
        new_dynamics, dyn_info = self.dynamics.update(
            dyn_batch, critic_value_params=value_params
        )
        info = {**dyn_info, **{f"critic/{k}": v for k, v in critic_info.items()}}
        # Flat aliases expected by older toy logs / scoreboard.
        if "phase1/loss" in dyn_info:
            info["loss"] = dyn_info["phase1/loss"]
        if "phase1/loss_idm" in dyn_info:
            info["idm_mse"] = dyn_info["phase1/loss_idm"]
        if "forward_bridge/path_mse" in dyn_info:
            info["path_mse"] = dyn_info["forward_bridge/path_mse"]
        return self.replace(dynamics=new_dynamics, critic=new_critic), info

    def _sample_candidates(self, observations, goals, seed, temperature=1.0):
        """Duck-typed PathBridger eval diagnostic used by toy render/train."""
        del temperature
        include_mean = bool(self.config.get("subgoal_include_mean", False))
        n = int(self.config.get("subgoal_eval_num_samples", 1))
        candidates, mu = self.dynamics.sample_subgoal_candidates(
            observations,
            goals,
            seed,
            num_candidates=n,
            include_mean=include_mean,
        )
        return candidates, mu

    def sample_plan(
        self,
        observations,
        goals,
        value_goals=None,
        seed=None,
        temperature=1.0,
    ):
        if value_goals is None:
            value_goals = goals
        if seed is None:
            seed = self.dynamics.rng
        # Subgoal net is φ-conditioned on task-goal prefix (0,1,2,3);
        # full-state value goals keep that prefix in the leading channels.
        cond_goals = value_goals
        if float(temperature) == 0.0:
            endpoint = self.dynamics.infer_subgoal_mean(observations, cond_goals)
        else:
            endpoint = self.dynamics.infer_subgoal_for_eval(
                observations,
                cond_goals,
                critic_agent=self.critic,
                rng=seed,
            )
        planned = self.dynamics.plan(observations, endpoint, goal=cond_goals)
        return planned["trajectory"]

    def sample_action_chunk(
        self,
        observations,
        goals,
        value_goals=None,
        seed=None,
        temperature=1.0,
    ):
        path = self.sample_plan(
            observations,
            goals,
            value_goals=value_goals,
            seed=seed,
            temperature=temperature,
        )
        h_a = max(1, int(self.config.get("action_chunk_horizon", 5)))
        k = int(self.config.get("dynamics_N", h_a))
        horizon = min(h_a, k, int(path.shape[1]) - 1)
        return self.dynamics._idm_actions_from_trajectories(path, horizon)

    def sample_actions(
        self,
        observations,
        goals,
        value_goals=None,
        seed=None,
        temperature=1.0,
    ):
        chunk = self.sample_action_chunk(
            observations,
            goals,
            value_goals=value_goals,
            seed=seed,
            temperature=temperature,
        )
        return chunk[:, 0]

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        dyn_cfg = _merge_dynamics_config(config)
        # Keep distribution from caller (pbg vs pbf).
        if "subgoal_distribution" in config:
            dyn_cfg["subgoal_distribution"] = config["subgoal_distribution"]
        crit_cfg = _merge_critic_config(
            {
                "action_chunk_horizon": int(
                    config.get("action_chunk_horizon", dyn_cfg["action_chunk_horizon"])
                ),
                "full_chunk_horizon": int(
                    config.get("full_chunk_horizon", dyn_cfg["dynamics_N"])
                ),
                "action_dim": int(np.asarray(ex_actions).shape[-1]),
                "lr": float(config.get("lr", dyn_cfg.get("lr", 3e-4))),
                "env_name": str(config.get("env_name", dyn_cfg.get("env_name", "car_race"))),
                "goal_representation": "full",
                "phi_goal_obs_indices": (),
            }
        )
        h_a = int(crit_cfg["action_chunk_horizon"])
        ex_obs = np.asarray(ex_observations, dtype=np.float32)
        ex_act = np.asarray(ex_actions, dtype=np.float32)
        if ex_act.ndim == 1:
            ex_act = ex_act[None, :]
        # Critic expects flattened action chunks (B, h_a * A).
        flat = np.tile(ex_act[:, None, :], (1, h_a, 1)).reshape(ex_act.shape[0], -1)

        dynamics = DynamicsAgent.create(
            seed, ex_obs, dyn_cfg, ex_actions=ex_act
        )
        critic = CriticAgent.create(
            seed + 1,
            ex_obs,
            None,
            flat,
            crit_cfg,
            ex_goals=ex_obs,
        )
        # Mirror critic value head dims into dynamics for borrowed value params.
        dyn_cfg["subgoal_value_hidden_dims"] = tuple(
            crit_cfg.get("value_hidden_dims", (256, 256))
        )
        dyn_cfg["subgoal_value_layer_norm"] = True
        merged = dict(dyn_cfg)
        merged["action_chunk_horizon"] = h_a
        merged["subgoal_include_mean"] = False
        return cls(dynamics=dynamics, critic=critic, config=merged)


# Back-compat names used elsewhere.
PBGAgent = PathBridgerAgent
PBFAgent = PathBridgerAgent
