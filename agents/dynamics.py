"""PathBridger dynamics agent (PBG / PBF via ``subgoal_distribution``).

Layout mirrors Pathbridger_flow:
  - ``agents/dynamics_utils.py``: closed-form bridge coefficients
  - ``agents/critic.py``: transitive V train + score
  - this module: subgoal / path residual / IDM + act-time selection
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import flax
import flax.core
import jax
import jax.numpy as jnp
import optax

from agents.critic import (
    pick_best_candidates,
    score_transitive_ratio,
    soft_update,
    transitive_value_loss,
)
from agents.dynamics_utils import (
    forward_bridge_coefficients,
    plan_forward_bridge,
    residual_endpoint_weights,
)
from agents.flax_utils import ModuleDict, TrainState, nonpytree_field
from agents.networks import (
    FlowVelocityNet,
    InverseDynamicsNet,
    PathResidualNet,
    ScalarValueNet,
    SubgoalNet,
)

SubgoalMode = Literal["diag_gaussian", "flow"]


def _subgoal_mode(config) -> SubgoalMode:
    mode = str(config.get("subgoal_distribution", "diag_gaussian")).lower()
    if mode in ("diag_gaussian", "gaussian", "pbg"):
        return "diag_gaussian"
    if mode in ("flow", "pbf"):
        return "flow"
    raise ValueError(f"Unknown subgoal_distribution: {mode!r}")


class PathBridgerAgent(flax.struct.PyTreeNode):
    """Toy PathBridger: subgoal proposals + transitive V pick + bridge + IDM.

    Subgoal nets (Gaussian / flow) predict displacement ``Delta = s_{t+K} - s_t``
    (PathBridger mainline). Bridge planning also runs in the displacement frame
    ``(z0=0, zK=Delta)`` then shifts back by ``s_t``. Value / IDM stay absolute.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    bridge_a: Any = nonpytree_field()
    bridge_b: Any = nonpytree_field()
    bridge_w: Any = nonpytree_field()

    @staticmethod
    def _subgoal_abs_from_raw(observations, raw_delta):
        return jnp.asarray(observations, dtype=jnp.float32) + jnp.asarray(
            raw_delta, dtype=jnp.float32
        )

    @staticmethod
    def _subgoal_target_delta(observations, target_abs):
        return jnp.asarray(target_abs, dtype=jnp.float32) - jnp.asarray(
            observations, dtype=jnp.float32
        )

    def _plan(self, s0, z_k_abs, grad_params=None):
        """Absolute path ``(B, K+1, D)`` via displacement-frame bridge + residual."""
        k = int(self.config["dynamics_N"])
        s0 = jnp.asarray(s0, dtype=jnp.float32)
        z_k_abs = jnp.asarray(z_k_abs, dtype=jnp.float32)
        z0 = jnp.zeros_like(s0)
        z_k = z_k_abs - s0
        t_norm = jnp.broadcast_to(
            jnp.arange(k + 1, dtype=jnp.float32)[None, :] / float(k),
            (s0.shape[0], k + 1),
        )
        # Residual is conditioned on absolute ``s_t`` (anchor) and local ``Delta``.
        residual = self.network.select("path_residual")(
            s0, z_k, t_norm, params=grad_params
        )
        path_local = plan_forward_bridge(
            z0, z_k, residual, a=self.bridge_a, b=self.bridge_b, w=self.bridge_w
        )
        return path_local + s0[:, None, :]

    def _flow_from_noise(self, observations, goals, z0, params=None):
        """Euler-integrate CFM velocity; ``z0`` is noise in *displacement* frame."""
        steps = int(self.config.get("flow_steps", 8))
        x = z0
        for i in range(steps):
            u = jnp.full((observations.shape[0], 1), (i + 0.5) / steps)
            v = self.network.select("flow")(x, u, observations, goals, params=params)
            x = x + v / steps
        return x

    def _sample_candidates(self, observations, goals, rng, *, temperature: float):
        """Propose absolute subgoal candidates (raw nets output ``Delta``).

        ``temperature <= 0`` returns the mean only. ``temperature > 0`` samples
        ``mean ± (std * temperature) * ε`` (Gaussian) or temperature-scaled flow
        noise (PBF). With ``N>1`` and ``include_mean``, candidate 0 stays the mean
        and the remaining ``N-1`` are stochastic.
        """
        mode = _subgoal_mode(self.config)
        n = int(self.config.get("subgoal_eval_num_samples", 4))
        include_mean = bool(self.config.get("subgoal_include_mean", False))
        b, d = observations.shape
        temperature = float(temperature)

        if mode == "diag_gaussian":
            mu_delta, log_std = self.network.select("subgoal")(observations, goals)
            std = jnp.exp(log_std)
            mu_abs = self._subgoal_abs_from_raw(observations, mu_delta)
            if temperature <= 0.0:
                return mu_abs[:, None, :], mu_abs
            if include_mean and n > 1:
                n_rand = n - 1
                noise = jax.random.normal(rng, (b, n_rand, d))
                sampled_delta = (
                    mu_delta[:, None, :] + std[:, None, :] * temperature * noise
                )
                sampled_abs = self._subgoal_abs_from_raw(
                    observations[:, None, :], sampled_delta
                )
                return jnp.concatenate([mu_abs[:, None, :], sampled_abs], axis=1), mu_abs
            noise = jax.random.normal(rng, (b, n, d))
            sampled_delta = mu_delta[:, None, :] + std[:, None, :] * temperature * noise
            sampled_abs = self._subgoal_abs_from_raw(
                observations[:, None, :], sampled_delta
            )
            return sampled_abs, mu_abs

        # flow: CFM ODE from noise in Delta-space; zero-noise path is the mean Delta.
        goal_dim = int(goals.shape[-1])
        z_mean0 = jnp.zeros((b, d), dtype=jnp.float32)
        mean_delta = self._flow_from_noise(observations, goals, z_mean0)
        mean_abs = self._subgoal_abs_from_raw(observations, mean_delta)
        if temperature <= 0.0:
            return mean_abs[:, None, :], mean_abs

        parts = []
        n_rand = n - 1 if include_mean and n > 1 else n
        if include_mean and n > 1:
            parts.append(mean_abs[:, None, :])
        if n_rand > 0:
            z_rand = jax.random.normal(rng, (b, n_rand, d)) * temperature
            flat_delta = self._flow_from_noise(
                jnp.repeat(observations[:, None, :], n_rand, axis=1).reshape(
                    b * n_rand, d
                ),
                jnp.repeat(goals[:, None, :], n_rand, axis=1).reshape(
                    b * n_rand, goal_dim
                ),
                z_rand.reshape(b * n_rand, d),
            ).reshape(b, n_rand, d)
            flat_abs = self._subgoal_abs_from_raw(
                observations[:, None, :], flat_delta
            )
            parts.append(flat_abs)
        candidates = jnp.concatenate(parts, axis=1)
        return candidates, mean_abs

    def _subgoal_gap_weight(
        self, observations, target_subgoals_abs, value_goals
    ):
        """Weight dataset targets by their frozen target-V improvement (absolute)."""
        if int(value_goals.shape[-1]) != int(observations.shape[-1]):
            raise ValueError(
                "subgoal_value_goals must be full states: "
                f"observations={observations.shape}, goals={value_goals.shape}"
            )
        current_value = jax.nn.sigmoid(
            self.network.select("target_value")(observations, value_goals)
        )
        target_value = jax.nn.sigmoid(
            self.network.select("target_value")(target_subgoals_abs, value_goals)
        )
        gap = jax.lax.stop_gradient(target_value - current_value)
        gap_scale = float(self.config.get("subgoal_value_gap_scale", 3.0))
        weight = jnp.exp(gap_scale * gap)
        weight_max = float(self.config.get("subgoal_value_weight_max", 100.0))
        if weight_max > 0.0:
            weight = jnp.minimum(weight, weight_max)
        return jax.lax.stop_gradient(weight), gap, current_value, target_value

    def _subgoal_loss(self, batch, grad_params, rng):
        mode = _subgoal_mode(self.config)
        s0 = batch["observations"]
        z_true_abs = batch["path_observations"][:, -1]
        z_true = self._subgoal_target_delta(s0, z_true_abs)
        weight, gap, current_value, target_value = self._subgoal_gap_weight(
            s0, z_true_abs, batch["subgoal_value_goals"]
        )
        gap_info = {
            "subgoal_gap_mean": jnp.mean(gap),
            "subgoal_gap_min": jnp.min(gap),
            "subgoal_gap_max": jnp.max(gap),
            "subgoal_weight_mean": jnp.mean(weight),
            "subgoal_weight_max": jnp.max(weight),
            "subgoal_current_value_mean": jnp.mean(current_value),
            "subgoal_target_value_mean": jnp.mean(target_value),
        }
        if mode == "diag_gaussian":
            mu, log_std = self.network.select("subgoal")(
                s0, batch["goals"], params=grad_params
            )
            inv_var = jnp.exp(-2.0 * log_std)
            mean_diff = z_true - mu
            nll_per = 0.5 * jnp.sum(
                mean_diff**2 * inv_var + 2.0 * log_std + jnp.log(2.0 * jnp.pi),
                axis=-1,
            )
            mean_mse = jnp.mean(mean_diff**2, axis=-1)
            loss = jnp.mean(weight * nll_per)
            return loss, {
                "subgoal_nll": jnp.mean(nll_per),
                "subgoal_weighted_nll": loss,
                "subgoal_mean_mse": jnp.mean(mean_mse),
                "subgoal_std_mean": jnp.mean(jnp.exp(log_std)),
                **gap_info,
            }

        # Conditional flow matching (rectified) in Delta-space.
        rng, u_rng, x0_rng = jax.random.split(rng, 3)
        eps = jax.random.normal(x0_rng, z_true.shape)
        t_min = float(self.config.get("flow_t_min", 1e-4))
        u = jax.random.uniform(
            u_rng,
            (z_true.shape[0], 1),
            minval=t_min,
            maxval=1.0 - t_min,
        )
        x_u = (1.0 - u) * eps + u * z_true
        target_v = z_true - eps
        pred_v = self.network.select("flow")(
            x_u, u, s0, batch["goals"], params=grad_params
        )
        mse_per = jnp.mean((pred_v - target_v) ** 2, axis=-1)
        raw_mse = jnp.mean(mse_per)
        loss = jnp.mean(weight * mse_per)
        return loss, {
            "flow_mse": raw_mse,
            "flow_weighted_mse": loss,
            **gap_info,
        }

    def total_loss(self, batch, grad_params, rng):
        true_path = batch["path_observations"]
        s0 = batch["observations"]
        z_true = true_path[:, -1]

        sub_loss, sub_info = self._subgoal_loss(batch, grad_params, rng)

        planned = self._plan(s0, jax.lax.stop_gradient(z_true), grad_params=grad_params)
        path_loss = jnp.mean((planned[:, 1:-1] - true_path[:, 1:-1]) ** 2)

        pred_act = self.network.select("idm")(
            batch["observations"],
            batch["next_observations"],
            params=grad_params,
        )
        idm_loss = jnp.mean((pred_act - batch["actions"]) ** 2)

        v_loss, v_info = transitive_value_loss(
            self.network,
            batch,
            grad_params,
            discount=float(self.config.get("discount", 0.99)),
        )

        w_path = float(self.config.get("path_loss_weight", 1.0))
        w_sub = float(self.config.get("subgoal_loss_weight", 1.0))
        w_idm = float(self.config.get("idm_loss_weight", 1.0))
        w_v = float(self.config.get("value_loss_weight", 1.0))
        loss = (
            w_sub * sub_loss
            + w_path * path_loss
            + w_idm * idm_loss
            + w_v * v_loss
        )
        return loss, {
            **sub_info,
            "path_mse": path_loss,
            "idm_mse": idm_loss,
            "loss": loss,
            **{f"value/{k}": v for k, v in v_info.items()},
        }

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_params = flax.core.copy(
            new_network.params,
            {
                "modules_target_value": soft_update(
                    new_network.params["modules_value"],
                    new_network.params["modules_target_value"],
                    float(self.config.get("tau", 0.005)),
                )
            },
        )
        new_network = new_network.replace(params=new_params)
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_plan(
        self,
        observations,
        goals,
        value_goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Propose a subgoal and return the bridged path ``(B, K+1, D)``."""
        if seed is None:
            seed = self.rng
        if value_goals is None:
            if int(goals.shape[-1]) != int(observations.shape[-1]):
                raise ValueError(
                    "PB evaluation requires a full-state value_goal separate "
                    "from the task goal."
                )
            value_goals = goals
        candidates, _ = self._sample_candidates(
            observations, goals, seed, temperature=float(temperature)
        )
        scores = score_transitive_ratio(
            self.network, observations, candidates, value_goals
        )
        z = pick_best_candidates(candidates, scores)
        return self._plan(observations, z)

    @partial(jax.jit, static_argnames=("horizon",))
    def _idm_actions_from_trajectories(self, trajectories, horizon: int):
        """Map a planned state path to an open-loop action chunk (Pathbridger_flow)."""
        prev_states = trajectories[:, :horizon, :]
        next_states = trajectories[:, 1 : horizon + 1, :]
        flat_prev = prev_states.reshape(-1, prev_states.shape[-1])
        flat_next = next_states.reshape(-1, next_states.shape[-1])
        pred = self.network.select("idm")(flat_prev, flat_next)
        return jnp.clip(pred, -1.0, 1.0).reshape(
            trajectories.shape[0], horizon, -1
        )

    def sample_action_chunk(
        self,
        observations,
        goals,
        value_goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Plan once and return ``(B, h_a, A)`` via ``action_chunk_horizon`` (h_a)."""
        path = self.sample_plan(
            observations,
            goals,
            value_goals=value_goals,
            seed=seed,
            temperature=temperature,
        )
        h_a = max(1, int(self.config.get("action_chunk_horizon", 1)))
        k = int(self.config.get("dynamics_N", h_a))
        horizon = min(h_a, k, int(path.shape[1]) - 1)
        return self._idm_actions_from_trajectories(path, horizon)

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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        state_dim = int(ex_observations.shape[-1])
        action_dim = int(ex_actions.shape[-1])
        goal_dim = int(config.get("goal_dim", state_dim))
        hidden = tuple(config["hidden_dims"])
        k = int(config["dynamics_N"])
        if k < 2:
            raise ValueError(f"PathBridger requires dynamics_N >= 2, got {k}")
        if k != int(config.get("subgoal_steps", k)):
            raise ValueError(
                "dynamics_N and subgoal_steps must match, got "
                f"{k} and {config.get('subgoal_steps')}"
            )
        mode = _subgoal_mode(config)
        # Subgoal / flow condition on task goals (often 4D); V uses full state.
        ex_goals = jnp.zeros(
            (ex_observations.shape[0], goal_dim), dtype=ex_observations.dtype
        )
        ex_value_goals = jnp.zeros(
            (ex_observations.shape[0], state_dim), dtype=ex_observations.dtype
        )

        a, b, _std = forward_bridge_coefficients(
            k,
            lambda_=float(config.get("dynamics_lambda", 1.0)),
            bridge_gamma_inv=float(config.get("bridge_gamma_inv", 0.0)),
            theta_total=float(config.get("theta_total", 1.0)),
            progress_alpha=float(config.get("progress_alpha", 0.8)),
        )
        w = residual_endpoint_weights(k)

        modules: dict[str, Any] = {
            "path_residual": PathResidualNet(
                hidden_dims=hidden, state_dim=state_dim, layer_norm=True
            ),
            "idm": InverseDynamicsNet(
                hidden_dims=hidden, action_dim=action_dim, layer_norm=True
            ),
            "value": ScalarValueNet(hidden_dims=hidden, layer_norm=True),
            "target_value": ScalarValueNet(hidden_dims=hidden, layer_norm=True),
        }
        init_kwargs: dict[str, Any] = {
            "path_residual": [
                ex_observations,
                ex_observations,
                jnp.broadcast_to(
                    jnp.arange(k + 1, dtype=jnp.float32)[None, :] / float(k),
                    (ex_observations.shape[0], k + 1),
                ),
            ],
            "idm": [ex_observations, ex_observations],
            "value": [ex_observations, ex_value_goals],
            "target_value": [ex_observations, ex_value_goals],
        }
        if mode == "diag_gaussian":
            modules["subgoal"] = SubgoalNet(
                hidden_dims=hidden, state_dim=state_dim, layer_norm=True
            )
            init_kwargs["subgoal"] = [ex_observations, ex_goals]
        else:
            modules["flow"] = FlowVelocityNet(
                hidden_dims=hidden, state_dim=state_dim, layer_norm=True
            )
            init_kwargs["flow"] = [
                ex_observations,
                jnp.zeros((ex_observations.shape[0], 1)),
                ex_observations,
                ex_goals,
            ]

        network_def = ModuleDict(modules)
        params = network_def.init(init_rng, **init_kwargs)["params"]
        params = flax.core.copy(
            params, {"modules_target_value": params["modules_value"]}
        )
        network = TrainState.create(
            network_def, params, tx=optax.adam(config.get("lr", 3e-4))
        )
        return cls(
            rng=rng,
            network=network,
            config=config,
            bridge_a=a,
            bridge_b=b,
            bridge_w=w,
        )


def default_config_pbg():
    return {
        "subgoal_distribution": "diag_gaussian",
        "lr": 3e-4,
        "hidden_dims": (256, 256),
        "batch_size": 256,
        "dynamics_N": 8,
        "subgoal_steps": 8,
        "dynamics_lambda": 1.0,
        "bridge_gamma_inv": 0.0,
        "theta_total": 1.0,
        "progress_alpha": 0.8,
        "discount": 0.99,
        "tau": 0.005,
        "path_loss_weight": 1.0,
        "subgoal_loss_weight": 1.0,
        "idm_loss_weight": 1.0,
        "value_loss_weight": 1.0,
        "subgoal_value_gap_scale": 3.0,
        "subgoal_value_weight_max": 100.0,
        "subgoal_eval_num_samples": 4,
        "subgoal_include_mean": False,
        "goal_dim": 4,
        # Pathbridger_flow name: env steps executed per replan (h_a).
        "action_chunk_horizon": 1,
    }


def default_config_pbf():
    cfg = default_config_pbg()
    cfg.update(
        {
            "subgoal_distribution": "flow",
            "flow_steps": 8,
            "flow_t_min": 1e-4,
        }
    )
    return cfg


# Back-compat aliases used by the train registry.
PBGAgent = PathBridgerAgent
PBFAgent = PathBridgerAgent
