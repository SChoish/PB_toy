"""PathBridger dynamics agent (PBG / PBF via ``subgoal_distribution``).

Layout mirrors Pathbridger_flow:
  - ``utils/dynamics.py``: closed-form bridge coefficients
  - ``agents/critic.py``: transitive V train + score
  - this module: subgoal / path residual / IDM + act-time selection
"""

from __future__ import annotations

from typing import Any, Literal

import flax
import flax.core
import jax
import jax.numpy as jnp
import optax

from hazard_env.agents.critic import (
    pick_best_candidates,
    score_transitive_ratio,
    soft_update,
    transitive_value_loss,
)
from hazard_env.utils.dynamics import (
    forward_bridge_coefficients,
    plan_forward_bridge,
    residual_endpoint_weights,
)
from hazard_env.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from hazard_env.utils.networks import (
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
    """Toy PathBridger: subgoal proposals + transitive V pick + bridge + IDM."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    bridge_a: Any = nonpytree_field()
    bridge_b: Any = nonpytree_field()
    bridge_w: Any = nonpytree_field()

    def _plan(self, s0, z_k, grad_params=None):
        k = int(self.config["dynamics_N"])
        t_norm = jnp.broadcast_to(
            jnp.arange(k + 1, dtype=jnp.float32)[None, :] / float(k),
            (s0.shape[0], k + 1),
        )
        residual = self.network.select("path_residual")(
            s0, z_k, t_norm, params=grad_params
        )
        return plan_forward_bridge(
            s0, z_k, residual, a=self.bridge_a, b=self.bridge_b, w=self.bridge_w
        )

    def _flow_from_noise(self, observations, goals, z0, params=None):
        steps = int(self.config.get("flow_steps", 8))
        x = z0
        for i in range(steps):
            u = jnp.full((observations.shape[0], 1), (i + 0.5) / steps)
            v = self.network.select("flow")(x, u, observations, goals, params=params)
            x = x + v / steps
        return x

    def _sample_candidates(self, observations, goals, rng):
        mode = _subgoal_mode(self.config)
        n = int(self.config.get("subgoal_num_candidates", 1))
        include_mean = bool(self.config.get("subgoal_include_mean", True))
        n_rand = n - 1 if include_mean else n
        b, d = observations.shape

        if mode == "diag_gaussian":
            mu, log_std = self.network.select("subgoal")(observations, goals)
            std = jnp.exp(log_std)
            if n_rand > 0:
                noise = jax.random.normal(rng, (b, n_rand, d))
                sampled = mu[:, None, :] + std[:, None, :] * noise
            else:
                sampled = jnp.zeros((b, 0, d), dtype=mu.dtype)
            if include_mean:
                return jnp.concatenate([mu[:, None, :], sampled], axis=1), mu
            return sampled, mu

        # flow: CFM ODE from noise; candidate 0 is the zero-noise (mean) path.
        z_mean0 = jnp.zeros((b, d), dtype=jnp.float32)
        z_rand = jax.random.normal(rng, (b, n_rand, d))
        parts = []
        if include_mean:
            parts.append(self._flow_from_noise(observations, goals, z_mean0)[:, None, :])
        if n_rand > 0:
            flat = self._flow_from_noise(
                jnp.repeat(observations[:, None, :], n_rand, axis=1).reshape(b * n_rand, d),
                jnp.repeat(goals[:, None, :], n_rand, axis=1).reshape(b * n_rand, d),
                z_rand.reshape(b * n_rand, d),
            ).reshape(b, n_rand, d)
            parts.append(flat)
        candidates = jnp.concatenate(parts, axis=1)
        return candidates, candidates[:, 0, :]

    def _subgoal_gap_weight(self, observations, target_subgoals, goals):
        """Weight dataset targets by their frozen target-V improvement."""
        current_value = jax.nn.sigmoid(
            self.network.select("target_value")(observations, goals)
        )
        target_value = jax.nn.sigmoid(
            self.network.select("target_value")(target_subgoals, goals)
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
        z_true = batch["path_observations"][:, -1]
        weight, gap, current_value, target_value = self._subgoal_gap_weight(
            s0, z_true, batch["goals"]
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

        # Conditional flow matching (rectified): x_u=(1-u)ε+u z, v*=z-ε.
        rng, u_rng, x0_rng = jax.random.split(rng, 3)
        eps = jax.random.normal(x0_rng, z_true.shape)
        u = jax.random.uniform(u_rng, (z_true.shape[0], 1))
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
        first_step_loss = jnp.mean((planned[:, 1] - true_path[:, 1]) ** 2)

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
            + w_path * (path_loss + first_step_loss)
            + w_idm * idm_loss
            + w_v * v_loss
        )
        return loss, {
            **sub_info,
            "path_mse": path_loss,
            "first_step_mse": first_step_loss,
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

    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        del temperature
        if seed is None:
            seed = self.rng
        candidates, _ = self._sample_candidates(observations, goals, seed)
        scores = score_transitive_ratio(self.network, observations, candidates, goals)
        z = pick_best_candidates(candidates, scores)
        path = self._plan(observations, z)
        actions = self.network.select("idm")(observations, path[:, 1])
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        state_dim = int(ex_observations.shape[-1])
        action_dim = int(ex_actions.shape[-1])
        goal_dim = int(config.get("goal_dim", state_dim))
        hidden = tuple(config["hidden_dims"])
        k = int(config["dynamics_N"])
        mode = _subgoal_mode(config)
        ex_goals = jnp.zeros(
            (ex_observations.shape[0], goal_dim), dtype=ex_observations.dtype
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
            "value": [ex_observations, ex_goals],
            "target_value": [ex_observations, ex_goals],
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
        "subgoal_num_candidates": 1,
        "subgoal_include_mean": True,
        "goal_dim": 4,
    }


def default_config_pbf():
    cfg = default_config_pbg()
    cfg.update(
        {
            "subgoal_distribution": "flow",
            "subgoal_num_candidates": 8,
            "flow_steps": 8,
            "flow_loss_weight": 1.0,
        }
    )
    return cfg


# Back-compat aliases used by the train registry.
PBGAgent = PathBridgerAgent
PBFAgent = PathBridgerAgent
