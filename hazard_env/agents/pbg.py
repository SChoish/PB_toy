"""PathBridger-Gaussian (PBG): subgoal + closed-form bridge residual + IDM."""

from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from hazard_env.agents.bridge import (
    forward_bridge_coefficients,
    plan_forward_bridge,
    residual_endpoint_weights,
)
from hazard_env.agents.flax_utils import ModuleDict, TrainState, nonpytree_field
from hazard_env.agents.networks import InverseDynamicsNet, PathResidualNet, SubgoalNet


class PBGAgent(flax.struct.PyTreeNode):
    """Lite PathBridger with a deterministic subgoal head.

    Training mirrors Pathbridger_flow dynamics (without critic / SPI):
      - subgoal:  ẑ(s, g) ≈ s_{t+K}
      - path:     plan(s, z) ≈ recorded path s_{t:t+K}
      - IDM:      â(s_i, s_{i+1}) ≈ a_i  (supervised on the data transition)

    Acting:
      - z = subgoal(s, g)
      - path = a_i s + b_i z + w_i r_θ(s, z, i/K)
      - a = IDM(s, path[:, 1])
    """

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

    def total_loss(self, batch, grad_params):
        true_path = batch["path_observations"]  # (B, K+1, D)
        s0 = batch["observations"]
        z_true = true_path[:, -1]

        pred_z = self.network.select("subgoal")(
            s0, batch["goals"], params=grad_params
        )
        sub_loss = jnp.mean((pred_z - z_true) ** 2)

        # Path supervised with true endpoint (stop-grad), matching PB path loss.
        planned = self._plan(s0, jax.lax.stop_gradient(z_true), grad_params=grad_params)
        # Interior + first step; endpoints are clamped to data.
        path_loss = jnp.mean((planned[:, 1:-1] - true_path[:, 1:-1]) ** 2)
        first_step_loss = jnp.mean((planned[:, 1] - true_path[:, 1]) ** 2)

        pred_act = self.network.select("idm")(
            batch["observations"],
            batch["next_observations"],
            params=grad_params,
        )
        idm_loss = jnp.mean((pred_act - batch["actions"]) ** 2)

        w_path = float(self.config.get("path_loss_weight", 1.0))
        w_sub = float(self.config.get("subgoal_loss_weight", 1.0))
        w_idm = float(self.config.get("idm_loss_weight", 1.0))
        loss = w_sub * sub_loss + w_path * (path_loss + first_step_loss) + w_idm * idm_loss
        return loss, {
            "subgoal_mse": sub_loss,
            "path_mse": path_loss,
            "first_step_mse": first_step_loss,
            "idm_mse": idm_loss,
            "loss": loss,
        }

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        del seed, temperature
        z = self.network.select("subgoal")(observations, goals)
        path = self._plan(observations, z)
        next_state = path[:, 1]
        actions = self.network.select("idm")(observations, next_state)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        state_dim = int(ex_observations.shape[-1])
        action_dim = int(ex_actions.shape[-1])
        hidden = tuple(config["hidden_dims"])
        k = int(config["dynamics_N"])

        a, b, _std = forward_bridge_coefficients(
            k,
            lambda_=float(config.get("dynamics_lambda", 1.0)),
            bridge_gamma_inv=float(config.get("bridge_gamma_inv", 0.0)),
            theta_total=float(config.get("theta_total", 1.0)),
            progress_alpha=float(config.get("progress_alpha", 0.8)),
        )
        w = residual_endpoint_weights(k)

        network_def = ModuleDict(
            {
                "subgoal": SubgoalNet(
                    hidden_dims=hidden, state_dim=state_dim, layer_norm=True
                ),
                "path_residual": PathResidualNet(
                    hidden_dims=hidden, state_dim=state_dim, layer_norm=True
                ),
                "idm": InverseDynamicsNet(
                    hidden_dims=hidden, action_dim=action_dim, layer_norm=True
                ),
            }
        )
        ex_t = jnp.broadcast_to(
            jnp.arange(k + 1, dtype=jnp.float32)[None, :] / float(k),
            (ex_observations.shape[0], k + 1),
        )
        params = network_def.init(
            init_rng,
            subgoal=[ex_observations, ex_observations],
            path_residual=[ex_observations, ex_observations, ex_t],
            idm=[ex_observations, ex_observations],
        )["params"]
        tx = optax.adam(config.get("lr", 3e-4))
        network = TrainState.create(network_def, params, tx=tx)
        return cls(
            rng=rng,
            network=network,
            config=config,
            bridge_a=a,
            bridge_b=b,
            bridge_w=w,
        )


def default_config():
    return {
        "lr": 3e-4,
        "hidden_dims": (256, 256),
        "batch_size": 256,
        "dynamics_N": 8,
        "subgoal_steps": 8,
        "dynamics_lambda": 1.0,
        "bridge_gamma_inv": 0.0,
        "theta_total": 1.0,
        "progress_alpha": 0.8,
        "path_loss_weight": 1.0,
        "subgoal_loss_weight": 1.0,
        "idm_loss_weight": 1.0,
    }
