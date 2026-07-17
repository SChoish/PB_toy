"""PathBridger-Flow (PBF): flow subgoal + closed-form bridge residual + IDM."""

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
from hazard_env.agents.networks import FlowVelocityNet, InverseDynamicsNet, PathResidualNet


class PBFAgent(flax.struct.PyTreeNode):
    """Lite PathBridger with a conditional flow-matching subgoal.

    Same bridge/IDM stack as PBG; only the endpoint estimator differs:
      - flow matching trains a velocity field toward s_{t+K}
      - at inference, Euler-integrate to obtain z, then closed-form bridge + IDM
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

    def _flow_subgoal(self, observations, goals, rng, params=None):
        """Integrate conditional flow from noise toward the endpoint."""
        steps = int(self.config.get("flow_steps", 8))
        x = jax.random.normal(rng, observations.shape)
        for i in range(steps):
            u = jnp.full((observations.shape[0], 1), (i + 0.5) / steps)
            v = self.network.select("flow")(
                x, u, observations, goals, params=params
            )
            x = x + v / steps
        return x

    def total_loss(self, batch, grad_params, rng):
        rng, u_rng, x0_rng, _ = jax.random.split(rng, 4)
        true_path = batch["path_observations"]
        s0 = batch["observations"]
        z_true = true_path[:, -1]

        # Conditional flow matching: x_u = (1-u) eps + u z*, v* = z* - eps.
        eps = jax.random.normal(x0_rng, z_true.shape)
        u = jax.random.uniform(u_rng, (z_true.shape[0], 1))
        x_u = (1.0 - u) * eps + u * z_true
        target_v = z_true - eps
        pred_v = self.network.select("flow")(
            x_u, u, s0, batch["goals"], params=grad_params
        )
        flow_loss = jnp.mean((pred_v - target_v) ** 2)

        planned = self._plan(s0, jax.lax.stop_gradient(z_true), grad_params=grad_params)
        path_loss = jnp.mean((planned[:, 1:-1] - true_path[:, 1:-1]) ** 2)
        first_step_loss = jnp.mean((planned[:, 1] - true_path[:, 1]) ** 2)

        pred_act = self.network.select("idm")(
            batch["observations"],
            batch["next_observations"],
            params=grad_params,
        )
        idm_loss = jnp.mean((pred_act - batch["actions"]) ** 2)

        w_path = float(self.config.get("path_loss_weight", 1.0))
        w_flow = float(self.config.get("flow_loss_weight", 1.0))
        w_idm = float(self.config.get("idm_loss_weight", 1.0))
        loss = w_flow * flow_loss + w_path * (path_loss + first_step_loss) + w_idm * idm_loss
        return loss, {
            "flow_mse": flow_loss,
            "path_mse": path_loss,
            "first_step_mse": first_step_loss,
            "idm_mse": idm_loss,
            "loss": loss,
        }

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        del temperature
        if seed is None:
            seed = self.rng
        z = self._flow_subgoal(observations, goals, seed)
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
                "flow": FlowVelocityNet(
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
        ex_u = jnp.zeros((ex_observations.shape[0], 1))
        ex_t = jnp.broadcast_to(
            jnp.arange(k + 1, dtype=jnp.float32)[None, :] / float(k),
            (ex_observations.shape[0], k + 1),
        )
        params = network_def.init(
            init_rng,
            flow=[ex_observations, ex_u, ex_observations, ex_observations],
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
        "flow_steps": 8,
        "dynamics_lambda": 1.0,
        "bridge_gamma_inv": 0.0,
        "theta_total": 1.0,
        "progress_alpha": 0.8,
        "path_loss_weight": 1.0,
        "flow_loss_weight": 1.0,
        "idm_loss_weight": 1.0,
    }
