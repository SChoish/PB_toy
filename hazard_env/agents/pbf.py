"""Simplified PathBridger-Flow (PBF): conditional flow subgoal + inverse dynamics."""

from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from hazard_env.agents.flax_utils import ModuleDict, TrainState, nonpytree_field
from hazard_env.agents.networks import FlowVelocityNet, InverseDynamicsNet


class PBFAgent(flax.struct.PyTreeNode):
    """Toy PathBridger with a conditional flow-matching subgoal.

    Training:
      - flow matching: x_u = (1-u) x0 + u x1, target vel = x1 - x0
      - IDM MSE: â(s, s') ≈ a

    Acting:
      - integrate flow from noise/current toward subgoal for a few Euler steps
      - IDM(s, s_next) with a short bridge mix
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def total_loss(self, batch, grad_params, rng):
        rng, u_rng, x0_rng = jax.random.split(rng, 3)
        x1 = batch["subgoals"]
        x0 = batch["observations"]
        # Optionally mix noise endpoint for fuller support.
        noise = jax.random.normal(x0_rng, x0.shape)
        x0 = 0.5 * x0 + 0.5 * noise
        u = jax.random.uniform(u_rng, (x0.shape[0], 1))
        x_u = (1.0 - u) * x0 + u * x1
        target_v = x1 - x0
        pred_v = self.network.select("flow")(
            x_u, u, batch["observations"], batch["goals"], params=grad_params
        )
        flow_loss = jnp.mean((pred_v - target_v) ** 2)

        pred_act = self.network.select("idm")(
            batch["observations"],
            batch["next_observations"],
            params=grad_params,
        )
        idm_loss = jnp.mean((pred_act - batch["actions"]) ** 2)
        loss = flow_loss + idm_loss
        return loss, {
            "flow_mse": flow_loss,
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

    def _integrate_subgoal(self, observations, goals, rng):
        steps = int(self.config.get("flow_steps", 4))
        x = observations
        # Start from observation (not pure noise) for stable short-horizon control.
        for i in range(steps):
            u = jnp.full((observations.shape[0], 1), (i + 0.5) / steps)
            v = self.network.select("flow")(x, u, observations, goals)
            x = x + v / steps
        return x

    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        del temperature
        if seed is None:
            seed = jax.random.PRNGKey(0)
        subgoal = self._integrate_subgoal(observations, goals, seed)
        alpha = self.config.get("bridge_alpha", 0.35)
        next_state = observations + alpha * (subgoal - observations)
        actions = self.network.select("idm")(observations, next_state)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        state_dim = int(ex_observations.shape[-1])
        action_dim = int(ex_actions.shape[-1])
        hidden = tuple(config["hidden_dims"])
        network_def = ModuleDict(
            {
                "flow": FlowVelocityNet(
                    hidden_dims=hidden, state_dim=state_dim, layer_norm=True
                ),
                "idm": InverseDynamicsNet(
                    hidden_dims=hidden, action_dim=action_dim, layer_norm=True
                ),
            }
        )
        ex_u = jnp.zeros((ex_observations.shape[0], 1))
        params = network_def.init(
            init_rng,
            flow=[ex_observations, ex_u, ex_observations, ex_observations],
            idm=[ex_observations, ex_observations],
        )["params"]
        tx = optax.adam(config.get("lr", 3e-4))
        network = TrainState.create(network_def, params, tx=tx)
        return cls(rng=rng, network=network, config=config)


def default_config():
    return {
        "lr": 3e-4,
        "hidden_dims": (256, 256),
        "bridge_alpha": 0.35,
        "flow_steps": 4,
        "batch_size": 256,
        "subgoal_steps": 8,
    }
