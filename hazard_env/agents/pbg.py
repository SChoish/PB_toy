"""Simplified PathBridger-Gaussian (PBG): subgoal regressor + inverse dynamics."""

from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from hazard_env.agents.flax_utils import ModuleDict, TrainState, nonpytree_field
from hazard_env.agents.networks import InverseDynamicsNet, SubgoalNet


class PBGAgent(flax.struct.PyTreeNode):
    """Toy PathBridger with a deterministic/Gaussian-style subgoal head.

    Training:
      - subgoal MSE:  ẑ(s, g) ≈ s_{t+k}
      - IDM MSE:      â(s, s') ≈ a

    Acting:
      - z = subgoal(s, g)
      - step toward a convex mix of z and g, then IDM(s, s_next)
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def total_loss(self, batch, grad_params):
        pred_sub = self.network.select("subgoal")(
            batch["observations"], batch["goals"], params=grad_params
        )
        sub_loss = jnp.mean((pred_sub - batch["subgoals"]) ** 2)

        pred_act = self.network.select("idm")(
            batch["observations"],
            batch["next_observations"],
            params=grad_params,
        )
        idm_loss = jnp.mean((pred_act - batch["actions"]) ** 2)
        loss = sub_loss + idm_loss
        return loss, {
            "subgoal_mse": sub_loss,
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
        subgoal = self.network.select("subgoal")(observations, goals)
        # Short bridge step: move a fraction of the way to the subgoal in state space.
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
                "subgoal": SubgoalNet(
                    hidden_dims=hidden, state_dim=state_dim, layer_norm=True
                ),
                "idm": InverseDynamicsNet(
                    hidden_dims=hidden, action_dim=action_dim, layer_norm=True
                ),
            }
        )
        params = network_def.init(
            init_rng,
            subgoal=[ex_observations, ex_observations],
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
        "batch_size": 256,
        "subgoal_steps": 8,
    }
