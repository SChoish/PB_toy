"""Simplified goal-conditioned behavioral cloning."""

from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from hazard_env.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from hazard_env.utils.networks import GCActor


class BCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params):
        dist = self.network.select("actor")(
            batch["observations"], batch["actor_goals"], params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])
        loss = -log_prob.mean()
        return loss, {
            "actor_loss": loss,
            "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
        }

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            loss, info = self.actor_loss(batch, params)
            return loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        dist = self.network.select("actor")(
            observations, goals, temperature=temperature
        )
        if seed is None:
            return jnp.clip(dist.mode(), -1.0, 1.0)
        return jnp.clip(dist.sample(seed=seed), -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = int(ex_actions.shape[-1])
        actor_def = GCActor(
            hidden_dims=tuple(config["hidden_dims"]),
            action_dim=action_dim,
            const_std=config.get("const_std", True),
            layer_norm=config.get("layer_norm", False),
        )
        network_def = ModuleDict({"actor": actor_def})
        params = network_def.init(
            init_rng,
            actor=[ex_observations, ex_observations],
        )["params"]
        tx = optax.adam(config.get("lr", 3e-4))
        network = TrainState.create(network_def, params, tx=tx)
        return cls(rng=rng, network=network, config=config)


def default_config():
    return {
        "lr": 3e-4,
        "hidden_dims": (256, 256),
        "const_std": True,
        "layer_norm": False,
        "batch_size": 256,
    }
