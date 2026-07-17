"""Simplified hierarchical IQL (HIQL) for the hazard toy."""

from __future__ import annotations

from typing import Any

import flax
import flax.core
import jax
import jax.numpy as jnp
import optax

from hazard_env.agents.flax_utils import ModuleDict, TrainState, nonpytree_field
from hazard_env.agents.networks import EnsembleGCValue, GCActor


class HIQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        next_vs = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_v = jnp.min(next_vs, axis=0)
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v

        vs_t = self.network.select("target_value")(
            batch["observations"], batch["value_goals"]
        )
        v_t = jnp.mean(vs_t, axis=0)
        adv = q - v_t

        vs = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        losses = [
            self.expectile_loss(adv, q - vs[i], self.config["expectile"]).mean()
            for i in range(vs.shape[0])
        ]
        value_loss = sum(losses)
        return value_loss, {"value_loss": value_loss, "v_mean": jnp.mean(vs)}

    def low_actor_loss(self, batch, grad_params):
        vs = self.network.select("value")(
            batch["observations"], batch["low_actor_goals"]
        )
        nvs = self.network.select("value")(
            batch["next_observations"], batch["low_actor_goals"]
        )
        adv = jnp.mean(nvs, axis=0) - jnp.mean(vs, axis=0)
        exp_a = jnp.minimum(jnp.exp(adv * self.config["low_alpha"]), 100.0)
        dist = self.network.select("low_actor")(
            batch["observations"], batch["low_actor_goals"], params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])
        loss = -(exp_a * log_prob).mean()
        return loss, {"actor_loss": loss, "adv": adv.mean()}

    def high_actor_loss(self, batch, grad_params):
        # High policy predicts a subgoal state (xy+vel) toward the goal.
        vs = self.network.select("value")(
            batch["observations"], batch["high_actor_goals"]
        )
        nvs = self.network.select("value")(
            batch["high_actor_targets"], batch["high_actor_goals"]
        )
        adv = jnp.mean(nvs, axis=0) - jnp.mean(vs, axis=0)
        exp_a = jnp.minimum(jnp.exp(adv * self.config["high_alpha"]), 100.0)
        dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )
        log_prob = dist.log_prob(batch["high_actor_targets"])
        loss = -(exp_a * log_prob).mean()
        return loss, {
            "actor_loss": loss,
            "mse": jnp.mean((dist.mode() - batch["high_actor_targets"]) ** 2),
        }

    def target_update(self, network):
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * (1.0 - self.config["tau"]) + tp * self.config["tau"],
            network.params["modules_value"],
            network.params["modules_target_value"],
        )
        new_params = flax.core.copy(
            network.params, {"modules_target_value": new_target}
        )
        return network.replace(params=new_params)

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            v_loss, v_info = self.value_loss(batch, params)
            low_loss, low_info = self.low_actor_loss(batch, params)
            high_loss, high_info = self.high_actor_loss(batch, params)
            info = {
                "value/value_loss": v_info["value_loss"],
                "value/v_mean": v_info["v_mean"],
                "low_actor/actor_loss": low_info["actor_loss"],
                "high_actor/actor_loss": high_info["actor_loss"],
                "high_actor/mse": high_info["mse"],
            }
            return v_loss + low_loss + high_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        # Hierarchical: high proposes subgoal, low acts toward it.
        high = self.network.select("high_actor")(observations, goals)
        subgoal = high.mode()
        low = self.network.select("low_actor")(
            observations, subgoal, temperature=temperature
        )
        if seed is None:
            return jnp.clip(low.mode(), -1.0, 1.0)
        return jnp.clip(low.sample(seed=seed), -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = int(ex_actions.shape[-1])
        state_dim = int(ex_observations.shape[-1])
        hidden = tuple(config["hidden_dims"])

        network_def = ModuleDict(
            {
                "value": EnsembleGCValue(hidden_dims=hidden, num_qs=2),
                "target_value": EnsembleGCValue(hidden_dims=hidden, num_qs=2),
                "low_actor": GCActor(
                    hidden_dims=hidden, action_dim=action_dim, const_std=True
                ),
                "high_actor": GCActor(
                    hidden_dims=hidden, action_dim=state_dim, const_std=True
                ),
            }
        )
        params = network_def.init(
            init_rng,
            value=[ex_observations, ex_observations],
            target_value=[ex_observations, ex_observations],
            low_actor=[ex_observations, ex_observations],
            high_actor=[ex_observations, ex_observations],
        )["params"]
        # Sync target with value at init.
        params = flax.core.copy(
            params, {"modules_target_value": params["modules_value"]}
        )
        tx = optax.adam(config.get("lr", 3e-4))
        network = TrainState.create(network_def, params, tx=tx)
        return cls(rng=rng, network=network, config=config)


def default_config():
    return {
        "lr": 3e-4,
        "hidden_dims": (256, 256),
        "discount": 0.99,
        "tau": 0.005,
        "expectile": 0.7,
        "low_alpha": 3.0,
        "high_alpha": 3.0,
        "batch_size": 256,
    }
