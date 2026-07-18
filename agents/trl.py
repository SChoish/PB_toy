"""Toy-scale port of Transitive RL (aoberai/trl)."""

from __future__ import annotations

from typing import Any

import flax
import flax.core
import jax
import jax.numpy as jnp
import optax

from agents.flax_utils import ModuleDict, TrainState, nonpytree_field
from agents.networks import ActionVectorField, EnsembleGCActionValue


class TRLAgent(flax.struct.PyTreeNode):
    """Official TRL product critic with flow rejection-sampling policy."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(logits, targets):
        return optax.sigmoid_binary_cross_entropy(logits, targets)

    def critic_loss(self, batch, grad_params):
        q_logits = self.network.select("critic")(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            params=grad_params,
        )
        qs = jax.nn.sigmoid(q_logits)

        first_logits = self.network.select("target_critic")(
            batch["observations"],
            batch["value_midpoint_goals"],
            batch["actions"],
        )
        first = jnp.where(
            (batch["value_midpoint_offsets"] <= 1)[None],
            self.config["discount"]
            ** batch["value_midpoint_offsets"][None],
            jax.nn.sigmoid(first_logits),
        )
        second_offsets = (
            batch["value_offsets"] - batch["value_midpoint_offsets"]
        )
        second_logits = self.network.select("target_critic")(
            batch["value_midpoint_observations"],
            batch["value_goals"],
            batch["value_midpoint_actions"],
        )
        second = jnp.where(
            (second_offsets <= 1)[None],
            self.config["discount"] ** second_offsets[None],
            jax.nn.sigmoid(second_logits),
        )
        targets = jax.lax.stop_gradient(jnp.clip(first * second, 1e-6, 1.0))
        weights = jnp.where(
            targets >= qs,
            self.config["expectile"],
            1.0 - self.config["expectile"],
        )
        distance = jax.lax.stop_gradient(
            jnp.log(targets) / jnp.log(self.config["discount"])
        )
        distance_weights = (1.0 / (1.0 + distance)) ** self.config["lam"]
        q_loss = (weights * distance_weights * self.bce_loss(q_logits, targets)).mean()
        return q_loss, {
            "critic_loss": q_loss,
            "q_mean": qs.mean(),
            "q_min": qs.min(),
            "q_max": qs.max(),
            "target_mean": targets.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch["actions"].shape
        x_rng, t_rng = jax.random.split(rng)
        x0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x1 = batch["actions"]
        times = jax.random.uniform(t_rng, (batch_size, 1))
        xt = (1.0 - times) * x0 + times * x1
        velocity = x1 - x0
        pred = self.network.select("actor")(
            batch["observations"],
            batch["actor_goals"],
            xt,
            times,
            params=grad_params,
        )
        loss = jnp.mean((pred - velocity) ** 2)
        return loss, {"actor_loss": loss}

    @jax.jit
    def update(self, batch):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(params):
            critic_loss, critic_info = self.critic_loss(batch, params)
            actor_loss, actor_info = self.actor_loss(batch, params, loss_rng)
            info = {
                "critic/critic_loss": critic_info["critic_loss"],
                "critic/q_mean": critic_info["q_mean"],
                "critic/q_min": critic_info["q_min"],
                "critic/q_max": critic_info["q_max"],
                "critic/target_mean": critic_info["target_mean"],
                "actor/actor_loss": actor_info["actor_loss"],
            }
            return critic_loss + actor_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        target = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1.0 - self.config["tau"]),
            new_network.params["modules_critic"],
            new_network.params["modules_target_critic"],
        )
        new_network = new_network.replace(
            params=flax.core.copy(
                new_network.params, {"modules_target_critic": target}
            )
        )
        return self.replace(network=new_network, rng=new_rng), info

    def _flow_actions(self, observations, goals, noises):
        actions = noises
        for i in range(int(self.config["flow_steps"])):
            times = jnp.full(
                (*actions.shape[:-1], 1),
                i / float(self.config["flow_steps"]),
            )
            actions = actions + self.network.select("actor")(
                observations, goals, actions, times
            ) / float(self.config["flow_steps"])
        return jnp.clip(actions, -1.0, 1.0)

    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        del temperature
        if seed is None:
            seed = self.rng
        n = int(self.config["num_samples"])
        keys = jax.random.split(seed, n)

        def sample_one(key):
            noise = jax.random.normal(key, (*observations.shape[:-1], self.config["action_dim"]))
            return self._flow_actions(observations, goals, noise)

        candidates = jax.vmap(sample_one)(keys)

        def score_one(actions):
            logits = self.network.select("critic")(observations, goals, actions)
            return jnp.min(jax.nn.sigmoid(logits), axis=0)

        scores = jax.vmap(score_one)(candidates)
        best = jnp.argmax(scores, axis=0)
        flat_candidates = candidates.reshape(n, -1, candidates.shape[-1])
        flat_best = best.reshape(-1)
        selected = flat_candidates[
            flat_best, jnp.arange(flat_candidates.shape[1])
        ]
        return selected.reshape(candidates.shape[1:])

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        observations = example_batch["observations"]
        actions = example_batch["actions"]
        goals = example_batch["actor_goals"]
        hidden = tuple(config["hidden_dims"])
        action_dim = int(actions.shape[-1])
        times = jnp.zeros((*actions.shape[:-1], 1), dtype=actions.dtype)

        critic = EnsembleGCActionValue(
            hidden_dims=hidden, num_qs=2, layer_norm=True
        )
        network_def = ModuleDict(
            {
                "critic": critic,
                "target_critic": EnsembleGCActionValue(
                    hidden_dims=hidden, num_qs=2, layer_norm=True
                ),
                "actor": ActionVectorField(
                    hidden_dims=hidden,
                    action_dim=action_dim,
                    layer_norm=True,
                ),
            }
        )
        params = network_def.init(
            init_rng,
            critic=[observations, goals, actions],
            target_critic=[observations, goals, actions],
            actor=[observations, goals, actions, times],
        )["params"]
        params = flax.core.copy(
            params, {"modules_target_critic": params["modules_critic"]}
        )
        network = TrainState.create(
            network_def, params, tx=optax.adam(config.get("lr", 3e-4))
        )
        config = dict(config)
        config["action_dim"] = action_dim
        return cls(rng=rng, network=network, config=config)


def default_config():
    return {
        "lr": 3e-4,
        "hidden_dims": (256, 256),
        "batch_size": 256,
        "discount": 0.99,
        "tau": 0.005,
        "lam": 0.0,
        "expectile": 0.7,
        "flow_steps": 8,
        "num_samples": 8,
        "value_p_curgoal": 0.0,
        "value_p_trajgoal": 1.0,
        "value_p_randomgoal": 0.0,
        "value_geom_sample": True,
        "actor_p_curgoal": 0.0,
        "actor_p_trajgoal": 0.5,
        "actor_p_randomgoal": 0.5,
        "actor_geom_sample": True,
    }
