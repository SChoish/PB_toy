"""Toy-scale port of Decoupled Q-Chunking (ColinQiyangLi/dqc)."""

from __future__ import annotations

from typing import Any

import flax
import flax.core
import jax
import jax.numpy as jnp
import optax

from hazard_env.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from hazard_env.utils.networks import (
    ActionVectorField,
    EnsembleGCActionValue,
    GCActionValue,
)


class DQCAgent(flax.struct.PyTreeNode):
    """Long chunk critic with a distilled one-step flow policy."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(logits, targets):
        return optax.sigmoid_binary_cross_entropy(logits, targets)

    def _valid_mask(self, batch):
        index = int(self.config["policy_chunk_size"]) - 1
        return batch["valids"][..., index]

    def chunk_critic_loss(self, batch, grad_params):
        next_v = jax.nn.sigmoid(
            self.network.select("value")(
                batch["high_value_next_observations"],
                batch["high_value_goals"],
            )
        )
        target = (
            batch["high_value_rewards"]
            + self.config["discount"] ** batch["high_value_backup_horizon"]
            * batch["high_value_masks"]
            * next_v
        )
        target = jax.lax.stop_gradient(jnp.clip(target, 0.0, 1.0))
        logits = self.network.select("chunk_critic")(
            batch["observations"],
            batch["high_value_goals"],
            batch["high_value_action_chunks"],
            params=grad_params,
        )
        q = jax.nn.sigmoid(logits)
        loss = self.bce_loss(logits, target[None]).mean()
        return loss, {
            "critic_loss": loss,
            "q_mean": q.mean(),
            "target_mean": target.mean(),
        }

    def action_critic_loss(self, batch, grad_params):
        chunk_target = jax.nn.sigmoid(
            self.network.select("chunk_critic")(
                batch["observations"],
                batch["high_value_goals"],
                batch["high_value_action_chunks"],
            )
        )
        target = jax.lax.stop_gradient(jnp.mean(chunk_target, axis=0))
        partial_actions = batch["high_value_action_chunks"][
            ..., : self.config["ac_action_dim"]
        ]
        logits = self.network.select("action_critic")(
            batch["observations"],
            batch["high_value_goals"],
            partial_actions,
            params=grad_params,
        )
        q = jax.nn.sigmoid(logits)
        weights = jnp.where(
            target[None] >= q,
            self.config["kappa_d"],
            1.0 - self.config["kappa_d"],
        )
        valid = self._valid_mask(batch)
        critic_loss = (
            weights * self.bce_loss(logits, target[None]) * valid[None]
        ).mean()

        target_qs = jax.nn.sigmoid(
            self.network.select("target_action_critic")(
                batch["observations"],
                batch["high_value_goals"],
                partial_actions,
            )
        )
        if self.config["q_agg"] == "min":
            target_q = target_qs.min(axis=0)
        else:
            target_q = target_qs.mean(axis=0)
        target_q = jax.lax.stop_gradient(jnp.clip(target_q, 1e-5, 1.0 - 1e-5))
        value_logits = self.network.select("value")(
            batch["observations"],
            batch["high_value_goals"],
            params=grad_params,
        )
        value = jax.nn.sigmoid(value_logits)
        value_weights = jnp.where(
            target_q >= value,
            self.config["kappa_b"],
            1.0 - self.config["kappa_b"],
        )
        target_q_logits = jnp.log(target_q) - jnp.log1p(-target_q)
        if self.config["implicit_backup_type"] == "expectile":
            value_error = self.bce_loss(value_logits, target_q)
        else:
            value_error = jnp.abs(value_logits - target_q_logits)
        value_loss = (value_weights * value_error * valid).mean()
        return critic_loss + value_loss, {
            "critic_loss": critic_loss,
            "value_loss": value_loss,
            "q_mean": q.mean(),
            "v_mean": value.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        batch_size = batch["observations"].shape[0]
        x_rng, t_rng = jax.random.split(rng)
        x1 = batch["high_value_action_chunks"][
            ..., : self.config["ac_action_dim"]
        ]
        x0 = jax.random.normal(x_rng, x1.shape)
        times = jax.random.uniform(t_rng, (batch_size, 1))
        xt = (1.0 - times) * x0 + times * x1
        pred = self.network.select("actor_bc")(
            batch["observations"], None, xt, times, params=grad_params
        )
        loss = (((pred - (x1 - x0)) ** 2).mean(axis=-1) * self._valid_mask(batch)).mean()
        return loss, {"bc_flow_loss": loss}

    @jax.jit
    def update(self, batch):
        new_rng, actor_rng = jax.random.split(self.rng)

        def loss_fn(params):
            chunk_loss, chunk_info = self.chunk_critic_loss(batch, params)
            action_loss, action_info = self.action_critic_loss(batch, params)
            actor_loss, actor_info = self.actor_loss(batch, params, actor_rng)
            info = {
                "chunk_critic/critic_loss": chunk_info["critic_loss"],
                "chunk_critic/q_mean": chunk_info["q_mean"],
                "chunk_critic/target_mean": chunk_info["target_mean"],
                "action_critic/critic_loss": action_info["critic_loss"],
                "action_critic/value_loss": action_info["value_loss"],
                "action_critic/q_mean": action_info["q_mean"],
                "action_critic/v_mean": action_info["v_mean"],
                "actor/bc_flow_loss": actor_info["bc_flow_loss"],
            }
            return chunk_loss + action_loss + actor_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        target = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1.0 - self.config["tau"]),
            new_network.params["modules_action_critic"],
            new_network.params["modules_target_action_critic"],
        )
        new_network = new_network.replace(
            params=flax.core.copy(
                new_network.params, {"modules_target_action_critic": target}
            )
        )
        return self.replace(network=new_network, rng=new_rng), info

    def _flow_actions(self, observations, noises):
        actions = noises
        for i in range(int(self.config["flow_steps"])):
            times = jnp.full(
                (*actions.shape[:-1], 1),
                i / float(self.config["flow_steps"]),
            )
            actions = actions + self.network.select("actor_bc")(
                observations, None, actions, times
            ) / float(self.config["flow_steps"])
        return jnp.clip(actions, -1.0, 1.0)

    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        del temperature
        if seed is None:
            seed = self.rng
        n = int(self.config["best_of_n"])
        keys = jax.random.split(seed, n)

        def sample_one(key):
            noise = jax.random.normal(
                key, (*observations.shape[:-1], self.config["ac_action_dim"])
            )
            return self._flow_actions(observations, noise)

        candidates = jax.vmap(sample_one)(keys)

        def score_one(actions):
            q = jax.nn.sigmoid(
                self.network.select("action_critic")(
                    observations, goals, actions
                )
            )
            return q.min(axis=0) if self.config["q_agg"] == "min" else q.mean(axis=0)

        scores = jax.vmap(score_one)(candidates)
        best = jnp.argmax(scores, axis=0)
        flat = candidates.reshape(n, -1, candidates.shape[-1])
        selected = flat[best.reshape(-1), jnp.arange(flat.shape[1])]
        selected = selected.reshape(candidates.shape[1:])
        return selected[..., : self.config["action_dim"]]

    def _sigmoid_v(self, observations, goals):
        return jax.nn.sigmoid(
            self.network.select("value")(observations, goals)
        )

    @classmethod
    def create(cls, seed, example_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        observations = example_batch["observations"]
        actions = example_batch["actions"]
        goals = example_batch["high_value_goals"]
        chunks = example_batch["high_value_action_chunks"]
        hidden = tuple(config["hidden_dims"])
        action_dim = int(actions.shape[-1])
        ac_action_dim = int(config["policy_chunk_size"]) * action_dim
        partial = chunks[..., :ac_action_dim]
        times = jnp.zeros((*observations.shape[:-1], 1), dtype=observations.dtype)

        network_def = ModuleDict(
            {
                "chunk_critic": EnsembleGCActionValue(
                    hidden_dims=hidden, num_qs=int(config["num_qs"]), layer_norm=True
                ),
                "action_critic": EnsembleGCActionValue(
                    hidden_dims=hidden, num_qs=int(config["num_qs"]), layer_norm=True
                ),
                "target_action_critic": EnsembleGCActionValue(
                    hidden_dims=hidden, num_qs=int(config["num_qs"]), layer_norm=True
                ),
                "value": GCActionValue(hidden_dims=hidden, layer_norm=True),
                "actor_bc": ActionVectorField(
                    hidden_dims=hidden, action_dim=ac_action_dim, layer_norm=True
                ),
            }
        )
        params = network_def.init(
            init_rng,
            chunk_critic=[observations, goals, chunks],
            action_critic=[observations, goals, partial],
            target_action_critic=[observations, goals, partial],
            value=[observations, goals],
            actor_bc=[observations, None, partial, times],
        )["params"]
        params = flax.core.copy(
            params,
            {
                "modules_target_action_critic": params[
                    "modules_action_critic"
                ]
            },
        )
        network = TrainState.create(
            network_def, params, tx=optax.adam(config.get("lr", 3e-4))
        )
        config = dict(config)
        config.update(
            action_dim=action_dim,
            ac_action_dim=ac_action_dim,
            ob_dims=tuple(observations.shape[1:]),
        )
        return cls(rng=rng, network=network, config=config)


def default_config():
    return {
        "lr": 3e-4,
        "hidden_dims": (256, 256),
        "batch_size": 256,
        "discount": 0.99,
        "tau": 0.005,
        "num_qs": 2,
        "q_agg": "mean",
        "flow_steps": 8,
        "backup_horizon": 8,
        "policy_chunk_size": 1,
        "kappa_d": 0.5,
        "implicit_backup_type": "quantile",
        "kappa_b": 0.9,
        "best_of_n": 8,
        "value_p_curgoal": 0.2,
        "value_p_trajgoal": 0.5,
        "value_p_randomgoal": 0.3,
        "value_geom_sample": False,
    }
