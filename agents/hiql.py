"""Hierarchical IQL aligned with OGBench HIQL (state-based, toy-sized)."""

from __future__ import annotations

from typing import Any

import flax
import flax.core
import jax
import jax.numpy as jnp
import optax

from agents.flax_utils import ModuleDict, TrainState, nonpytree_field
from agents.networks import GCActor, GoalRepNet, HIQLValue


class HIQLAgent(flax.struct.PyTreeNode):
    """State-based HIQL aligned with the OGBench reference implementation."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * (diff**2)

    def _project_goal(self, states_or_goals):
        goal_dim = int(self.config.get("goal_dim", states_or_goals.shape[-1]))
        if goal_dim < states_or_goals.shape[-1]:
            return states_or_goals[..., :goal_dim]
        return states_or_goals

    def _goal_rep(self, observations, goals, grad_params=None):
        goals = self._project_goal(goals)
        return self.network.select("goal_rep")(
            jnp.concatenate([observations, goals], axis=-1),
            params=grad_params,
        )

    def _value(self, observations, goals, *, params=None, target: bool = False):
        name = "target_value" if target else "value"
        goals = self._project_goal(goals)
        return self.network.select(name)(observations, goals, params=params)

    def value_loss(self, batch, grad_params):
        next_vs_t = self._value(
            batch["next_observations"], batch["value_goals"], target=True
        )
        next_v_t = jnp.min(next_vs_t, axis=0)
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v_t

        vs_t = self._value(batch["observations"], batch["value_goals"], target=True)
        v_t = jnp.mean(vs_t, axis=0)
        adv = q - v_t

        q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_vs_t[0]
        q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_vs_t[1]

        vs = self.network.select("value")(
            batch["observations"],
            self._project_goal(batch["value_goals"]),
            params=grad_params,
        )
        value_loss = (
            self.expectile_loss(adv, q1 - vs[0], self.config["expectile"]).mean()
            + self.expectile_loss(adv, q2 - vs[1], self.config["expectile"]).mean()
        )
        return value_loss, {
            "value_loss": value_loss,
            "v_mean": jnp.mean(vs),
        }

    def low_actor_loss(self, batch, grad_params):
        vs = self._value(batch["observations"], batch["low_actor_goals"])
        nvs = self._value(batch["next_observations"], batch["low_actor_goals"])
        adv = jnp.mean(nvs, axis=0) - jnp.mean(vs, axis=0)
        exp_a = jnp.minimum(jnp.exp(adv * self.config["low_alpha"]), 100.0)

        goal_reps = self._goal_rep(
            batch["observations"], batch["low_actor_goals"], grad_params=grad_params
        )
        if not self.config.get("low_actor_rep_grad", False):
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select("low_actor")(
            batch["observations"],
            goal_reps,
            goal_encoded=True,
            params=grad_params,
        )
        log_prob = dist.log_prob(batch["actions"])
        loss = -(exp_a * log_prob).mean()
        return loss, {"actor_loss": loss, "adv": adv.mean()}

    def high_actor_loss(self, batch, grad_params):
        vs = self._value(batch["observations"], batch["high_actor_goals"])
        nvs = self._value(batch["high_actor_targets"], batch["high_actor_goals"])
        adv = jnp.mean(nvs, axis=0) - jnp.mean(vs, axis=0)
        exp_a = jnp.minimum(jnp.exp(adv * self.config["high_alpha"]), 100.0)

        dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )
        target = jax.lax.stop_gradient(
            self._goal_rep(batch["observations"], batch["high_actor_targets"])
        )
        log_prob = dist.log_prob(target)
        loss = -(exp_a * log_prob).mean()
        return loss, {
            "actor_loss": loss,
            "mse": jnp.mean((dist.mode() - target) ** 2),
            "adv": adv.mean(),
        }

    def target_update(self, network):
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1.0 - self.config["tau"]),
            network.params["modules_value"],
            network.params["modules_target_value"],
        )
        return network.replace(
            params=flax.core.copy(
                network.params, {"modules_target_value": new_target}
            )
        )

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
                "low_actor/adv": low_info["adv"],
                "high_actor/actor_loss": high_info["actor_loss"],
                "high_actor/mse": high_info["mse"],
                "high_actor/adv": high_info["adv"],
            }
            return v_loss + low_loss + high_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        if seed is None:
            high = self.network.select("high_actor")(
                observations, goals, temperature=1.0
            )
            goal_reps = high.mode()
            goal_reps = goal_reps / jnp.linalg.norm(
                goal_reps, axis=-1, keepdims=True
            ) * jnp.sqrt(goal_reps.shape[-1])
            low = self.network.select("low_actor")(
                observations, goal_reps, goal_encoded=True, temperature=1.0
            )
            return jnp.clip(low.mode(), -1.0, 1.0)

        high_seed, low_seed = jax.random.split(seed)
        high = self.network.select("high_actor")(
            observations, goals, temperature=temperature
        )
        goal_reps = high.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(
            goal_reps, axis=-1, keepdims=True
        ) * jnp.sqrt(goal_reps.shape[-1])
        low = self.network.select("low_actor")(
            observations, goal_reps, goal_encoded=True, temperature=temperature
        )
        return jnp.clip(low.sample(seed=low_seed), -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = int(ex_actions.shape[-1])
        hidden = tuple(config["hidden_dims"])
        rep_dim = int(config.get("rep_dim", 10))
        goal_dim = int(config.get("goal_dim", ex_observations.shape[-1]))
        ex_goals = jnp.zeros(
            (ex_observations.shape[0], goal_dim), dtype=ex_observations.dtype
        )
        network_def = ModuleDict(
            {
                "goal_rep": GoalRepNet(
                    hidden_dims=hidden, rep_dim=rep_dim, layer_norm=True
                ),
                "value": HIQLValue(
                    hidden_dims=hidden, rep_dim=rep_dim, num_qs=2, layer_norm=True
                ),
                "target_value": HIQLValue(
                    hidden_dims=hidden, rep_dim=rep_dim, num_qs=2, layer_norm=True
                ),
                "low_actor": GCActor(
                    hidden_dims=hidden, action_dim=action_dim, const_std=True
                ),
                "high_actor": GCActor(
                    hidden_dims=hidden, action_dim=rep_dim, const_std=True
                ),
            }
        )
        params = network_def.init(
            init_rng,
            goal_rep=[jnp.concatenate([ex_observations, ex_goals], axis=-1)],
            value=[ex_observations, ex_goals],
            target_value=[ex_observations, ex_goals],
            low_actor=[
                ex_observations,
                jnp.zeros(
                    (ex_observations.shape[0], rep_dim),
                    dtype=ex_observations.dtype,
                ),
            ],
            high_actor=[ex_observations, ex_goals],
        )["params"]
        params = flax.core.copy(
            params, {"modules_target_value": params["modules_value"]}
        )
        network = TrainState.create(
            network_def, params, tx=optax.adam(config.get("lr", 3e-4))
        )
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
        "subgoal_steps": 8,
        "rep_dim": 10,
        "goal_dim": 4,
        "target_goal_encoder": "independent",
        "low_actor_rep_grad": False,
        "gc_negative": True,
        "value_p_curgoal": 0.2,
        "value_p_trajgoal": 0.5,
        "value_p_randomgoal": 0.3,
        "value_geom_sample": True,
        "actor_p_randomgoal": 0.0,
        "actor_geom_sample": False,
    }
