"""TR-HIQL: HIQL actors + PathBridger transitive reachability critic."""

from __future__ import annotations

from typing import Any

import flax
import flax.core
import jax
import jax.numpy as jnp
import optax

from agents.critic import soft_update, transitive_value_loss
from agents.flax_utils import ModuleDict, TrainState, nonpytree_field
from agents.networks import GCActor, ScalarValueNet


class TRHIQLAgent(flax.struct.PyTreeNode):
    """Hierarchical actors like HIQL; critic is TRL-lite sigmoid V."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _project_goal(self, goals):
        goal_dim = int(self.config.get("goal_dim", goals.shape[-1]))
        if goal_dim < goals.shape[-1]:
            return goals[..., :goal_dim]
        return goals

    def _sigmoid_v(self, observations, goals):
        return jax.nn.sigmoid(
            self.network.select("value")(observations, self._project_goal(goals))
        )

    def low_actor_loss(self, batch, grad_params):
        # Critic uses compact goals; low-actor follows full-state subgoals.
        value_goals = batch.get("high_actor_goals", batch["low_actor_goals"])
        v_s = self._sigmoid_v(batch["observations"], value_goals)
        v_sp = self._sigmoid_v(batch["next_observations"], value_goals)
        adv = v_sp - v_s
        exp_a = jnp.minimum(jnp.exp(adv * self.config["low_alpha"]), 100.0)
        dist = self.network.select("low_actor")(
            batch["observations"], batch["low_actor_goals"], params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])
        loss = -(exp_a * log_prob).mean()
        return loss, {"actor_loss": loss, "adv": adv.mean()}

    def high_actor_loss(self, batch, grad_params):
        v_s = self._sigmoid_v(batch["observations"], batch["high_actor_goals"])
        v_z = self._sigmoid_v(batch["high_actor_targets"], batch["high_actor_goals"])
        adv = v_z - v_s
        exp_a = jnp.minimum(jnp.exp(adv * self.config["high_alpha"]), 100.0)
        dist = self.network.select("high_actor")(
            batch["observations"],
            self._project_goal(batch["high_actor_goals"]),
            params=grad_params,
        )
        log_prob = dist.log_prob(batch["high_actor_targets"])
        loss = -(exp_a * log_prob).mean()
        return loss, {
            "actor_loss": loss,
            "mse": jnp.mean((dist.mode() - batch["high_actor_targets"]) ** 2),
            "adv": adv.mean(),
        }

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            v_loss, v_info = transitive_value_loss(
                self.network,
                batch,
                params,
                discount=float(self.config.get("discount", 0.99)),
                goal_dim=int(self.config.get("goal_dim", 4)),
            )
            low_loss, low_info = self.low_actor_loss(batch, params)
            high_loss, high_info = self.high_actor_loss(batch, params)
            info = {
                "value/value_loss": v_info["value_loss"],
                "value/value_self": v_info["value_self"],
                "value/value_base": v_info["value_base"],
                "value/value_tri": v_info["value_tri"],
                "low_actor/actor_loss": low_info["actor_loss"],
                "low_actor/adv": low_info["adv"],
                "high_actor/actor_loss": high_info["actor_loss"],
                "high_actor/mse": high_info["mse"],
                "high_actor/adv": high_info["adv"],
            }
            return v_loss + low_loss + high_loss, info

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
        goals = self._project_goal(goals)
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
        goal_dim = int(config.get("goal_dim", state_dim))
        hidden = tuple(config["hidden_dims"])
        ex_goals = jnp.zeros(
            (ex_observations.shape[0], goal_dim), dtype=ex_observations.dtype
        )

        network_def = ModuleDict(
            {
                "value": ScalarValueNet(hidden_dims=hidden, layer_norm=True),
                "target_value": ScalarValueNet(hidden_dims=hidden, layer_norm=True),
                "low_actor": GCActor(
                    hidden_dims=hidden, action_dim=action_dim, const_std=True
                ),
                "high_actor": GCActor(
                    hidden_dims=hidden, action_dim=state_dim, const_std=True
                ),
            }
        )
        # Value / high-actor take compact goals; low-actor follows full-state subgoals.
        params = network_def.init(
            init_rng,
            value=[ex_observations, ex_goals],
            target_value=[ex_observations, ex_goals],
            low_actor=[ex_observations, ex_observations],
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
        "low_alpha": 3.0,
        "high_alpha": 3.0,
        "batch_size": 256,
        "subgoal_steps": 25,
        "goal_dim": 4,
    }
