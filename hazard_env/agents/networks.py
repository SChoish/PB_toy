"""Small network building blocks for hazard_env agents."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class GCActor(nn.Module):
    """Gaussian policy over concatenated (obs, goal) [or encoded goal]."""

    hidden_dims: Sequence[int]
    action_dim: int
    const_std: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    layer_norm: bool = False

    def setup(self):
        self.trunk = MLP(
            (*self.hidden_dims,),
            activate_final=True,
            layer_norm=self.layer_norm,
        )
        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(1e-2)
        )
        if not self.const_std:
            self.log_stds = self.param(
                "log_stds", nn.initializers.zeros, (self.action_dim,)
            )

    def __call__(self, observations, goals=None, temperature: float = 1.0):
        inputs = observations if goals is None else jnp.concatenate(
            [observations, goals], axis=-1
        )
        h = self.trunk(inputs)
        means = self.mean_net(h)
        if self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = jnp.broadcast_to(self.log_stds, means.shape)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )


class GCValue(nn.Module):
    """Scalar V(s, g)."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        x = MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)
        return jnp.squeeze(x, axis=-1)


class EnsembleGCValue(nn.Module):
    hidden_dims: Sequence[int]
    num_qs: int = 2
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, goals):
        ensemble = nn.vmap(
            GCValue,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        return ensemble(
            hidden_dims=self.hidden_dims, layer_norm=self.layer_norm, name="ensemble"
        )(observations, goals)


class SubgoalNet(nn.Module):
    """Predict a future state / subgoal from (obs, goal)."""

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        return MLP(
            (*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)


class InverseDynamicsNet(nn.Module):
    """Predict action from (obs, next_obs)."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, next_observations):
        x = jnp.concatenate([observations, next_observations], axis=-1)
        return MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)


class FlowVelocityNet(nn.Module):
    """Conditional flow velocity v(x_u, u | obs, goal)."""

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x_u, u, observations, goals):
        if u.ndim == 1:
            u = u[:, None]
        x = jnp.concatenate([x_u, u, observations, goals], axis=-1)
        return MLP(
            (*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)
