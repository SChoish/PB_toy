"""Small network building blocks for toy-suite agents."""

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

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded: bool = False,
        temperature: float = 1.0,
    ):
        del goal_encoded  # goals are already features; always concat when present
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


class LengthNormalize(nn.Module):
    """Normalize vectors to length ``sqrt(dim)`` (OGBench HIQL)."""

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(
            x.shape[-1]
        )


class GoalRepNet(nn.Module):
    """State-dependent subgoal representation ``φ([s; g])``."""

    hidden_dims: Sequence[int]
    rep_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs_goal_concat):
        x = MLP(
            (*self.hidden_dims, self.rep_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(obs_goal_concat)
        return LengthNormalize()(x)


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
    """Ensemble V(s, g) returning stacked ``(num_qs, B)`` values."""

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


class HIQLValue(nn.Module):
    """OGBench HIQL value with its own state-dependent goal encoder."""

    hidden_dims: Sequence[int]
    rep_dim: int
    num_qs: int = 2
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, goals):
        goal_reps = GoalRepNet(
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            layer_norm=self.layer_norm,
            name="goal_rep",
        )(jnp.concatenate([observations, goals], axis=-1))
        return EnsembleGCValue(
            hidden_dims=self.hidden_dims,
            num_qs=self.num_qs,
            layer_norm=self.layer_norm,
            name="value_net",
        )(observations, goal_reps)


class GCActionValue(nn.Module):
    """Goal-conditioned scalar value with an optional action input."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, goals=None, actions=None):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        x = jnp.concatenate(inputs, axis=-1)
        x = MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)
        return jnp.squeeze(x, axis=-1)


class EnsembleGCActionValue(nn.Module):
    """Ensemble of action-conditioned goal values."""

    hidden_dims: Sequence[int]
    num_qs: int = 2
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, goals=None, actions=None):
        ensemble = nn.vmap(
            GCActionValue,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        return ensemble(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            name="ensemble",
        )(observations, goals, actions)


class ActionVectorField(nn.Module):
    """Flow-matching velocity field over action vectors."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, goals=None, actions=None, times=None):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        if times is not None:
            inputs.append(times)
        x = jnp.concatenate(inputs, axis=-1)
        return MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)


class SubgoalNet(nn.Module):
    """Diagonal-Gaussian subgoal q(z|s,g): returns ``(mu, log_std)``."""

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 1.0

    @nn.compact
    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        h = MLP(
            (*self.hidden_dims,),
            activate_final=True,
            layer_norm=self.layer_norm,
        )(x)
        mu = nn.Dense(self.state_dim, kernel_init=default_init(1e-2), name="mu")(h)
        log_std = nn.Dense(
            self.state_dim, kernel_init=default_init(1e-2), name="log_std"
        )(h)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


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


class PathResidualNet(nn.Module):
    """Endpoint-preserving residual on the closed-form bridge mean.

    Inputs:
      anchor: absolute current state ``s_t``, shape ``(B, D)``
      z_k: bridge endpoint, shape ``(B, D)``
      t_norm: times ``i/K``, shape ``(B, K+1)``
    Returns residual of shape ``(B, K+1, D)``.
    """

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, anchor, z_k, t_norm):
        if t_norm.ndim == 1:
            t_norm = t_norm[None, :]
        b, t = t_norm.shape
        anchor_b = jnp.broadcast_to(anchor[:, None, :], (b, t, anchor.shape[-1]))
        z_b = jnp.broadcast_to(z_k[:, None, :], (b, t, z_k.shape[-1]))
        t_b = t_norm[..., None]
        x = jnp.concatenate([anchor_b, z_b, t_b], axis=-1)
        x = x.reshape(b * t, -1)
        y = MLP(
            (*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)
        return y.reshape(b, t, self.state_dim)


class ScalarValueNet(nn.Module):
    """Logit V(s, g) for transitive reachability (sigmoid → (0,1])."""

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
