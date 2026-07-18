"""Transitive reachability critic (TRL-lite), PathBridger-style."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

_VALUE_EXPECTILE = 0.7
_VALUE_BASE_HORIZON = 5.0
_VALUE_DISTANCE_WEIGHT_POWER = 1.0
_VALUE_DISTANCE_WEIGHT_CLIP_MIN = 0.05
_VALUE_DISTANCE_WEIGHT_CLIP_MAX = 1.0


def _bce_expectile_loss(logits, targets, tau: float):
    probs = jax.nn.sigmoid(logits)
    weights = jnp.where(targets >= probs, float(tau), 1.0 - float(tau))
    return weights * optax.sigmoid_binary_cross_entropy(logits, targets)


def soft_update(online_params, target_params, tau: float):
    return jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1.0 - tau),
        online_params,
        target_params,
    )


def transitive_value_loss(
    network,
    batch,
    grad_params,
    *,
    discount: float,
    goal_dim: int | None = None,
    eps: float = 1e-4,
):
    """Self + geometric base + product-transitive losses on sigmoid V."""
    obs = batch["observations"]
    state_dim = int(obs.shape[-1])
    effective_goal_dim = state_dim if goal_dim is None else int(goal_dim)

    def project(name: str):
        value = jnp.asarray(batch[name], dtype=jnp.float32)
        if goal_dim is None and int(value.shape[-1]) != state_dim:
            raise ValueError(
                f"{name} must contain full {state_dim}-D states, got {value.shape}"
            )
        if int(value.shape[-1]) < effective_goal_dim:
            raise ValueError(
                f"{name} needs at least {effective_goal_dim} channels, got {value.shape}"
            )
        return value[..., :effective_goal_dim]

    goals = project("value_goals")
    split = batch["trans_v_split_observations"]
    left_goals = project("trans_v_left_goals")
    right_goals = project("trans_v_right_goals")
    base_goals = project("value_base_goals")
    base_offsets = batch["value_base_offsets"].astype(jnp.float32)
    tri_valid = batch["trans_v_valid_mask"].astype(jnp.float32)
    self_goals = obs[..., :effective_goal_dim]

    v_self_logits = network.select("value")(obs, self_goals, params=grad_params)
    loss_self = optax.sigmoid_binary_cross_entropy(
        v_self_logits, jnp.ones_like(v_self_logits)
    ).mean()

    v_base_logits = network.select("value")(obs, base_goals, params=grad_params)
    base_target = jnp.clip(jnp.power(discount, base_offsets), eps, 1.0)
    loss_base = optax.sigmoid_binary_cross_entropy(v_base_logits, base_target).mean()

    v_tri_logits = network.select("value")(obs, goals, params=grad_params)
    left = jnp.clip(
        jax.nn.sigmoid(network.select("target_value")(obs, left_goals)), eps, 1.0
    )
    right = jnp.clip(
        jax.nn.sigmoid(network.select("target_value")(split, right_goals)), eps, 1.0
    )
    value_offsets = jnp.asarray(
        batch.get("value_offsets", jnp.ones_like(tri_valid)),
        dtype=jnp.float32,
    )
    split_offsets = jnp.asarray(
        batch.get("trans_v_split_offsets", value_offsets),
        dtype=jnp.float32,
    )
    right_offsets = value_offsets - split_offsets
    exact_left = jnp.clip(jnp.power(discount, split_offsets), eps, 1.0)
    exact_right = jnp.clip(jnp.power(discount, right_offsets), eps, 1.0)
    left = jnp.where(split_offsets <= _VALUE_BASE_HORIZON, exact_left, left)
    right = jnp.where(right_offsets <= _VALUE_BASE_HORIZON, exact_right, right)
    tri_target = jax.lax.stop_gradient(jnp.clip(left * right, eps, 1.0))
    loss_tri_per = _bce_expectile_loss(
        v_tri_logits, tri_target, _VALUE_EXPECTILE
    )
    v_tri = jax.nn.sigmoid(v_tri_logits)
    distance = jnp.maximum(
        jnp.log(jax.lax.stop_gradient(jnp.clip(v_tri, eps, 1.0)))
        / jnp.log(jnp.asarray(discount, dtype=jnp.float32)),
        0.0,
    )
    distance_weight = jnp.clip(
        1.0 / jnp.power(1.0 + distance, _VALUE_DISTANCE_WEIGHT_POWER),
        _VALUE_DISTANCE_WEIGHT_CLIP_MIN,
        _VALUE_DISTANCE_WEIGHT_CLIP_MAX,
    )
    tri_weight = tri_valid * distance_weight
    loss_tri = jnp.sum(loss_tri_per * tri_weight) / jnp.maximum(
        jnp.sum(tri_weight), 1.0
    )

    loss = loss_self + loss_base + loss_tri
    return loss, {
        "value_self": loss_self,
        "value_base": loss_base,
        "value_tri": loss_tri,
        "value_loss": loss,
    }


def score_transitive_ratio(
    network, observations, candidates, value_goals, *, eps: float = 1e-3
):
    """``V(s,z) V(z,g) / (V(s,g)+eps)``; candidates shape ``(B, N, D)`` full-state."""
    b, n, d = candidates.shape
    if int(value_goals.shape[-1]) != d:
        raise ValueError(
            f"value_goals must contain full {d}-D states, got {value_goals.shape}"
        )
    obs_rep = jnp.repeat(observations[:, None, :], n, axis=1).reshape(b * n, -1)
    g_rep = jnp.repeat(value_goals[:, None, :], n, axis=1).reshape(
        b * n, d
    )
    z_flat = candidates.reshape(b * n, d)
    v_sz = jax.nn.sigmoid(network.select("value")(obs_rep, z_flat))
    v_zg = jax.nn.sigmoid(network.select("value")(z_flat, g_rep))
    v_sg = jax.nn.sigmoid(network.select("value")(obs_rep, g_rep))
    return ((v_sz * v_zg) / (v_sg + eps)).reshape(b, n)


def pick_best_candidates(candidates, scores):
    idx = jnp.argmax(scores, axis=1)
    return jnp.take_along_axis(candidates, idx[:, None, None], axis=1)[:, 0, :]
