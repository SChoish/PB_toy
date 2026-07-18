"""Transitive reachability critic (TRL-lite), PathBridger-style."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def soft_update(online_params, target_params, tau: float):
    return jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1.0 - tau),
        online_params,
        target_params,
    )


def transitive_value_loss(network, batch, grad_params, *, discount: float, eps: float = 1e-4):
    """Self + geometric base + product-transitive losses on sigmoid V."""
    obs = batch["observations"]
    goals = batch["value_goals"]
    split = batch["trans_v_split_observations"]
    left_goals = batch["trans_v_left_goals"]
    right_goals = batch["trans_v_right_goals"]
    base_goals = batch["value_base_goals"]
    base_offsets = batch["value_base_offsets"].astype(jnp.float32)
    tri_valid = batch["trans_v_valid_mask"].astype(jnp.float32)
    goal_dim = goals.shape[-1]
    self_goals = obs[..., :goal_dim]

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
    tri_target = jax.lax.stop_gradient(jnp.clip(left * right, eps, 1.0))
    loss_tri_per = optax.sigmoid_binary_cross_entropy(v_tri_logits, tri_target)
    loss_tri = jnp.sum(loss_tri_per * tri_valid) / jnp.maximum(jnp.sum(tri_valid), 1.0)

    loss = loss_self + loss_base + loss_tri
    return loss, {
        "value_self": loss_self,
        "value_base": loss_base,
        "value_tri": loss_tri,
        "value_loss": loss,
    }


def score_transitive_ratio(network, observations, candidates, goals, *, eps: float = 1e-4):
    """``V(s,z) V(z,g) / (V(s,g)+eps)``; candidates shape ``(B, N, D)``."""
    b, n, d = candidates.shape
    goal_dim = goals.shape[-1]
    obs_rep = jnp.repeat(observations[:, None, :], n, axis=1).reshape(b * n, -1)
    g_rep = jnp.repeat(goals[:, None, :], n, axis=1).reshape(b * n, -1)
    z_flat = candidates.reshape(b * n, d)
    z_as_goal = z_flat[..., :goal_dim]
    v_sz = jax.nn.sigmoid(network.select("value")(obs_rep, z_as_goal))
    v_zg = jax.nn.sigmoid(network.select("value")(z_flat, g_rep))
    v_sg = jax.nn.sigmoid(network.select("value")(obs_rep, g_rep))
    return ((v_sz * v_zg) / (v_sg + eps)).reshape(b, n)


def pick_best_candidates(candidates, scores):
    idx = jnp.argmax(scores, axis=1)
    return jnp.take_along_axis(candidates, idx[:, None, None], axis=1)[:, 0, :]
