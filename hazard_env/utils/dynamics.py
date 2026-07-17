"""Closed-form linear-SDE bridge (PathBridger-compatible, mean path).

Port of the prefix-progress schedule used in Pathbridger_flow:
  mu_i = a_i s_0 + b_i z_K
  path_i = mu_i + w_i * residual(s_0, z_K, i/K)
  w_i = i*(K-i)/K^2   (zeros at endpoints)
"""

from __future__ import annotations

import jax.numpy as jnp


def desired_prefix_progress(n: int, progress_alpha: float = 0.8) -> jnp.ndarray:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    idx = jnp.arange(n + 1, dtype=jnp.float32)
    c = (idx / float(n)) ** float(progress_alpha)
    return c.at[0].set(0.0).at[-1].set(1.0)


def prefix_progress_theta_fwd(
    n: int,
    theta_total: float = 1.0,
    progress_alpha: float = 0.8,
) -> jnp.ndarray:
    c = desired_prefix_progress(n, progress_alpha=progress_alpha)
    total = jnp.asarray(theta_total, dtype=jnp.float32)
    theta = jnp.arcsinh(c * jnp.sinh(total))
    theta_fwd = theta[1:] - theta[:-1]
    return jnp.maximum(theta_fwd, 1e-12)


def _linear_dynamics_arrays(theta_fwd, g2_fwd, step_var_fwd, gamma_inv: float = 0.0):
    theta_fwd = jnp.asarray(theta_fwd, dtype=jnp.float32)
    g2_fwd = jnp.asarray(g2_fwd, dtype=jnp.float32)
    step_var_fwd = jnp.asarray(step_var_fwd, dtype=jnp.float32)
    n = int(theta_fwd.shape[0])
    a_step = jnp.exp(theta_fwd)
    q2 = step_var_fwd * a_step**2

    ps = [jnp.asarray(0.0, dtype=jnp.float32)]
    for i in range(n):
        ps.append(a_step[i] ** 2 * ps[-1] + q2[i])
    p = jnp.stack(ps)

    phis = [None] * (n + 1)
    phis[n] = jnp.asarray(1.0, dtype=jnp.float32)
    for i in range(n - 1, -1, -1):
        phis[i] = a_step[i] * phis[i + 1]
    phi = jnp.stack(phis)

    oms = [None] * (n + 1)
    oms[n] = jnp.asarray(0.0, dtype=jnp.float32)
    for i in range(n - 1, -1, -1):
        oms[i] = q2[i] * phi[i + 1] ** 2 + oms[i + 1]
    omega = jnp.stack(oms)

    gamma_inv_arr = jnp.asarray(gamma_inv, dtype=jnp.float32)
    denom = jnp.maximum(p[-1] + gamma_inv_arr, 1e-12)
    beta = p * phi / denom
    bridge_var = p * (omega + gamma_inv_arr) / denom
    bridge_var = jnp.maximum(bridge_var, 0.0)
    if float(gamma_inv) == 0.0:
        beta = beta.at[0].set(0.0).at[-1].set(1.0)
        bridge_var = bridge_var.at[0].set(0.0).at[-1].set(0.0)
    return beta, bridge_var


def forward_bridge_coefficients(
    k: int,
    *,
    lambda_: float = 1.0,
    bridge_gamma_inv: float = 0.0,
    theta_total: float = 1.0,
    progress_alpha: float = 0.8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(a, b, std)`` each of shape ``(K+1,)`` for ``mu = a s0 + b zK``."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    theta_fwd = prefix_progress_theta_fwd(
        k, theta_total=theta_total, progress_alpha=progress_alpha
    )
    g2_fwd = 2.0 * float(lambda_) ** 2 * theta_fwd
    step_var_fwd = float(lambda_) ** 2 * (1.0 - jnp.exp(-2.0 * theta_fwd))
    beta, bridge_var = _linear_dynamics_arrays(
        theta_fwd, g2_fwd, step_var_fwd, gamma_inv=bridge_gamma_inv
    )
    b = beta
    std = jnp.sqrt(jnp.maximum(bridge_var, 0.0))
    a = 1.0 - b
    a = a.at[0].set(1.0).at[-1].set(0.0)
    b = b.at[0].set(0.0).at[-1].set(1.0)
    std = std.at[0].set(0.0).at[-1].set(0.0)
    return a, b, std


def residual_endpoint_weights(k: int) -> jnp.ndarray:
    """``w_i = i (K - i) / K^2``, zero at endpoints."""
    idx = jnp.arange(k + 1, dtype=jnp.float32)
    return idx * (float(k) - idx) / float(k * k)


def plan_forward_bridge(
    s0: jnp.ndarray,
    z_k: jnp.ndarray,
    residual: jnp.ndarray,
    *,
    a: jnp.ndarray,
    b: jnp.ndarray,
    w: jnp.ndarray,
) -> jnp.ndarray:
    """Build ``(B, K+1, D)`` path with endpoint clamp."""
    mu = a[None, :, None] * s0[:, None, :] + b[None, :, None] * z_k[:, None, :]
    path = mu + w[None, :, None] * residual
    return path.at[:, 0, :].set(s0).at[:, -1, :].set(z_k)
