"""Shared PathBridger math + plotting (env-agnostic)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

from .envs import ToyHazardEnv, l2

CMAP = LinearSegmentedColormap.from_list(
    "value_field",
    [
        "#1a4f86",
        "#3a7eb0",
        "#6fadc0",
        "#a8c9a0",
        "#d9d88a",
        "#f0c95a",
        "#f0a830",
    ],
)

REPLAN_FRAC = 0.45  # h_a / K — solid executed prefix (ref.png)


# ------------------------------------------------------------------
# Endpoint-pinned bridge — paper Eq. (28), (47)
# ------------------------------------------------------------------
def bridge_schedules(K):
    i = np.arange(K + 1, dtype=float)
    alpha = (i / K) ** 0.8
    mask = i * (K - i) / (K**2)
    return alpha, mask


def pinned_bridge(s_t, delta_star, residuals):
    """hat_Delta_i = alpha_i Delta* + m_i R_i  →  s_t + hat_Delta_i."""
    K = len(residuals) - 1
    alpha, mask = bridge_schedules(K)
    deltas = alpha[:, None] * delta_star[None, :] + mask[:, None] * residuals
    return np.asarray(s_t, dtype=float)[None, :] + deltas


def split_bridge(path, frac=REPLAN_FRAC):
    path = np.asarray(path, dtype=float)
    idx = int(np.clip(round(frac * (len(path) - 1)), 1, len(path) - 2))
    return path[: idx + 1], path[idx:], path[idx]


def densify_path(points, n_per_seg=6):
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 2:
        return np.asarray(pts)
    segs = []
    for i in range(len(pts) - 1):
        segs.append(np.linspace(pts[i], pts[i + 1], n_per_seg, endpoint=False))
    segs.append(pts[-1][None, :])
    return np.vstack(segs)


def data_geometry_residual(s_t, delta_star, data_traj, K):
    """Stand-in / boost for R_theta: pull interior toward nearest data states."""
    alpha, _ = bridge_schedules(K)
    residuals = np.zeros((K + 1, 2))
    for i in range(1, K):
        ref = np.asarray(s_t, dtype=float) + alpha[i] * delta_star
        d = np.linalg.norm(data_traj - ref[None, :], axis=1)
        residuals[i] = data_traj[int(np.argmin(d))] - ref
    return residuals


def push_out_of_hazard(path, env, margin=0.04):
    """Soft projection so the drawn bridge does not enter the hazard disk."""
    out = np.asarray(path, dtype=float).copy()
    center = env.hazard_center
    r = env.hazard_radius + margin
    for i in range(len(out)):
        d = np.linalg.norm(out[i] - center)
        if d < r and d > 1e-8:
            out[i] = center + (out[i] - center) / d * r
    return out


def _nearest_index(traj, point):
    d = np.linalg.norm(np.asarray(traj, float) - np.asarray(point, float)[None, :], axis=1)
    return int(np.argmin(d))


def compose_ref_style_bridge(
    s_t,
    z_star,
    data_traj,
    env,
    K: int,
    ha_frac: float = REPLAN_FRAC,
    n_dense: int = 5,
):
    """
    Match ref.png storytelling:
      - executed prefix hugs the data trajectory (safe support)
      - at replan, peel toward z* with an endpoint-pinned residual arc
    """
    s_t = np.asarray(s_t, dtype=float)
    z_star = np.asarray(z_star, dtype=float)
    data_traj = np.asarray(data_traj, dtype=float)

    i0 = _nearest_index(data_traj, s_t)
    i_z = _nearest_index(data_traj, z_star)
    if i_z <= i0 + 2:
        i_z = min(len(data_traj) - 1, i0 + max(K, 8))

    # Replan sits partway along the data arc toward the nearest data point to z*.
    i_replan = int(np.clip(round(i0 + ha_frac * (i_z - i0)), i0 + 2, i_z - 1))
    prefix_raw = data_traj[i0 : i_replan + 1].copy()
    prefix_raw[0] = s_t
    replan = prefix_raw[-1].copy()

    # Remainder: short pinned bridge from replan → z* pulled toward data geometry.
    K_rem = max(8, K // 2)
    delta = z_star - replan
    residuals = data_geometry_residual(replan, delta, data_traj[i_replan:], K_rem)
    # Bias residual so the peel stays on the safe (data) side of the hazard.
    rem_raw = pinned_bridge(replan, delta, residuals)
    rem_raw = push_out_of_hazard(rem_raw, env, margin=0.045)
    rem_raw[0] = replan
    rem_raw[-1] = z_star

    prefix = densify_path(prefix_raw, n_per_seg=n_dense)
    remainder = densify_path(rem_raw, n_per_seg=n_dense)
    # Ensure join continuity for plotting.
    remainder[0] = prefix[-1]
    return prefix, remainder, prefix[-1]


# ------------------------------------------------------------------
# Value-greedy path toward z* (failure mode)
# ------------------------------------------------------------------
def gradient_ascent_path(
    start,
    target,
    value_grad_fn,
    step_size=0.03,
    tolerance=0.04,
    max_steps=200,
):
    point = np.asarray(start, dtype=float).copy()
    path = [point.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(point - target) <= tolerance:
            break
        value, gx, gy = value_grad_fn(point[0], point[1])
        direction = np.array([gx, gy], dtype=float)
        n = np.linalg.norm(direction)
        if n < 1e-12:
            break
        direction /= n
        local = step_size
        while local > 1e-5:
            cand = point + local * direction
            if value_grad_fn(cand[0], cand[1])[0] >= value - 1e-12:
                break
            local *= 0.5
        if local <= 1e-5:
            break
        point = cand
        path.append(point.copy())
    if np.linalg.norm(path[-1] - target) > 0.01:
        path.append(np.asarray(target, dtype=float).copy())
    return np.asarray(path)


def chord_hits_hazard(s_t, z, env):
    """Return whether the closed line segment intersects the hazard disk."""
    start = np.asarray(s_t, dtype=float)
    end = np.asarray(z, dtype=float)
    delta = end - start
    squared_length = float(delta @ delta)
    if squared_length == 0.0:
        closest = start
    else:
        fraction = float(
            np.clip(((np.asarray(env.hazard_center) - start) @ delta) / squared_length, 0.0, 1.0)
        )
        closest = start + fraction * delta
    return float(np.linalg.norm(closest - env.hazard_center)) <= float(env.hazard_radius)


# ------------------------------------------------------------------
# Plot (no labels / legend — faint value field only)
# ------------------------------------------------------------------
def plot_concept(
    env: ToyHazardEnv,
    V,
    xs,
    ys,
    greedy_path,
    bridge_prefix,
    bridge_remainder,
    replan,
    data_traj,
    s_t,
    z_star,
    goal,
    out_path,
    data_subgoal=None,
):
    X, Y = np.meshgrid(xs, ys)
    V_n = (V - V.min()) / (V.max() - V.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.contourf(X, Y, V_n, levels=100, cmap=CMAP, alpha=0.98)

    # Hazard (annotation only — not encoded in V)
    ax.add_patch(
        Circle(
            env.hazard_center,
            env.hazard_radius * 1.45,
            facecolor="#e53935",
            edgecolor="none",
            alpha=0.08,
            zorder=2,
        )
    )
    ax.add_patch(
        Circle(
            env.hazard_center,
            env.hazard_radius,
            facecolor=(0.90, 0.14, 0.12, 0.45),
            edgecolor="#9f1c1c",
            linewidth=2.4,
            hatch="///",
            zorder=3,
        )
    )

    # Data trajectory (support)
    ax.plot(
        data_traj[:, 0],
        data_traj[:, 1],
        color="#111111",
        lw=4.4,
        solid_capstyle="round",
        zorder=5,
    )

    # Value-greedy failure mode (through hazard)
    ax.plot(
        greedy_path[:, 0],
        greedy_path[:, 1],
        color="#1e88e5",
        lw=5.0,
        solid_capstyle="round",
        zorder=7,
    )
    ax.annotate(
        "",
        xy=greedy_path[-1],
        xytext=greedy_path[max(0, len(greedy_path) - 8)],
        arrowprops=dict(arrowstyle="-|>", color="#1e88e5", lw=3.2, mutation_scale=20),
        zorder=8,
    )

    # Bridge: thick solid prefix + dashed remainder (ref.png)
    ax.plot(
        bridge_prefix[:, 0],
        bridge_prefix[:, 1],
        color="#4a148c",
        lw=6.6,
        solid_capstyle="round",
        zorder=9,
    )
    ax.plot(
        bridge_remainder[:, 0],
        bridge_remainder[:, 1],
        color="#7b1fa2",
        lw=4.8,
        ls=(0, (2.8, 2.4)),
        solid_capstyle="round",
        zorder=9,
    )

    # Estimated path between data subgoal and z* (dotted, ref.png)
    if data_subgoal is not None:
        mid = 0.5 * (np.asarray(data_subgoal, float) + np.asarray(z_star, float))
        # slight bow away from hazard for readability
        away = mid - np.asarray(env.hazard_center, float)
        nrm = np.linalg.norm(away)
        if nrm > 1e-8:
            mid = mid + 0.08 * away / nrm
        curve = densify_path([data_subgoal, mid, z_star], n_per_seg=8)
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            color="#9c27b0",
            lw=2.8,
            ls=(0, (1.2, 2.2)),
            solid_capstyle="round",
            zorder=6,
            alpha=0.85,
        )

    ax.scatter(*replan, s=110, facecolor="#4a148c", edgecolor="white", linewidth=2.0, zorder=12)

    state_kw = dict(s=400, facecolor="white", linewidth=2.8, zorder=11)
    ax.scatter(*s_t, edgecolor="#111111", **state_kw)
    if data_subgoal is not None:
        ax.scatter(*data_subgoal, edgecolor="#111111", **state_kw)
    ax.scatter(
        *z_star,
        s=440,
        facecolor="white",
        edgecolor="#6a1b9a",
        linewidth=3.4,
        zorder=12,
    )
    ax.scatter(*goal, edgecolor="#111111", **state_kw)

    ax.set_xlim(env.cfg.box_low, env.cfg.box_high)
    ax.set_ylim(env.cfg.box_low, env.cfg.box_high)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(2.0)

    fig.savefig(out_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def print_checks(env, greedy, prefix, remainder, data_traj, z_star):
    haz = env.hazard_center
    r = env.hazard_radius
    g_min = np.linalg.norm(greedy - haz, axis=1).min()
    b_min = np.linalg.norm(np.vstack([prefix, remainder]) - haz, axis=1).min()
    d_min = np.linalg.norm(data_traj - z_star, axis=1).min()
    pref_len = float(np.linalg.norm(np.diff(prefix, axis=0), axis=1).sum())
    rem_len = float(np.linalg.norm(np.diff(remainder, axis=0), axis=1).sum())
    print(f"Greedy enters hazard: {g_min < r}  (dist={g_min:.3f}, r={r})")
    print(f"Bridge enters hazard: {b_min < r}  (dist={b_min:.3f})")
    print(f"Data traj min dist to z*: {d_min:.3f}")
    print(f"Prefix/remainder arc length: {pref_len:.3f} / {rem_len:.3f}")
