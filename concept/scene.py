"""Shared illustrative scene: env, hand value field, crossing geometry."""

from __future__ import annotations

import numpy as np

from envs import ToyEnvConfig, ToyHazardEnv, l2, obstacle_avoiding_goal_action
from fig_shared import chord_hits_hazard

GAMMA = 0.97
K = 28
H_A_FRAC = 0.48


def make_env(seed: int = 0) -> ToyHazardEnv:
    np.random.seed(seed)
    cfg = ToyEnvConfig(
        box_low=-1.0,
        box_high=1.0,
        hazard_center=(0.02, -0.02),
        hazard_radius=0.11,
        goal_radius=0.08,
        step_size=0.045,
        max_episode_steps=110,
        action_noise_std=0.010,
        repulsion_scale=0.34,
        repulsion_margin=0.36,
    )
    return ToyHazardEnv(cfg)


def _gauss(x, y, c, sx, sy, amp=1.0):
    return amp * np.exp(-0.5 * (((x - c[0]) / sx) ** 2 + ((y - c[1]) / sy) ** 2))


def value_to_goal(x, y, goal):
    """Hand-shaped V(s,g): global max at Goal; hazard NOT encoded."""
    g = np.asarray(goal, dtype=float)
    v = _gauss(x, y, g, 0.28, 0.24, amp=1.55)
    v += _gauss(x, y, g, 0.55, 0.48, amp=0.38)
    v += _gauss(x, y, np.array([0.42, 0.18]), 0.30, 0.26, amp=0.48)
    v += _gauss(x, y, np.array([-0.20, 0.40]), 0.28, 0.24, amp=0.22)
    v -= _gauss(x, y, np.array([-0.55, -0.40]), 0.36, 0.30, amp=0.18)
    v -= _gauss(x, y, np.array([0.35, -0.55]), 0.30, 0.26, amp=0.14)
    envelope = _gauss(x, y, np.array([0.05, 0.05]), 0.95, 0.85, amp=1.0)
    v += 0.04 * envelope * np.sin(3.0 * x) * np.cos(2.5 * y)
    v += _gauss(x, y, np.array([0.02, -0.02]), 0.16, 0.14, amp=0.04)
    return 1.0 / (1.0 + np.exp(-2.1 * (v - 0.48)))


def value_between(a, b):
    d = l2(np.asarray(a), np.asarray(b)) / 0.05
    return GAMMA**d


def _rollout_to(env, start, target, prefer_side="north", max_steps=80):
    s, g = env.reset(np.asarray(start, np.float32), np.asarray(target, np.float32))
    states = [s.copy()]
    for _ in range(max_steps):
        a = obstacle_avoiding_goal_action(env, s, g, prefer_side=prefer_side)
        ns, _, done, _ = env.step(a)
        states.append(ns.copy())
        s = ns
        if l2(s, target) <= env.goal_radius * 1.35:
            break
        if done and l2(s, target) <= env.goal_radius * 1.8:
            break
    return np.asarray(states, dtype=float)


def build_crossing_scene(env):
    """
    ref.png geometry:
      s_t bottom-left → northern via above hazard → Goal bottom-right.
    """
    s_t = np.array([-0.78, -0.55], dtype=float)
    via = np.array([-0.05, 0.52], dtype=float)
    goal = np.array([0.82, -0.35], dtype=float)
    leg1 = _rollout_to(env, s_t, via, prefer_side="north")
    leg2 = _rollout_to(env, leg1[-1], goal, prefer_side="north")
    return s_t, goal, np.vstack([leg1, leg2[1:]])


def place_subgoals(data_traj, env, s_t, goal):
    """
    data_subgoal ≈ northern apex (s_{t+K}).
    z* further along support with s_t→z* chord crossing the hazard.
    """
    i_apex = int(np.clip(np.argmax(data_traj[:, 1]), 4, len(data_traj) - 6))
    data_subgoal = data_traj[i_apex].copy()

    cands = []
    for i in range(i_apex + 3, min(len(data_traj) - 1, i_apex + 28)):
        base = data_traj[i]
        cands.append(base)
        cands.append(base + np.array([0.05, 0.02]))
        cands.append(base + np.array([0.08, -0.04]))
    cands.append(data_traj[min(len(data_traj) - 2, i_apex + 12)] + np.array([0.04, 0.06]))
    cands = np.asarray(cands, dtype=float)

    scored = []
    for z in cands:
        if not chord_hits_hazard(s_t, z, env):
            continue
        d_data = float(np.linalg.norm(data_traj - z[None, :], axis=1).min())
        if d_data > 0.18:
            continue
        s_tr = value_between(s_t, z) * value_to_goal(*z, goal)
        scored.append((s_tr - 0.5 * d_data, z))

    if scored:
        scored.sort(key=lambda t: t[0], reverse=True)
        z_star = scored[0][1]
    else:
        z_star = data_traj[min(len(data_traj) - 2, i_apex + 14)].copy()
        if not chord_hits_hazard(s_t, z_star, env):
            z_star = z_star + np.array([0.0, -0.08])
        if not chord_hits_hazard(s_t, z_star, env):
            z_star = np.array([0.45, 0.12])

    return data_subgoal, np.asarray(z_star, dtype=float)
