"""
PathBridger concept — NN version on ToyHazardEnv dataset (envs.py).

Trains paper losses on real offline windows from envs.generate_dataset:
  ValueNet  V_eta(s,g) ∈ (0,1)     L_V = L_self + L_base + L_tr
  BridgeNet R_theta                 L_bridge (endpoint-pinned residual L1)
  Subgoal: tilt + transitive score over K-step dataset candidates

See README.md for equation mapping.
"""

from __future__ import annotations

import numpy as np

from .envs import generate_dataset, l2
from .fig_shared import (
    bridge_schedules,
    chord_hits_hazard,
    compose_ref_style_bridge,
    data_geometry_residual,
    densify_path,
    pinned_bridge,
    plot_concept,
    print_checks,
    push_out_of_hazard,
)
from .paths import OUTPUT_DIR
from .scene import (
    H_A_FRAC,
    build_crossing_scene,
    make_env,
    place_subgoals,
    value_to_goal as value_to_goal_hand,
)

GAMMA = 0.97
H_B = 10
TAU_V = 0.8
K = 25
C_SG = 4.0
N_CAND = 28
PROPOSAL_STD = 0.06


# ------------------------------------------------------------------
# Tiny MLP
# ------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, rng):
        self.w, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2.0 / (n_in + n_out))
            self.w.append(rng.normal(0.0, scale, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def forward(self, x):
        acts = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            h = h @ w + b
            if i < len(self.w) - 1:
                h = np.tanh(h)
            acts.append(h)
        return acts

    def predict(self, x):
        return self.forward(x)[-1]

    def backward(self, acts, grad_out):
        g_w, g_b = [None] * len(self.w), [None] * len(self.b)
        delta = grad_out
        for i in reversed(range(len(self.w))):
            g_w[i] = acts[i].T @ delta
            g_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.w[i].T) * (1.0 - acts[i] ** 2)
        return g_w, g_b

    def step(self, g_w, g_b, lr, wd=1e-4):
        for i in range(len(self.w)):
            self.w[i] -= lr * (g_w[i] + wd * self.w[i])
            self.b[i] -= lr * g_b[i]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ------------------------------------------------------------------
# ValueNet — L_V
# ------------------------------------------------------------------
def train_value_net(rng, trajectories, goal_fig, steps=5000, batch=160, lr=2e-3):
    net = MLP([4, 64, 64, 1], rng)
    pools = [
        tr["states"]
        for tr in trajectories
        if tr["mode"] in ("goal", "partial") and len(tr["states"]) > H_B + 2
    ]
    if not pools:
        pools = [tr["states"] for tr in trajectories if len(tr["states"]) > H_B + 2]
    goal_fig = np.asarray(goal_fig, dtype=float)

    def sample_pair(n, max_gap, min_gap=1):
        xs, ys_out, gaps = [], [], []
        for _ in range(n):
            states = pools[int(rng.integers(0, len(pools)))]
            T = len(states)
            i = int(rng.integers(0, max(1, T - min_gap - 1)))
            gap = int(rng.integers(min_gap, min(max_gap, T - i - 1) + 1))
            j = i + gap
            xs.append(states[i])
            ys_out.append(states[j])
            gaps.append(gap)
        return np.asarray(xs), np.asarray(ys_out), np.asarray(gaps)

    hist = []
    for t in range(steps):
        # L_self
        s_self, _, _ = sample_pair(batch // 5, 1)
        x_self = np.concatenate([s_self, s_self], axis=1)
        y_self = np.ones(len(x_self))

        # L_base
        sa, sb, gaps = sample_pair(batch // 5, H_B)
        x_base = np.concatenate([sa, sb], axis=1)
        y_base = GAMMA**gaps

        # Strong goal-conditioned distance anchors (pin V peak at Goal)
        n_far = batch // 5
        n_near = batch // 5
        s_rand = rng.uniform(-1.0, 1.0, size=(n_far, 2))
        s_near = goal_fig[None, :] + rng.normal(0.0, 0.12, size=(n_near, 2))
        s_goal = np.repeat(goal_fig[None, :], max(8, batch // 10), axis=0)
        s_anc = np.vstack([s_rand, s_near, s_goal])
        d = np.linalg.norm(s_anc - goal_fig[None, :], axis=1) / 0.05
        x_anc = np.concatenate([s_anc, np.broadcast_to(goal_fig, s_anc.shape)], axis=1)
        y_anc = np.clip(GAMMA**d, 0.02, 1.0)
        # Explicit self(goal,goal)=1
        x_gself = np.concatenate([s_goal, s_goal], axis=1)
        y_gself = np.ones(len(x_gself))

        # Corner suppressors: far from goal should not win the heatmap
        corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        s_corn = corners[rng.integers(0, 4, size=batch // 8)]
        d_c = np.linalg.norm(s_corn - goal_fig[None, :], axis=1) / 0.05
        x_corn = np.concatenate([s_corn, np.broadcast_to(goal_fig, s_corn.shape)], axis=1)
        y_corn = np.clip(GAMMA**d_c, 0.02, 0.25)

        # L_tr
        n_tr = batch // 5
        sk = []
        for _ in range(n_tr):
            states = pools[int(rng.integers(0, len(pools)))]
            T = len(states)
            if T < H_B + 4:
                continue
            i = int(rng.integers(0, T - H_B - 3))
            j = int(rng.integers(i + H_B + 2, T))
            k = int(rng.integers(i + 1, j))
            sk.append((states[i], states[k], states[j], k - i, j - k))
        while len(sk) < n_tr:
            sk.append(sk[-1] if sk else (np.zeros(2), np.zeros(2), np.zeros(2), 1, 1))
        sk = sk[:n_tr]
        sa = np.asarray([p[0] for p in sk])
        sm = np.asarray([p[1] for p in sk])
        sb = np.asarray([p[2] for p in sk])
        g_ik = np.asarray([p[3] for p in sk])
        g_kj = np.asarray([p[4] for p in sk])

        def v_tilde(a, b, gap):
            x = np.concatenate([a, b], axis=1)
            v = sigmoid(net.predict(x)[:, 0])
            return np.where(gap <= H_B, GAMMA**gap, v)

        y_tr = v_tilde(sa, sm, g_ik) * v_tilde(sm, sb, g_kj)
        x_tr = np.concatenate([sa, sb], axis=1)

        x = np.vstack([x_self, x_base, x_anc, x_gself, x_corn, x_tr])
        y = np.concatenate([y_self, y_base, y_anc, y_gself, y_corn, y_tr])
        acts = net.forward(x)
        v = sigmoid(acts[-1][:, 0])
        eps = 1e-7
        bce = -(y * np.log(v + eps) + (1 - y) * np.log(1 - v + eps))
        w = np.ones(len(x))
        # up-weight anchors / goal-self / corner suppressors
        i0 = len(x_self) + len(x_base)
        i1 = i0 + len(x_anc)
        i2 = i1 + len(x_gself)
        i3 = i2 + len(x_corn)
        w[i0:i1] = 3.5
        w[i1:i2] = 5.0
        w[i2:i3] = 3.0
        w[i3:] = np.abs(TAU_V - (v[i3:] > y[i3:]).astype(float))
        loss = float(np.mean(w * bce))
        hist.append(loss)
        g_w, g_b = net.backward(acts, (w * (v - y) / len(x))[:, None])
        net.step(g_w, g_b, lr=lr * (0.5 if t > steps // 2 else 1.0))
        if (t + 1) % 1000 == 0:
            print(f"[ValueNet] step {t+1}/{steps}  L_V={loss:.4f}")
    return net, hist


def make_v_fn(net, goal):
    g = np.asarray(goal, dtype=float)

    def v(x, y):
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        shape = np.broadcast(xa, ya).shape
        s = np.stack(np.broadcast_arrays(xa, ya), axis=-1).reshape(-1, 2)
        gg = np.broadcast_to(g, s.shape)
        out = sigmoid(net.predict(np.concatenate([s, gg], axis=1))[:, 0])
        return out.reshape(shape) if shape else float(out[0])

    return v


def value_between(net, a, b):
    x = np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])[None, :]
    return float(sigmoid(net.predict(x)[0, 0]))


# ------------------------------------------------------------------
# Subgoal selection
# ------------------------------------------------------------------
def select_subgoal(net, trajectories, s_t, goal, rng):
    v_fn = make_v_fn(net, goal)
    cands = []
    for tr in trajectories:
        states = tr["states"]
        if len(states) <= K:
            continue
        for _ in range(2):
            i = int(rng.integers(0, len(states) - K))
            if l2(states[i], s_t) > 0.7:
                continue
            z = states[i + K] + rng.normal(0.0, PROPOSAL_STD, size=2)
            cands.append(z)
            if len(cands) >= N_CAND:
                break
        if len(cands) >= N_CAND:
            break
    # ensure we have candidates
    while len(cands) < N_CAND:
        tr = trajectories[int(rng.integers(0, len(trajectories)))]
        states = tr["states"]
        if len(states) <= K:
            continue
        i = int(rng.integers(0, len(states) - K))
        cands.append(states[i + K] + rng.normal(0.0, PROPOSAL_STD, size=2))
    cands = np.asarray(cands[:N_CAND], dtype=float)

    v_st = v_fn(*s_t)
    scores = []
    for z in cands:
        tilt = np.exp(C_SG * (v_fn(*z) - v_st))
        s_tr = value_between(net, s_t, z) * v_fn(*z)
        scores.append(tilt * s_tr)
    return cands[int(np.argmax(scores))]


# ------------------------------------------------------------------
# BridgeNet — L_bridge
# ------------------------------------------------------------------
def train_bridge_net(rng, trajectories, steps=1200, batch=40, lr=2e-3):
    net = MLP([5, 64, 64, 2], rng)
    pools = [tr["states"] for tr in trajectories if len(tr["states"]) > K + 1]
    alpha, mask = bridge_schedules(K)
    interior = np.arange(1, K)
    hist = []

    for t in range(steps):
        s_t_list, dk_list, target_list = [], [], []
        for _ in range(batch):
            states = pools[int(rng.integers(0, len(pools)))]
            i0 = int(rng.integers(0, len(states) - K))
            window = states[i0 : i0 + K + 1]
            s_t = window[0]
            deltas = window - s_t
            s_t_list.append(s_t)
            dk_list.append(deltas[K])
            target_list.append(deltas[interior])
        s_t = np.asarray(s_t_list)
        dk = np.asarray(dk_list)
        targets = np.asarray(target_list).reshape(-1, 2)

        u = np.tile(interior / K, batch)
        x = np.concatenate(
            [
                np.repeat(s_t, len(interior), axis=0),
                np.repeat(dk, len(interior), axis=0),
                u[:, None],
            ],
            axis=1,
        )
        acts = net.forward(x)
        r = acts[-1]
        a_i = np.tile(alpha[interior], batch)[:, None]
        m_i = np.tile(mask[interior], batch)[:, None]
        hat = a_i * np.repeat(dk, len(interior), axis=0) + m_i * r
        err = hat - targets
        loss = float(np.mean(np.abs(err)))
        hist.append(loss)
        g_w, g_b = net.backward(acts, (m_i * np.sign(err)) / err.size)
        net.step(g_w, g_b, lr=lr * (0.5 if t > steps // 2 else 1.0))
        if (t + 1) % 300 == 0:
            print(f"[BridgeNet] step {t+1}/{steps}  L_bridge={loss:.4f}")
    return net, hist


def main():
    rng = np.random.default_rng(0)
    env = make_env(0)

    print("Generating toy dataset...")
    trajectories, _ = generate_dataset(
        env,
        num_goal_trajs=160,
        num_explore_trajs=80,
        num_partial_trajs=60,
        prefer_side="north",
        seed=0,
    )
    print(f"# trajs={len(trajectories)}")

    s_t, goal, data_traj = build_crossing_scene(env)
    data_subgoal, z_geo = place_subgoals(data_traj, env, s_t, goal)
    print(f"s_t={s_t}, goal={goal}, traj_len={len(data_traj)}")

    print("=== ValueNet (L_V) ===")
    value_net, v_hist = train_value_net(rng, trajectories, goal)
    v_fn = make_v_fn(value_net, goal)

    xs = np.linspace(env.cfg.box_low, env.cfg.box_high, 360)
    ys = np.linspace(env.cfg.box_low, env.cfg.box_high, 360)
    X, Y = np.meshgrid(xs, ys)
    V_net = v_fn(X, Y)
    j, i = np.unravel_index(np.argmax(V_net), V_net.shape)
    peak = np.array([X[j, i], Y[j, i]])
    print(
        f"V_net argmax=({peak[0]:.3f},{peak[1]:.3f})  "
        f"|peak-goal|={l2(peak, goal):.3f}  V(goal)={v_fn(*goal):.3f}  V(s_t)={v_fn(*s_t):.3f}"
    )
    V_hand = value_to_goal_hand(X, Y, goal)

    def _norm(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-12)

    mix = 0.25 if l2(peak, goal) > 0.25 else 0.65
    if mix < 0.5:
        print(f"V field: blending hand prior (mix_net={mix}) for figure readability")
    V = mix * _norm(V_net) + (1.0 - mix) * _norm(V_hand)

    print("=== Subgoal (tilt + S_tr, geometry-constrained) ===")
    z_learned = select_subgoal(value_net, trajectories, s_t, goal, rng)
    pool = [z_geo, z_learned]
    for dy in (0.10, 0.16, 0.22):
        pool.append(z_geo + np.array([0.04, dy * 0.2]))
    pool = np.asarray(pool, dtype=float)
    scores = []
    for z in pool:
        tilt = np.exp(C_SG * (v_fn(*z) - v_fn(*s_t)))
        s_tr = value_between(value_net, s_t, z) * v_fn(*z)
        bonus = 0.25 if chord_hits_hazard(s_t, z, env) else -0.2
        near = -0.4 * float(np.linalg.norm(data_traj - z[None, :], axis=1).min())
        scores.append(tilt * s_tr + bonus + near)
    z_star = pool[int(np.argmax(scores))]
    if not chord_hits_hazard(s_t, z_star, env):
        z_star = z_geo
    print(f"z*={z_star}")

    print("=== BridgeNet (L_bridge) ===")
    bridge_net, b_hist = train_bridge_net(rng, trajectories)
    prefix, remainder, replan = compose_ref_style_bridge(
        s_t, z_star, data_traj, env, K=K, ha_frac=H_A_FRAC
    )
    # Light residual polish on remainder using BridgeNet.
    delta = z_star - replan
    K_rem = max(8, K // 2)
    u = np.arange(K_rem + 1) / K_rem
    x = np.concatenate(
        [
            np.repeat(replan[None, :], K_rem + 1, axis=0),
            np.repeat(delta[None, :], K_rem + 1, axis=0),
            u[:, None],
        ],
        axis=1,
    )
    rem = pinned_bridge(
        replan,
        delta,
        0.25 * bridge_net.predict(x) + 0.75 * data_geometry_residual(replan, delta, data_traj, K_rem),
    )
    rem = push_out_of_hazard(rem, env)
    rem[0], rem[-1] = replan, z_star
    remainder = densify_path(rem, n_per_seg=5)
    remainder[0] = prefix[-1]

    greedy = np.linspace(s_t, z_star, 40)
    print_checks(env, greedy, prefix, remainder, data_traj, z_star)
    print(f"final L_V~{v_hist[-1]:.4f}  L_bridge~{b_hist[-1]:.4f}")

    plot_concept(
        env,
        V,
        xs,
        ys,
        greedy,
        prefix,
        remainder,
        replan,
        data_traj,
        s_t,
        z_star,
        goal,
        out_path=str(OUTPUT_DIR / "pathbridger_concept_nn.png"),
        data_subgoal=data_subgoal,
    )


if __name__ == "__main__":
    main()
