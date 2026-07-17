import numpy as np
from dataclasses import dataclass


# ============================================================
# Utility functions
# ============================================================

def l2(x, y):
    return np.linalg.norm(x - y)

def unit(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def clip_to_box(x, low, high):
    return np.clip(x, low, high)

def linearly_interpolate(a, b, n):
    xs = []
    for t in np.linspace(0.0, 1.0, n):
        xs.append((1 - t) * a + t * b)
    return np.array(xs)


# ============================================================
# Toy environment
# ============================================================

@dataclass
class ToyEnvConfig:
    box_low: float = -1.0
    box_high: float = 1.0

    hazard_center: tuple = (0.10, -0.05)
    hazard_radius: float = 0.18

    goal_radius: float = 0.08
    step_size: float = 0.045
    max_episode_steps: int = 90

    # trajectory generation
    action_noise_std: float = 0.015
    repulsion_scale: float = 0.22
    repulsion_margin: float = 0.28


class ToyHazardEnv:
    def __init__(self, cfg: ToyEnvConfig):
        self.cfg = cfg
        self.low = np.array([cfg.box_low, cfg.box_low], dtype=np.float32)
        self.high = np.array([cfg.box_high, cfg.box_high], dtype=np.float32)
        self.hazard_center = np.array(cfg.hazard_center, dtype=np.float32)
        self.hazard_radius = cfg.hazard_radius
        self.goal_radius = cfg.goal_radius
        self.step_size = cfg.step_size
        self.max_episode_steps = cfg.max_episode_steps

        self.state = None
        self.goal = None
        self.t = 0

    # ----------------------------
    # geometry
    # ----------------------------
    def in_box(self, x):
        return np.all(x >= self.low) and np.all(x <= self.high)

    def in_hazard(self, x):
        return l2(x, self.hazard_center) <= self.hazard_radius

    def near_hazard(self, x, margin=None):
        if margin is None:
            margin = self.cfg.repulsion_margin
        return l2(x, self.hazard_center) <= (self.hazard_radius + margin)

    def reached_goal(self, x, goal):
        return l2(x, goal) <= self.goal_radius

    def sample_safe_point(self, margin=0.03):
        """Sample a point inside the box and outside the hazard."""
        while True:
            x = np.random.uniform(self.low, self.high)
            if l2(x, self.hazard_center) > (self.hazard_radius + margin):
                return x.astype(np.float32)

    # ----------------------------
    # env api
    # ----------------------------
    def reset(self, state=None, goal=None):
        if state is None:
            state = self.sample_safe_point()
        if goal is None:
            goal = self.sample_safe_point()

        self.state = np.array(state, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.t = 0
        return self.state.copy(), self.goal.copy()

    def step(self, action):
        """
        action: 2D vector, interpreted as desired displacement direction.
        """
        a = np.array(action, dtype=np.float32)
        a = unit(a) * min(np.linalg.norm(a), self.step_size)

        next_state = self.state + a
        next_state = clip_to_box(next_state, self.low, self.high)

        # If next state enters hazard, project back to hazard boundary slightly outside.
        if self.in_hazard(next_state):
            vec = next_state - self.hazard_center
            if np.linalg.norm(vec) < 1e-8:
                vec = np.array([1.0, 0.0], dtype=np.float32)
            vec = unit(vec)
            next_state = self.hazard_center + vec * (self.hazard_radius + 1e-3)

        self.state = next_state
        self.t += 1

        done = self.reached_goal(self.state, self.goal) or (self.t >= self.max_episode_steps)
        reward = 1.0 if self.reached_goal(self.state, self.goal) else 0.0

        info = {}
        return self.state.copy(), reward, done, info


# ============================================================
# Trajectory policies for dataset generation
# ============================================================

def obstacle_avoiding_goal_action(env: ToyHazardEnv, s, g, prefer_side: str | None = "north"):
    """
    Goal-directed action with repulsion from the hazard.

    prefer_side:
      "north" / "south" — bias the tangential detour (matches ref.png northern arc)
      None — pick whichever candidate stays farther from the hazard
    """
    cfg = env.cfg
    toward_goal = unit(g - s)

    vec_from_hazard = s - env.hazard_center
    dist = np.linalg.norm(vec_from_hazard)
    repulse = np.zeros(2, dtype=np.float32)

    safe_band = env.hazard_radius + cfg.repulsion_margin
    if dist < safe_band:
        strength = cfg.repulsion_scale * (safe_band - dist) / max(cfg.repulsion_margin, 1e-6)
        repulse = strength * unit(vec_from_hazard)

    # Tangential bias for going around the hazard (CCW vs CW relative to goal).
    tangent = np.array([-toward_goal[1], toward_goal[0]], dtype=np.float32)
    # "north" ≈ positive world-y component of the tangent kick.
    if prefer_side == "north":
        if tangent[1] < 0:
            tangent = -tangent
        tang_w = 0.32
    elif prefer_side == "south":
        if tangent[1] > 0:
            tangent = -tangent
        tang_w = 0.32
    else:
        tang_w = 0.18

    cand1 = toward_goal + repulse + tang_w * tangent
    cand2 = toward_goal + repulse - tang_w * tangent

    nxt1 = s + env.step_size * unit(cand1)
    nxt2 = s + env.step_size * unit(cand2)
    d1 = l2(nxt1, env.hazard_center)
    d2 = l2(nxt2, env.hazard_center)

    if prefer_side in ("north", "south"):
        # Keep preferred side unless it collapses into the hazard disk.
        action_dir = cand1 if d1 >= env.hazard_radius + 0.02 else cand2
        if d1 < env.hazard_radius + 0.02 and d2 < env.hazard_radius + 0.02:
            action_dir = cand1 if d1 >= d2 else cand2
    else:
        action_dir = cand1 if d1 >= d2 else cand2

    action_dir = action_dir + np.random.randn(2) * cfg.action_noise_std
    return unit(action_dir) * env.step_size


def exploratory_action(env: ToyHazardEnv, s, heading=None):
    """
    Non-goal-reaching exploratory / local motion that still avoids hazard.
    """
    cfg = env.cfg

    if heading is None:
        heading = unit(np.random.randn(2))
    else:
        heading = unit(0.85 * heading + 0.15 * np.random.randn(2))

    vec_from_hazard = s - env.hazard_center
    dist = np.linalg.norm(vec_from_hazard)
    repulse = np.zeros(2, dtype=np.float32)

    safe_band = env.hazard_radius + cfg.repulsion_margin
    if dist < safe_band:
        strength = cfg.repulsion_scale * (safe_band - dist) / max(cfg.repulsion_margin, 1e-6)
        repulse = 1.2 * strength * unit(vec_from_hazard)

    action = heading + repulse + np.random.randn(2) * cfg.action_noise_std
    return unit(action) * env.step_size, unit(action)


# ============================================================
# Dataset generation
# ============================================================

def generate_goal_trajectory(
    env: ToyHazardEnv,
    min_start_goal_dist=0.8,
    prefer_side: str | None = "north",
    start=None,
    goal=None,
):
    """
    Goal-reaching trajectory that usually avoids the hazard (northern bias by default).
    """
    if start is None or goal is None:
        while True:
            s0 = env.sample_safe_point() if start is None else np.asarray(start, dtype=np.float32)
            g = env.sample_safe_point() if goal is None else np.asarray(goal, dtype=np.float32)
            if l2(s0, g) > min_start_goal_dist:
                break
    else:
        s0 = np.asarray(start, dtype=np.float32)
        g = np.asarray(goal, dtype=np.float32)

    s, g = env.reset(s0, g)

    states = [s.copy()]
    actions = []
    next_states = []
    rewards = []
    dones = []

    for _ in range(env.max_episode_steps):
        a = obstacle_avoiding_goal_action(env, s, g, prefer_side=prefer_side)
        ns, r, done, _ = env.step(a)

        states.append(ns.copy())
        actions.append(a.copy())
        next_states.append(ns.copy())
        rewards.append(r)
        dones.append(done)

        s = ns
        if done:
            break

    traj = {
        "mode": "goal",
        "start": s0.copy(),
        "goal": g.copy(),
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
    }
    return traj


def generate_partial_trajectory(env: ToyHazardEnv, prefer_side: str | None = "north"):
    """Truncated goal-reaching demo (offline support for mid-horizon windows)."""
    traj = generate_goal_trajectory(env, prefer_side=prefer_side)
    states = traj["states"]
    if len(states) < 12:
        traj["mode"] = "partial"
        return traj
    keep = int(np.random.randint(max(8, len(states) // 3), max(9, 3 * len(states) // 4)))
    keep = min(keep, len(states))
    traj = dict(traj)
    traj["mode"] = "partial"
    traj["states"] = states[:keep]
    n_act = keep - 1
    traj["actions"] = traj["actions"][:n_act]
    traj["next_states"] = traj["next_states"][:n_act]
    traj["rewards"] = traj["rewards"][:n_act]
    traj["dones"] = traj["dones"][:n_act]
    if n_act > 0:
        traj["dones"][-1] = 1.0
        traj["rewards"][:] = 0.0
    return traj


def generate_exploration_trajectory(env: ToyHazardEnv, horizon_range=(25, 60)):
    """
    Not-necessarily-goal-reaching trajectory.
    We still assign some dummy goal because goal-conditioned datasets
    often store a goal, but the trajectory itself need not reach it.
    """
    s0 = env.sample_safe_point()
    dummy_goal = env.sample_safe_point()
    s, _ = env.reset(s0, dummy_goal)

    H = np.random.randint(horizon_range[0], horizon_range[1] + 1)

    states = [s.copy()]
    actions = []
    next_states = []
    rewards = []
    dones = []

    heading = unit(np.random.randn(2))

    for t in range(H):
        a, heading = exploratory_action(env, s, heading)
        ns, r, _, _ = env.step(a)

        done = (t == H - 1)

        states.append(ns.copy())
        actions.append(a.copy())
        next_states.append(ns.copy())
        rewards.append(0.0)  # exploration trajectory는 reward 0으로 두어도 됨
        dones.append(done)

        s = ns

    traj = {
        "mode": "explore",
        "start": s0.copy(),
        "goal": dummy_goal.copy(),
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
    }
    return traj


def generate_dataset(
    env: ToyHazardEnv,
    num_goal_trajs=120,
    num_explore_trajs=80,
    num_partial_trajs=60,
    prefer_side: str | None = "north",
    seed: int | None = None,
):
    """
    Offline mix:
      goal (~full reaches) + partial (truncated) + explore (roaming).
    Prefer a consistent northern detour so concept figures match ref.png.
    """
    if seed is not None:
        np.random.seed(seed)

    trajectories = []

    # Seed a few left→right corridor demos so K-step windows near the figure scene exist.
    corridor = [
        (np.array([-0.80, -0.55], np.float32), np.array([0.82, -0.35], np.float32)),
        (np.array([-0.75, -0.45], np.float32), np.array([0.78, -0.25], np.float32)),
        (np.array([-0.85, -0.65], np.float32), np.array([0.85, -0.40], np.float32)),
        (np.array([-0.70, -0.30], np.float32), np.array([0.75, -0.15], np.float32)),
    ]
    for s0, g in corridor:
        trajectories.append(
            generate_goal_trajectory(env, start=s0, goal=g, prefer_side=prefer_side, min_start_goal_dist=0.5)
        )

    for _ in range(num_goal_trajs):
        side = prefer_side if np.random.rand() < 0.75 else None
        trajectories.append(generate_goal_trajectory(env, prefer_side=side))

    for _ in range(num_partial_trajs):
        trajectories.append(generate_partial_trajectory(env, prefer_side=prefer_side))

    for _ in range(num_explore_trajs):
        trajectories.append(generate_exploration_trajectory(env))

    transitions = []
    for traj_idx, traj in enumerate(trajectories):
        T = len(traj["actions"])
        for t in range(T):
            transitions.append({
                "traj_idx": traj_idx,
                "mode": traj["mode"],
                "t": t,
                "s": traj["states"][t],
                "a": traj["actions"][t],
                "s_next": traj["next_states"][t],
                "goal": traj["goal"],
                "reward": traj["rewards"][t],
                "done": traj["dones"][t],
            })

    return trajectories, transitions


# ============================================================
# Visualization
# ============================================================

def draw_env(ax, env: ToyHazardEnv):
    import matplotlib.pyplot as plt

    # box
    ax.set_xlim(env.cfg.box_low, env.cfg.box_high)
    ax.set_ylim(env.cfg.box_low, env.cfg.box_high)
    ax.set_aspect("equal")

    # hazard
    hazard = plt.Circle(
        env.hazard_center,
        env.hazard_radius,
        color="red",
        alpha=0.35,
        ec="firebrick",
        lw=2,
    )
    ax.add_patch(hazard)

    ax.text(
        env.hazard_center[0],
        env.hazard_center[1] - env.hazard_radius - 0.05,
        "Hazard",
        color="firebrick",
        ha="center",
        va="top",
        fontsize=11,
    )

    ax.set_xticks([])
    ax.set_yticks([])


def plot_dataset(env: ToyHazardEnv, trajectories, n_show=80):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_env(ax, env)

    idxs = np.random.choice(len(trajectories), size=min(n_show, len(trajectories)), replace=False)

    for i in idxs:
        traj = trajectories[i]
        states = traj["states"]

        if traj["mode"] == "goal":
            color = "black"
            alpha = 0.45
            lw = 1.6
        else:
            color = "gray"
            alpha = 0.28
            lw = 1.2

        ax.plot(states[:, 0], states[:, 1], color=color, alpha=alpha, lw=lw)

    ax.set_title("Toy dataset trajectories (hazard-avoiding)")
    plt.show()


def plot_examples(env: ToyHazardEnv, trajectories, n_goal=4, n_explore=4):
    import matplotlib.pyplot as plt

    goal_trajs = [tr for tr in trajectories if tr["mode"] == "goal"]
    exp_trajs = [tr for tr in trajectories if tr["mode"] == "explore"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.reshape(2, 4)

    for ax in axes.flat:
        draw_env(ax, env)

    for i in range(min(n_goal, len(goal_trajs))):
        traj = goal_trajs[i]
        states = traj["states"]
        goal = traj["goal"]
        start = traj["start"]

        axes[0, i].plot(states[:, 0], states[:, 1], color="black", lw=2)
        axes[0, i].scatter(start[0], start[1], c="blue", s=40, label="start")
        axes[0, i].scatter(goal[0], goal[1], c="gold", edgecolors="black", s=60, label="goal")
        axes[0, i].set_title("goal-reaching")

    for i in range(min(n_explore, len(exp_trajs))):
        traj = exp_trajs[i]
        states = traj["states"]
        start = traj["start"]

        axes[1, i].plot(states[:, 0], states[:, 1], color="gray", lw=2)
        axes[1, i].scatter(start[0], start[1], c="blue", s=40)
        axes[1, i].set_title("exploration")

    plt.tight_layout()
    plt.show()


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    cfg = ToyEnvConfig(
        box_low=-1.0,
        box_high=1.0,
        hazard_center=(0.08, -0.02),
        hazard_radius=0.17,      # 너무 크지 않게
        goal_radius=0.08,
        step_size=0.045,
        max_episode_steps=90,
        action_noise_std=0.015,
        repulsion_scale=0.22,
        repulsion_margin=0.30,
    )

    env = ToyHazardEnv(cfg)

    trajectories, transitions = generate_dataset(
        env,
        num_goal_trajs=140,
        num_explore_trajs=100,
    )

    print(f"# trajectories: {len(trajectories)}")
    print(f"# transitions : {len(transitions)}")

    # dataset visualization
    plot_dataset(env, trajectories, n_show=120)
    plot_examples(env, trajectories, n_goal=4, n_explore=4)