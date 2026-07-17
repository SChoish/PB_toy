"""
PathBridger concept — numerical version (no learning).

Hand-shaped V(s,g) + dataset geometry + endpoint-pinned bridge.
"""

from __future__ import annotations

import shutil

import numpy as np

from envs import generate_dataset, l2
from fig_shared import (
    chord_hits_hazard,
    compose_ref_style_bridge,
    gradient_ascent_path,
    plot_concept,
    print_checks,
)
from paths import OUTPUT_DIR
from scene import (
    H_A_FRAC,
    K,
    build_crossing_scene,
    make_env,
    place_subgoals,
    value_to_goal,
)


def main():
    env = make_env(0)

    print("Generating toy dataset...")
    generate_dataset(
        env,
        num_goal_trajs=140,
        num_explore_trajs=70,
        num_partial_trajs=50,
        prefer_side="north",
        seed=0,
    )
    s_t, goal, data_traj = build_crossing_scene(env)
    data_subgoal, z_star = place_subgoals(data_traj, env, s_t, goal)

    print(f"s_t={s_t}, goal={goal}, traj_len={len(data_traj)}")
    print(f"data_subgoal={data_subgoal}, z*={z_star}")

    xs = np.linspace(env.cfg.box_low, env.cfg.box_high, 420)
    ys = np.linspace(env.cfg.box_low, env.cfg.box_high, 420)
    X, Y = np.meshgrid(xs, ys)
    V = value_to_goal(X, Y, goal)
    j, i = np.unravel_index(np.argmax(V), V.shape)
    print(f"V argmax=({X[j,i]:.3f},{Y[j,i]:.3f}), |peak-goal|={l2([X[j,i],Y[j,i]], goal):.3f}")
    print(f"chord hits hazard: {chord_hits_hazard(s_t, z_star, env)}  V(z*,g)={value_to_goal(*z_star, goal):.3f}")

    prefix, remainder, replan = compose_ref_style_bridge(
        s_t, z_star, data_traj, env, K=K, ha_frac=H_A_FRAC
    )

    def greedy_vg(x, y, v_scale=0.15, pull=3.4, eps=1e-4):
        v = value_to_goal(x, y, goal)
        gx = (value_to_goal(x + eps, y, goal) - value_to_goal(x - eps, y, goal)) / (2 * eps)
        gy = (value_to_goal(x, y + eps, goal) - value_to_goal(x, y - eps, goal)) / (2 * eps)
        dx, dy = x - z_star[0], y - z_star[1]
        return (
            v_scale * v - 0.5 * pull * (dx * dx + dy * dy),
            v_scale * gx - pull * dx,
            v_scale * gy - pull * dy,
        )

    greedy = np.linspace(s_t, z_star, 40)
    if not chord_hits_hazard(s_t, z_star, env):
        greedy = gradient_ascent_path(s_t, z_star, greedy_vg, step_size=env.step_size * 0.75)

    print_checks(env, greedy, prefix, remainder, data_traj, z_star)

    out = OUTPUT_DIR / "pathbridger_concept_numerical.png"
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
        out_path=str(out),
        data_subgoal=data_subgoal,
    )
    shutil.copyfile(out, OUTPUT_DIR / "pathbridger_concept.png")


if __name__ == "__main__":
    main()
