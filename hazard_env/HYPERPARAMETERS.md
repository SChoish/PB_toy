# Agent hyperparameters

Defaults live in each agent’s `default_config()` under `hazard_env/agents/`.
Training entrypoint: `python -m hazard_env.agents.train --agent {bc,hiql,pbg,pbf}`.

These are **toy reimplementations** inspired by OGBench (GCBC / HIQL) and
Pathbridger_flow (PBG / PBF), not full imports of those codebases.

---

## Shared training / data settings

| Setting | Value | Notes |
|---------|-------|--------|
| Optimizer | Adam | all agents |
| Learning rate `lr` | `3e-4` | |
| Batch size | `256` | |
| MLP width | `(256, 256)` | GELU; see per-agent LN |
| Observation | `[x, y, vx, vy]` | `observation_mode="state"` |
| Action (env) | `(angle ∈ [-π,π], thrust ∈ [0,1])` | |
| Action (network) | `[angle/π, 2·thrust−1] ∈ [-1,1]^2` | clip after sample |
| Dataset | `datasets/hazard2d-navigate.npz` | ~96k transitions |
| Goal relabel | `goal_relabel_prob=0.8` | else commanded / terminal |
| Subgoal horizon `K` | `8` | `subgoal_steps` / `dynamics_N` |
| Path tensor | `(B, K+1, 4)` | true `s_t … s_{t+K}` for PBG/PBF |
| Eval | tasks 1–5, 3 eps each | `max_episode_steps=300` |
| Seed (recent 50k run) | `0` | |
| GPU cap (recent run) | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.25` (~4GB on 16GB) | `PREALLOCATE=false` |

Sparse success for HIQL rewards: `‖next_xy − goal_xy‖ ≤ 0.08` (env `goal_radius`).

---

## BC (goal-conditioned BC / GCBC-style)

**File:** `agents/bc.py`  
**Policy:** `π(a | s, g)` Gaussian; loss `-E[log π(a|s,g)]`.

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `lr` | `3e-4` | Adam |
| `hidden_dims` | `(256, 256)` | actor MLP |
| `const_std` | `True` | `σ = 1` (fixed); NLL ∝ MSE of mean |
| `layer_norm` | `False` | |
| `batch_size` | `256` | |
| Tanh squash | off | raw Gaussian mean, then clip to `[-1,1]` |
| Encoder | none | concat `[s; g]` |

**Inference:** `a = clip(mode(π(·|s,g)))`, then denormalize to env units.

---

## HIQL (simplified hierarchical IQL)

**File:** `agents/hiql.py`  
**Modules:** ensemble value `V₁,V₂`, target `V̄`, low actor `π_lo(a|s,g)`, high actor `π_hi(z|s,g)` predicting a **state-space** subgoal (no separate `goal_rep` like full OGBench HIQL).

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `lr` | `3e-4` | Adam (shared) |
| `hidden_dims` | `(256, 256)` | all nets |
| `batch_size` | `256` | |
| `discount` | `0.99` | TD target `r + γ m V̄(s',g)` |
| `tau` | `0.005` | target soft update: `V̄ ← τ V + (1-τ) V̄` |
| `expectile` | `0.7` | asymmetric MSE weight for IVL |
| `low_alpha` | `3.0` | AWR temperature for low actor |
| `high_alpha` | `3.0` | AWR temperature for high actor |
| Value ensemble | `2` | `min` for next-V, mean for adv |
| Low actor out | `action_dim=2` | `const_std=True` |
| High actor out | `state_dim=4` | predicts subgoal state |

**Target update:** `V̄ ← τ V + (1-τ) V̄` with `τ=0.005` (OGBench-style Polyak).

**Losses:** value expectile + AWR-weighted BC for low/high.  
**Inference:** `z = mode(π_hi(·|s,g))`, then `a = mode(π_lo(·|s,z))`.

---

## PBG (PathBridger-Gaussian lite)

**File:** `agents/pbg.py`, bridge math in `agents/bridge.py`  
**Stack:** deterministic subgoal MLP + closed-form linear-SDE bridge + path residual + IDM  
(no critic / SPI / action chunks from full Pathbridger_flow).

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `lr` | `3e-4` | Adam |
| `hidden_dims` | `(256, 256)` | subgoal / residual / IDM; **LayerNorm=True** |
| `batch_size` | `256` | |
| `dynamics_N` / `subgoal_steps` | `8` | bridge horizon `K` |
| `dynamics_lambda` | `1.0` | linear-SDE diffusion scale `λ` |
| `bridge_gamma_inv` | `0.0` | hard endpoint bridge |
| `theta_total` | `1.0` | cumulative OU rate `Θ_K` |
| `progress_alpha` | `0.8` | prefix-progress `c_i=(i/K)^α` |
| `subgoal_loss_weight` | `1.0` | `‖ẑ(s,g) − s_{t+K}‖²` |
| `path_loss_weight` | `1.0` | interior path MSE + first-step MSE |
| `idm_loss_weight` | `1.0` | `‖IDM(s,s') − a‖²` |

**Bridge (fixed schedule, not learned):**
- Coefficients `(a_i, b_i, std_i)` from prefix-progress linear-SDE (same construction as Pathbridger_flow).
- Mean: `μ_i = a_i s + b_i z`.
- Residual weights: `w_i = i(K−i)/K²` (0 at endpoints).
- Planned path: `path_i = μ_i + w_i · r_θ(s, z, i/K)`, endpoints clamped.

**Inference:** `z = subgoal(s,g)` → plan path → `a = IDM(s, path_1)`.

---

## PBF (PathBridger-Flow lite)

**File:** `agents/pbf.py`  
Same bridge + IDM as PBG; endpoint is a **conditional flow** instead of a point MLP.

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `lr` | `3e-4` | Adam |
| `hidden_dims` | `(256, 256)` | flow / residual / IDM; LayerNorm=True |
| `batch_size` | `256` | |
| `dynamics_N` / `subgoal_steps` | `8` | bridge `K` |
| `flow_steps` | `8` | Euler steps at inference |
| `dynamics_lambda` | `1.0` | same as PBG |
| `bridge_gamma_inv` | `0.0` | hard bridge |
| `theta_total` | `1.0` | |
| `progress_alpha` | `0.8` | |
| `flow_loss_weight` | `1.0` | CFM: `x_u=(1-u)ε+u z*`, `v*=z*-ε` |
| `path_loss_weight` | `1.0` | same path supervision as PBG |
| `idm_loss_weight` | `1.0` | |

**Inference:** integrate flow from noise → `z` (`flow_steps` Euler) → closed-form bridge → `IDM(s, path_1)`.

---

## Recent training recipe (50k)

```bash
export PYTHONPATH=/path/to/toy_examples
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25   # ~4GB on 16GB GPU

python -m hazard_env.agents.train --agent bc   --steps 50000 --eval-every 10000 --log-every 2000 --seed 0
# likewise: hiql | pbg | pbf
```

Log file for the batched run: `hazard_env/runs/train_50k.log`.
