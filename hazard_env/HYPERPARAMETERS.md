# Agent hyperparameters

Defaults live under `hazard_env/agents/` (`bc`, `hiql`, `dynamics`).
Training: `python -m hazard_env.agents.train --agent {bc,hiql,tr_hiql,trl,dqc,pbg,pbf}`.

Layout (mirrors Pathbridger_flow):

```
hazard_env/
  agents/   # bc, hiql, dynamics (PBG/PBF), critic (TRL-lite), train
  utils/    # flax_utils, networks, dynamics (bridge), datasets
```

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
| Dataset | `datasets/{env}_{policy}_{size}.npz` | policy∈{navigate,noisy,random}, size∈{1k,10k,100k} |
| Goal relabel | `goal_relabel_prob=0.8` | else commanded / terminal |
| Subgoal horizon `K` | `8` | `subgoal_steps` / `dynamics_N` |
| Path tensor | `(B, K+1, 4)` | true `s_t … s_{t+K}` for PBG/PBF |
| Goals | full 4D state | commanded/eval: `[gx, gy, 0, 0]` |
| Eval | tasks 1–5 × 5 eps (=25); BC/HIQL `T=0`, PB `T=1` | `max_episode_steps=300` |
| Seed (recent runs) | `0` | |
| GPU cap | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.25` | `PREALLOCATE=false` |

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

## HIQL (OGBench-aligned hierarchical IQL)

**File:** `agents/hiql.py`  
**Modules:** OGBench-style independent value/target goal encoders φ([s;g]),
ensemble value `V(s,φ)`, target `V̄`,
low actor `π_lo(a|s,φ)`, high actor `π_hi(z|s,g)` predicting **representation** targets.

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `lr` | `3e-4` | Adam (shared) |
| `hidden_dims` | `(256, 256)` | all nets; value/goal_rep LayerNorm |
| `batch_size` | `256` | |
| `discount` | `0.99` | TD + geometric value-goal sampling |
| `tau` | `0.005` | target soft update |
| `expectile` | `0.7` | IVL expectile |
| `low_alpha` / `high_alpha` | `3.0` | AWR temperatures |
| `rep_dim` | `10` | φ / high-actor output dim |
| `subgoal_steps` | `8` | low/high K |
| `goal_dim` | `4` | full-state goals only |
| `gc_negative` | `True` | rewards `-1/0` with index equality |
| Value mixture | `0.2/0.5/0.3` | cur / traj / random |
| Value ensemble | `2` | min for next-V sign; per-head `q1/q2` |

**Dataset:** `HGCNavigateDataset` (OGBench HGC-style).
**Eval:** BC / HIQL-family at `T=0` (mode); PBG / PBF at `T=1` (sample).

**Inference:** `z~π_hi` (renorm to `√d`), then `a~π_lo(·|s,z)`.

---

## TR-HIQL (HIQL actors + transitive critic)

**File:** `agents/tr_hiql.py`
HIQL의 low/high actor + AWR는 그대로 두고, critic만 PathBridger TRL-lite
(`ScalarValueNet` + self/base/product-transitive)로 교체.

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `lr` | `3e-4` | Adam (shared) |
| `hidden_dims` | `(256, 256)` | value LayerNorm=True; actors LN=False |
| `batch_size` | `256` | |
| `discount` | `0.99` | geometric base `γ^j` + tri product |
| `tau` | `0.005` | target V Polyak |
| `low_alpha` / `high_alpha` | `3.0` | AWR on sigmoid-V advantage |
| `subgoal_steps` | `8` | path triples for transitive V |
| Expectile / ensemble | none | replaced by TRL-lite |

**Actor advantages (sigmoid V):**
- low: `V(s',g) − V(s,g)`
- high: `V(z,g) − V(s,g)` with `z = s_{t+K}`

**Inference:** same as HIQL — `z = mode(π_hi)`, `a = mode(π_lo(·|s,z))`.

---

## TRL (official Transitive RL toy port)

**File:** `agents/trl.py`  
**Source:** `aoberai/trl`; product-transitive ensemble critic + flow
rejection-sampling actor.

| Hyperparameter | Value |
|---|---|
| `hidden_dims` / `batch_size` | `(256,256)` / `256` |
| `discount` / `tau` / `expectile` | `0.99` / `0.005` / `0.7` |
| `lam` | `0.0` |
| `flow_steps` / candidates | `8` / `8` |
| value goals | future trajectory, geometric |
| actor goals | trajectory/random = `0.5/0.5` |

## DQC (official Decoupled Q-Chunking toy port)

**File:** `agents/dqc.py`  
**Source:** `ColinQiyangLi/dqc`; full chunk critic + distilled partial
critic + implicit V + flow BC.

| Hyperparameter | Value |
|---|---|
| `hidden_dims` / `batch_size` | `(256,256)` / `256` |
| critic chunk / policy chunk | `8` / `1` |
| `discount` / `tau` | `0.99` / `0.005` |
| `kappa_d` / `kappa_b` | `0.5` / `0.9` |
| `flow_steps` / best-of-N | `8` / `8` |

Both agents use terminal-safe compact trajectory samplers and emit one normalized
2D action per environment step.

---

## PBG (PathBridger-Gaussian lite)

**Files:** `agents/dynamics.py` (`subgoal_distribution=diag_gaussian`), bridge in `utils/dynamics.py`, value in `agents/critic.py`
**Stack:** subgoal mean + **transitive sigmoid V** + closed-form bridge + path residual + IDM

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `lr` | `3e-4` | Adam |
| `hidden_dims` | `(256, 256)` | subgoal / residual / IDM / value; LayerNorm=True |
| `batch_size` | `256` | |
| `dynamics_N` / `subgoal_steps` | `8` | bridge horizon `K` |
| `dynamics_lambda` | `1.0` | linear-SDE diffusion scale `λ` |
| `bridge_gamma_inv` | `0.0` | hard endpoint bridge |
| `theta_total` | `1.0` | cumulative OU rate `Θ_K` |
| `progress_alpha` | `0.8` | prefix-progress `c_i=(i/K)^α` |
| `discount` | `0.99` | geometric base target `γ^j` for V |
| `tau` | `0.005` | target V Polyak |
| `subgoal_loss_weight` | `1.0` | `‖ẑ(s,g) − s_{t+K}‖²` |
| `path_loss_weight` | `1.0` | interior + first-step path MSE |
| `idm_loss_weight` | `1.0` | `‖IDM(s,s') − a‖²` |
| `value_loss_weight` | `1.0` | self + base + product-transitive |
| `subgoal_value_gap_scale` | `3.0` | subgoal weight `exp(alpha·[V(z,g)−V(s,g)])`; HIQL alpha와 통일 |
| `subgoal_value_weight_max` | `100.0` | maximum detached gap weight; HIQL cap과 통일 |
| Subgoal loss | Gaussian **NLL** | `q(z\|s,g)=N(μ,diag(σ²))`; gap-weighted |
| `subgoal_num_candidates` | `1` | eval: mean (include_mean) only |
| `subgoal_include_mean` | `True` | pin μ as the single candidate |

**Value / subgoal selection (transitive):**
- Train: `V(s,s)→1`, `V(s,s_{t+j})→γ^j`, `V(s,g) ← V̄(s,z)V̄(z,g)` on path triples.
- Subgoal regression/flow matching: multiply each sample by detached
  `min(exp(3·[V̄(s_{t+K},g)−V̄(s,g)]), 100)`.
- Act (PBG): Gaussian mean endpoint 1개 → bridge → IDM.

**Bridge (fixed schedule):**
- `μ_i = a_i s + b_i z`, `path_i = μ_i + w_i r_θ`, `w_i=i(K−i)/K²`.

---

## PBF (PathBridger-Flow lite)

**Files:** same `agents/dynamics.py` with `subgoal_distribution=flow`.
Same transitive V + bridge + IDM as PBG; endpoint proposals from **conditional flow**.

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `lr` | `3e-4` | Adam |
| `hidden_dims` | `(256, 256)` | flow / residual / IDM / value; LayerNorm=True |
| `batch_size` | `256` | |
| `dynamics_N` / `subgoal_steps` | `8` | bridge `K` |
| `flow_steps` | `8` | Euler steps per proposal |
| `dynamics_lambda` | `1.0` | |
| `bridge_gamma_inv` | `0.0` | |
| `theta_total` | `1.0` | |
| `progress_alpha` | `0.8` | |
| `discount` / `tau` | `0.99` / `0.005` | same TRL-lite V |
| `flow_loss_weight` | `1.0` | CFM |
| gap scale / max weight | `3.0` / `100.0` | same detached value-gap weighting |
| Flow loss | CFM MSE | `x_u=(1-u)ε+u z`, `‖v_θ−(z−ε)‖²` |
| `subgoal_num_candidates` | `8` | zero-noise endpoint 1 + noisy flow endpoints 7 |
| `path_loss_weight` | `1.0` | |
| `idm_loss_weight` | `1.0` | |
| `value_loss_weight` | `1.0` | |

**Inference:** flow endpoints 8개 → transitive-ratio best 선택 → bridge → `IDM(s, path_1)`.

---

## Recent training recipe (full-only 10k)

```bash
export PYTHONPATH=/path/to/toy_examples
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

python -m hazard_env.agents.train \
  --agent {bc,hiql,tr_hiql,pbg,pbf} \
  --steps 10000 --eval-every 5000 --log-every 1000 --seed 0 \
  --env hazard_plain \
  --checkpoint-dir hazard_env/checkpoints/hazard_plain/full_10k/{agent} \
  --render-dir hazard_env/renders/hazard_plain/full_10k/{agent}
```

Use `--env hazard_grav` or `--env hazard_anti_grav` for the signed-field
variants. Artifacts are grouped under
`checkpoints/{hazard_plain,hazard_grav,hazard_anti_grav}/` and
`renders/{hazard_plain,hazard_grav,hazard_anti_grav}/`.