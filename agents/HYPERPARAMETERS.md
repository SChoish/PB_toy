# Agent hyperparameters

Defaults live under `agents/` (`bc`, `hiql`, `dynamics`).
Training: `python -m car_race.train` / `python -m swingby.train` `--agent {bc,hiql,tr_hiql,trl,dqc,pbg,pbf}`.

Layout (mirrors Pathbridger_flow):

```
agents/              # shared agents + HYPERPARAMETERS + diagnostic rendering
car_race/            # annular navigation + lap racing
swingby/             # orbital flyby / swing-by
```

These are **toy environments + training loops**. PBG/PBF agent cores are
**from PathBridger** (`agents/pathbridger/`); other agents remain
lightweight reimplementations inspired by OGBench / Pathbridger_flow.

---

## Shared training / data settings

| Setting | Value | Notes |
|---------|-------|--------|
| Optimizer | Adam | all agents |
| Learning rate `lr` | `3e-4` | |
| Batch size | `256` | |
| MLP width | `(256, 256)` | GELU; see per-agent LN |
| Observation | env-specific | car_race / swingby state vectors |
| Action (env) | `(angle ∈ [-π,π], thrust ∈ [0,1])` | |
| Action (network) | `[angle/π, 2·thrust−1] ∈ [-1,1]^2` | clip after sample |
| Dataset | `datasets/{env}_{policy}_{size}.npz` | policy∈{navigate,noisy,random}, size∈{1k,10k,100k} |
| Goal relabel | `goal_relabel_prob=0.8` | else commanded / terminal |
| Subgoal horizon `K` | `25` | `subgoal_steps` / `dynamics_N` |
| Action chunk `h_a` | `5` | PBG/PBF `action_chunk_horizon` (env steps per replan) |
| Path tensor | `(B, K+1, 4)` | true `s_t … s_{t+K}` for PBG/PBF |
| Goals | env-specific | car_race / swingby task goal prefix (φ) |
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
| `subgoal_steps` | `25` | low/high K |
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
| `subgoal_steps` | `25` | path triples for transitive V |
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
| critic chunk / policy chunk | `25` / `5` |
| `discount` / `tau` | `0.99` / `0.005` |
| `kappa_d` / `kappa_b` | `0.5` / `0.9` |
| `flow_steps` / best-of-N | `8` / `8` |

Both agents use terminal-safe compact trajectory samplers and emit one normalized
2D action per environment step.

---

## PBG (PathBridger-Gaussian)

**Source:** from `../PathBridger`, packaged as `agents/pathbridger/` (DynamicsAgent + CriticAgent).
Toy wrapper: `agents/dynamics.py` (`PathBridgerAgent`).

**Stack:** subgoal mean + **transitive sigmoid V** (CriticAgent) + closed-form bridge + path residual + IDM

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `lr` | `3e-4` | Adam |
| `hidden_dims` | `(256, 256)` | toy suite width (override PathBridger default 512^3) |
| `batch_size` | `256` | |
| `dynamics_N` / `subgoal_steps` | `25` | bridge horizon `K` |
| `action_chunk_horizon` | `5` | open-loop env steps per replan (`h_a`) |
| `forward_bridge_path_loss_horizon` | `5` | path loss prefix |
| `dynamics_lambda` | `1.0` | linear-SDE diffusion scale `λ` |
| `bridge_gamma_inv` | `0.0` | hard endpoint bridge |
| `discount` | `0.99` | geometric base target for V |
| `subgoal_eval_num_samples` | `1` | T=1: single stochastic Gaussian sample |
| `subgoal_include_mean` | `False` | match mainline eval; no pinned mean |
| actor φ | `(0,1,2,3)` | car_race `[x,y,progress,dir]`; swingby `[gx,gy,gvx,gvy]` |
| critic / value | `full` | full-state transitive V |

**Value / subgoal selection (transitive):**
- Train: full-state same-trajectory tuples with `i<k<j`,
  `V(s_i,s_i)→1`, `V(s_i,s_{i+j})→γ^j`, and
  `V(s_i,s_j) ← V̄(s_i,s_k)V̄(s_k,s_j)`.
- Subgoal regression/flow matching: multiply each sample by detached
  `min(exp(3·[V̄(s_{t+K},g)−V̄(s,g)]), 100)`.
- Act (PBG): T=0 is the Gaussian mean diagnostic; T=1 uses **one**
  stochastic sample (no BoN).

**Bridge (fixed schedule):**
- `μ_i = a_i s + b_i z`, `path_i = μ_i + w_i r_θ`, `w_i=i(K−i)/K²`.

---

## PBF (PathBridger-Flow)

Same PathBridger stack (`agents/pathbridger`) as PBG with `subgoal_distribution=flow`.

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `dynamics_N` / `subgoal_steps` | `25` | bridge `K` |
| `action_chunk_horizon` | `5` | open-loop env steps per replan (`h_a`) |
| `subgoal_flow_steps` | `8` | Euler steps per proposal |
| `subgoal_eval_num_samples` | `8` | eight stochastic flow endpoints at T=1 |
| `subgoal_include_mean` | `False` | no zero-noise endpoint in T=1 BoN |
| actor φ / critic | `(0,1,2,3)` / `full` | same recipe as PBG |

**Inference:** eight stochastic flow endpoints → transitive-ratio best selection
→ bridge → IDM action chunk (`h_a`). T=0 remains a zero-noise diagnostic.

---

## Recent training recipe (full-only 10k)

```bash
export PYTHONPATH=/path/to/toy_examples
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

python -m car_race.train --env car_race_plain --agent pbg --task navigation
python -m swingby.train --env swingby_planet --agent hiql