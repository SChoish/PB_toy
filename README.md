# PathBridger toy examples

Offline GCRL toy suite: shared agents and two continuous control environments
(CarRace, Orbital Swing-by), plus concept figures.

**Repo:** https://github.com/SChoish/PB_toy

```
toy_examples/
├── agents/        # BC, HIQL, TR-HIQL, TRL, DQC, PBG, PBF
├── car_race/      # annular navigation + lap racing
├── swingby/       # orbital flyby / swing-by
└── concept/       # PathBridger concept figures
```

Release surface is env / train / generate / agents / tests. Local demos, sweeps,
plots, refs, datasets, checkpoints, and renders are gitignored.

## Install

```bash
python -m pip install -e .
python -m pip install -e ".[research]"   # JAX/Flax agents + training
python -m pip install -e ".[render]"     # optional image/GIF helpers
python -m pip install -e ".[test]"       # pytest
```

Requires Python 3.10–3.12. Without an editable install, set
`PYTHONPATH=/path/to/toy_examples`.

## Agents

| Agent | Role |
|-------|------|
| `bc` | behavior cloning |
| `hiql` / `tr_hiql` | hierarchical / transitive HIQL |
| `trl` / `dqc` | trajectory / action-chunk critics |
| `pbg` / `pbf` | PathBridger (Gaussian / flow) — core in `agents/pathbridger/` |

PathBridger eval reports **T=0** (Gaussian mean / flow zero-noise diagnostic)
and **T=1** (PBG: 1 stochastic sample; PBF: transitive-value best of 8 flow
endpoints; no pinned mean). Other families use their default temperature (BC/HIQL: 0, TRL/DQC: 1).

Defaults: [`agents/HYPERPARAMETERS.md`](agents/HYPERPARAMETERS.md).

## Benchmark contract

Both environments expose five fixed evaluation tasks with task IDs 1 through 5
via `env.reset(options={"task_id": n})`; `info["goal"]` is exactly the 4-D goal
passed to the agent and `info["success"]` reports task completion. Datasets use
regular transition tuples with explicit `next_observations`. Train/validation
splits use disjoint seeds and retain whole episodes. `terminals` marks trajectory
boundaries, while goal-conditioned Bellman masks are computed separately from
the relabeled goal success and true absorbing failures (collision/death/escape),
not from time-limit boundaries.

## CarRace

| Env | Physics |
|-----|---------|
| `car_race_plain` | baseline (`rolling_drag=0.40`) |
| `car_race_grav` | inward field |
| `car_race_anti_grav` | outward field |
| `car_race_ice` | no field; low cornering, steering, acceleration, and braking grip |

Tasks: `navigation` | `lap_1p` … `lap_8p`.

Ice retains lateral momentum while the chassis turns. Its low rolling drag also
makes coasting and stopping distances longer than on the plain surface.

```bash
python -m car_race.generate_dataset --generate-all --task navigation
python -m car_race.generate_dataset --generate-all --task lap
python -m car_race.train --env car_race_plain --agent pbg --task navigation \
  --dataset-size 100k --steps 50000
```

## Powered Orbital Flyby (`swingby` API)

Body presets: `planet` | `black_hole`.  This is a fuel-limited powered rocket
flyby around a fixed central body, not a moving-planet gravity assist that can add
inertial-frame energy.  Every fixed evaluation task now starts on the incoming
left branch and targets an outgoing state on the right after a body pass.

| Contract | Definition |
|----------|------------|
| State | `[x, y, vx, vy, fuel_fraction]` |
| Runtime action | `[inertial_thrust_angle, throttle]`, in `[-pi, pi] x [0, 1]` |
| Network action | Cartesian thrust `throttle * [cos(angle), sin(angle)]` |
| Goal | `[goal_x, goal_y, goal_vx, goal_vy]` |
| Physics | Fixed central field, variable wet mass, fuel-limited thrust, velocity-Verlet with `dt=0.04` and 8 substeps |


```bash
# Default swingby dataset: one balanced dataset spanning T1 through T5.
python -m swingby.generate_dataset --generate-all
python -m swingby.train --env swingby_planet --agent hiql \
  --dataset-size 100k --steps 50000

# Legacy coast-aligned distribution remains available for ablations.
python -m swingby.generate_dataset --generate-all --dataset-mode ballistic
```
`swingby` balances transition coverage across the five fixed task families
while separating train and evaluation initial conditions. Training uses the
frozen `dataset` task table and inner rotation bands. The fixed evaluation
uses an outgoing powered T1, the original T2/T4 flyby geometry, a hard T3, and a
77.5% hard T5 interpolation. Each canonical task runs once plus 24 variants from
disjoint held-out rotation bands; T2 mixes near- and opposite-side bands so its
variants are challenging without being uniformly far out of distribution.
The expert dataset keeps only successful
trajectories, terminates each trajectory on its commanded goal, and trains with
a 50/50 mix of exact commanded goals and future HER goals whenever that
commanded goal was actually reached. Network actions use continuous Cartesian thrust
`throttle * [cos(angle), sin(angle)]`; this removes the angle wrap discontinuity
and gives coasting the unique action `[0, 0]`. The raw NPZ still stores physical
`(angle, throttle)` actions. Matrix jobs use `*_swingby_*` datasets and isolated
`*_swingby_*` checkpoints by default. Older versioned files remain readable
for reproducibility but are never selected by the default matrix scripts.

The expert first predicts the unpowered trajectory. Reachable ballistic goals
coast through periapsis; collision or goal-miss predictions trigger an inbound
angular-momentum burn before the phase-space correction. Goal success requires
position proximity, velocity-error tolerance, at least half the requested speed,
and velocity cosine alignment of at least `0.75`; a stationary rocket cannot
satisfy a slow capture goal. All five fixed tasks
are regression-tested for both body presets.

## Concept figures

```bash
python -m concept.codes_numerical
python -m concept.codes_nn
python -m concept.codes             # both → concept/outputs/
```

| Element | Meaning |
|---------|---------|
| Heatmap | \(V(\cdot,g)\), peak at goal |
| Black curve | offline trajectory |
| Blue line | value-greedy / endpoint-only |
| Purple path | bridge prefix / planned remainder |
| Purple node | selected \(z^\star=\hat s_{t+K}\) |

## CLI entry points

`car-race-generate`, `car-race-train`, `swingby-generate`, `swingby-train`,
`pb-concept`.

## Tests

```bash
python -m pytest
```

CI runs pytest on Python 3.10–3.12, a research-import smoke check, and
wheel/sdist builds.
