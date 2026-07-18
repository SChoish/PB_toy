# PathBridger toy examples

Offline GCRL toy suite: shared agents and three continuous control environments
(Hazard2D, CarRace, Orbital Swing-by), plus concept figures.

**Repo:** https://github.com/SChoish/PB_toy

```
toy_examples/
├── agents/        # BC, HIQL, TR-HIQL, TRL, DQC, PBG, PBF
├── hazard_env/    # ContinuousHazard2D
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
| `pbg` / `pbf` | PathBridger (Gaussian / flow subgoal) |

PathBridger eval reports **T=0** (Gaussian mean / flow zero-noise diagnostic)
and **T=1** (transitive-value best of four stochastic endpoints, with no pinned
mean). Other families use their default temperature (BC/HIQL: 0, TRL/DQC: 1).

Defaults: [`hazard_env/HYPERPARAMETERS.md`](hazard_env/HYPERPARAMETERS.md).

## Hazard2D

| Env | Field |
|-----|-------|
| `hazard_plain` | off |
| `hazard_grav` | attract |
| `hazard_anti_grav` | repel |

```bash
python -m hazard_env.generate_navigate --generate-all
python -m hazard_env.train --env hazard_plain --agent pbg --steps 50000 \
  --dataset-policy navigate --dataset-size 100k
```

Eval tasks: `env.reset(options={"task_id": 1})` … `5`.

## CarRace

| Env | Physics |
|-----|---------|
| `car_race_plain` | baseline (`rolling_drag=0.40`) |
| `car_race_grav` | inward field |
| `car_race_anti_grav` | outward field |
| `car_race_ice` | no field; low cornering, steering, acceleration, and braking grip |

Tasks: `navigation` | `lap_2p` | `lap_4p` | `lap_8p`.

Ice retains lateral momentum while the chassis turns. Its low rolling drag also
makes coasting and stopping distances longer than on the plain surface.

```bash
python -m car_race.generate_dataset --generate-all
python -m car_race.train --env car_race_plain --agent pbg --task navigation \
  --dataset-size 100k --steps 50000
```

## Orbital Swing-by

Body presets: `planet` | `black_hole`.

```bash
python -m swingby.generate_dataset --generate-all
```

The expert first predicts the unpowered trajectory. Reachable ballistic goals
coast through periapsis; collision or goal-miss predictions trigger an inbound
angular-momentum burn before the phase-space correction. All five fixed tasks
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

`car-race-generate`, `car-race-train`, `hazard-generate`, `hazard-train`,
`swingby-generate`, `pb-concept`.

## Tests

```bash
python -m pytest
```

CI runs pytest on Python 3.10–3.12, a research-import smoke check, and
wheel/sdist builds.
