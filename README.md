# PathBridger toy examples

Illustrative 2D figures and a continuous hazard environment for PathBridger / offline GCRL experiments.

Repo: https://github.com/SChoish/PB_toy

```
toy_examples/
├── concept/          # single-panel concept figure
├── hazard_env/       # Hazard2D modes + datasets + toy agents
│   ├── env.py
│   ├── generate_navigate.py
│   ├── plot_tasks.py
│   ├── HYPERPARAMETERS.md
│   ├── agents/       # bc, hiql, dynamics (pbg/pbf), critic, train
│   ├── utils/        # flax_utils, networks, dynamics (bridge), datasets
│   ├── tests/
│   ├── datasets/     # hazard_plain / hazard_grav / hazard_anti_grav
│   ├── checkpoints/  # grouped by environment mode
│   └── renders/      # grouped by environment mode
└── refs/             # reference sketches / paper pages
```

## Hazard env

```bash
export PYTHONPATH=/path/to/toy_examples:$PYTHONPATH

# collect datasets: env × {navigate,noisy,random} × {1k,10k,100k}
python -m hazard_env.generate_navigate --generate-all
# or one combo
python -m hazard_env.generate_navigate --env hazard_plain --policy noisy --size 10k

# signed field variants are included via --env / --generate-all

# plot fixed eval tasks 1–5
python -m hazard_env.plot_tasks --env hazard_grav
python -m hazard_env.plot_coverage --env hazard_anti_grav --policy navigate --size 100k

# train a toy agent
python -m hazard_env.agents.train --env hazard_plain --agent bc --steps 50000 \
  --dataset-policy navigate --dataset-size 100k
# envs: hazard_plain | hazard_grav | hazard_anti_grav
# agents: bc | hiql | tr_hiql | pbg | pbf
```

All three modes use `ContinuousHazard2DEnv` and `Hazard2DConfig`.
`gravity_strength=0` disables the field, positive values attract toward the
hazard, and negative values repel away from it. Field magnitude follows the
inverse square of the distance from the hazard center.

Eval tasks: `env.reset(options={"task_id": 1})` … `5` (easy → hard).

The environment rejects invalid physical configurations and follows the Gymnasium
episode lifecycle: call `reset()` after either `terminated` or `truncated` becomes true.

Hyperparameters: see [`hazard_env/HYPERPARAMETERS.md`](hazard_env/HYPERPARAMETERS.md).

## Concept figure

```bash
python -m concept.codes_numerical   # → concept/outputs/pathbridger_concept_numerical.png
python -m concept.codes_nn          # → concept/outputs/pathbridger_concept_nn.png
python -m concept.codes             # both
```

| 요소 | 의미 |
|------|------|
| Heatmap | \(V(\cdot,g)\), peak at Goal (hazard not encoded) |
| Black curve | offline data trajectory |
| Blue line | value-greedy / endpoint-only (through hazard) |
| Purple solid / dashed | bridge executed prefix / planned remainder |
| Purple-border node | selected \(z^\star=\hat s_{t+K}\) |

## Refs

- `refs/ref.png` — labeled sketch (style reference)
- `refs/_pb_fig1.png`, `_pb_page4.png`, `PathBridger.pdf` — architecture / paper
