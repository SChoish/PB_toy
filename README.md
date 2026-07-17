# PathBridger toy examples

Illustrative 2D figures and a continuous hazard environment for PathBridger / offline GCRL experiments.

Repo: https://github.com/SChoish/PB_toy

```
toy_examples/
├── concept/          # single-panel concept figure
├── hazard_env/       # ContinuousHazard2DEnv + navigate dataset + toy agents
│   ├── env.py
│   ├── generate_navigate.py
│   ├── plot_tasks.py
│   ├── HYPERPARAMETERS.md
│   ├── agents/       # bc, hiql, pbg, pbf (JAX toys)
│   ├── tests/
│   └── datasets/
└── refs/             # reference sketches / paper pages
```

## Hazard env

```bash
export PYTHONPATH=/path/to/toy_examples:$PYTHONPATH

# collect navigate dataset
python -m hazard_env.generate_navigate --num-episodes 200

# plot fixed eval tasks 1–5
python -m hazard_env.plot_tasks

# train a toy agent
python -m hazard_env.agents.train --agent bc --steps 50000
# agents: bc | hiql | pbg | pbf
```

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
