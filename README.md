# PathBridger toy examples

Illustrative 2D figures and a continuous hazard environment for PathBridger / offline GCRL experiments.

Repo: https://github.com/SChoish/PB_toy

```
toy_examples/
├── concept/          # single-panel concept figure
├── hazard_bridge/    # 3-panel "why the bridge matters"
├── hazard_env/       # ContinuousHazard2DEnv + navigate dataset + toy agents
│   ├── env.py
│   ├── generate_navigate.py
│   ├── plot_tasks.py
│   ├── agents/       # bc, hiql, pbg, pbf (simplified JAX)
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
python -m hazard_env.agents.train --agent bc --steps 5000
# agents: bc | hiql | pbg | pbf
```

Eval tasks: `env.reset(options={"task_id": 1})` … `5` (easy → hard).

## Concept figure

```bash
cd concept
python codes_numerical.py   # → outputs/pathbridger_concept_numerical.png
python codes_nn.py          # → outputs/pathbridger_concept_nn.png
python codes.py             # both
```

| 요소 | 의미 |
|------|------|
| Heatmap | \(V(\cdot,g)\), peak at Goal (hazard not encoded) |
| Black curve | offline data trajectory |
| Blue line | value-greedy / endpoint-only (through hazard) |
| Purple solid / dashed | bridge executed prefix / planned remainder |
| Purple-border node | selected \(z^\star=\hat s_{t+K}\) |

## Hazard-bridge 3-panel

```bash
cd hazard_bridge
python run_experiment.py
# → outputs/toy_pathbridger.{svg,png}
```

## Refs

- `refs/ref.png` — labeled sketch (style reference)
- `refs/_pb_fig1.png`, `_pb_page4.png`, `PathBridger.pdf` — architecture / paper
