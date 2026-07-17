# Hazard-bridge toy (3-panel)

Shows why an endpoint alone is not enough: the same selected endpoint can
induce an unsafe straight controller, while an endpoint-pinned bridge follows
offline trajectory support around hazards.

```bash
python run_experiment.py
# optional: python -m unittest tests.test_pipeline
```

Outputs in `outputs/`:
- `toy_hazard_dataset.json`
- `learned_models.npz`
- `metrics.json`
- `toy_pathbridger.svg` / `.png`
