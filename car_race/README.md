# PathBridger CarRace

Continuous-control Gymnasium environment for navigation and lap racing in a safe
annulus between two collision hazards.

| Env | Physics |
|-----|---------|
| `car_race_plain` | baseline (`rolling_drag=0.40`) |
| `car_race_grav` | inward inverse-square field |
| `car_race_anti_grav` | outward inverse-square field |
| `car_race_ice` | no field; low tire grip and persistent lateral momentum |

Core runtime depends on NumPy and Gymnasium. Training uses the shared top-level
`agents` package (`pip install -e ".[research]"`).

## Quick start

```python
import numpy as np

from car_race import CarRaceConfig, CarRaceEnv, mode_config_kwargs

env = CarRaceEnv(
    CarRaceConfig(task_mode="navigation", **mode_config_kwargs("car_race_ice")),
    observation_mode="state_goal",
)
observation, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    action = np.array([0.0, 0.5], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

Actions are normalized `[steering, throttle_or_brake]` in `[-1, 1]`.
State layout:

```text
[x, y, task_progress, direction, cos(heading), sin(heading),
 drive_speed, health, external_velocity_x, external_velocity_y]
```

Observation modes: `state`, `state_goal`, `goal_dict`.
Tasks: `navigation` | `lap_2p` | `lap_4p` | `lap_8p`.

Ice uses reduced cornering grip (`0.20`), longitudinal acceleration/braking grip
(`0.53`), and steering response (`0.55`). The chassis can point into a turn while
the observed `external_velocity` retains the previous travel direction, producing
real drift without changing the state or dataset dimensions.

## Registered environments

Call `car_race.register_environment()` before `gymnasium.make(...)`.

- `CarRaceNavigation-v0`, `CarRaceIceNavigation-v0`, …
- `CarRaceLap2p-v0` / `Lap4p` / `Lap8p` and field-specific variants

## Dataset + train

```bash
python -m car_race.generate_dataset --env car_race_ice --policy expert --size 100k
python -m car_race.train --env car_race_ice --agent pbg --task navigation \
  --dataset-size 100k --steps 50000
```

Policies: `expert` | `noisy` | `random`. Sizes: `1k` | `10k` | `100k`
(minimum transitions; whole episodes are kept).

## Tests

```bash
python -m pytest car_race/tests
```
