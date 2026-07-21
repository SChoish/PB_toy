# CarParking

`car_race`와 같은 연속 제어 인터페이스를 쓰는 저속 주차 Gymnasium 환경입니다.
차량은 원이 아니라 회전 직사각형으로 충돌 판정되며, 중심점만 슬롯에 들어가서는
성공하지 않습니다.

## 시나리오

| Gymnasium ID | 주차 방식 |
|---|---|
| `CarParking-v0` | 평행/T자 전진/T자 후진/사선 혼합 |
| `CarParkingParallel-v0` | 평행주차 |
| `CarParkingTForward-v0` | T자 전면주차 |
| `CarParkingTReverse-v0` | T자 후면주차 |
| `CarParkingAngled-v0` | 사선주차 |

혼합 환경의 고정 평가 task 1–5는 각각 하단 평행, 상단 평행, T자 후진,
T자 전진, 사선주차입니다.

## 빠른 시작

```python
import numpy as np

from car_parking import CarParkingConfig, CarParkingEnv

env = CarParkingEnv(
    CarParkingConfig(maneuver="t_reverse"),
    observation_mode="state_goal",
    render_mode="rgb_array",
)
observation, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    # [steering, throttle_or_brake], both normalized to [-1, 1].
    action = np.array([0.0, 0.2], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

등록된 환경도 사용할 수 있습니다.

```python
import gymnasium as gym
import car_parking

car_parking.register_environment()
env = gym.make("CarParkingParallel-v0", observation_mode="goal_dict")
```

## 관측과 성공 조건

상태는 다음 10차원입니다.

```text
[x, y, cos(yaw), sin(yaw), speed, normalized_steering,
 distance_to_slot, normalized_yaw_error, inside_slot, health]
```

목표는 `[goal_x, goal_y, cos(goal_yaw), sin(goal_yaw)]`입니다.
`state`, `state_goal`, `goal_dict` 관측 모드를 지원합니다.

주차 성공에는 다음 조건이 모두 필요합니다.

- 차체의 네 모서리가 슬롯 여유 영역 안에 있음
- 목표 헤딩과의 오차가 기본 10도 이하
- 속도가 기본 `0.025` 이하
- 위 상태를 기본 8 step 연속 유지

고정 평가 시작은 `env.reset(options={"task_id": 1})`처럼 선택합니다.
`info["goal"]`, `info["is_success"]`, `info["collision"]`,
`info["dead"]`, `info["health"]`, `info["health_loss"]`,
`info["step_impulse"]`, `info["fully_inside_slot"]`,
`info["dwell_count"]`를 제공합니다.

충돌은 기본적으로 즉시 episode를 끝내지 않고 impact impulse에 비례해 health를
깎습니다. 기본 damage capacity는 의도적으로 작아 저속 접촉도 큰 손상을 주며,
health가 0이 되면 `health_depleted`로 종료됩니다. 즉시 충돌 종료는
`CarParkingConfig(terminate_on_collision=True)` ablation에서만 사용합니다.
`info["collision"]`은 해당 step의 접촉이고, `info["dead"]`만 누적 손상에 의한
absorbing failure를 뜻합니다.

주행 통로 폭은 기본 `0.48`이며 `CarParkingConfig(aisle_width=...)`로
조절할 수 있습니다. 맞은편 보도 경계는 장식이 아니라 실제 충돌 영역입니다.

## 테스트

```bash
python -m pytest car_parking/tests
```

## Hybrid A* expert

Expert는 환경과 같은 bicycle geometry와 oriented-box 충돌 판정으로 전진/후진
lattice 경로를 찾습니다. 경로 pose를 state에 복사하지 않고, 조향 지연과
gear switch를 처리하면서 모든 이동과 dwell을 실제 `env.step()`으로 실행합니다.

```python
from car_parking import CarParkingEnv, rollout_expert

env = CarParkingEnv()
result = rollout_expert(env, task_id=1, seed=0)
print(result.success, result.steps)
```

Task 1–5 acceptance와 선택적인 시작 pose jitter는 다음처럼 검증합니다.

```bash
python -m car_parking.validate_expert --episodes-per-task 20 --seed 0
python -m car_parking.validate_expert --episodes-per-task 5 --seed 0 \
  --jitter-position 0.005 --jitter-heading-deg 1.0
```

## 고정 task 데모

Task 1–5의 개별 PNG와 전체 overview는 [`demo/`](demo/)에 있습니다.
시각화 변경 후 다음 명령으로 다시 생성할 수 있습니다.

```bash
python -m car_parking.render_demos
```

## 데이터셋 설계

Goal leakage를 피하는 raw schema, Hybrid A* expert, HER 규칙과 검증 기준은
[`DATASET_DESIGN.md`](DATASET_DESIGN.md)에 정리되어 있습니다.
