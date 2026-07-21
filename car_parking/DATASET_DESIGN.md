# CarParking dataset design

## 결론

첫 데이터셋은 네 maneuver를 균등하게 섞은 성공 trajectory 중심의 물리-state
데이터셋으로 만든다. Expert는 Hybrid A*로 충돌 없는 전진/후진 경로를 찾고,
실제 `CarParkingEnv.step()`을 사용하는 저속 controller가 그 경로를 추종한다.
Planner가 만든 좌표를 곧바로 저장하지 않고 반드시 환경 rollout을 저장한다.

`car_race`와 동일하게 transition, episode boundary, future-goal sampling 계약을
유지하되, CarParking의 goal-relative state feature는 loader에서 다시 계산해야
한다.

## 가장 중요한 주의점: goal leakage

환경의 10차원 state는 다음과 같다.

```text
[x, y, cos(yaw), sin(yaw), speed, normalized_steering,
 distance_to_slot, normalized_yaw_error, inside_slot, health]
```

6–8번 feature는 현재 commanded goal에 종속된다. 이 state를 그대로 저장한 뒤
다른 future goal로 HER를 하면 observation에는 원래 goal 정보가 남고, 별도 goal
입력에는 relabeled goal이 들어가는 불일치가 생긴다.

따라서 NPZ에는 아래 7차원 Markov 물리 상태만 기본 observation으로 저장한다.

```text
[x, y, cos(yaw), sin(yaw), speed, normalized_steering, health]
```

Loader는 batch에서 선택된 goal마다 다음을 다시 계산해 10차원 state를 만든다.

- `distance_to_slot`: 현재 `(x, y)`와 sampled goal 위치의 거리
- `normalized_yaw_error`: 현재 yaw와 sampled goal yaw의 wrapped 차이 / π
- `inside_slot`: sampled goal pose와 해당 episode의 slot 크기로 재계산
- `health`: raw state의 마지막 값을 그대로 복사한다. goal relabeling과 무관하다.

Path, midpoint, value goal 등 full-state를 사용하는 agent field도 같은 sampled
goal을 기준으로 일관되게 재계산한다.

## Raw NPZ schema

| Field | Shape / dtype | 의미 |
|---|---|---|
| `observations` | `(N, 7) float32` | health를 포함한 goal-independent Markov 상태 |
| `actions` | `(N, 2) float32` | normalized steering/throttle |
| `next_observations` | `(N, 7) float32` | 다음 health를 포함한 물리 상태 |
| `commanded_goals` | `(N, 4) float32` | `[x, y, cos(yaw), sin(yaw)]` |
| `terminals` | `(N,) bool` | episode의 마지막 transition |
| `successes` | `(N,) bool` | dwell까지 끝난 실제 주차 성공 |
| `collisions` | `(N,) bool` | 해당 transition의 접촉 event |
| `deaths` | `(N,) bool` | health depletion absorbing failure |
| `health_losses` | `(N,) float32` | 해당 transition의 health 감소량 |
| `impact_impulses` | `(N,) float32` | 해당 transition의 누적 충격량 |
| `timeouts` | `(N,) bool` | time-limit boundary |
| `episode_ids` | `(N,) int32` | trajectory grouping |
| `maneuver_ids` | `(N,) int8` | parallel/T-forward/T-reverse/angled |
| `layout_variants` | `(N,) int16` | layout 재현용 variant |
| `slot_lengths` | `(N,) float32` | goal-relative state 재구성용 |
| `slot_widths` | `(N,) float32` | goal-relative state 재구성용 |

Whole episode만 저장한다. `terminals`는 collection boundary이고 `deaths`만
주차 환경의 absorbing failure다. 접촉 후 살아 있으면 다음 transition을 계속
저장하며, timeout은 Bellman absorbing으로 취급하지 않는다.

## Expert 생성

### 1. Hybrid A* pose planner

검색 state는 `(x, y, yaw, gear)`이고 환경과 동일한 wheelbase와 최대 steering을
사용한다. 초기 권장 설정은 다음과 같다.

- xy resolution: `0.02`
- yaw resolution: `5°`
- primitive 길이: `0.035–0.045`
- steering primitive: `[-1, -0.5, 0, 0.5, 1]`
- gear: forward / reverse
- primitive 내부 collision check 간격: 최대 `0.01`
- goal tolerance: position `0.015`, yaw `5°`

Cost에는 path length 외에 gear switch, steering 변화, reverse 주행, obstacle
clearance를 넣는다. Reverse 자체를 과도하게 벌점 주면 평행주차와 T-reverse가
망가지므로 gear switch 비용을 reverse 비용보다 크게 둔다.

현재 최소 회전 반경은 대략 다음과 같다.

```text
wheelbase / tan(max_steer) = 0.13 / tan(0.62) ≈ 0.183
```

따라서 단일 원호를 가정한 rule-based expert보다 multi-point maneuver를 찾을 수
있는 lattice search가 적합하다.

### 2. 실제 dynamics 추종

Hybrid A* path를 그대로 transition으로 쓰지 않는다. 방향별 pure-pursuit 또는
Stanley controller와 speed controller로 실제 환경을 step한다.

- forward target speed: 약 `0.14–0.18`
- reverse target speed: 약 `0.10–0.14`
- gear switch 전에는 `abs(speed) < 0.02`까지 제동
- goal 부근에서는 position/yaw 오차에 따라 속도를 연속적으로 낮춤
- planner path에서 벗어나면 현재 pose에서 한 번 재계획
- slot에 들어간 뒤 환경의 `dwell_steps` 전체를 수집

성공하지 못한 expert episode는 기본 expert 파일에서 제외하고 rejection count를
통계로 남긴다. Planner path가 존재해도 controller rollout이 충돌하면 실패다.

### 3. Behavior mixture

권장 benchmark 파일은 다음 mixture를 사용한다.

- 70%: deterministic expert
- 25%: expert action에 작은 correlated noise + replan recovery
- 5%: collision 직전까지 가는 near-miss/recovery trajectory

완전 random policy는 좁은 주차장에서 대부분 즉시 충돌하므로 별도 ablation
파일로만 만든다. Expert benchmark에 무작정 섞지 않는다.

## Task와 split 균형

Transition 수가 아니라 episode 수를 먼저 maneuver별로 균등 배분한다.

- parallel 25%
- T-forward 25%
- T-reverse 25%
- angled 25%
- 각 maneuver에서 upper/lower mirror 50/50

Train/validation은 서로 다른 seed와 start pose jitter band를 사용한다. 현재
layout은 variant가 주기적으로 반복되므로 goal 위치만으로 완전한 holdout이 되지
않는다. 정식 benchmark 전에 continuous slot shift/slot clearance randomization을
reset option으로 추가하는 것이 좋다. 첫 smoke dataset은 기존 layout에서 start
`x/y/yaw` jitter만 사용한다.

크기는 CarRace와 맞춘다.

- `1k`: schema와 학습 smoke test
- `10k`: planner/controller 및 agent 개발
- `100k`: 비교 실험
- validation은 train transition budget의 10%

## Goal sampling과 reward/mask

초기 권장 goal mix는 commanded 50%, same-trajectory future HER 50%다.
Cross-trajectory random goal은 다른 obstacle/layout에서는 도달 가능성이 보장되지
않으므로 v1에서는 사용하지 않는다.

주차 commanded goal과 일반 future HER goal은 종료 의미가 다르다.

- commanded goal: reward는 실제 `successes=True`인 마지막 dwell transition에서 1
- future HER goal: 선택된 future index에 도착하는 transition에서 1
- death: mask 0인 absorbing failure
- collision contact: health가 남아 있으면 mask를 유지
- timeout: episode boundary지만 absorbing mask는 유지

Future HER는 단순 pose radius 비교보다 선택한 same-trajectory index를 기준으로
성공 transition을 정하는 편이 안전하다. 이렇게 하면 이동 중 speed를 가진
중간 pose도 유효한 subgoal이 되고, 최종 commanded goal만 정지+dwell을 요구한다.

마지막 dwell 구간을 삭제하면 agent가 슬롯 안에서 정지 상태를 유지하는 action을
배우지 못하므로 반드시 보존한다. Batch sampler에서 final approach와 dwell
transition을 1.5–2배 정도 oversample하는 것은 유효하지만 raw 파일을 복제할
필요는 없다.

## 구현 순서

1. `hybrid_astar.py`: collision-aware forward/reverse pose planner
2. `parking_policy.py`: path tracker, speed controller, replan
3. `generate_dataset.py`: balanced episode collection과 NPZ 저장
4. `datasets.py`: goal-relative state reconstruction과 HER sampler
5. `train.py`: shared agents 연결

첫 단계에서는 planner/controller가 canonical task 1–5를 모두 성공하는지 먼저
검증한다. 이것이 통과하기 전에는 대용량 dataset을 생성하지 않는다.

## Acceptance gates

- canonical task별 expert 100회 성공률 `>= 95%`
- expert contact rate `<= 1%`
- expert death rate `= 0%`
- noisy-recovery 성공률 `>= 70%`
- maneuver별 episode 수 편차 `<= 2%`
- 모든 episode 마지막에만 `terminals=True`
- death와 timeout label이 동시에 켜지지 않음
- 같은 raw state에 두 goal을 적용했을 때 0–5번 physical feature와 health는
  동일하고 6–8번 goal-relative feature만 달라지는 unit test
- 고정 seed에서 byte-identical 1k dataset 재생성
- 모든 agent용 path/action-chunk window가 episode boundary를 넘지 않음
