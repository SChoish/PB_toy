#!/usr/bin/env bash
# Job matrix for PB_toy: tr_hiql / pbg / pbf / trl / dqc
# CarRace: ice, grav, anti_grav × navigation, lap_2p, lap_4p, lap_8p
# Swingby: planet, blackhole
#
# Usage:
#   bash scripts/plan_matrix.sh              # print jobs
#   bash scripts/plan_matrix.sh --count      # print counts only
#   bash scripts/run_matrix.sh               # launch (separate script)

set -euo pipefail

# Override with space-separated list, e.g. AGENTS="tr_hiql trl dqc"
if [[ -n "${AGENTS:-}" ]]; then
  # shellcheck disable=SC2206
  AGENTS=(${AGENTS})
else
  AGENTS=(tr_hiql pbg pbf trl dqc)
fi
# Override with space-separated list, e.g. CAR_ENVS="car_race_grav car_race_anti_grav"
if [[ -n "${CAR_ENVS:-}" ]]; then
  # shellcheck disable=SC2206
  CAR_ENVS=(${CAR_ENVS})
else
  CAR_ENVS=(car_race_ice car_race_grav car_race_anti_grav)
fi
CAR_TASKS=(navigation lap_2p lap_4p lap_8p)
SWING_ENVS=(swingby_planet swingby_blackhole)

STEPS="${STEPS:-50000}"
SEED="${SEED:-0}"
SIZE="${SIZE:-100k}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-500}"

count_only=0
if [[ "${1:-}" == "--count" ]]; then
  count_only=1
fi

car=0
swing=0

echo "# PB_toy train matrix  steps=${STEPS} seed=${SEED} size=${SIZE}"
echo "# format: kind|env|task|agent|checkpoint_dir"

for env in "${CAR_ENVS[@]}"; do
  for task in "${CAR_TASKS[@]}"; do
    for agent in "${AGENTS[@]}"; do
      ckpt="checkpoints/car_race/${env}_${task}_${agent}_s${SEED}"
      if [[ "${count_only}" -eq 0 ]]; then
        echo "car_race|${env}|${task}|${agent}|${ckpt}"
      fi
      car=$((car + 1))
    done
  done
done

for env in "${SWING_ENVS[@]}"; do
  for agent in "${AGENTS[@]}"; do
    ckpt="checkpoints/swingby/${env}_${agent}_s${SEED}"
    if [[ "${count_only}" -eq 0 ]]; then
      echo "swingby|${env}|-|${agent}|${ckpt}"
    fi
    swing=$((swing + 1))
  done
done

echo "# totals: car_race=${car} swingby=${swing} all=$((car + swing))"
