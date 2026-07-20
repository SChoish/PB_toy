#!/usr/bin/env bash
# Job matrix for PB_toy: tr_hiql / hiql / pbg / pbf / trl
# CarRace: ice, grav, anti_grav × navigation, lap_2p × expert/noisy/random
# Swingby: planet, blackhole × expert/noisy/random
#
# Usage:
#   bash scripts/plan_matrix.sh              # print jobs
#   bash scripts/plan_matrix.sh --count      # print counts only
#   bash scripts/run_matrix.sh               # launch (separate script)

set -euo pipefail

# Override with space-separated list, e.g. AGENTS="tr_hiql hiql trl"
if [[ -n "${AGENTS:-}" ]]; then
  # shellcheck disable=SC2206
  AGENTS=(${AGENTS})
else
  AGENTS=(tr_hiql hiql pbg pbf trl)
fi
# Override with space-separated list, e.g. CAR_ENVS="car_race_grav car_race_anti_grav"
if [[ -n "${CAR_ENVS:-}" ]]; then
  # shellcheck disable=SC2206
  CAR_ENVS=(${CAR_ENVS})
else
  CAR_ENVS=(car_race_ice car_race_grav car_race_anti_grav)
fi
# Override with space-separated list, e.g. CAR_TASKS="navigation"
if [[ -n "${CAR_TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  CAR_TASKS=(${CAR_TASKS})
else
  CAR_TASKS=(navigation lap_2p)
fi
# Dataset policies: expert keeps legacy ckpt names; noisy/random append _${policy}.
if [[ -n "${POLICIES:-}" ]]; then
  # shellcheck disable=SC2206
  POLICIES=(${POLICIES})
else
  POLICIES=(expert noisy random)
fi
# Override with space-separated list; set empty to skip swingby:
#   SWING_ENVS= bash scripts/plan_matrix.sh
if [[ -n "${SWING_ENVS+x}" ]]; then
  # shellcheck disable=SC2206
  SWING_ENVS=(${SWING_ENVS})
else
  SWING_ENVS=(swingby_planet swingby_blackhole)
fi
# Dataset naming / ckpt prefix: ballistic (legacy) or swingby (schema swingby → ckpt *_swingby_*).
#   SWING_DATASET_MODE=swingby bash scripts/plan_matrix.sh
SWING_DATASET_MODE="${SWING_DATASET_MODE:-swingby}"

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

ckpt_name() {
  # $1=prefix (env_task or env), $2=agent, $3=policy
  local prefix="$1" agent="$2" policy="$3"
  if [[ "${policy}" == "expert" ]]; then
    echo "checkpoints/${kind_dir}/${prefix}_${agent}_s${SEED}"
  else
    echo "checkpoints/${kind_dir}/${prefix}_${agent}_${policy}_s${SEED}"
  fi
}

echo "# PB_toy train matrix  steps=${STEPS} seed=${SEED} size=${SIZE} policies=${POLICIES[*]} swing_mode=${SWING_DATASET_MODE}"
echo "# format: kind|env|task|agent|policy|checkpoint_dir"

kind_dir=car_race
for env in "${CAR_ENVS[@]}"; do
  for task in "${CAR_TASKS[@]}"; do
    for policy in "${POLICIES[@]}"; do
      for agent in "${AGENTS[@]}"; do
        # Lap tasks already encode the schema in the task name (lap_2p / lap_4p / …).
        prefix="${env}_${task}"
        ckpt="$(ckpt_name "${prefix}" "${agent}" "${policy}")"
        if [[ "${count_only}" -eq 0 ]]; then
          echo "car_race|${env}|${task}|${agent}|${policy}|${ckpt}"
        fi
        car=$((car + 1))
      done
    done
  done
done

kind_dir=swingby
for env in "${SWING_ENVS[@]}"; do
  for policy in "${POLICIES[@]}"; do
    for agent in "${AGENTS[@]}"; do
      if [[ "${SWING_DATASET_MODE}" == "swingby" ]]; then
        prefix="${env}_swingby"
      else
        prefix="${env}"
      fi
      ckpt="$(ckpt_name "${prefix}" "${agent}" "${policy}")"
      if [[ "${count_only}" -eq 0 ]]; then
        echo "swingby|${env}|-|${agent}|${policy}|${ckpt}"
      fi
      swing=$((swing + 1))
    done
  done
done

echo "# totals: car_race=${car} swingby=${swing} all=$((car + swing))"
