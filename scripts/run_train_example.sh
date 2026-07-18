#!/usr/bin/env bash
# Example training launches for car_race / swingby (after datasets exist).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate pb_toy
fi

# Pick a free GPU (override with CUDA_VISIBLE_DEVICES=...).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_MEM_FRACTION

AGENT="${AGENT:-pbg}"
STEPS="${STEPS:-50000}"
SEED="${SEED:-0}"
SIZE="${SIZE:-100k}"

usage() {
  cat <<EOF
Usage: $0 <car_race|swingby> [env_name]

Examples:
  $0 car_race car_race_plain
  AGENT=pbf STEPS=50000 $0 car_race car_race_ice
  $0 swingby swingby_planet
  TASK=lap_4p $0 car_race car_race_plain
EOF
}

kind="${1:-}"
env_name="${2:-}"
if [[ -z "${kind}" ]]; then
  usage
  exit 1
fi

case "${kind}" in
  car_race)
    env_name="${env_name:-car_race_plain}"
    task="${TASK:-navigation}"
    out="checkpoints/car_race/${env_name}_${task}_${AGENT}_s${SEED}"
    python -m car_race.train \
      --env "${env_name}" --agent "${AGENT}" --task "${task}" \
      --dataset-size "${SIZE}" --steps "${STEPS}" --seed "${SEED}" \
      --eval-every 5000 --log-every 500 \
      --checkpoint-dir "${out}" \
      --render-dir "${out}/renders"
    ;;
  swingby)
    env_name="${env_name:-swingby_planet}"
    out="checkpoints/swingby/${env_name}_${AGENT}_s${SEED}"
    python -m swingby.train \
      --env "${env_name}" --agent "${AGENT}" \
      --dataset-size "${SIZE}" --steps "${STEPS}" --seed "${SEED}" \
      --eval-every 5000 --log-every 500 --num-eval-envs 25 \
      --checkpoint-dir "${out}" \
      --render-dir "${out}/renders"
    ;;
  *)
    usage
    exit 1
    ;;
esac
