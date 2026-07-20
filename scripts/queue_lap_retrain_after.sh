#!/usr/bin/env bash
# Wait for the active matrix orch(es) to finish, ensure swingby is done,
# then retrain lap_2p on the new *_lap_* datasets (seed=0, fresh ckpts).
#
#   nohup bash scripts/queue_lap_retrain_after.sh \
#     >> nohup_logs/queue_lap_retrain_after.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate pb_toy 2>/dev/null || true
fi

GPUS="${GPUS:-3,2}"
PACK="${PACK:-6}"
STEPS="${STEPS:-50000}"
SEED="${SEED:-0}"
SIZE="${SIZE:-100k}"
AGENTS="${AGENTS:-tr_hiql hiql pbg pbf trl}"
POLICIES="${POLICIES:-expert noisy random}"
CAR_ENVS="${CAR_ENVS:-car_race_ice car_race_grav car_race_anti_grav}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-checkpoints/car_race_pre_lap_retrain_$(date +%Y%m%d)}"

log() { echo "$(date -Is) $*"; }

wait_for_matrix_orches() {
  log "waiting for existing run_matrix.sh orchestrators to exit..."
  while pgrep -f 'bash scripts/run_matrix.sh' >/dev/null 2>&1; do
    # Ignore this script's own children once we launch later phases.
    local pids
    pids="$(pgrep -f 'bash scripts/run_matrix.sh' || true)"
    if [[ -z "${pids}" ]]; then
      break
    fi
    log "  still running: $(echo "${pids}" | tr '\n' ' ')"
    sleep 60
  done
  log "no run_matrix orchestrators left"
}

wait_for_trains() {
  log "waiting for live car_race/swingby trains to exit..."
  while pgrep -f 'python -m (car_race|swingby)\.train' >/dev/null 2>&1; do
    local n
    n="$(pgrep -cf 'python -m (car_race|swingby)\.train' || true)"
    log "  live trains=${n}"
    sleep 60
  done
  log "no live trains left"
}

run_swingby_phase() {
  log "=== PHASE swingby ==="
  GPUS="${GPUS}" PACK="${PACK}" STEPS="${STEPS}" SEED="${SEED}" SIZE="${SIZE}" \
    AGENTS="${AGENTS}" POLICIES="${POLICIES}" \
    CAR_ENVS= CAR_TASKS= \
    SWING_ENVS='swingby_planet swingby_blackhole' \
    bash scripts/run_matrix.sh
  log "=== PHASE swingby DONE ==="
}

archive_old_lap_ckpts() {
  mkdir -p "${ARCHIVE_ROOT}"
  local moved=0
  shopt -s nullglob
  for d in checkpoints/car_race/*lap_2p*; do
    [[ -d "${d}" ]] || continue
    local base
    base="$(basename "${d}")"
    if [[ -e "${ARCHIVE_ROOT}/${base}" ]]; then
      rm -rf "${ARCHIVE_ROOT}/${base}"
    fi
    mv "${d}" "${ARCHIVE_ROOT}/"
    moved=$((moved + 1))
  done
  shopt -u nullglob
  # Stale gallery dirs for lap (optional).
  shopt -s nullglob
  for d in renders/*lap_2p*; do
    rm -rf "${d}"
  done
  shopt -u nullglob
  log "archived ${moved} lap_2p ckpt dirs -> ${ARCHIVE_ROOT}"
}

run_lap_retrain_phase() {
  log "=== PHASE lap_retrain (new *_lap_* datasets) ==="
  archive_old_lap_ckpts
  GPUS="${GPUS}" PACK="${PACK}" STEPS="${STEPS}" SEED="${SEED}" SIZE="${SIZE}" \
    AGENTS="${AGENTS}" POLICIES="${POLICIES}" \
    CAR_ENVS="${CAR_ENVS}" CAR_TASKS='lap_2p' \
    SWING_ENVS= \
    bash scripts/run_matrix.sh
  log "=== PHASE lap_retrain DONE ==="
}

log "queue_lap_retrain_after starting (gpus=${GPUS} pack=${PACK})"
wait_for_matrix_orches
wait_for_trains
run_swingby_phase
wait_for_trains
run_lap_retrain_phase
log "=== QUEUE_ALL_DONE ==="
