#!/usr/bin/env bash
# 1) Retrain lap_2p on new *_lap_* datasets (now).
# 2) Wait until canonical swingby NPZs exist.
# 3) Train swingby with SWING_DATASET_MODE=swingby (ckpt prefix *_swingby_*).
#
#   nohup bash scripts/queue_lap_then_swingby.sh \
#     >> nohup_logs/queue_lap_then_swingby.log 2>&1 &
#
# Resume after lap already launched:
#   SKIP_LAP=1 nohup bash scripts/queue_lap_then_swingby.sh \
#     >> nohup_logs/queue_lap_then_swingby.log 2>&1 &
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
SWING_ENVS="${SWING_ENVS:-swingby_planet swingby_blackhole}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-checkpoints/car_race_pre_lap_retrain_$(date +%Y%m%d)}"
POLL_SEC="${POLL_SEC:-60}"
SKIP_LAP="${SKIP_LAP:-0}"

log() { echo "$(date -Is) $*"; }

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
  for d in renders/*lap_2p*; do
    rm -rf "${d}"
  done
  shopt -u nullglob
  log "archived ${moved} lap_2p ckpt dirs -> ${ARCHIVE_ROOT}"
}

swingby_stem_candidates() {
  local env="$1" policy="$2"
  echo "${env}_swingby_${policy}_${SIZE}"
}

resolve_swingby_pair() {
  # Prints "train_path|val_path" if a matching train+val pair exists.
  local env="$1" policy="$2" stem train val
  for stem in $(swingby_stem_candidates "${env}" "${policy}"); do
    train="swingby/datasets/${stem}.npz"
    val="swingby/datasets/${stem}_val.npz"
    if [[ -f "${train}" && -f "${val}" ]]; then
      echo "${train}|${val}"
      return 0
    fi
  done
  return 1
}

missing_swingby_combos() {
  local env policy
  # shellcheck disable=SC2206
  local envs=(${SWING_ENVS})
  # shellcheck disable=SC2206
  local pols=(${POLICIES})
  for env in "${envs[@]}"; do
    for policy in "${pols[@]}"; do
      if ! resolve_swingby_pair "${env}" "${policy}" >/dev/null; then
        echo "${env}/${policy}"
      fi
    done
  done
}

verify_swingby_schema() {
  # Soft check: if numpy readable, require dataset_schema == swingby.
  local train="$1"
  python - "$train" <<'PY' 2>/dev/null || return 0
import sys
import numpy as np
path = sys.argv[1]
try:
    with np.load(path, allow_pickle=True) as z:
        schema = z["dataset_schema"].item() if "dataset_schema" in z.files else None
except Exception:
    sys.exit(0)
if schema is not None and schema != "swingby":
    print(f"bad_schema {path} schema={schema!r}", flush=True)
    sys.exit(1)
sys.exit(0)
PY
}

wait_for_swingby_datasets() {
  log "waiting for canonical swingby datasets (size=${SIZE})..."
  while true; do
    local missing
    missing="$(missing_swingby_combos || true)"
    if [[ -z "${missing}" ]]; then
      local env policy pair train ok=1
      # shellcheck disable=SC2206
      local envs=(${SWING_ENVS})
      # shellcheck disable=SC2206
      local pols=(${POLICIES})
      for env in "${envs[@]}"; do
        for policy in "${pols[@]}"; do
          pair="$(resolve_swingby_pair "${env}" "${policy}")"
          train="${pair%%|*}"
          if ! verify_swingby_schema "${train}"; then
            log "  schema check failed for ${train}; still waiting"
            ok=0
          else
            log "  ready ${train}"
          fi
        done
      done
      if [[ "${ok}" -eq 1 ]]; then
        log "all swingby train+val NPZs present"
        break
      fi
    else
      local n
      n="$(printf '%s\n' "${missing}" | grep -c . || true)"
      log "  missing ${n} env/policy combos (e.g. $(printf '%s\n' "${missing}" | head -1))"
    fi
    sleep "${POLL_SEC}"
  done
}

run_lap_phase() {
  log "=== PHASE lap_retrain (new *_lap_* datasets) ==="
  archive_old_lap_ckpts
  GPUS="${GPUS}" PACK="${PACK}" STEPS="${STEPS}" SEED="${SEED}" SIZE="${SIZE}" \
    AGENTS="${AGENTS}" POLICIES="${POLICIES}" \
    CAR_ENVS="${CAR_ENVS}" CAR_TASKS='lap_2p' \
    SWING_ENVS= \
    bash scripts/run_matrix.sh
  log "=== PHASE lap_retrain DONE ==="
}

run_swingby_phase() {
  log "=== PHASE swingby (ckpt *_swingby_*) ==="
  GPUS="${GPUS}" PACK="${PACK}" STEPS="${STEPS}" SEED="${SEED}" SIZE="${SIZE}" \
    AGENTS="${AGENTS}" POLICIES="${POLICIES}" \
    CAR_ENVS= CAR_TASKS= \
    SWING_ENVS="${SWING_ENVS}" \
    SWING_DATASET_MODE=swingby \
    bash scripts/run_matrix.sh
  log "=== PHASE swingby DONE ==="
}

log "queue_lap_then_swingby starting (gpus=${GPUS} pack=${PACK} skip_lap=${SKIP_LAP})"

if [[ "${SKIP_LAP}" != "1" ]]; then
  run_lap_phase
else
  log "SKIP_LAP=1 — skipping lap phase"
fi

wait_for_swingby_datasets
run_swingby_phase
log "=== QUEUE_ALL_DONE ==="
