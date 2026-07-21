#!/usr/bin/env bash
# Generate CarParking datasets: policy ∈ {expert,noisy,mixture,random} × size ∈ {1k,10k,100k}
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/ext_csv/miniconda3/envs/pb_toy/bin/python}"
LOGDIR="${LOGDIR:-$ROOT/nohup_logs/car_parking_dataset}"
WORKERS="${WORKERS:-3}"
mkdir -p "$LOGDIR" car_parking/datasets
MASTER="$LOGDIR/master.log"
: >"$MASTER"
ts() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

POLICIES="${POLICIES:-expert noisy mixture random}"
JOBS=()
for policy in $POLICIES; do
  for size in 1k 10k 100k; do
    JOBS+=("${policy}|${size}")
  done
done

log "START parking dataset regen jobs=${#JOBS[@]} workers=$WORKERS"
running=0
pids=()
labels=()
failures=0

minimum_steps() {
  case "$1" in
    1k) echo 1000 ;;
    10k) echo 10000 ;;
    100k) echo 100000 ;;
    *) return 1 ;;
  esac
}

validate_pair() {
  local policy="$1" size="$2" out train_n val_n
  out="car_parking/datasets/car_parking_${policy}_${size}.npz"
  train_n="$(minimum_steps "$size")"
  val_n=$((train_n / 10))
  [[ -f "$out" && -f "${out%.npz}_val.npz" ]] || return 1
  "$PY" - "$out" "${out%.npz}_val.npz" "$train_n" "$val_n" <<'PY'
import sys
from car_parking.generate_dataset import validate_dataset_file

validate_dataset_file(sys.argv[1], minimum_steps=int(sys.argv[3]), require_schema=True)
validate_dataset_file(sys.argv[2], minimum_steps=int(sys.argv[4]), require_schema=True)
PY
}

wait_one() {
  local i pid label rc
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      rc=$?
      label="${labels[$i]}"
      if [[ "$rc" -eq 0 ]]; then
        IFS='_' read -r policy size <<<"$label"
        if validate_pair "$policy" "$size"; then
          log "DONE $label"
        else
          log "FAIL $label validation"
          failures=$((failures + 1))
        fi
      else
        log "FAIL $label rc=$rc"
        failures=$((failures + 1))
      fi
      unset 'pids[i]'
      unset 'labels[i]'
      # compact arrays
      pids=("${pids[@]}")
      labels=("${labels[@]}")
      running=$((running - 1))
      return 0
    fi
  done
  sleep 5
}

for spec in "${JOBS[@]}"; do
  IFS='|' read -r policy size <<<"$spec"
  out="car_parking/datasets/car_parking_${policy}_${size}.npz"
  if validate_pair "$policy" "$size"; then
    log "SKIP valid $policy $size"
    continue
  fi
  rm -f "$out" "${out%.npz}_val.npz"
  while [[ "$running" -ge "$WORKERS" ]]; do
    wait_one
  done
  label="${policy}_${size}"
  log "LAUNCH $label"
  (
    export CUDA_VISIBLE_DEVICES=""
    export JAX_PLATFORMS=cpu
    export PYTHONUNBUFFERED=1
    "$PY" -u -m car_parking.generate_dataset --policy "$policy" --size "$size"
  ) >"$LOGDIR/${label}.log" 2>&1 &
  pids+=("$!")
  labels+=("$label")
  running=$((running + 1))
  sleep 2
done

while [[ "$running" -gt 0 ]]; do
  wait_one
done

if [[ "$failures" -gt 0 ]]; then
  log "=== CAR_PARKING_FAILED count=$failures ==="
  exit 1
fi
log "=== CAR_PARKING_ALL_DONE ==="
ls -lah car_parking/datasets/
