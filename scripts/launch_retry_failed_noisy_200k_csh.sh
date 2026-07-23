#!/usr/bin/env bash
# Relaunch unfinished/failed noisy→200k jobs only, respecting PID budget.
# Leaves healthy in-flight trainers alone (SKIP_RUNNING / MAX_GLOBAL_TRAINS).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/nohup_logs/queue_noisy_200k"
mkdir -p "$LOGDIR" "$ROOT/nohup_logs/pblogs_sync"

SEED="${SEED:-0}"
K="${K:-10}"
HA="${HA:-2}"
SIZE="${SIZE:-100k}"
STEPS="${STEPS:-200000}"
ONLY_FILE="${ONLY_FILE:-$LOGDIR/retry_only.txt}"

# Rebuild need list unless caller supplied ONLY_FILE already populated.
if [[ "${REFRESH_ONLY:-1}" == "1" ]]; then
  AGENTS=(hiql tr_hiql pbg pbf)
  : >"$ONLY_FILE"
  for env in car_race_ice car_race_grav car_race_anti_grav; do
    for task in lap_1p lap_2p lap_4p; do
      for agent in "${AGENTS[@]}"; do
        ckpt="checkpoints/car_race/${env}_${task}_${agent}_noisy_s${SEED}_k${K}"
        [[ "$agent" == pbg || "$agent" == pbf ]] && ckpt="${ckpt}_ha${HA}"
        ckpt="${ckpt}_${SIZE}"
        tag=$(basename "$ckpt")
        if [[ -f "$ckpt/step_${STEPS}.msgpack" && -f "$ckpt/step_${STEPS}.json" ]]; then
          continue
        fi
        if pgrep -f "checkpoint-dir ${ckpt}" >/dev/null 2>&1; then
          continue
        fi
        echo "$tag" >>"$ONLY_FILE"
      done
    done
  done
  for env in swingby_planet swingby_blackhole; do
    for agent in "${AGENTS[@]}"; do
      ckpt="checkpoints/swingby/${env}_swingby_${agent}_noisy_s${SEED}_k${K}"
      [[ "$agent" == pbg || "$agent" == pbf ]] && ckpt="${ckpt}_ha${HA}"
      ckpt="${ckpt}_${SIZE}"
      tag=$(basename "$ckpt")
      if [[ -f "$ckpt/step_${STEPS}.msgpack" && -f "$ckpt/step_${STEPS}.json" ]]; then
        continue
      fi
      if pgrep -f "checkpoint-dir ${ckpt}" >/dev/null 2>&1; then
        continue
      fi
      echo "$tag" >>"$ONLY_FILE"
    done
  done
  for agent in "${AGENTS[@]}"; do
    ckpt="checkpoints/car_parking/car_parking_${agent}_noisy_s${SEED}_k${K}"
    [[ "$agent" == pbg || "$agent" == pbf ]] && ckpt="${ckpt}_ha${HA}"
    ckpt="${ckpt}_${SIZE}"
    tag=$(basename "$ckpt")
    if [[ -f "$ckpt/step_${STEPS}.msgpack" && -f "$ckpt/step_${STEPS}.json" ]]; then
      continue
    fi
    if pgrep -f "checkpoint-dir ${ckpt}" >/dev/null 2>&1; then
      continue
    fi
    echo "$tag" >>"$ONLY_FILE"
  done
fi

n_only=$(grep -c . "$ONLY_FILE" || true)
echo "retry_only count=$n_only file=$ONLY_FILE"
if [[ "$n_only" -eq 0 ]]; then
  echo "nothing to relaunch"
  exit 0
fi

export PY="${PY:-/home/ext_csh/miniconda3/envs/pbtoy/bin/python}"
export GPUS="${GPUS:-0,1}"
export PACK="${PACK:-4}"
export STEPS SIZE K HA SEED
export MEM_FRACTION="${MEM_FRACTION:-0.03}"
export MIN_PID_HEADROOM="${MIN_PID_HEADROOM:-1500}"
export MAX_GLOBAL_TRAINS="${MAX_GLOBAL_TRAINS:-12}"
export STAGGER_SEC="${STAGGER_SEC:-15}"
export ONLY_FILE
export APPEND_MASTER=1
export PB_LOG_HOST="${PB_LOG_HOST:-csh_server}"
export PB_LOGS_ROOT="${PB_LOGS_ROOT:-/home/ext_csh/PB_logs}"
export PB_REPO="${PB_REPO:-/home/ext_csh/PB_logs}"
export PBLOGS_PUSH="${PBLOGS_PUSH:-1}"
export PBLOGS_WATCH="${PBLOGS_WATCH:-1}"
export PBLOGS_INTERVAL_SEC="${PBLOGS_INTERVAL_SEC:-600}"
export DRY_RUN="${DRY_RUN:-0}"
export NEXT_QUEUE="${NEXT_QUEUE:-skip}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

NV_LIBS="$(find /home/ext_csh/miniconda3/envs/pbtoy/lib/python3.11/site-packages/nvidia -type d -name lib -printf '%p:' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"

# Stop idle wait-only scheduler if it is the previous full queue.
if [[ -f "$LOGDIR/queue.pid" ]]; then
  old=$(cat "$LOGDIR/queue.pid")
  if kill -0 "$old" 2>/dev/null; then
    echo "stopping previous queue scheduler pid=$old (trainers keep running)"
    kill "$old" 2>/dev/null || true
    sleep 2
  fi
fi

echo "[$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')] === RETRY_FAILED start n=$n_only max_global=$MAX_GLOBAL_TRAINS headroom=$MIN_PID_HEADROOM ===" \
  >>"$LOGDIR/master.log"

nohup env \
  PY="$PY" GPUS="$GPUS" PACK="$PACK" STEPS="$STEPS" SIZE="$SIZE" K="$K" HA="$HA" SEED="$SEED" \
  MEM_FRACTION="$MEM_FRACTION" MIN_PID_HEADROOM="$MIN_PID_HEADROOM" MAX_GLOBAL_TRAINS="$MAX_GLOBAL_TRAINS" \
  STAGGER_SEC="$STAGGER_SEC" ONLY_FILE="$ONLY_FILE" APPEND_MASTER=1 \
  PB_LOG_HOST="$PB_LOG_HOST" PB_LOGS_ROOT="$PB_LOGS_ROOT" PB_REPO="$PB_REPO" \
  PBLOGS_PUSH="$PBLOGS_PUSH" PBLOGS_WATCH="$PBLOGS_WATCH" PBLOGS_INTERVAL_SEC="$PBLOGS_INTERVAL_SEC" \
  DRY_RUN="$DRY_RUN" NEXT_QUEUE="$NEXT_QUEUE" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  bash "$ROOT/scripts/run_queue_noisy_200k_csh.sh" \
  >"$LOGDIR/nohup_retry.out" 2>&1 </dev/null &
echo $! | tee "$LOGDIR/queue.pid"
echo "launched retry queue pid=$(cat "$LOGDIR/queue.pid") PACK=$PACK MAX_GLOBAL_TRAINS=$MAX_GLOBAL_TRAINS"
