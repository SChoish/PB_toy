#!/usr/bin/env bash
# Launch noisy→200k queue on csh_server at max PACK, with PB_logs sync/push.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/nohup_logs/queue_noisy_200k"
mkdir -p "$LOGDIR" "$ROOT/nohup_logs/pblogs_sync"

export PY="${PY:-/home/ext_csh/miniconda3/envs/pbtoy/bin/python}"
export GPUS="${GPUS:-0,1}"
export PACK="${PACK:-26}"
export STEPS="${STEPS:-200000}"
export SIZE="${SIZE:-100k}"
export K="${K:-10}"
export HA="${HA:-2}"
export SEED="${SEED:-0}"
export MEM_FRACTION="${MEM_FRACTION:-0.03}"
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

if [[ -f "$LOGDIR/master.log" ]]; then
  mv "$LOGDIR/master.log" "$LOGDIR/master_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S).log"
fi

nohup env \
  PY="$PY" GPUS="$GPUS" PACK="$PACK" STEPS="$STEPS" SIZE="$SIZE" K="$K" HA="$HA" SEED="$SEED" \
  MEM_FRACTION="$MEM_FRACTION" \
  PB_LOG_HOST="$PB_LOG_HOST" PB_LOGS_ROOT="$PB_LOGS_ROOT" PB_REPO="$PB_REPO" \
  PBLOGS_PUSH="$PBLOGS_PUSH" PBLOGS_WATCH="$PBLOGS_WATCH" PBLOGS_INTERVAL_SEC="$PBLOGS_INTERVAL_SEC" \
  DRY_RUN="$DRY_RUN" NEXT_QUEUE="$NEXT_QUEUE" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  bash "$ROOT/scripts/run_queue_noisy_200k_csh.sh" \
  >"$LOGDIR/nohup_csh_server.out" 2>&1 </dev/null &
echo $! | tee "$LOGDIR/queue.pid"
echo "launched queue pid=$(cat "$LOGDIR/queue.pid") PACK=$PACK host=$PB_LOG_HOST"
