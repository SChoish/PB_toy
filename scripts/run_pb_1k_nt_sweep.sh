#!/usr/bin/env bash
# Parallel NT sweep for PB_toy 1k final PBG+PBF checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /home/svcho/anaconda3/etc/profile.d/conda.sh
conda activate offrl

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.08}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

AGENTS="${AGENTS:-pbg,pbf}"
NS="${NS:-1,2,4,8,16,32}"
TS="${TS:-0,0.25,0.5,1.0}"
EPISODES="${EPISODES:-25}"
WORKERS="${WORKERS:-2}"

LOG_DIR="nohup_logs/pb_1k_nt_sweep"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER="$LOG_DIR/master_${STAMP}.log"

echo "Launching parallel NT sweep agents=$AGENTS workers=$WORKERS ns=$NS ts=$TS -> $MASTER"
nohup python -u scripts/eval_pb_1k_nt_sweep.py \
  --apply \
  --agents "$AGENTS" \
  --ns "$NS" \
  --ts "$TS" \
  --episodes "$EPISODES" \
  --workers "$WORKERS" \
  --logs-root /home/svcho/PB_logs \
  >"$MASTER" 2>&1 &
echo "pid=$!"
echo "$MASTER"
