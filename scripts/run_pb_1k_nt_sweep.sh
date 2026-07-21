#!/usr/bin/env bash
# Parallel NT sweep for PB_toy 1k final PBG+PBF checkpoints.
# Host-local defaults: pb_toy conda, /home/ext_csv/PB_logs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-pb_toy}"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.08}"
# Default CPU (user note); set JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 for GPU.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
if [[ "${JAX_PLATFORMS}" == "cpu" ]]; then
  unset CUDA_VISIBLE_DEVICES || true
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

AGENTS="${AGENTS:-pbg,pbf}"
NS="${NS:-1,2,4,8,16,32}"
TS="${TS:-0,0.25,0.5,1.0}"
EPISODES="${EPISODES:-25}"
WORKERS="${WORKERS:-2}"
LOGS_ROOT="${LOGS_ROOT:-/home/ext_csv/PB_logs}"
PY="${PY:-/home/ext_csv/miniconda3/envs/pb_toy/bin/python}"

LOG_DIR="nohup_logs/pb_1k_nt_sweep"
mkdir -p "$LOG_DIR" "$LOGS_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER="$LOG_DIR/master_${STAMP}.log"

echo "Launching parallel NT sweep agents=$AGENTS workers=$WORKERS ns=$NS ts=$TS logs-root=$LOGS_ROOT jax=$JAX_PLATFORMS -> $MASTER"
nohup "$PY" -u scripts/eval_pb_1k_nt_sweep.py \
  --apply \
  --agents "$AGENTS" \
  --ns "$NS" \
  --ts "$TS" \
  --episodes "$EPISODES" \
  --workers "$WORKERS" \
  --logs-root "$LOGS_ROOT" \
  >"$MASTER" 2>&1 &
echo "pid=$!"
echo "$MASTER"
