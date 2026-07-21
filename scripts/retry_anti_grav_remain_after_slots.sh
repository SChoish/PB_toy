#!/usr/bin/env bash
# Wait for cgroup PID headroom, then retry the 3 anti_grav jobs that died on pthread/ptxas.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/ext_csv/miniconda3/envs/pb_toy/bin/python}"
LOGDIR="$ROOT/nohup_logs/anti_grav_remain_10k_k10"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
ts() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

PIDS_MAX=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo 8192)
# Need ~400–600 spare PIDs for one JAX train process
NEED_SPARE="${NEED_SPARE:-1200}"
POLL_SEC="${POLL_SEC:-30}"

JOBS=(
  "pbf|car_race_anti_grav|lap_8p|noisy|car_race_anti_grav_lap_8p_pbf_noisy_s0_k10_ha2_10k|1"
  "tr_hiql|car_race_anti_grav|lap_8p|random|car_race_anti_grav_lap_8p_tr_hiql_random_s0_k10_10k|0"
  "trl|car_race_anti_grav|lap_8p|random|car_race_anti_grav_lap_8p_trl_random_s0_k10_10k|1"
)

spare() {
  local cur
  cur=$(cat /sys/fs/cgroup/pids.current 2>/dev/null || echo 0)
  echo $((PIDS_MAX - cur))
}

wait_slots() {
  while true; do
    local s trains
    s=$(spare)
    trains=$(pgrep -af 'python -u -m car_race.train' | grep -v pgrep | wc -l)
    log "WAIT spare_pids=$s need=$NEED_SPARE trains=$trains"
    if (( s >= NEED_SPARE )); then
      return 0
    fi
    sleep "$POLL_SEC"
  done
}

launch_one() {
  local agent=$1 env=$2 task=$3 policy=$4 tag=$5 gpu=$6
  local dataset="$ROOT/car_race/datasets/${env}_lap_${policy}_10k.npz"
  local ckpt="$ROOT/checkpoints/car_race/${tag}"
  local logf="$LOGDIR/${tag}.retry2.log"
  if [[ -f "$ckpt/step_10000.msgpack" && -f "$ckpt/step_10000.json" ]]; then
    log "SKIP_DONE $tag"
    return 0
  fi
  mkdir -p "$ckpt"
  log "RETRY2 gpu=$gpu $tag (spare=$(spare))"
  (
    sg nvidia -c "export CUDA_VISIBLE_DEVICES=$gpu \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.08 \
      JAX_PLATFORMS=cuda \
      XLA_FLAGS='--xla_gpu_autotune_level=1' \
      PYTHONPATH='$ROOT' PYTHONUNBUFFERED=1 WANDB_MODE=offline; \
      exec \"$PY\" -u -m car_race.train \
      --env \"$env\" --agent \"$agent\" --task \"$task\" \
      --dataset \"$dataset\" --dataset-size 10k \
      --steps 10000 --seed 0 \
      --eval-every 2000 --log-every 500 \
      --num-eval-envs 25 --subgoal-steps 10 \
      --action-chunk-horizon 2 --checkpoint-dir \"$ckpt\" \
      --render-dir \"$ckpt/renders\""
  ) >"$logf" 2>&1 &
  local pid=$!
  sleep 45
  if ! kill -0 "$pid" 2>/dev/null && ! pgrep -af "checkpoint-dir .*$tag" | grep -q car_race.train; then
    log "FAIL_EARLY $tag — see $logf"
    rg -n 'Traceback|Failed|LLVM ERROR|Check failed|step=' "$logf" | head -8 | tee -a "$MASTER" || true
    return 1
  fi
  if rg -q 'Traceback|Failed to launch|LLVM ERROR|Check failed' "$logf"; then
    log "FAIL_LOG $tag — see $logf"
    return 1
  fi
  log "OK_STARTED $tag"
  return 0
}

log "START retry2 queue jobs=${#JOBS[@]} need_spare=$NEED_SPARE"
for spec in "${JOBS[@]}"; do
  IFS='|' read -r agent env task policy tag gpu <<<"$spec"
  wait_slots
  launch_one "$agent" "$env" "$task" "$policy" "$tag" "$gpu" || {
    log "Will re-wait and retry once more: $tag"
    sleep 60
    wait_slots
    launch_one "$agent" "$env" "$task" "$policy" "$tag" "$gpu" || log "GIVE_UP $tag"
  }
  # let process settle / free compile threads before next
  sleep 60
done
log "=== RETRY2_QUEUE_DONE ==="
