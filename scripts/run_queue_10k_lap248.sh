#!/usr/bin/env bash
# CarRace 10k matrix after 100k queue:
#   ice/grav/anti_grav × lap_2p/4p/8p × noisy/random × {hiql,tr_hiql,pbg,pbf,trl}
#   = 90 jobs.  SIZE=10k STEPS=10000 K=10 h_a=2 seed=0
# Includes the 10 anti_grav remainders + 3 svcho handoff resumes (auto-resume).
#
# Usage:
#   DRY_RUN=1 bash scripts/run_queue_10k_lap248.sh
#   bash scripts/run_queue_10k_lap248.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate pb_toy
fi

PY="${PY:-/home/ext_csv/miniconda3/envs/pb_toy/bin/python}"
GPUS="${GPUS:-0,1}"
PACK="${PACK:-8}"
STEPS="${STEPS:-10000}"
SEED="${SEED:-0}"
SIZE="${SIZE:-10k}"
K="${K:-10}"
HA="${HA:-2}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-500}"
DRY_RUN="${DRY_RUN:-0}"
STAGGER_SEC="${STAGGER_SEC:-3}"
MEM_FRACTION="${MEM_FRACTION:-0.08}"
MIN_PID_HEADROOM="${MIN_PID_HEADROOM:-900}"
# Optional: rsync/copy root containing the 3 handoff tags (step_4000*).
HANDOFF_SRC="${HANDOFF_SRC:-}"

LOGDIR="$ROOT/nohup_logs/queue_10k_lap248"
mkdir -p "$LOGDIR" checkpoints/car_race
MASTER="$LOGDIR/master.log"
ts() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

AGENTS=(hiql tr_hiql pbg pbf trl)
CAR_ENVS=(car_race_ice car_race_grav car_race_anti_grav)
CAR_TASKS=(lap_2p lap_4p lap_8p)
CAR_POLICIES=(noisy random)

# Prefer resume from @4000 rather than training from scratch.
HANDOFF_TAGS=(
  car_race_anti_grav_lap_4p_pbg_noisy_s0_k10_ha2_10k
  car_race_anti_grav_lap_4p_pbf_noisy_s0_k10_ha2_10k
  car_race_anti_grav_lap_4p_pbg_random_s0_k10_ha2_10k
)

import_handoff() {
  local tag src dest
  if [[ -z "$HANDOFF_SRC" ]]; then
    return 0
  fi
  if [[ ! -d "$HANDOFF_SRC" ]]; then
    log "HANDOFF_SRC missing: $HANDOFF_SRC"
    return 0
  fi
  for tag in "${HANDOFF_TAGS[@]}"; do
    src="$HANDOFF_SRC/$tag"
    dest="$ROOT/checkpoints/car_race/$tag"
    if [[ ! -d "$src" ]]; then
      log "HANDOFF_MISS $tag (no $src)"
      continue
    fi
    mkdir -p "$dest"
    # Copy only if dest has nothing newer; never overwrite a finished run.
    if [[ -f "$dest/step_${STEPS}.msgpack" ]]; then
      log "HANDOFF_SKIP_DONE $tag"
      continue
    fi
    log "HANDOFF_IMPORT $tag <- $src"
    if [[ "$DRY_RUN" == "1" ]]; then
      continue
    fi
    rsync -a --ignore-existing "$src/" "$dest/"
  done
}

build_jobs() {
  local agent env task policy prefix ckpt
  JOBS=()
  for env in "${CAR_ENVS[@]}"; do
    for task in "${CAR_TASKS[@]}"; do
      for policy in "${CAR_POLICIES[@]}"; do
        for agent in "${AGENTS[@]}"; do
          prefix="${env}_${task}"
          ckpt="checkpoints/car_race/${prefix}_${agent}_${policy}_s${SEED}_k${K}"
          if [[ "$agent" == pbg || "$agent" == pbf ]]; then
            ckpt="${ckpt}_ha${HA}"
          fi
          ckpt="${ckpt}_${SIZE}"
          JOBS+=("${env}|${task}|${agent}|${policy}|${ckpt}")
        done
      done
    done
  done
}

HAS_CAR_TRAIN=0
"$PY" -c "import car_race.train" 2>/dev/null && HAS_CAR_TRAIN=1

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_GPU_IDX=()

reap() {
  local i pid
  local -a keep_pids=() keep_g=()
  for i in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      keep_pids+=("${pid}")
      keep_g+=("${ACTIVE_GPU_IDX[$i]}")
    else
      wait "${pid}" 2>/dev/null || true
    fi
  done
  ACTIVE_PIDS=("${keep_pids[@]}")
  ACTIVE_GPU_IDX=("${keep_g[@]}")
}

_train_pids() {
  pgrep -f '/python -u -m car_race\.train' 2>/dev/null || true
}

declare -A GPU_UUID=()
init_gpu_uuids() {
  local idx uuid
  GPU_UUID=()
  while IFS=',' read -r idx uuid; do
    idx="${idx// /}"
    uuid="${uuid// /}"
    [[ -n "$idx" && -n "$uuid" ]] || continue
    GPU_UUID["$idx"]="$uuid"
  done < <(sg nvidia -c 'nvidia-smi --query-gpu=index,uuid --format=csv,noheader' 2>/dev/null || true)
}

count_on_gpu() {
  local gpu="$1" uuid count=0 pid
  uuid="${GPU_UUID[$gpu]:-}"
  if [[ -z "$uuid" ]]; then
    echo 0
    return
  fi
  local apps
  apps=$(sg nvidia -c 'nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader' 2>/dev/null || true)
  for pid in $(_train_pids); do
    if printf '%s\n' "$apps" | awk -F',' -v u="$uuid" -v p="$pid" '
      {gsub(/^ +| +$/,"",$1); gsub(/^ +| +$/,"",$2); if($1==u && $2==p) found=1}
      END{exit !found}'; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

pick_gpu() {
  local i best=-1 best_count=999 c
  if [[ -r /sys/fs/cgroup/pids.current && -r /sys/fs/cgroup/pids.max ]]; then
    local current maximum
    current=$(</sys/fs/cgroup/pids.current)
    maximum=$(</sys/fs/cgroup/pids.max)
    if [[ "$maximum" != "max" ]] \
      && (( maximum - current < MIN_PID_HEADROOM )); then
      echo -1
      return
    fi
  fi
  for i in "${!GPU_ARR[@]}"; do
    c=$(count_on_gpu "${GPU_ARR[$i]}")
    if [[ "$c" -lt "$PACK" && "$c" -lt "$best_count" ]]; then
      best=$i
      best_count=$c
    fi
  done
  echo "$best"
}

is_handoff_tag() {
  local tag="$1" t
  for t in "${HANDOFF_TAGS[@]}"; do
    [[ "$t" == "$tag" ]] && return 0
  done
  return 1
}

run_job() {
  local gpu="$1" env="$2" task="$3" agent="$4" policy="$5" ckpt="$6"
  local dataset tag logf
  dataset="$ROOT/car_race/datasets/${env}_lap_${policy}_${SIZE}.npz"
  tag="$(basename "$ckpt")"
  logf="$LOGDIR/${tag}.log"

  if [[ ! -f "$dataset" ]]; then
    log "SKIP_MISSING_DATA $tag ($dataset)"
    echo "${env}|${task}|${agent}|${policy}|${ckpt}|missing_data" >>"$LOGDIR/deferred.txt"
    return 0
  fi
  if [[ "$HAS_CAR_TRAIN" != "1" ]]; then
    log "SKIP_NO_TRAIN_MODULE $tag"
    echo "${env}|${task}|${agent}|${policy}|${ckpt}|no_train_module" >>"$LOGDIR/deferred.txt"
    return 0
  fi
  if [[ -f "$ckpt/step_${STEPS}.msgpack" && -f "$ckpt/step_${STEPS}.json" ]]; then
    log "SKIP_DONE $tag"
    return 0
  fi
  # Handoff cells: resume if local ckpt exists; otherwise train from 0.
  if is_handoff_tag "$tag" && [[ ! -f "$ckpt/step_4000.msgpack" && ! -f "$ckpt/step_2000.msgpack" ]]; then
    log "HANDOFF_FROM_ZERO $tag"
  fi
  if pgrep -f "checkpoint-dir ${ckpt}" >/dev/null 2>&1; then
    log "SKIP_RUNNING $tag"
    return 0
  fi

  log "LAUNCH gpu=$gpu $tag"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  mkdir -p "$ckpt" "$ckpt/renders"
  (
    sg nvidia -c "export CUDA_VISIBLE_DEVICES=$gpu \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
      JAX_PLATFORMS=cuda \
      PYTHONPATH='$ROOT' \
      PYTHONUNBUFFERED=1 \
      WANDB_MODE=offline; \
      exec \"$PY\" -u -m car_race.train \
      --env \"$env\" --agent \"$agent\" --task \"$task\" \
      --dataset \"$dataset\" --dataset-size \"$SIZE\" \
      --steps $STEPS --seed $SEED \
      --eval-every $EVAL_EVERY --log-every $LOG_EVERY \
      --num-eval-envs 25 --subgoal-steps $K \
      --action-chunk-horizon $HA \
      --checkpoint-dir \"$ckpt\" \
      --render-dir \"$ckpt/renders\""
  ) >"$logf" 2>&1 &
  ACTIVE_PIDS+=("$!")
  sleep "$STAGGER_SEC"
}

: >"$MASTER"
: >"$LOGDIR/deferred.txt"
import_handoff
build_jobs
init_gpu_uuids

log "QUEUE_10k_lap248 jobs=${#JOBS[@]} pack=$PACK gpus=${GPUS} steps=$STEPS K=$K h_a=$HA size=$SIZE seed=$SEED dry=$DRY_RUN"
log "matrix: envs=${#CAR_ENVS[@]} tasks=${#CAR_TASKS[@]} policies=${#CAR_POLICIES[@]} agents=${#AGENTS[@]}"
log "gpu_uuids: $(for g in "${GPU_ARR[@]}"; do echo -n "$g=${GPU_UUID[$g]:-?} "; done)"

if [[ "$DRY_RUN" != "1" ]]; then
  log "GPU preflight"
  sg nvidia -c \
    'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
fi

for line in "${JOBS[@]}"; do
  IFS='|' read -r env task agent policy ckpt <<<"$line"
  if [[ "$DRY_RUN" == "1" ]]; then
    idx=$(( ${#ACTIVE_PIDS[@]} % ${#GPU_ARR[@]} ))
    run_job "${GPU_ARR[$idx]}" "$env" "$task" "$agent" "$policy" "$ckpt"
    ACTIVE_PIDS+=("dry")
    ACTIVE_GPU_IDX+=("$idx")
    continue
  fi
  while true; do
    reap
    idx="$(pick_gpu)"
    if [[ "$idx" -ge 0 ]]; then
      break
    fi
    sleep 10
  done
  before=${#ACTIVE_PIDS[@]}
  run_job "${GPU_ARR[$idx]}" "$env" "$task" "$agent" "$policy" "$ckpt"
  after=${#ACTIVE_PIDS[@]}
  if [[ "$after" -gt "$before" ]]; then
    ACTIVE_GPU_IDX+=("$idx")
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY_RUN done scheduled=${#JOBS[@]} deferred=$(wc -l <"$LOGDIR/deferred.txt")"
  exit 0
fi

log "All submitted; waiting for ${#ACTIVE_PIDS[@]} active"
while [[ ${#ACTIVE_PIDS[@]} -gt 0 ]]; do
  reap
  sleep 15
done
log "=== QUEUE_10k_lap248_DONE === deferred=$(wc -l <"$LOGDIR/deferred.txt") (see deferred.txt)"
