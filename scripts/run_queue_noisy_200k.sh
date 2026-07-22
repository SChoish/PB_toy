#!/usr/bin/env bash
# Noisy-only resume queue → 200k steps (dataset SIZE still 100k).
#   5 agents × (CarRace ice/grav/anti_grav × lap_1p/2p/4p × noisy
#              + SwingBy planet/blackhole × noisy
#              + CarParking noisy)
#   = 60 jobs.  Same ckpt dirs as the 100k queue so partial runs resume.
#
# Usage:
#   DRY_RUN=1 bash scripts/run_queue_noisy_200k.sh
#   PACK=8 bash scripts/run_queue_noisy_200k.sh
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
STEPS="${STEPS:-200000}"
SEED="${SEED:-0}"
SIZE="${SIZE:-100k}"
K="${K:-10}"
HA="${HA:-2}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-500}"
DRY_RUN="${DRY_RUN:-0}"
STAGGER_SEC="${STAGGER_SEC:-3}"
MEM_FRACTION="${MEM_FRACTION:-0.03}"
MIN_PID_HEADROOM="${MIN_PID_HEADROOM:-900}"
PARK_DATA_WAIT_SEC="${PARK_DATA_WAIT_SEC:-21600}"
NEXT_QUEUE="${NEXT_QUEUE:-skip}"

LOGDIR="$ROOT/nohup_logs/queue_noisy_200k"
mkdir -p "$LOGDIR" checkpoints/car_race checkpoints/swingby checkpoints/car_parking
MASTER="$LOGDIR/master.log"
ts() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

AGENTS=(hiql tr_hiql pbg pbf trl)
CAR_ENVS=(car_race_ice car_race_grav car_race_anti_grav)
CAR_TASKS=(lap_1p lap_2p lap_4p)
CAR_POLICIES=(noisy)
SWING_ENVS=(swingby_planet swingby_blackhole)
SWING_POLICIES=(noisy)
PARK_POLICIES=(noisy)

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
          JOBS+=("car_race|${env}|${task}|${agent}|${policy}|${ckpt}")
        done
      done
    done
  done
  for env in "${SWING_ENVS[@]}"; do
    for policy in "${SWING_POLICIES[@]}"; do
      for agent in "${AGENTS[@]}"; do
        prefix="${env}_swingby"
        ckpt="checkpoints/swingby/${prefix}_${agent}_${policy}_s${SEED}_k${K}"
        if [[ "$agent" == pbg || "$agent" == pbf ]]; then
          ckpt="${ckpt}_ha${HA}"
        fi
        ckpt="${ckpt}_${SIZE}"
        JOBS+=("swingby|${env}|-|${agent}|${policy}|${ckpt}")
      done
    done
  done
  for policy in "${PARK_POLICIES[@]}"; do
    for agent in "${AGENTS[@]}"; do
      ckpt="checkpoints/car_parking/car_parking_${agent}_${policy}_s${SEED}_k${K}"
      if [[ "$agent" == pbg || "$agent" == pbf ]]; then
        ckpt="${ckpt}_ha${HA}"
      fi
      ckpt="${ckpt}_${SIZE}"
      JOBS+=("car_parking|car_parking|-|${agent}|${policy}|${ckpt}")
    done
  done
}

HAS_CAR_TRAIN=0
HAS_SWING_TRAIN=0
HAS_PARK_TRAIN=0
"$PY" -c "import car_race.train" 2>/dev/null && HAS_CAR_TRAIN=1
"$PY" -c "import swingby.train" 2>/dev/null && HAS_SWING_TRAIN=1
"$PY" -c "import car_parking.train" 2>/dev/null && HAS_PARK_TRAIN=1

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
  pgrep -f '/python -u -m (car_race|swingby|car_parking)\.train' 2>/dev/null || true
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

dataset_for() {
  local kind="$1" env="$2" task="$3" policy="$4"
  case "$kind" in
    car_race)
      if [[ "$task" == navigation ]]; then
        echo "$ROOT/car_race/datasets/${env}_${policy}_${SIZE}.npz"
      else
        echo "$ROOT/car_race/datasets/${env}_lap_${policy}_${SIZE}.npz"
      fi
      ;;
    swingby)
      echo "$ROOT/swingby/datasets/${env}_swingby_${policy}_${SIZE}.npz"
      ;;
    car_parking)
      echo "$ROOT/car_parking/datasets/car_parking_${policy}_${SIZE}.npz"
      ;;
  esac
}

run_job() {
  local gpu="$1" kind="$2" env="$3" task="$4" agent="$5" policy="$6" ckpt="$7"
  local dataset tag logf module has
  dataset="$(dataset_for "$kind" "$env" "$task" "$policy")"
  tag="$(basename "$ckpt")"
  logf="$LOGDIR/${tag}.log"

  if [[ "$kind" == "car_parking" && "$DRY_RUN" != "1" ]]; then
    local waited=0
    while [[ ! -f "$dataset" && "$waited" -lt "$PARK_DATA_WAIT_SEC" ]]; do
      if (( waited % 600 == 0 )); then
        log "WAIT_DATA $tag waited=${waited}s ($dataset)"
      fi
      sleep 60
      waited=$((waited + 60))
    done
  fi
  if [[ ! -f "$dataset" ]]; then
    log "SKIP_MISSING_DATA $tag ($dataset)"
    echo "$kind|$env|$task|$agent|$policy|$ckpt|missing_data" >>"$LOGDIR/deferred.txt"
    return 0
  fi

  case "$kind" in
    car_race) module=car_race.train; has=$HAS_CAR_TRAIN ;;
    swingby) module=swingby.train; has=$HAS_SWING_TRAIN ;;
    car_parking) module=car_parking.train; has=$HAS_PARK_TRAIN ;;
  esac
  if [[ "$has" != "1" ]]; then
    log "SKIP_NO_TRAIN_MODULE $tag ($module) — deferred"
    echo "$kind|$env|$task|$agent|$policy|$ckpt|no_train_module" >>"$LOGDIR/deferred.txt"
    return 0
  fi

  if [[ -f "$ckpt/step_${STEPS}.msgpack" && -f "$ckpt/step_${STEPS}.json" ]]; then
    log "SKIP_DONE $tag"
    return 0
  fi
  if pgrep -f "checkpoint-dir ${ckpt}" >/dev/null 2>&1; then
    log "SKIP_RUNNING $tag"
    return 0
  fi

  # Resume hint for logs (train.py resumes automatically from latest < STEPS).
  local latest=0
  if [[ -d "$ckpt" ]]; then
    latest=$(
      find "$ckpt" -maxdepth 1 -name 'step_*.json' -printf '%f\n' 2>/dev/null \
        | sed -n 's/^step_\([0-9]*\)\.json$/\1/p' \
        | sort -n | tail -1
    )
    latest="${latest:-0}"
  fi
  if [[ "$latest" -gt 0 ]]; then
    log "LAUNCH gpu=$gpu $tag resume_from=${latest} → ${STEPS}"
  else
    log "LAUNCH gpu=$gpu $tag from_scratch → ${STEPS}"
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  mkdir -p "$ckpt" "$ckpt/renders"
  local task_flag=""
  if [[ "$kind" == "car_race" ]]; then
    task_flag="--task $task"
  fi
  (
    # shellcheck disable=SC2086
    sg nvidia -c "export CUDA_VISIBLE_DEVICES=$gpu \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
      JAX_PLATFORMS=cuda \
      PYTHONPATH='$ROOT' \
      PYTHONUNBUFFERED=1 \
      WANDB_MODE=offline; \
      exec \"$PY\" -u -m $module \
      --env \"$env\" --agent \"$agent\" $task_flag \
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
build_jobs
init_gpu_uuids

log "QUEUE_noisy_200k jobs=${#JOBS[@]} pack=$PACK gpus=${GPUS} steps=$STEPS K=$K h_a=$HA size=$SIZE seed=$SEED dry=$DRY_RUN"
log "breakdown: car=$((3*3*1*5))=45 swing=$((2*1*5))=10 park=$((1*5))=5 total=${#JOBS[@]}"
log "gpu_uuids: $(for g in "${GPU_ARR[@]}"; do echo -n "$g=${GPU_UUID[$g]:-?} "; done)"

if [[ "$DRY_RUN" != "1" && "${PBLOGS_WATCH:-1}" == "1" ]]; then
  SYNC_PID_FILE="$ROOT/nohup_logs/pblogs_sync/watcher.pid"
  mkdir -p "$(dirname "$SYNC_PID_FILE")"
  if [[ -f "$SYNC_PID_FILE" ]] && kill -0 "$(cat "$SYNC_PID_FILE")" 2>/dev/null; then
    log "PBLOGS_WATCH already running pid=$(cat "$SYNC_PID_FILE")"
  else
    nohup bash -c "WATCH=1 PUSH=${PBLOGS_PUSH:-1} INTERVAL_SEC=${PBLOGS_INTERVAL_SEC:-600} \
      bash '$ROOT/scripts/sync_pb_toy_to_pblogs.sh'" \
      >"$ROOT/nohup_logs/pblogs_sync/watcher.out" 2>&1 </dev/null &
    echo $! >"$SYNC_PID_FILE"
    log "PBLOGS_WATCH started pid=$! interval=${PBLOGS_INTERVAL_SEC:-600}s push=${PBLOGS_PUSH:-1}"
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  log "GPU preflight"
  sg nvidia -c \
    'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
fi

for line in "${JOBS[@]}"; do
  IFS='|' read -r kind env task agent policy ckpt <<<"$line"
  if [[ "$DRY_RUN" == "1" ]]; then
    idx=$(( ${#ACTIVE_PIDS[@]} % ${#GPU_ARR[@]} ))
    run_job "${GPU_ARR[$idx]}" "$kind" "$env" "$task" "$agent" "$policy" "$ckpt"
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
  run_job "${GPU_ARR[$idx]}" "$kind" "$env" "$task" "$agent" "$policy" "$ckpt"
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
log "=== QUEUE_noisy_200k_DONE === deferred=$(wc -l <"$LOGDIR/deferred.txt") (see deferred.txt)"

if [[ "${PBLOGS_SYNC:-1}" == "1" ]]; then
  log "PBLOGS_SYNC once"
  PUSH="${PBLOGS_PUSH:-1}" bash "$ROOT/scripts/sync_pb_toy_to_pblogs.sh" \
    >>"$LOGDIR/pblogs_sync.log" 2>&1 || log "WARN pblogs sync failed"
fi

if [[ "$NEXT_QUEUE" == "0" || "$NEXT_QUEUE" == "skip" ]]; then
  log "NEXT_QUEUE skipped"
elif [[ -n "$NEXT_QUEUE" ]]; then
  log "NEXT_QUEUE: $NEXT_QUEUE"
  eval "$NEXT_QUEUE"
  log "NEXT_QUEUE finished"
fi
