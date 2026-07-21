#!/usr/bin/env bash
# Remaining anti_grav 10k K=10 h_a=2 jobs (svcho handoff remainder).
# NOTE: cgroup pids.max≈8192 — do not pack >~6–7 JAX trains with PathBridger.
# Failed jobs (pthread/ptxas): scripts/retry_anti_grav_remain_after_slots.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/ext_csv/miniconda3/envs/pb_toy/bin/python}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate pb_toy

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.08}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

STEPS=10000
EVAL_EVERY=2000
LOG_EVERY=500
K=10
HA=2
SIZE=10k
SEED=0
LOGDIR="$ROOT/nohup_logs/anti_grav_remain_10k_k10"
mkdir -p "$LOGDIR" checkpoints/car_race
MASTER="$LOGDIR/master.log"
: >"$MASTER"
ts() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

# agent|env|task|policy|tag
JOBS=(
  "pbf|car_race_anti_grav|lap_2p|random|car_race_anti_grav_lap_2p_pbf_random_s0_k10_ha2_10k"
  "tr_hiql|car_race_anti_grav|lap_4p|noisy|car_race_anti_grav_lap_4p_tr_hiql_noisy_s0_k10_10k"
  "pbf|car_race_anti_grav|lap_4p|random|car_race_anti_grav_lap_4p_pbf_random_s0_k10_ha2_10k"
  "pbf|car_race_anti_grav|lap_8p|noisy|car_race_anti_grav_lap_8p_pbf_noisy_s0_k10_ha2_10k"
  "trl|car_race_anti_grav|lap_8p|noisy|car_race_anti_grav_lap_8p_trl_noisy_s0_k10_10k"
  "hiql|car_race_anti_grav|lap_8p|random|car_race_anti_grav_lap_8p_hiql_random_s0_k10_10k"
  "tr_hiql|car_race_anti_grav|lap_8p|random|car_race_anti_grav_lap_8p_tr_hiql_random_s0_k10_10k"
  "pbg|car_race_anti_grav|lap_8p|random|car_race_anti_grav_lap_8p_pbg_random_s0_k10_ha2_10k"
  "pbf|car_race_anti_grav|lap_8p|random|car_race_anti_grav_lap_8p_pbf_random_s0_k10_ha2_10k"
  "trl|car_race_anti_grav|lap_8p|random|car_race_anti_grav_lap_8p_trl_random_s0_k10_10k"
)

log "START remain anti_grav jobs=${#JOBS[@]} steps=$STEPS K=$K h_a=$HA size=$SIZE"
idx=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r agent env task policy tag <<<"$spec"
  dataset="$ROOT/car_race/datasets/${env}_lap_${policy}_${SIZE}.npz"
  ckpt="$ROOT/checkpoints/car_race/${tag}"
  logf="$LOGDIR/${tag}.log"
  gpu=$((idx % 2))
  idx=$((idx + 1))

  if [[ ! -f "$dataset" ]]; then
    log "MISSING_DATA $tag ($dataset)"
    continue
  fi
  if [[ -f "$ckpt/step_${STEPS}.msgpack" && -f "$ckpt/step_${STEPS}.json" ]]; then
    log "SKIP_DONE $tag"
    continue
  fi

  log "LAUNCH gpu=$gpu $tag"
  mkdir -p "$ckpt"
  (
    sg nvidia -c "export CUDA_VISIBLE_DEVICES=$gpu \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.08} \
      JAX_PLATFORMS=cuda \
      PYTHONPATH='$ROOT'\${PYTHONPATH:+:\$PYTHONPATH} \
      PYTHONUNBUFFERED=1 \
      WANDB_MODE=offline; \
      exec \"$PY\" -u -m car_race.train \
      --env \"$env\" --agent \"$agent\" --task \"$task\" \
      --dataset \"$dataset\" --dataset-size \"$SIZE\" \
      --steps $STEPS --seed $SEED \
      --eval-every $EVAL_EVERY --log-every $LOG_EVERY \
      --num-eval-envs 25 --subgoal-steps $K \
      --action-chunk-horizon $HA --checkpoint-dir \"$ckpt\" \
      --render-dir \"$ckpt/renders\""
  ) >"$logf" 2>&1 &
  sleep 3
done

wait
log "=== ANTI_GRAV_REMAIN_10K_K10_DONE ==="
