#!/usr/bin/env bash
# Run the 120-job PB_toy 10k matrix with K=10 and PB h_a=2.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /home/svcho/anaconda3/etc/profile.d/conda.sh
conda activate offrl

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.25}"

PACK="${PACK:-6}"
STEPS="${STEPS:-10000}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-500}"
K="${SUBGOAL_STEPS:-10}"
HA="${ACTION_CHUNK_HORIZON:-2}"
SIZE=10k
SEED=0
# Distance-weight power for PB/TR-HIQL, and TRL lam (same role). Current code defaults to 0.
# Set this explicitly when running a different code/config so log names remain honest.
WEIGHT_LABEL="${WEIGHT_LABEL:-w0}"
if [[ "$WEIGHT_LABEL" != "w0" && "$WEIGHT_LABEL" != "w1" ]]; then
  echo "WEIGHT_LABEL must be w0 or w1, got: $WEIGHT_LABEL" >&2
  exit 2
fi

export AGENTS="${AGENTS:-hiql tr_hiql pbg pbf trl}"
export CAR_ENVS="${CAR_ENVS:-car_race_ice car_race_grav car_race_anti_grav}"
export CAR_TASKS="${CAR_TASKS:-navigation lap_2p}"
export POLICIES="${POLICIES:-expert noisy random}"
# Empty SWING_ENVS skips swingby (use ${VAR-default}, not :- , so "" is preserved).
export SWING_ENVS="${SWING_ENVS-swingby_planet swingby_blackhole}"
export SWING_DATASET_MODE=swingby
export SIZE SEED STEPS EVAL_EVERY LOG_EVERY

LOG_DIR="nohup_logs/train_10k_k10"
mkdir -p "$LOG_DIR" checkpoints/car_race checkpoints/swingby
MASTER="$LOG_DIR/master_${WEIGHT_LABEL}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER") 2>&1

mapfile -t JOBS < <(bash scripts/plan_matrix.sh | grep -v '^#')
echo "jobs=${#JOBS[@]} pack=$PACK size=$SIZE steps=$STEPS K=$K h_a=$HA weight=$WEIGHT_LABEL"

wait_for_slot() {
  while [[ "$(jobs -pr | wc -l)" -ge "$PACK" ]]; do
    wait -n || true
  done
}

for line in "${JOBS[@]}"; do
  IFS='|' read -r kind env task agent policy base_ckpt <<< "$line"
  ckpt="${base_ckpt}_k${K}"
  if [[ "$agent" == "pbg" || "$agent" == "pbf" ]]; then
    ckpt="${ckpt}_ha${HA}"
  fi
  ckpt="${ckpt}_${SIZE}"
  tag="$(basename "$ckpt")"
  # hiql has no distance-weight/lam; leave its log name untagged.
  case "$agent" in
    tr_hiql|pbg|pbf|trl)
      log_weight="$WEIGHT_LABEL"
      log="$LOG_DIR/${tag}_${log_weight}.log"
      ;;
    *)
      log_weight=""
      log="$LOG_DIR/${tag}.log"
      ;;
  esac

  if [[ "$kind" == "car_race" ]]; then
    if [[ "$task" == "navigation" ]]; then
      dataset="car_race/datasets/${env}_${policy}_${SIZE}.npz"
    else
      dataset="car_race/datasets/${env}_lap_${policy}_${SIZE}.npz"
    fi
  else
    dataset="swingby/datasets/${env}_swingby_${policy}_${SIZE}.npz"
  fi

  if [[ ! -f "$dataset" ]]; then
    echo "MISSING $dataset"
    continue
  fi
  if [[ -f "$ckpt/step_${STEPS}.msgpack" && -f "$ckpt/step_${STEPS}.json" ]]; then
    echo "SKIP_DONE $tag"
    continue
  fi

  wait_for_slot
  if [[ -n "$log_weight" ]]; then
    echo "LAUNCH $tag log_weight=$log_weight"
  else
    echo "LAUNCH $tag"
  fi
  mkdir -p "$ckpt"
  if [[ "$kind" == "car_race" ]]; then
    (
      python -m car_race.train \
        --env "$env" --agent "$agent" --task "$task" \
        --dataset "$dataset" --dataset-size "$SIZE" \
        --steps "$STEPS" --seed "$SEED" \
        --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
        --num-eval-envs 25 --subgoal-steps "$K" \
        --action-chunk-horizon "$HA" --checkpoint-dir "$ckpt" \
        --render-dir "$ckpt/renders"
    ) >"$log" 2>&1 &
  else
    (
      python -m swingby.train \
        --env "$env" --agent "$agent" \
        --dataset "$dataset" --dataset-size "$SIZE" \
        --steps "$STEPS" --seed "$SEED" \
        --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
        --num-eval-envs 5 --subgoal-steps "$K" \
        --action-chunk-horizon "$HA" --checkpoint-dir "$ckpt" \
        --render-dir "$ckpt/renders"
    ) >"$log" 2>&1 &
  fi
done

wait
echo "=== TRAIN_10K_K10_DONE ==="
