#!/usr/bin/env bash
# Render env/overlay videos for completed matrix checkpoints.
#
#   bash scripts/render_matrix.sh              # all ready ckpts missing renders
#   DRY_RUN=1 bash scripts/render_matrix.sh
#   GPUS=3,2 PACK=4 bash scripts/render_matrix.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate pb_toy
fi

GPUS="${GPUS:-3,2}"
PACK="${PACK:-4}"
STEPS="${STEPS:-50000}"
SEED="${SEED:-0}"
SIZE="${SIZE:-100k}"
DRY_RUN="${DRY_RUN:-0}"
SWING_DATASET_MODE="${SWING_DATASET_MODE:-swingby}"

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
export PATH="${SITE}/nvidia/cuda_nvcc/bin:${PATH}"
export LD_LIBRARY_PATH="${SITE}/nvidia/cudnn/lib:${SITE}/nvidia/cublas/lib:${SITE}/nvidia/cuda_runtime/lib:${SITE}/nvidia/cusolver/lib:${SITE}/nvidia/cusparse/lib:${SITE}/nvidia/cufft/lib:${SITE}/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
mkdir -p nohup_logs/render

mapfile -t JOBS < <(STEPS="${STEPS}" SEED="${SEED}" SWING_DATASET_MODE="${SWING_DATASET_MODE}" bash scripts/plan_matrix.sh | grep -v '^#')

need_render() {
  local ckpt="$1"
  local render_dir="${ckpt}/renders"
  [[ -f "${ckpt}/step_${STEPS}.msgpack" ]] || return 1
  [[ -f "${render_dir}/env/task5.mp4" && -f "${render_dir}/overlay/task5.mp4" ]] && return 1
  return 0
}

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

pick_gpu() {
  local -a counts
  local i idx best=-1 best_count=999
  for i in "${!GPU_ARR[@]}"; do counts[$i]=0; done
  for i in "${!ACTIVE_GPU_IDX[@]}"; do
    idx="${ACTIVE_GPU_IDX[$i]}"
    counts[$idx]=$(( counts[idx] + 1 ))
  done
  for i in "${!GPU_ARR[@]}"; do
    if [[ "${counts[$i]}" -lt "${PACK}" && "${counts[$i]}" -lt "${best_count}" ]]; then
      best=$i
      best_count=${counts[$i]}
    fi
  done
  echo "${best}"
}

queued=0
for line in "${JOBS[@]}"; do
  # plan_matrix: kind|env|task|agent|policy|checkpoint_dir
  IFS='|' read -r kind env task agent policy ckpt <<< "${line}"
  if ! need_render "${ckpt}"; then
    continue
  fi
  queued=$((queued + 1))
  if [[ "${kind}" == "car_race" ]]; then
    if [[ "${policy}" == "expert" ]]; then
      tag="${env}_${task}_${agent}_s${SEED}"
    else
      tag="${env}_${task}_${agent}_${policy}_s${SEED}"
    fi
    if [[ "${task}" == "navigation" ]]; then
      dataset="${ROOT}/car_race/datasets/${env}_${policy}_${SIZE}.npz"
    else
      dataset="${ROOT}/car_race/datasets/${env}_lap_${policy}_${SIZE}.npz"
    fi
  else
    if [[ "${SWING_DATASET_MODE}" == "swingby" ]]; then
      if [[ "${policy}" == "expert" ]]; then
        tag="${env}_swingby_${agent}_s${SEED}"
      else
        tag="${env}_swingby_${agent}_${policy}_s${SEED}"
      fi
      dataset="${ROOT}/swingby/datasets/${env}_swingby_${policy}_${SIZE}.npz"
    else
      if [[ "${policy}" == "expert" ]]; then
        tag="${env}_${agent}_s${SEED}"
      else
        tag="${env}_${agent}_${policy}_s${SEED}"
      fi
      dataset="${ROOT}/swingby/datasets/${env}_${policy}_${SIZE}.npz"
    fi
  fi
  render_dir="${ckpt}/renders"
  log="nohup_logs/render/${tag}.log"

  while true; do
    reap
    idx="$(pick_gpu)"
    if [[ "${idx}" -ge 0 ]]; then
      break
    fi
    sleep 5
  done
  gpu="${GPU_ARR[$idx]}"
  echo "RENDER gpu=${gpu} ${tag} policy=${policy} -> ${log}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    ACTIVE_PIDS+=("dry")
    ACTIVE_GPU_IDX+=("${idx}")
    continue
  fi

  if [[ ! -f "${dataset}" ]]; then
    echo "SKIP_MISSING_DATA ${tag} (${dataset})"
    continue
  fi

  if [[ "${kind}" == "car_race" ]]; then
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      unset XLA_PYTHON_CLIENT_MEM_FRACTION
      # Resume at STEPS (instant) then render.
      exec python -m car_race.train \
        --env "${env}" --agent "${agent}" --task "${task}" \
        --dataset "${dataset}" --dataset-size "${SIZE}" --steps "${STEPS}" --seed "${SEED}" \
        --eval-every 0 --log-every 100000 --num-eval-envs 1 \
        --checkpoint-dir "${ckpt}" \
        --render-dir "${render_dir}"
    ) >"${log}" 2>&1 &
  else
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      unset XLA_PYTHON_CLIENT_MEM_FRACTION
      exec python -m swingby.train \
        --env "${env}" --agent "${agent}" \
        --dataset "${dataset}" --dataset-size "${SIZE}" --steps "${STEPS}" --seed "${SEED}" \
        --eval-every 0 --log-every 100000 --num-eval-envs 1 \
        --checkpoint-dir "${ckpt}" \
        --render-dir "${render_dir}"
    ) >"${log}" 2>&1 &
  fi
  ACTIVE_PIDS+=("$!")
  ACTIVE_GPU_IDX+=("${idx}")
done

echo "queued_renders=${queued} dry_run=${DRY_RUN}"
if [[ "${DRY_RUN}" != "1" && ${#ACTIVE_PIDS[@]} -gt 0 ]]; then
  while [[ ${#ACTIVE_PIDS[@]} -gt 0 ]]; do
    before=${#ACTIVE_PIDS[@]}
    reap
    if [[ ${#ACTIVE_PIDS[@]} -lt "${before}" ]]; then
      bash scripts/collect_renders.sh || true
    fi
    sleep 8
  done
  bash scripts/collect_renders.sh || true
  echo "=== RENDER_MATRIX_DONE ==="
fi
