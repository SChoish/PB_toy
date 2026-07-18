#!/usr/bin/env bash
# Launch the planned PB_toy matrix with GPU packing.
#
#   DRY_RUN=1 bash scripts/run_matrix.sh          # print schedule
#   bash scripts/run_matrix.sh                    # actually train
#
# Env overrides: GPUS PACK STEPS SEED SIZE EVAL_EVERY LOG_EVERY
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate pb_toy
fi

GPUS="${GPUS:-3,2}"
PACK="${PACK:-8}"
STEPS="${STEPS:-50000}"
SEED="${SEED:-0}"
SIZE="${SIZE:-100k}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-500}"
DRY_RUN="${DRY_RUN:-0}"

# Prefer pip-bundled CUDA nvcc/ptxas over older system /usr/bin/ptxas.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
export PATH="${SITE}/nvidia/cuda_nvcc/bin:${PATH}"
export LD_LIBRARY_PATH="${SITE}/nvidia/cudnn/lib:${SITE}/nvidia/cublas/lib:${SITE}/nvidia/cuda_runtime/lib:${SITE}/nvidia/cusolver/lib:${SITE}/nvidia/cusparse/lib:${SITE}/nvidia/cufft/lib:${SITE}/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
N_SLOTS=$(( ${#GPU_ARR[@]} * PACK ))
mkdir -p nohup_logs/matrix checkpoints/car_race checkpoints/swingby

mapfile -t JOBS < <(STEPS="${STEPS}" SEED="${SEED}" bash scripts/plan_matrix.sh | grep -v '^#')
echo "jobs=${#JOBS[@]} slots=${N_SLOTS} (${#GPU_ARR[@]} GPUs × pack ${PACK}) steps=${STEPS} dry_run=${DRY_RUN}"

# Active PIDs and their assigned GPU index into GPU_ARR.
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_GPU_IDX=()

reap() {
  local i pid finished=0
  local -a keep_pids=() keep_g=()
  for i in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      keep_pids+=("${pid}")
      keep_g+=("${ACTIVE_GPU_IDX[$i]}")
    else
      wait "${pid}" 2>/dev/null || true
      finished=1
    fi
  done
  ACTIVE_PIDS=("${keep_pids[@]}")
  ACTIVE_GPU_IDX=("${keep_g[@]}")
  # Train already renders via --render-dir; refresh the flat gallery on exit.
  if [[ "${finished}" -eq 1 ]]; then
    bash scripts/collect_renders.sh >/dev/null 2>&1 || true
  fi
}

count_trains_on_gpu() {
  # Count live car_race/swingby trains whose CUDA_VISIBLE_DEVICES equals $1.
  local gpu="$1" pid count=0
  for pid in $(pgrep -f 'python -m (car_race|swingby)\.train' 2>/dev/null || true); do
    if tr '\0' '\n' <"/proc/${pid}/environ" 2>/dev/null \
      | grep -qx "CUDA_VISIBLE_DEVICES=${gpu}"; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

pick_gpu() {
  # Count running jobs per GPU (this orch + any already-running trains); return least-loaded under PACK.
  local -a counts
  local i idx best=-1 best_count=999
  for i in "${!GPU_ARR[@]}"; do
    counts[$i]=$(count_trains_on_gpu "${GPU_ARR[$i]}")
  done
  for i in "${!GPU_ARR[@]}"; do
    if [[ "${counts[$i]}" -lt "${PACK}" && "${counts[$i]}" -lt "${best_count}" ]]; then
      best=$i
      best_count=${counts[$i]}
    fi
  done
  echo "${best}"
}

run_job() {
  local gpu="$1" kind="$2" env="$3" task="$4" agent="$5" ckpt="$6"
  local tag log render_dir

  if [[ "${kind}" == "car_race" ]]; then
    tag="${env}_${task}_${agent}_s${SEED}"
  else
    tag="${env}_${agent}_s${SEED}"
  fi
  log="nohup_logs/matrix/${tag}.log"
  render_dir="${ckpt}/renders"

  # Already finished: skip if renders exist; otherwise render-only (instant resume).
  local render_only=0 eval_every="${EVAL_EVERY}" log_every="${LOG_EVERY}" n_eval=25
  if [[ -f "${ckpt}/step_${STEPS}.msgpack" && -f "${ckpt}/step_${STEPS}.json" ]]; then
    if [[ -f "${render_dir}/env/task5.mp4" && -f "${render_dir}/overlay/task5.mp4" ]]; then
      echo "SKIP_DONE ${tag}"
      return 0
    fi
    render_only=1
    eval_every=0
    log_every=100000
    n_eval=1
    echo "RENDER_ONLY ${tag} (ckpt done, videos missing)"
  fi
  # Skip if a live train is already writing this checkpoint dir.
  if pgrep -f "checkpoint-dir ${ckpt}" >/dev/null 2>&1; then
    echo "SKIP_RUNNING ${tag}"
    return 0
  fi

  echo "LAUNCH gpu=${gpu} ${tag} -> ${log}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  mkdir -p "${ckpt}"
  if [[ "${kind}" == "car_race" ]]; then
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      unset XLA_PYTHON_CLIENT_MEM_FRACTION
      exec python -m car_race.train \
        --env "${env}" --agent "${agent}" --task "${task}" \
        --dataset-size "${SIZE}" --steps "${STEPS}" --seed "${SEED}" \
        --eval-every "${eval_every}" --log-every "${log_every}" \
        --num-eval-envs "${n_eval}" \
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
        --dataset-size "${SIZE}" --steps "${STEPS}" --seed "${SEED}" \
        --eval-every "${eval_every}" --log-every "${log_every}" \
        --num-eval-envs "${n_eval}" \
        --checkpoint-dir "${ckpt}" \
        --render-dir "${render_dir}"
    ) >"${log}" 2>&1 &
  fi
  ACTIVE_PIDS+=("$!")
  # caller sets ACTIVE_GPU_IDX
}

for line in "${JOBS[@]}"; do
  IFS='|' read -r kind env task agent ckpt <<< "${line}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    # Round-robin assign for schedule preview.
    idx=$(( ${#ACTIVE_PIDS[@]} % ${#GPU_ARR[@]} ))
    gpu="${GPU_ARR[$idx]}"
    run_job "${gpu}" "${kind}" "${env}" "${task}" "${agent}" "${ckpt}"
    ACTIVE_PIDS+=("dry")
    ACTIVE_GPU_IDX+=("${idx}")
    continue
  fi

  while true; do
    reap
    idx="$(pick_gpu)"
    if [[ "${idx}" -ge 0 ]]; then
      break
    fi
    sleep 8
  done
  gpu="${GPU_ARR[$idx]}"
  before=${#ACTIVE_PIDS[@]}
  run_job "${gpu}" "${kind}" "${env}" "${task}" "${agent}" "${ckpt}"
  after=${#ACTIVE_PIDS[@]}
  if [[ "${after}" -gt "${before}" ]]; then
    ACTIVE_GPU_IDX+=("${idx}")
  fi
done

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "All jobs submitted (${#ACTIVE_PIDS[@]} still running); waiting..."
  while [[ ${#ACTIVE_PIDS[@]} -gt 0 ]]; do
    reap
    sleep 10
  done
  echo "=== MATRIX_DONE ==="
else
  echo "# dry-run finished (${#JOBS[@]} jobs scheduled)"
fi
