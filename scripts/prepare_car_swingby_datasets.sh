#!/usr/bin/env bash
# Prepare car_race + swingby offline datasets for training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer the dedicated conda env when available.
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx pb_toy; then
    conda activate pb_toy
  fi
fi

SIZE="${SIZE:-100k}"
POLICY="${POLICY:-expert}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

extra=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  extra+=(--skip-existing)
fi

echo "=== car_race datasets (policy=${POLICY} size=${SIZE}) ==="
for env in car_race_plain car_race_grav car_race_anti_grav car_race_ice; do
  python -m car_race.generate_dataset \
    --env "${env}" --policy "${POLICY}" --size "${SIZE}" "${extra[@]}"
done

echo "=== swingby datasets (policy=${POLICY} size=${SIZE}) ==="
for env in swingby_planet swingby_blackhole; do
  python -m swingby.generate_dataset \
    --env "${env}" --policy "${POLICY}" --size "${SIZE}" "${extra[@]}"
done

echo "=== DONE ==="
ls -lh car_race/datasets/*_${POLICY}_${SIZE}.npz 2>/dev/null || true
ls -lh swingby/datasets/*_${POLICY}_${SIZE}.npz 2>/dev/null || true
