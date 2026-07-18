#!/usr/bin/env bash
# Mirror finished checkpoint renders into a flat gallery under renders/.
#
#   bash scripts/collect_renders.sh
#
# Layout:
#   renders/<run_tag>/env/taskN.mp4
#   renders/<run_tag>/overlay/taskN.mp4
# Uses relative symlinks so videos stay next to checkpoints without copying.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-renders}"
mkdir -p "${OUT}"

linked=0
skipped=0

while IFS= read -r -d '' mp4; do
  # checkpoints/car_race/<tag>/renders/env/task1.mp4
  rel="${mp4#checkpoints/}"
  # car_race/<tag>/renders/...
  kind="${rel%%/*}"
  rest="${rel#*/}"
  tag="${rest%%/renders/*}"
  suffix="${rest#*/renders/}"
  dest="${OUT}/${tag}/${suffix}"
  mkdir -p "$(dirname "${dest}")"
  if [[ -L "${dest}" || -f "${dest}" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  # relative link from dest dir back to the mp4
  dest_dir="$(dirname "${dest}")"
  # depth: renders/<tag>/env -> ../../../checkpoints/...
  target_rel="$(realpath --relative-to="${dest_dir}" "${mp4}")"
  ln -s "${target_rel}" "${dest}"
  linked=$((linked + 1))
done < <(find checkpoints -path '*/renders/*/*.mp4' -print0 2>/dev/null)

echo "collect_renders linked=${linked} skipped=${skipped} out=${OUT}"
