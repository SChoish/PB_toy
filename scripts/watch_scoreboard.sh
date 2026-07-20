#!/usr/bin/env bash
# Periodically refresh scores.md + learning curves from checkpoints / logs.
#
#   bash scripts/watch_scoreboard.sh          # every 5m (default)
#   INTERVAL=120 bash scripts/watch_scoreboard.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
INTERVAL="${INTERVAL:-300}"
OUT="${OUT:-scores.md}"
mkdir -p nohup_logs

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate pb_toy 2>/dev/null || true
fi

log() { echo "$(date -Is) $*"; }

while true; do
  log "scoreboard + curves refresh"
  python scripts/make_scoreboard_and_learning_curves.py --out "${OUT}" \
    >> nohup_logs/scoreboard.log 2>&1 || log "refresh failed"
  sleep "${INTERVAL}"
done
