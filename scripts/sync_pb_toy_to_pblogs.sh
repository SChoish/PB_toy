#!/usr/bin/env bash
# Sync PB_toy eval/train logs → PathBridger/logs (PB_logs) and optionally push.
#
# Usage:
#   bash scripts/sync_pb_toy_to_pblogs.sh              # export only
#   PUSH=1 bash scripts/sync_pb_toy_to_pblogs.sh       # export + github push
#   WATCH=1 INTERVAL_SEC=600 bash scripts/sync_pb_toy_to_pblogs.sh  # loop
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/ext_csv/miniconda3/envs/pb_toy/bin/python}"
LOGS_ROOT="${PB_LOGS_ROOT:-/home/ext_csv/PathBridger/logs}"
PB_REPO="${PB_REPO:-/home/ext_csv/PathBridger}"
export PB_LOG_HOST="${PB_LOG_HOST:-$(hostname -s)}"
export PB_LOGS_ROOT="$LOGS_ROOT"

PUSH="${PUSH:-0}"
WATCH="${WATCH:-0}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"
OVERWRITE="${OVERWRITE:-0}"

LOGDIR="$ROOT/nohup_logs/pblogs_sync"
mkdir -p "$LOGDIR"
ts() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*"; }

once() {
  local extra=()
  [[ "$OVERWRITE" == "1" ]] && extra+=(--overwrite)
  log "export → $LOGS_ROOT (host=$PB_LOG_HOST)"
  "$PY" -u scripts/export_results_to_pblogs.py --apply --k 10 --h-a 2 "${extra[@]}"
  if [[ "$PUSH" == "1" ]]; then
    log "backup_logs_to_github.sh"
    MSG="${MSG:-pb_toy sync [$PB_LOG_HOST] $(ts)}" \
      bash "$PB_REPO/scripts/backup_logs_to_github.sh"
  fi
  log "sync done"
}

if [[ "$WATCH" == "1" ]]; then
  log "WATCH every ${INTERVAL_SEC}s PUSH=$PUSH"
  while true; do
    once || log "WARN sync failed rc=$?"
    sleep "$INTERVAL_SEC"
  done
else
  once
fi
