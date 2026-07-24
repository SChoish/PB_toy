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
PY="${PY:-/home/ext_csh/miniconda3/envs/pbtoy/bin/python}"
LOGS_ROOT="${PB_LOGS_ROOT:-/home/ext_csh/PB_logs}"
PB_REPO="${PB_REPO:-/home/ext_csh/PB_logs}"
export PB_LOG_HOST="${PB_LOG_HOST:-$(hostname -s)}"
export PB_LOGS_ROOT="$LOGS_ROOT"

PUSH="${PUSH:-0}"
WATCH="${WATCH:-0}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"
OVERWRITE="${OVERWRITE:-0}"
UPDATE_RESULTS_MD="${UPDATE_RESULTS_MD:-1}"

LOGDIR="$ROOT/nohup_logs/pblogs_sync"
mkdir -p "$LOGDIR"
ts() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*"; }

once() {
  local extra=()
  [[ "$OVERWRITE" == "1" ]] && extra+=(--overwrite)
  log "export → $LOGS_ROOT (host=$PB_LOG_HOST)"
  "$PY" -u scripts/export_results_to_pblogs.py --apply --k 10 --h-a 2 "${extra[@]}"
  if [[ "$UPDATE_RESULTS_MD" == "1" ]]; then
    log "parse_pb_noisy100k_nt_sweep.py"
    "$PY" -u scripts/parse_pb_noisy100k_nt_sweep.py || log "WARN nt sweep parse failed rc=$?"
    log "update_noisy100k_results_md.py"
    "$PY" -u scripts/update_noisy100k_results_md.py || log "WARN results md update failed rc=$?"
    log "plot_noisy100k_learning_curves.py"
    "$PY" -u scripts/plot_noisy100k_learning_curves.py || log "WARN learning curves plot failed rc=$?"
    log "update_noisy10k_results_md.py"
    "$PY" -u scripts/update_noisy10k_results_md.py || log "WARN 10k results md update failed rc=$?"
    log "plot_noisy10k_learning_curves.py"
    "$PY" -u scripts/plot_noisy10k_learning_curves.py || log "WARN 10k learning curves plot failed rc=$?"
  fi
  if [[ "$PUSH" == "1" ]]; then
    # Mirror results into PB_logs so they ride the existing logs push even
    # before a PB_toy deploy key is installed.
    if [[ -d "$LOGS_ROOT/pb_toy" ]]; then
      log "mirror results → PB_logs/pb_toy/"
      for f in \
        PB_toy_results_20260723_noisy100k_eval100k_200k_csh.md \
        PB_toy_learning_curves_noisy100k_csh.png \
        PB_toy_results_noisy10k_eval100k_200k_csh.md \
        PB_toy_learning_curves_noisy10k_csh.png \
        PB_toy_nt_sweep_noisy100k_200k.md \
        PB_toy_nt_sweep_noisy100k_200k.json
      do
        [[ -f "$ROOT/$f" ]] && cp -a "$ROOT/$f" "$LOGS_ROOT/pb_toy/$f"
      done
    fi
    log "backup_logs_to_github.sh (PB_logs)"
    MSG="${MSG:-pb_toy sync [$PB_LOG_HOST] $(ts)}" \
      bash "$PB_REPO/scripts/backup_logs_to_github.sh" \
      || log "WARN PB_logs push failed rc=$?"
    log "backup_pb_toy_to_github.sh (PB_toy)"
    MSG="${MSG:-pb_toy results [$PB_LOG_HOST] $(ts)}" \
      bash "$ROOT/scripts/backup_pb_toy_to_github.sh" \
      || log "WARN PB_toy push failed rc=$? (add deploy key if denied)"
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
