#!/usr/bin/env bash
# After PathBridger → 100k queue finishes, run:
#   1) CarRace 10k matrix (ice/grav/anti_grav × lap_2p/4p/8p × noisy/random × 5 agents)
#      — covers the 10 anti_grav remainders + 3 svcho handoff resumes
#   2) PB 1k N/T sweep (pbg/pbf final ckpts; skip already-written results)
#
# Invoked by run_queue_100k_k10.sh via NEXT_QUEUE, or manually:
#   bash scripts/run_queue_post_100k.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOGDIR="$ROOT/nohup_logs/queue_post_100k"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"
ts() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

SKIP_MATRIX="${SKIP_MATRIX:-0}"
# NT sweep disabled on this host (no 1k ckpts / not requested).
SKIP_NT_SWEEP="${SKIP_NT_SWEEP:-1}"
HANDOFF_SRC="${HANDOFF_SRC:-}"

: >>"$MASTER"
log "=== POST_100k START === SKIP_MATRIX=$SKIP_MATRIX SKIP_NT_SWEEP=$SKIP_NT_SWEEP"

# ---------- Phase 1: 10k lap248 matrix (handoff from 0 if no ckpt) ----------
if [[ "$SKIP_MATRIX" != "1" ]]; then
  log "PHASE1: run_queue_10k_lap248.sh"
  HANDOFF_SRC="$HANDOFF_SRC" bash "$ROOT/scripts/run_queue_10k_lap248.sh" \
    | tee -a "$MASTER"
  log "PHASE1 done"
else
  log "PHASE1 skipped"
fi

if [[ "$SKIP_NT_SWEEP" != "1" ]]; then
  log "PHASE2: NT sweep not configured — set SKIP_NT_SWEEP=0 and provide _1k ckpts to enable"
else
  log "PHASE2 NT sweep skipped"
fi

log "=== POST_100k DONE ==="
