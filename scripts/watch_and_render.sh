#!/usr/bin/env bash
# DEPRECATED: do not use.
#
# Training already renders on completion via --render-dir (run_matrix / train.py),
# then collect_renders.sh refreshes renders/. For backfill only:
#
#   bash scripts/render_matrix.sh
#   bash scripts/collect_renders.sh
#
set -euo pipefail
echo "watch_and_render.sh is deprecated; use train --render-dir (automatic) or scripts/render_matrix.sh" >&2
exit 1
