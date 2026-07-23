#!/usr/bin/env bash
# Commit+push PB_toy result artifacts (and sync helper scripts) on csh_server.
# Uses ~/.ssh/pb_toy_deploy (Deploy key with write on SChoish/PB_toy).
# Does NOT commit WIP train/dataset edits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST_LABEL="${PB_LOG_HOST:-csh_server}"
MSG="${MSG:-pb_toy results [$HOST_LABEL] $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S %Z')}"
SSH_KEY="${PB_TOY_SSH_KEY:-/home/ext_csh/.ssh/pb_toy_deploy}"
REMOTE_SSH="${PB_TOY_REMOTE_SSH:-git@github.com-pb-toy:SChoish/PB_toy.git}"
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o IdentitiesOnly=yes -i $SSH_KEY}"

# One-shot author (no git config mutation).
GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-csh_server}"
GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-csh_server@local}"
GIT_COMMITTER_NAME="${GIT_COMMITTER_NAME:-$GIT_AUTHOR_NAME}"
GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL:-$GIT_AUTHOR_EMAIL}"
export GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL

PATHS=(
  PB_toy_results_20260723_noisy100k_eval100k_200k_csh.md
  PB_toy_learning_curves_noisy100k_csh.png
  scripts/update_noisy100k_results_md.py
  scripts/plot_noisy100k_learning_curves.py
  scripts/sync_pb_toy_to_pblogs_csh.sh
  scripts/backup_pb_toy_to_github.sh
  scripts/eval_pending_checkpoints_cpu.py
  scripts/launch_retry_failed_noisy_200k_csh.sh
  scripts/launch_noisy_200k_csh_server.sh
  scripts/run_queue_noisy_200k_csh.sh
)

add_args=()
force_args=()
for p in "${PATHS[@]}"; do
  [[ -e "$p" ]] || continue
  if git check-ignore -q "$p" 2>/dev/null; then
    force_args+=("$p")
  else
    add_args+=("$p")
  fi
done
if [[ ${#add_args[@]} -eq 0 && ${#force_args[@]} -eq 0 ]]; then
  echo "nothing to add"
  exit 0
fi

[[ ${#add_args[@]} -gt 0 ]] && git add -- "${add_args[@]}"
[[ ${#force_args[@]} -gt 0 ]] && git add -f -- "${force_args[@]}"

if git diff --cached --quiet; then
  echo "nothing to commit"
  exit 0
fi

git commit -m "$MSG"
if ! git push "$REMOTE_SSH" HEAD:main; then
  echo "push failed; pull --rebase then retry"
  git pull --rebase "$REMOTE_SSH" main || true
  git push "$REMOTE_SSH" HEAD:main
fi
echo "pb_toy pushed ok"
