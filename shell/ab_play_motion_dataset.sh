#!/usr/bin/env bash
# A/B play: same old checkpoint, old vs new motion dataset.
#
# Controls: only --motion_file / --motion_path changes; task + checkpoint fixed.
#
# Usage:
#   LOAD_RUN=2026-06-09_03-45-57_resumed_dribble_dribble \
#   CHECKPOINT=model_84000.pt \
#   bash shell/ab_play_motion_dataset.sh
#
# Optional: record dual-view MP4 for each run
#   RECORD=1 bash shell/ab_play_motion_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TASK="${TASK:-Tracking-CG-G1-Dribbling-RNN-v0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_dribbling}"
LOAD_RUN="${LOAD_RUN:?set LOAD_RUN to your *_dribble run dir name}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT e.g. model_84000.pt}"
NUM_ENVS="${NUM_ENVS:-1}"

OLD_ROOT="${OLD_MOTION_ROOT:-motions/dribble-distance}"
NEW_ROOT="${NEW_MOTION_ROOT:-motions/dribble-distance-modified}"
SEG1_OLD="${OLD_ROOT}/FAST-seg1_unitree_g1.npz"
SEG1_NEW="${NEW_ROOT}/FAST-seg1_unitree_g1.npz"

PLAY_EXTRA=(--headless)
if [[ "${RECORD:-0}" == "1" ]]; then
  PLAY_EXTRA+=(--dual_view --video)
fi

cd "${REPO_ROOT}"

echo "═══════════════════════════════════════════════════════════════"
echo " A/B motion dataset play"
echo " task=${TASK}"
echo " load_run=${LOAD_RUN}  checkpoint=${CHECKPOINT}"
echo " old=${OLD_ROOT}"
echo " new=${NEW_ROOT}"
echo "═══════════════════════════════════════════════════════════════"

if [[ -f "${SEG1_OLD}" && -f "${SEG1_NEW}" ]]; then
  if cmp -s "${SEG1_OLD}" "${SEG1_NEW}"; then
    echo "[INFO] seg1: byte-identical in old and new datasets"
  else
    echo "[WARN] seg1: files differ between old and new!"
  fi
else
  echo "[WARN] seg1 missing under old or new root"
fi

run_play() {
  local label="$1"
  shift
  echo ""
  echo "───────────────────────────────────────────────────────────────"
  echo " RUN: ${label}"
  echo "───────────────────────────────────────────────────────────────"
  python scripts/rsl_rl/play_multi.py \
    --task "${TASK}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --num_envs "${NUM_ENVS}" \
    "${PLAY_EXTRA[@]}" \
    "$@"
}

# 1) Single clip seg1 — isolates dataset file content (should match if identical npz)
run_play "A seg1 OLD dataset" --motion_file "${SEG1_OLD}"

run_play "B seg1 NEW dataset" --motion_file "${SEG1_NEW}"

# 2) Full directories — sequential sampling order / clip list differs on new set
run_play "C full OLD directory" --motion_path "${OLD_ROOT}"

run_play "D full NEW directory" --motion_path "${NEW_ROOT}"

echo ""
echo "[DONE] Compare A vs B (seg1): arms should look the same (same npz)."
echo "       If A/B differ → not the dataset; check task, checkpoint, or play_multi version."
echo "       If A/B same but C vs D differ → clip list / multi-clip sampling (not seg1 content)."
