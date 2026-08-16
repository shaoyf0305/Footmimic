#!/usr/bin/env bash
# Evaluate per-reference quality for the active Stage-2 control task.
#
# Usage:
#   bash shell/eval_dribble_references.sh [LOAD_RUN] [CHECKPOINT] [NUM_ROLLOUTS]
#
# Example (local 1.18 case):
#   bash shell/eval_dribble_references.sh 2026-06-09_23-57-01_resumed model_100000.pt 3

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LOAD_RUN="${1:-2026-06-09_23-57-01_resumed}"
CHECKPOINT="${2:-model_100000.pt}"
NUM_ROLLOUTS="${3:-3}"
MOTION_PATH="${CONTROL_MOTION_PATH:-motions/master-v2}"
TASK="Tracking-CG-G1-Dribbling-RNN-control"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate isaaclab_211
source /data/isaacsim/setup_conda_env.sh

cd "${REPO_ROOT}"

python scripts/rsl_rl/eval_references.py \
  --task "${TASK}" \
  --experiment_name g1_dribbling \
  --motion_path "${MOTION_PATH}" \
  --load_run "${LOAD_RUN}" \
  --checkpoint "${CHECKPOINT}" \
  --num_rollouts "${NUM_ROLLOUTS}" \
  --headless \
  "$@"
