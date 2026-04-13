#!/usr/bin/env bash
#
# Progressive Dribbling Training (2-Stage)
#
# Stage 1: Motion tracking (learn locomotion + balance)
#   - Default:       Tracking-Terrain-G1-RNN-v0 (vanilla terrain)
#   - --ankle-disturb: Tracking-Flat-G1-Dribbling-AnkleDisturb-RNN-v0
#                      (zero ankle reward + random ankle torques)
# Stage 2: Flat ground with dribbling rewards (learn ball control)
#
# Usage:
#   bash shell/progressive_dribbling_train.sh [RUN_NAME] [--ankle-disturb]
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/g1_dribbling"

# Parse arguments
RUN_NAME="${1:-dribbling}"
ANKLE_DISTURB=false
for arg in "$@"; do
    if [[ "${arg}" == "--ankle-disturb" ]]; then
        ANKLE_DISTURB=true
    fi
done

# Select Stage 1 task
if [[ "${ANKLE_DISTURB}" == "true" ]]; then
    STAGE1_TASK="Tracking-Flat-G1-Dribbling-AnkleDisturb-RNN-v0"
    echo ">>> Ankle Disturbance mode ENABLED <<<"
else
    STAGE1_TASK="Tracking-Terrain-G1-RNN-v0"
fi

cd "${REPO_ROOT}"

# ── Stage 1: Locomotion foundation ──────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo " Stage 1: ${STAGE1_TASK}"
echo " Run:  ${RUN_NAME}"
echo "════════════════════════════════════════════════════════════════"

python scripts/rsl_rl/train_multi.py --task "${STAGE1_TASK}" \
    --motion_path motions/soccer-standard \
    --run_name "${RUN_NAME}" \
    --experiment_name g1_dribbling \
    --num_envs 2000 \
    --max_iterations 4000 \
    --headless

# ── Resolve Stage 1 checkpoint ──────────────────────────────────────
LOAD_RUN="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"

if [[ -z "${LOAD_RUN}" ]]; then
    echo "Failed to resolve Stage 1 checkpoint from ${EXPERIMENT_DIR}"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Stage 2: Dribbling on flat ground"
echo " Task: Tracking-Flat-G1-Dribbling-RNN-v0"
echo " Resuming from: ${LOAD_RUN}"
echo "════════════════════════════════════════════════════════════════"

# ── Stage 2: Dribbling rewards ──────────────────────────────────────
python scripts/rsl_rl/train_multi.py --task Tracking-Flat-G1-Dribbling-RNN-v0 \
    --motion_path motions/soccer-standard \
    --load_run "${LOAD_RUN}" \
    --run_name "${RUN_NAME}_dribble" \
    --experiment_name g1_dribbling \
    --num_envs 2000 \
    --resume True \
    --headless
