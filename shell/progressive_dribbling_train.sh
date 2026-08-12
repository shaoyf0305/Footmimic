#!/usr/bin/env bash
# Train the only supported pipeline:
#   Stage 1: Tracking-CG-G1-Motion-RNN-mimic
#   Stage 2: Tracking-CG-G1-Dribbling-RNN-control

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_dribbling}"
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"
MOTION_PATH="${DRIBBLE_MOTION_PATH:-motions/dribble}"
RUN_NAME="${1:-dribbling}"
NUM_ENVS="${NUM_ENVS:-2000}"
STAGE1_ITERATIONS="${STAGE1_ITERATIONS:-4000}"

STAGE1_TASK="Tracking-CG-G1-Motion-RNN-mimic"
STAGE2_TASK="Tracking-CG-G1-Dribbling-RNN-control"

cd "${REPO_ROOT}"

echo "════════════════════════════════════════════════════════════════"
echo " Stage 1: ${STAGE1_TASK}"
echo " motion_path: ${MOTION_PATH}"
echo " run_name:    ${RUN_NAME}"
echo "════════════════════════════════════════════════════════════════"

python scripts/rsl_rl/train_multi.py --task "${STAGE1_TASK}" \
    --motion_path "${MOTION_PATH}" \
    --run_name "${RUN_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${STAGE1_ITERATIONS}" \
    --headless

LOAD_RUN="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"

if [[ -z "${LOAD_RUN}" ]]; then
    echo "Failed to resolve Stage 1 checkpoint from ${EXPERIMENT_DIR}"
    exit 1
fi

echo
echo "════════════════════════════════════════════════════════════════"
echo " Stage 2: ${STAGE2_TASK}"
echo " resume:      ${LOAD_RUN}"
echo " motion_path: ${MOTION_PATH}"
echo "════════════════════════════════════════════════════════════════"

python scripts/rsl_rl/train_multi.py --task "${STAGE2_TASK}" \
    --motion_path "${MOTION_PATH}" \
    --load_run "${LOAD_RUN}" \
    --run_name "${RUN_NAME}_control" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --resume True \
    --headless

echo
echo "Stage 2 play command:"
echo "  python scripts/rsl_rl/play_multi.py --task ${STAGE2_TASK} \\"
echo "    --motion_path \"${MOTION_PATH}\" --load_run \"<RUN_DIR>_control\" --checkpoint model_XXXX.pt"
