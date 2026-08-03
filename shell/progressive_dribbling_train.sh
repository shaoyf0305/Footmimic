#!/usr/bin/env bash
#
# Progressive dribbling training.
#
# Default: flat motion imitation -> flat dribbling.
# CG baselines:
#   --cg-control       legacy continuous command baseline
#   --cg-full-control  frozen IDLE/DRIBBLE/STOP baseline
#   --cg-unified-control  polar-only unified interface, fixed right instep touch
#   --cg-unified-3stage    reference mimic -> reference-contact dribble -> free task control
#
# Usage:
#   DRIBBLE_MOTION_PATH=motions/my_dribble \
#     bash shell/progressive_dribbling_train.sh [RUN_NAME] \
#       [--cg-control | --cg-full-control | --cg-unified-control | --cg-unified-3stage]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/g1_dribbling"
MOTION_PATH="${DRIBBLE_MOTION_PATH:-motions/dribble}"
# Used only by --cg-unified-control. ``any`` is the dataset-compatible
# default; ``instep`` accepts either dorsal instep side, while directional
# instep modes require a matching surface label on every contact.
CONTACT_SURFACE="${DRIBBLE_CONTACT_SURFACE:-any}"

case "${CONTACT_SURFACE}" in
    any|instep|inside_instep|outside_instep) ;;
    *)
        echo "DRIBBLE_CONTACT_SURFACE must be any, instep, inside_instep, or outside_instep; got: ${CONTACT_SURFACE}" >&2
        exit 2
        ;;
esac

RUN_NAME="dribbling"
MODE="flat"
STAGE2_EXTRA_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --cg-control) MODE="cg-control" ;;
        --cg-full-control) MODE="cg-full-control" ;;
        --cg-unified-control) MODE="cg-unified-control" ;;
        --cg-unified-3stage) MODE="cg-unified-3stage" ;;
        --*) ;;
        *) RUN_NAME="${arg}" ;;
    esac
done

case "${MODE}" in
    flat)
        STAGE1_TASK="Tracking-Flat-G1-Motion-RNN-v0"
        STAGE2_TASK="Tracking-Flat-G1-Dribbling-RNN-v0"
        ;;
    cg-control)
        STAGE1_TASK="Tracking-CG-G1-Motion-RNN-mimic"
        STAGE2_TASK="Tracking-CG-G1-Dribbling-RNN-control"
        ;;
    cg-full-control)
        STAGE1_TASK="Tracking-CG-G1-Motion-RNN-mimic"
        STAGE2_TASK="Tracking-CG-G1-Dribbling-RNN-full-control"
        ;;
    cg-unified-control)
        # This pair has the same 163-D actor input layout.  Stage 1 fixes the
        # polar command to [speed, cos(heading), sin(heading)] = [0, 1, 0]
        # and task state to IDLE, so resume needs no zero-padding.
        STAGE1_TASK="Tracking-CG-G1-Motion-RNN-unified-mimic"
        STAGE2_TASK="Tracking-CG-G1-Dribbling-RNN-unified-control"
        STAGE2_EXTRA_ARGS=("dribble_contact_surface=${CONTACT_SURFACE}")
        ;;
    cg-unified-3stage)
        STAGE1_TASK="Tracking-CG-G1-Motion-RNN-unified-s1-mimic"
        STAGE2_TASK="Tracking-CG-G1-Dribbling-RNN-unified-s2-reference"
        STAGE3_TASK="Tracking-CG-G1-Dribbling-RNN-unified-s3-task"
        ;;
esac

cd "${REPO_ROOT}"

if [[ "${MODE}" == "cg-unified-3stage" ]]; then
    S1_RUN_NAME="${RUN_NAME}_s1"
    S2_RUN_NAME="${RUN_NAME}_s2"
    S3_RUN_NAME="${RUN_NAME}_s3"

    echo "Stage 1: ${STAGE1_TASK}"
    echo "motion_path: ${MOTION_PATH}"
    echo "run_name: ${S1_RUN_NAME}"
    python scripts/rsl_rl/train_multi.py --task "${STAGE1_TASK}" \
        --motion_path "${MOTION_PATH}" \
        --run_name "${S1_RUN_NAME}" \
        --experiment_name g1_dribbling \
        --num_envs 2000 \
        --max_iterations 4000 \
        --headless

    LOAD_S1="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${S1_RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"
    if [[ -z "${LOAD_S1}" ]]; then
        echo "Failed to resolve Stage 1 checkpoint from ${EXPERIMENT_DIR}" >&2
        exit 1
    fi

    echo "Stage 2: ${STAGE2_TASK}"
    echo "resume: ${LOAD_S1}"
    echo "run_name: ${S2_RUN_NAME}"
    python scripts/rsl_rl/train_multi.py --task "${STAGE2_TASK}" \
        --motion_path "${MOTION_PATH}" \
        --load_run "${LOAD_S1}" \
        --run_name "${S2_RUN_NAME}" \
        --experiment_name g1_dribbling \
        --num_envs 2000 \
        --max_iterations 4000 \
        --resume True \
        --headless

    LOAD_S2="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${S2_RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"
    if [[ -z "${LOAD_S2}" ]]; then
        echo "Failed to resolve Stage 2 checkpoint from ${EXPERIMENT_DIR}" >&2
        exit 1
    fi

    echo "Stage 3: ${STAGE3_TASK}"
    echo "resume: ${LOAD_S2}"
    echo "run_name: ${S3_RUN_NAME}"
    python scripts/rsl_rl/train_multi.py --task "${STAGE3_TASK}" \
        --motion_path "${MOTION_PATH}" \
        --load_run "${LOAD_S2}" \
        --run_name "${S3_RUN_NAME}" \
        --experiment_name g1_dribbling \
        --num_envs 2000 \
        --max_iterations 4000 \
        --resume True \
        --headless

    echo "Play Stage 3:"
    echo "  python scripts/rsl_rl/play_multi.py --task ${STAGE3_TASK} \\\""
    echo "    --motion_path \"${MOTION_PATH}\" --load_run \"<RUN_DIR>_s3\" --checkpoint model_XXXX.pt"
    exit 0
fi

echo "Stage 1: ${STAGE1_TASK}"
echo "motion_path: ${MOTION_PATH}"
echo "run_name: ${RUN_NAME}"

python scripts/rsl_rl/train_multi.py --task "${STAGE1_TASK}" \
    --motion_path "${MOTION_PATH}" \
    --run_name "${RUN_NAME}" \
    --experiment_name g1_dribbling \
    --num_envs 2000 \
    --max_iterations 4000 \
    --headless

LOAD_RUN="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"
if [[ -z "${LOAD_RUN}" ]]; then
    echo "Failed to resolve Stage 1 checkpoint from ${EXPERIMENT_DIR}"
    exit 1
fi

echo "Stage 2: ${STAGE2_TASK}"
echo "resume: ${LOAD_RUN}"
echo "motion_path: ${MOTION_PATH}"

python scripts/rsl_rl/train_multi.py --task "${STAGE2_TASK}" \
    --motion_path "${MOTION_PATH}" \
    --load_run "${LOAD_RUN}" \
    --run_name "${RUN_NAME}_dribble" \
    --experiment_name g1_dribbling \
    --num_envs 2000 \
    --resume True \
    "${STAGE2_EXTRA_ARGS[@]}" \
    --headless

echo "Play Stage 2:"
echo "  python scripts/rsl_rl/play_multi.py --task ${STAGE2_TASK} \\\"
echo "    --motion_path \"${MOTION_PATH}\" --load_run \"<RUN_DIR>_dribble\" --checkpoint model_XXXX.pt"
