#!/usr/bin/env bash

# Apply the same Essay13 evaluation suite to a table of trained ablations.

set -euo pipefail

SCRIPT_REL="scripts/rsl_rl/run_essay13_ablation_evaluation.sh"
SUITE_REL="scripts/rsl_rl/run_essay13_baseline_suite.sh"
PROJECT_IN_CONTAINER="/workspace/projects/Footmimic"
ISAACLAB_LAUNCHER="/workspace/isaaclab/isaaclab.sh"

if [[ "${1:-}" == "--inside-container" ]]; then
    shift
elif [[ ! -x "$ISAACLAB_LAUNCHER" ]]; then
    if [[ -z "${WORK:-}" || ! -x "$WORK/run_isaaclab.sh" ]]; then
        echo "[ERROR] Run inside the Isaac container or export WORK for run_isaaclab.sh." >&2
        exit 2
    fi
    printf -v FORWARDED_ARGS ' %q' "$@"
    exec "$WORK/run_isaaclab.sh" bash -lc \
        "source ~/isaac_env.sh; cd $PROJECT_IN_CONTAINER; bash $SCRIPT_REL --inside-container$FORWARDED_ARGS"
fi

CHECKPOINT_TABLE=""
EXPERIMENT_NAME="g1_dribbling_essay_ablation"
MOTION_PATH="motions/master-v2"
PROFILE="core"
EVAL_SEEDS="13"
VIDEOS="representative"
DEVICE="cuda:0"
RESULT_DIR=""
RESUME=0
DRY_RUN=0
CONTINUE_ON_ERROR=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/rsl_rl/run_essay13_ablation_evaluation.sh \
      --checkpoint-table FILE [options]

Checkpoint table format (whitespace-separated, one row per policy):
  variant  load_run  checkpoint

Lines beginning with # and a first row beginning with "variant" are skipped.

Options:
  --experiment-name NAME
  --motion-path PATH
  --profile smoke|core|paper
  --eval-seeds 13,23,37
  --videos none|representative|all
  --device DEVICE
  --result-dir PATH
  --resume
  --dry-run
  --continue-on-error
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-table) CHECKPOINT_TABLE="$2"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --motion-path) MOTION_PATH="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --eval-seeds) EVAL_SEEDS="$2"; shift 2 ;;
        --videos) VIDEOS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$CHECKPOINT_TABLE" || ! -f "$CHECKPOINT_TABLE" ]]; then
    echo "[ERROR] --checkpoint-table must name an existing file." >&2
    exit 2
fi
if [[ -z "$RESULT_DIR" ]]; then
    RESULT_DIR="output/essay13_ablation_evaluation/$(date -u +%Y%m%d_%H%M%S)_${PROFILE}"
fi
RESULT_DIR="$(realpath -m "$RESULT_DIR")"
mkdir -p "$RESULT_DIR"

declare -A TASK_BY_VARIANT=(
    [full]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-Full"
    [no_ball_velocity]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoBallVelocity"
    [no_recovery]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoRecovery"
    [no_stage1]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoStage1"
    [no_interaction_reference]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoInteractionReference"
)

SUMMARY="$RESULT_DIR/evaluation_manifest.tsv"
printf 'variant\tload_run\tcheckpoint\tstatus\texit_code\tresult_dir\n' > "$SUMMARY"
FAILED=0
COUNT=0
while read -r variant load_run checkpoint extra; do
    if [[ -z "${variant:-}" || "${variant:0:1}" == "#" || "$variant" == "variant" ]]; then
        continue
    fi
    if [[ -n "${extra:-}" || -z "${load_run:-}" || -z "${checkpoint:-}" ]]; then
        echo "[ERROR] Invalid checkpoint-table row for variant ${variant:-<empty>}." >&2
        exit 2
    fi
    if [[ -z "${TASK_BY_VARIANT[$variant]+x}" ]]; then
        echo "[ERROR] Unknown checkpoint-table variant: $variant" >&2
        exit 2
    fi
    COUNT=$((COUNT + 1))
    variant_dir="$RESULT_DIR/$variant"
    cmd=(
        bash "$SUITE_REL" --inside-container
        --task "${TASK_BY_VARIANT[$variant]}"
        --motion-path "$MOTION_PATH"
        --experiment-name "$EXPERIMENT_NAME"
        --load-run "$load_run"
        --checkpoint "$checkpoint"
        --profile "$PROFILE"
        --eval-seeds "$EVAL_SEEDS"
        --videos "$VIDEOS"
        --device "$DEVICE"
        --result-dir "$variant_dir"
    )
    [[ "$RESUME" -eq 1 ]] && cmd+=(--resume)
    [[ "$DRY_RUN" -eq 1 ]] && cmd+=(--dry-run --no-archive)

    printf '[EVAL] variant=%s run=%s checkpoint=%s\n' "$variant" "$load_run" "$checkpoint"
    if "${cmd[@]}"; then
        exit_code=0
    else
        exit_code=$?
    fi
    if [[ "$exit_code" -eq 0 ]]; then
        status="passed"
    else
        status="failed"
        FAILED=$((FAILED + 1))
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$variant" "$load_run" "$checkpoint" "$status" "$exit_code" "$variant_dir" >> "$SUMMARY"
    if [[ "$exit_code" -ne 0 && "$CONTINUE_ON_ERROR" -eq 0 ]]; then
        exit "$exit_code"
    fi
done < "$CHECKPOINT_TABLE"

if [[ "$COUNT" -eq 0 ]]; then
    echo "[ERROR] Checkpoint table contains no policies." >&2
    exit 2
fi
echo "[DONE] evaluation manifest: $SUMMARY"
if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
