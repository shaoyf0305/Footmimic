#!/usr/bin/env bash

# Evaluate the frozen Essay13 full-method checkpoint with a physical ball and
# with a collision-free pelvis-locked ghost ball. This is an inference-only
# diagnostic and never changes or retrains the checkpoint.

set -uo pipefail

SCRIPT_REL="scripts/rsl_rl/run_essay13_locomotion_diagnostic.sh"
PROJECT_IN_CONTAINER="/workspace/projects/Footmimic"
ISAACLAB_LAUNCHER="/workspace/isaaclab/isaaclab.sh"

if [[ "${1:-}" == "--inside-container" ]]; then
    shift
elif [[ ! -x "$ISAACLAB_LAUNCHER" ]]; then
    if [[ -z "${WORK:-}" || ! -x "$WORK/run_isaaclab.sh" ]]; then
        echo "[ERROR] Run inside the Isaac container or set WORK to the host workspace." >&2
        exit 2
    fi
    printf -v FORWARDED_ARGS ' %q' "$@"
    exec "$WORK/run_isaaclab.sh" bash -lc \
        "source ~/isaac_env.sh; cd $PROJECT_IN_CONTAINER; bash $SCRIPT_REL --inside-container$FORWARDED_ARGS"
fi

TASK="Tracking-CG-G1-Dribbling-RNN-control"
MOTION_PATH="motions/master-v2"
EXPERIMENT_NAME="g1_dribbling_essay"
LOAD_RUN="2026-08-20_02-48-27_s2_13"
CHECKPOINT="model_88000.pt"
DEVICE="cuda:0"
SEEDS_CSV="13"
SPEEDS_CSV="0.80,1.20,1.50,1.65"
MODES_CSV="physical,pelvis_locked"
RESULT_DIR=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/rsl_rl/run_essay13_locomotion_diagnostic.sh [options]

Options:
  --seeds CSV          Evaluation seeds (default: 13)
  --speeds CSV         Straight-line command speeds (default: 0.80,1.20,1.50,1.65)
  --modes CSV          physical,pelvis_locked or either one
  --device DEVICE      Isaac/RSL-RL device (default: cuda:0)
  --result-dir PATH    Explicit output directory
  --load-run RUN       Override formal full-method run
  --checkpoint FILE    Override checkpoint filename
  -h, --help

The default run uses the frozen formal full-method checkpoint and performs no
training. Each case lasts 20 s (1,000 control steps) at zero heading.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) SEEDS_CSV="$2"; shift 2 ;;
        --speeds) SPEEDS_CSV="$2"; shift 2 ;;
        --modes) MODES_CSV="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --load-run) LOAD_RUN="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a SPEEDS <<< "$SPEEDS_CSV"
IFS=',' read -r -a MODES <<< "$MODES_CSV"

for mode in "${MODES[@]}"; do
    if [[ "$mode" != "physical" && "$mode" != "pelvis_locked" ]]; then
        echo "[ERROR] Unsupported mode: $mode" >&2
        exit 2
    fi
done

if [[ -z "$RESULT_DIR" ]]; then
    RESULT_DIR="output/essay13_locomotion_diagnostic/$(date -u +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RESULT_DIR"
MANIFEST="$RESULT_DIR/manifest.tsv"
printf 'case_id\tmode\tseed\tspeed\tstatus\tdiagnostic\n' > "$MANIFEST"

echo "[INFO] Result directory: $RESULT_DIR"
echo "[INFO] Checkpoint: logs/rsl_rl/$EXPERIMENT_NAME/$LOAD_RUN/$CHECKPOINT"

overall_status=0
for seed in "${SEEDS[@]}"; do
    for speed in "${SPEEDS[@]}"; do
        speed_tag="${speed/./p}"
        for mode in "${MODES[@]}"; do
            case_id="${mode}_s${speed_tag}_seed${seed}"
            case_dir="$RESULT_DIR/$case_id"
            diagnostic="$case_dir/diagnostic.npz"
            mkdir -p "$case_dir"

            cmd=(
                "$ISAACLAB_LAUNCHER" -p scripts/rsl_rl/play_multi.py
                --task "$TASK"
                --motion_path "$MOTION_PATH"
                --experiment_name "$EXPERIMENT_NAME"
                --load_run "$LOAD_RUN"
                --checkpoint "$CHECKPOINT"
                --num_envs 1
                --seed "$seed"
                --device "$DEVICE"
                --headless
                --locomotion_cmd_speed "$speed"
                --locomotion_cmd_heading 0
                --locomotion_cmd_duration 20
                --locomotion_cmd_hold_last
                --disable_interval_pushes
                --evaluation_reference_phase 0
                --evaluation_ball_mode "$mode"
                --video_length 1000
                --diagnostic
                --diagnostic_stride 1
                --diagnostic_path "$diagnostic"
                --evaluation_case_id "$case_id"
                --stop_on_done
            )

            echo "[INFO] Running $case_id"
            "${cmd[@]}" > "$case_dir/stdout.log" 2>&1
            status=$?
            if [[ "$status" -ne 0 || ! -s "$diagnostic" ]]; then
                overall_status=1
                echo "[WARN] $case_id failed with status $status"
            fi
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$case_id" "$mode" "$seed" "$speed" "$status" "$diagnostic" >> "$MANIFEST"
        done
    done
done

"$ISAACLAB_LAUNCHER" -p scripts/rsl_rl/analyze_essay13_locomotion_diagnostic.py \
    --result-dir "$RESULT_DIR"
analysis_status=$?
if [[ "$analysis_status" -ne 0 ]]; then
    overall_status=1
fi

exit "$overall_status"
