#!/usr/bin/env bash

# One-command launcher for configuration-controlled Essay13 ablation training.

set -euo pipefail

SCRIPT_REL="scripts/rsl_rl/run_essay13_ablation_training.sh"
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

MOTION_PATH="motions/master-v2"
EXPERIMENT_NAME="g1_dribbling_essay_ablation"
STAGE1_EXPERIMENT_NAME="g1_dribbling_essay"
STAGE1_LOAD_RUN=""
STAGE1_CHECKPOINT=""
STAGE1_MIGRATION=""
VARIANTS_CSV=""
SEEDS_CSV="13"
NUM_ENVS=4096
MAX_ITERATIONS=100000
DEVICE="cuda:0"
RESULT_DIR=""
DRY_RUN=0
CONTINUE_ON_ERROR=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/rsl_rl/run_essay13_ablation_training.sh [options]

Required when training any variant except no_stage1:
  --stage1-load-run RUN
  --stage1-checkpoint FILE
  --stage1-migration none|legacy-residual|bounded-policy

Selection:
  --variants NAME,NAME     Train a subset of the five reported conditions.
  --seeds 13,23,37

Training:
  --motion-path PATH
  --experiment-name NAME
  --stage1-experiment-name NAME
  --num-envs N
  --max-iterations N
  --device DEVICE
  --result-dir PATH
  --dry-run
  --continue-on-error
  -h, --help

Reported conditions:
  full, no_ball_velocity, no_recovery, no_stage1,
  no_interaction_reference
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variants) VARIANTS_CSV="$2"; shift 2 ;;
        --seeds) SEEDS_CSV="$2"; shift 2 ;;
        --motion-path) MOTION_PATH="$2"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --stage1-experiment-name) STAGE1_EXPERIMENT_NAME="$2"; shift 2 ;;
        --stage1-load-run) STAGE1_LOAD_RUN="$2"; shift 2 ;;
        --stage1-checkpoint) STAGE1_CHECKPOINT="$2"; shift 2 ;;
        --stage1-migration) STAGE1_MIGRATION="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$STAGE1_MIGRATION" in
    ""|none|legacy-residual|bounded-policy) ;;
    *) echo "[ERROR] Invalid --stage1-migration value." >&2; exit 2 ;;
esac
if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ || ! "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --num-envs and --max-iterations must be positive integers." >&2
    exit 2
fi
if [[ ! -e "$MOTION_PATH" ]]; then
    echo "[ERROR] Motion path not found: $MOTION_PATH" >&2
    exit 2
fi

declare -A TASK_BY_VARIANT=(
    [full]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-Full"
    [no_ball_velocity]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoBallVelocity"
    [no_recovery]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoRecovery"
    [no_stage1]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoStage1"
    [no_interaction_reference]="Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoInteractionReference"
)

FORMAL_VARIANTS=(
    full
    no_ball_velocity
    no_recovery
    no_stage1
    no_interaction_reference
)

if [[ -n "$VARIANTS_CSV" ]]; then
    IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"
else
    VARIANTS=("${FORMAL_VARIANTS[@]}")
fi
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

NEEDS_STAGE1=0
for variant in "${VARIANTS[@]}"; do
    if [[ -z "${TASK_BY_VARIANT[$variant]+x}" ]]; then
        echo "[ERROR] Unknown variant: $variant" >&2
        exit 2
    fi
    if [[ "$variant" != "no_stage1" ]]; then
        NEEDS_STAGE1=1
    fi
done
for seed in "${SEEDS[@]}"; do
    if [[ ! "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "[ERROR] Invalid seed: $seed" >&2
        exit 2
    fi
done
if [[ "$NEEDS_STAGE1" -eq 1 ]]; then
    if [[ -z "$STAGE1_LOAD_RUN" || -z "$STAGE1_CHECKPOINT" || -z "$STAGE1_MIGRATION" ]]; then
        echo "[ERROR] Stage-I run, checkpoint, and migration mode are required." >&2
        echo "        Use --stage1-migration none only for an already compatible checkpoint." >&2
        exit 2
    fi
fi

if [[ -z "$RESULT_DIR" ]]; then
    RESULT_DIR="output/essay13_ablation_launches/$(date -u +%Y%m%d_%H%M%S)"
fi
RESULT_DIR="$(realpath -m "$RESULT_DIR")"
mkdir -p "$RESULT_DIR/logs"

MANIFEST="$RESULT_DIR/launch_manifest.tsv"
printf 'variant\tseed\ttask\tstatus\texit_code\n' > "$MANIFEST"
{
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
    echo "motion_path=$MOTION_PATH"
    echo "experiment_name=$EXPERIMENT_NAME"
    echo "stage1_experiment_name=$STAGE1_EXPERIMENT_NAME"
    echo "stage1_load_run=$STAGE1_LOAD_RUN"
    echo "stage1_checkpoint=$STAGE1_CHECKPOINT"
    echo "stage1_migration=$STAGE1_MIGRATION"
    echo "seeds=$SEEDS_CSV"
    echo "num_envs=$NUM_ENVS"
    echo "max_iterations=$MAX_ITERATIONS"
    echo "device=$DEVICE"
} > "$RESULT_DIR/launch_config.txt"
git status --short > "$RESULT_DIR/git_status.txt" 2>/dev/null || true
git diff HEAD > "$RESULT_DIR/git_diff.patch" 2>/dev/null || true

FAILED=0
for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        task="${TASK_BY_VARIANT[$variant]}"
        run_name="e13_${variant}_seed${seed}"
        log_path="$RESULT_DIR/logs/${run_name}.log"
        cmd=(
            "$ISAACLAB_LAUNCHER" -p scripts/rsl_rl/train_multi.py
            --task "$task"
            --motion_path "$MOTION_PATH"
            --num_envs "$NUM_ENVS"
            --max_iterations "$MAX_ITERATIONS"
            --seed "$seed"
            --device "$DEVICE"
            --headless
            --experiment_name "$EXPERIMENT_NAME"
            --run_name "$run_name"
        )

        if [[ "$variant" != "no_stage1" ]]; then
            cmd+=(
                --resume True
                --resume_experiment_name "$STAGE1_EXPERIMENT_NAME"
                --load_run "$STAGE1_LOAD_RUN"
                --checkpoint "$STAGE1_CHECKPOINT"
            )
            case "$STAGE1_MIGRATION" in
                legacy-residual) cmd+=(--migrate_legacy_upper_body_residual) ;;
                bounded-policy) cmd+=(--migrate_bounded_upper_body_policy) ;;
                none) ;;
            esac
        fi

        printf '[RUN] variant=%s seed=%s\n' "$variant" "$seed"
        printf '%q ' "${cmd[@]}" > "$RESULT_DIR/logs/${run_name}.command"
        printf '\n' >> "$RESULT_DIR/logs/${run_name}.command"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            printf '%s\t%s\t%s\tdry-run\t0\n' "$variant" "$seed" "$task" >> "$MANIFEST"
            continue
        fi

        set +o pipefail
        "${cmd[@]}" 2>&1 | tee "$log_path"
        exit_code=${PIPESTATUS[0]}
        set -o pipefail
        if [[ "$exit_code" -eq 0 ]]; then
            status="passed"
        else
            status="failed"
            FAILED=$((FAILED + 1))
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$variant" "$seed" "$task" "$status" "$exit_code" >> "$MANIFEST"
        if [[ "$exit_code" -ne 0 && "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            echo "[ERROR] Training failed; see $log_path" >&2
            exit "$exit_code"
        fi
    done
done

echo "[DONE] launch manifest: $MANIFEST"
if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
