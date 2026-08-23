#!/usr/bin/env bash

# One-command Essay13 checkpoint evaluation suite.
#
# From the host:
#   bash scripts/rsl_rl/run_essay13_baseline_suite.sh --profile core
#
# The script re-enters the Isaac container through $WORK/run_isaaclab.sh when
# needed. Every case receives a stable diagnostic path and its own stdout log.

set -uo pipefail

SCRIPT_REL="scripts/rsl_rl/run_essay13_baseline_suite.sh"
PROJECT_IN_CONTAINER="/workspace/projects/Footmimic"
ISAACLAB_LAUNCHER="/workspace/isaaclab/isaaclab.sh"

if [[ "${1:-}" == "--inside-container" ]]; then
    shift
elif [[ ! -x "$ISAACLAB_LAUNCHER" ]]; then
    if [[ -z "${WORK:-}" ]]; then
        echo "[ERROR] WORK is not set and $ISAACLAB_LAUNCHER is unavailable." >&2
        echo "        Run from the Isaac container or export WORK for run_isaaclab.sh." >&2
        exit 2
    fi
    if [[ ! -x "$WORK/run_isaaclab.sh" ]]; then
        echo "[ERROR] Cannot execute $WORK/run_isaaclab.sh" >&2
        exit 2
    fi
    printf -v FORWARDED_ARGS ' %q' "$@"
    exec "$WORK/run_isaaclab.sh" bash -lc \
        "source ~/isaac_env.sh; cd $PROJECT_IN_CONTAINER; bash $SCRIPT_REL --inside-container$FORWARDED_ARGS"
fi

TASK="Tracking-CG-G1-Dribbling-RNN-control"
MOTION_PATH="motions/master-v2"
LOAD_RUN="2026-08-20_02-48-27_s2_13"
CHECKPOINT="model_88000.pt"
EXPERIMENT_NAME="g1_dribbling_essay"
BASELINE_COMMIT="a589bd71168bd876fe4db93a3d887039c94005a8"
DEVICE="cuda:0"
NUM_ENVS=1
PROFILE="core"
EVAL_SEEDS_CSV="13"
VIDEOS="representative"
RESULT_DIR=""
RESUME=0
DRY_RUN=0
FAIL_FAST=0
MAKE_ARCHIVE=1
ALLOW_BASELINE_DRIFT=0
SOURCE_CONTRACT="baseline"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/rsl_rl/run_essay13_baseline_suite.sh [options]

Profiles:
  smoke  Two short cases for checking the environment and output paths.
  core   Practical baseline suite. Uses a chained 3x3 command grid.
  paper  Expanded suite. Uses independent 5x5 cells and the full phase/offset grid.

Options:
  --profile smoke|core|paper
  --eval-seeds 13,23,37
  --videos none|representative|all
  --load-run RUN
  --checkpoint FILE
  --motion-path PATH
  --task TASK
  --experiment-name NAME
  --device DEVICE
  --result-dir PATH
  --resume                 Skip cases whose diagnostic.npz already exists.
  --dry-run                Write commands without launching Isaac Sim.
  --fail-fast              Stop after the first process failure.
  --no-archive             Do not create the final tar.gz bundle.
  --source-contract MODE   baseline or committed (for trained ablations).
  --allow-baseline-drift   Continue despite source differences from Essay13.
  -h, --help

The defaults reproduce the supplied Essay13 checkpoint command.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --eval-seeds) EVAL_SEEDS_CSV="$2"; shift 2 ;;
        --videos) VIDEOS="$2"; shift 2 ;;
        --load-run) LOAD_RUN="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --motion-path) MOTION_PATH="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --fail-fast) FAIL_FAST=1; shift ;;
        --no-archive) MAKE_ARCHIVE=0; shift ;;
        --source-contract) SOURCE_CONTRACT="$2"; shift 2 ;;
        --allow-baseline-drift) ALLOW_BASELINE_DRIFT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$PROFILE" in
    smoke|core|paper) ;;
    *) echo "[ERROR] --profile must be smoke, core, or paper." >&2; exit 2 ;;
esac
case "$VIDEOS" in
    none|representative|all) ;;
    *) echo "[ERROR] --videos must be none, representative, or all." >&2; exit 2 ;;
esac
case "$SOURCE_CONTRACT" in
    baseline|committed) ;;
    *) echo "[ERROR] --source-contract must be baseline or committed." >&2; exit 2 ;;
esac
if [[ "$NUM_ENVS" -ne 1 ]]; then
    echo "[ERROR] The diagnostic currently records env 0 only. NUM_ENVS must remain 1." >&2
    exit 2
fi
if [[ ! -e "$MOTION_PATH" ]]; then
    echo "[ERROR] Motion path not found: $MOTION_PATH" >&2
    exit 2
fi

# These approved runtime differences do not change the Essay13 training MDP.
# The command hook is evaluation-only and defaults to disabled.  The bounded
# actor patch adds finite-value checks and an exploration-std safety floor; it
# is retained as an approved safety backport and recorded in every result pack.
APPROVED_SOURCE_DIFFS=(
    "source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_multi_motion_soccer.py"
    "source/whole_body_tracking/soccer/utils/bounded_actor_critic.py"
)
if [[ "$SOURCE_CONTRACT" == "baseline" ]]; then
    if ! git cat-file -e "${BASELINE_COMMIT}^{commit}" 2>/dev/null; then
        echo "[ERROR] Frozen Essay13 commit is unavailable: $BASELINE_COMMIT" >&2
        exit 2
    fi
    mapfile -t SOURCE_DIFFS < <(
        git diff --name-only "$BASELINE_COMMIT" -- source/whole_body_tracking/soccer
    )
else
    mapfile -t SOURCE_DIFFS < <(
        git diff --name-only HEAD -- \
            source/whole_body_tracking/soccer \
            scripts/rsl_rl/play_multi.py \
            "$SCRIPT_REL"
    )
fi
mapfile -t UNTRACKED_SOURCE < <(
    git ls-files --others --exclude-standard -- source/whole_body_tracking/soccer
)
BASELINE_DRIFTS=()
for path in "${SOURCE_DIFFS[@]}"; do
    approved=0
    if [[ "$SOURCE_CONTRACT" == "baseline" ]]; then
        for approved_path in "${APPROVED_SOURCE_DIFFS[@]}"; do
            if [[ "$path" == "$approved_path" ]]; then
                approved=1
                break
            fi
        done
    fi
    if [[ "$approved" -eq 0 ]]; then
        BASELINE_DRIFTS+=("$path")
    fi
done
for path in "${UNTRACKED_SOURCE[@]}"; do
    # Editor/backup copies are not imported by Python and cannot affect the
    # rollout.  Keep them out of the scientific source contract while still
    # rejecting any other untracked module under the active package.
    case "$path" in
        *.bak|*.orig|*~) continue ;;
        *) BASELINE_DRIFTS+=("$path (untracked)") ;;
    esac
done

if [[ ${#BASELINE_DRIFTS[@]} -gt 0 ]]; then
    echo "[ERROR] Source violates the $SOURCE_CONTRACT evaluation contract:" >&2
    printf '        %s\n' "${BASELINE_DRIFTS[@]}" >&2
    if [[ "$SOURCE_CONTRACT" == "baseline" ]]; then
        echo "[INFO] Tracked source diff against the frozen Essay13 commit:" >&2
        git diff --stat "$BASELINE_COMMIT" -- source/whole_body_tracking/soccer >&2 || true
        echo "[INFO] Inspect it with:" >&2
        echo "        git diff $BASELINE_COMMIT -- <path>" >&2
    fi
    if [[ "$ALLOW_BASELINE_DRIFT" -eq 0 ]]; then
        if [[ "$SOURCE_CONTRACT" == "baseline" ]]; then
            echo "        Restore or isolate changes outside the approved baseline backports." >&2
        else
            echo "        Commit or isolate all source changes before evaluating an ablation." >&2
        fi
        echo "        Use --allow-baseline-drift only for debugging, never for paper data." >&2
        exit 2
    fi
    echo "[WARN] Continuing because --allow-baseline-drift was supplied." >&2
fi

IFS=',' read -r -a EVAL_SEEDS <<< "$EVAL_SEEDS_CSV"
if [[ ${#EVAL_SEEDS[@]} -eq 0 ]]; then
    echo "[ERROR] --eval-seeds must contain at least one integer." >&2
    exit 2
fi
for seed in "${EVAL_SEEDS[@]}"; do
    if [[ ! "$seed" =~ ^-?[0-9]+$ ]]; then
        echo "[ERROR] Invalid evaluation seed: $seed" >&2
        exit 2
    fi
done

if [[ -z "$RESULT_DIR" ]]; then
    RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
    RESULT_DIR="output/essay13_baseline_suite/${RUN_STAMP}_${PROFILE}"
fi
RESULT_DIR="$(realpath -m "$RESULT_DIR")"
mkdir -p "$RESULT_DIR/cases"
mkdir -p "$RESULT_DIR/source_snapshot"

cp "$SCRIPT_REL" "$RESULT_DIR/source_snapshot/"
cp scripts/rsl_rl/play_multi.py "$RESULT_DIR/source_snapshot/"
cp source/whole_body_tracking/soccer/tasks/tracking/mdp/commands_multi_motion_soccer.py \
    "$RESULT_DIR/source_snapshot/"
cp source/whole_body_tracking/soccer/utils/bounded_actor_critic.py \
    "$RESULT_DIR/source_snapshot/"
for source_file in \
    source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_dribbling_env_cfg.py \
    source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_dribbling_ablation_env_cfg.py \
    source/whole_body_tracking/soccer/tasks/tracking/mdp/observations_anchor.py \
    source/whole_body_tracking/soccer/tasks/tracking/mdp/rewards_dribbling.py; do
    if [[ -f "$source_file" ]]; then
        cp "$source_file" "$RESULT_DIR/source_snapshot/"
    fi
done
{
    printf '%s\0' "$SCRIPT_REL" scripts/rsl_rl/play_multi.py
    git ls-files -z source/whole_body_tracking/soccer
} | xargs -0 -r sha256sum > "$RESULT_DIR/source_sha256.txt"

CHECKPOINT_PATH="logs/rsl_rl/$EXPERIMENT_NAME/$LOAD_RUN/$CHECKPOINT"
if [[ -f "$CHECKPOINT_PATH" ]]; then
    sha256sum "$CHECKPOINT_PATH" > "$RESULT_DIR/checkpoint_sha256.txt"
else
    echo "unresolved_path=$CHECKPOINT_PATH" > "$RESULT_DIR/checkpoint_sha256.txt"
fi
if [[ -d "$MOTION_PATH" ]]; then
    find "$MOTION_PATH" -maxdepth 1 -type f -name '*.npz' -print0 \
        | sort -z \
        | xargs -0 -r sha256sum \
        > "$RESULT_DIR/motion_sha256.txt"
else
    sha256sum "$MOTION_PATH" > "$RESULT_DIR/motion_sha256.txt"
fi

MANIFEST="$RESULT_DIR/manifest.tsv"
FAILED_CASES="$RESULT_DIR/failed_cases.txt"
if [[ ! -e "$MANIFEST" ]]; then
    printf 'case_id\tcategory\tseed\tsteps\tcontrolled\tstatus\texit_code\tdiagnostic\n' > "$MANIFEST"
fi
: > "$FAILED_CASES"

{
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "profile=$PROFILE"
    echo "task=$TASK"
    echo "motion_path=$MOTION_PATH"
    echo "load_run=$LOAD_RUN"
    echo "checkpoint=$CHECKPOINT"
    echo "experiment_name=$EXPERIMENT_NAME"
    echo "baseline_commit=$BASELINE_COMMIT"
    echo "source_contract=$SOURCE_CONTRACT"
    echo "baseline_drift_override=$ALLOW_BASELINE_DRIFT"
    echo "device=$DEVICE"
    echo "eval_seeds=$EVAL_SEEDS_CSV"
    echo "videos=$VIDEOS"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_branch=$(git branch --show-current 2>/dev/null || echo unknown)"
} > "$RESULT_DIR/suite_config.txt"
git status --short > "$RESULT_DIR/git_status.txt" 2>/dev/null || true
git diff HEAD > "$RESULT_DIR/git_diff.patch" 2>/dev/null || true

TOTAL_CASES=0
PASSED_CASES=0
SKIPPED_CASES=0
FAILED_COUNT=0

is_representative_video_case() {
    case "$1" in
        baseline_heading|transition_coupled|recovery_pos_left) return 0 ;;
        *) return 1 ;;
    esac
}

run_case() {
    local case_id="$1"
    local category="$2"
    local seed="$3"
    local steps="$4"
    local controlled="$5"
    shift 5

    TOTAL_CASES=$((TOTAL_CASES + 1))
    local case_dir="$RESULT_DIR/cases/$case_id/seed_$seed"
    local diagnostic="$case_dir/diagnostic.npz"
    local stdout_log="$case_dir/stdout.log"
    mkdir -p "$case_dir"

    if [[ "$RESUME" -eq 1 && -s "$diagnostic" ]]; then
        echo "[SKIP] $case_id seed=$seed"
        printf '%s\t%s\t%s\t%s\t%s\tskipped\t0\t%s\n' \
            "$case_id" "$category" "$seed" "$steps" "$controlled" "$diagnostic" >> "$MANIFEST"
        SKIPPED_CASES=$((SKIPPED_CASES + 1))
        return 0
    fi

    local cmd=(
        "$ISAACLAB_LAUNCHER" -p scripts/rsl_rl/play_multi.py
        --task "$TASK"
        --motion_path "$MOTION_PATH"
        --load_run "$LOAD_RUN"
        --checkpoint "$CHECKPOINT"
        --num_envs "$NUM_ENVS"
        --device "$DEVICE"
        --headless
        --experiment_name "$EXPERIMENT_NAME"
        --seed "$seed"
        --video_length "$steps"
        --diagnostic
        --diagnostic_stride 1
        --diagnostic_path "$diagnostic"
        --evaluation_case_id "$case_id"
    )
    if [[ "$controlled" -eq 1 ]]; then
        cmd+=(--disable_interval_pushes)
    fi
    if [[ "$VIDEOS" == "all" ]] || \
       [[ "$VIDEOS" == "representative" ]] && is_representative_video_case "$case_id"; then
        cmd+=(--dual_view --cam_layout task_front_side --video_output_dir "$case_dir/video")
    fi
    cmd+=("$@")

    {
        echo "case_id=$case_id"
        echo "category=$category"
        echo "seed=$seed"
        echo "steps=$steps"
        echo "controlled=$controlled"
        printf 'command='
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$case_dir/case_metadata.txt"

    echo "[RUN ] $case_id seed=$seed steps=$steps"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[DRY ] '
        printf '%q ' "${cmd[@]}"
        printf '\n'
        printf '%s\t%s\t%s\t%s\t%s\tdry-run\t0\t%s\n' \
            "$case_id" "$category" "$seed" "$steps" "$controlled" "$diagnostic" >> "$MANIFEST"
        return 0
    fi

    set +o pipefail
    "${cmd[@]}" 2>&1 | tee "$stdout_log"
    local status=${PIPESTATUS[0]}
    set -o pipefail
    if [[ "$status" -eq 0 && -s "$diagnostic" ]]; then
        echo "[PASS] $case_id seed=$seed"
        printf '%s\t%s\t%s\t%s\t%s\tpassed\t0\t%s\n' \
            "$case_id" "$category" "$seed" "$steps" "$controlled" "$diagnostic" >> "$MANIFEST"
        PASSED_CASES=$((PASSED_CASES + 1))
        return 0
    fi

    echo "[FAIL] $case_id seed=$seed exit=$status" >&2
    printf '%s\t%s\t%s\t%s\t%s\tfailed\t%s\t%s\n' \
        "$case_id" "$category" "$seed" "$steps" "$controlled" "$status" "$diagnostic" >> "$MANIFEST"
    printf '%s\tseed=%s\texit=%s\n' "$case_id" "$seed" "$status" >> "$FAILED_CASES"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    if [[ "$FAIL_FAST" -eq 1 ]]; then
        return "$status"
    fi
    return 0
}

run_regression_case() {
    local seed="$1"
    run_case baseline_heading regression "$seed" 1800 0 \
        --locomotion_cmd_speed 1.5 1.5 1.5 \
        --locomotion_cmd_heading 0 0.65 -0.65 \
        --locomotion_cmd_duration 5 5 5 \
        --locomotion_cmd_reset_on_end
}

run_smoke_cases() {
    local seed="$1"
    run_case smoke_steady_1p0 steady "$seed" 250 1 \
        --locomotion_cmd_speed 1.0 \
        --locomotion_cmd_heading 0 \
        --locomotion_cmd_duration 6 \
        --locomotion_cmd_hold_last --stop_on_done
    run_case smoke_recovery_left recovery "$seed" 350 1 \
        --locomotion_cmd_speed 1.0 \
        --locomotion_cmd_heading 0 \
        --locomotion_cmd_duration 8 \
        --locomotion_cmd_hold_last --stop_on_done \
        --evaluation_ball_perturb_step 150 \
        --evaluation_ball_position_delta 0 0.15
}

run_core_grid() {
    local seed="$1"
    local speeds=(0.40 0.40 0.40 1.00 1.00 1.00 1.65 1.65 1.65)
    local headings=(-0.65 0 0.65 0.65 0 -0.65 -0.65 0 0.65)
    local durations=(12 12 12 12 12 12 12 12 12)
    run_case steady_grid_3x3_chained steady-grid "$seed" 5400 1 \
        --locomotion_cmd_speed "${speeds[@]}" \
        --locomotion_cmd_heading "${headings[@]}" \
        --locomotion_cmd_duration "${durations[@]}" \
        --locomotion_cmd_hold_last
}

run_paper_grid() {
    local seed="$1"
    local speeds=(0.40 0.80 1.20 1.50 1.65)
    local headings=(-0.75 -0.375 0 0.375 0.75)
    local speed heading speed_tag heading_tag case_id
    for speed in "${speeds[@]}"; do
        for heading in "${headings[@]}"; do
            speed_tag="${speed//./p}"
            heading_tag="${heading//-/m}"
            heading_tag="${heading_tag//./p}"
            case_id="steady_s${speed_tag}_h${heading_tag}"
            run_case "$case_id" steady-grid "$seed" 1000 1 \
                --locomotion_cmd_speed "$speed" \
                --locomotion_cmd_heading "$heading" \
                --locomotion_cmd_duration 21 \
                --locomotion_cmd_hold_last --stop_on_done
        done
    done
}

run_transition_cases() {
    local seed="$1"
    run_case transition_heading transition "$seed" 900 1 \
        --locomotion_cmd_speed 1.5 1.5 1.5 \
        --locomotion_cmd_heading 0 0.65 -0.65 \
        --locomotion_cmd_duration 5 5 8 \
        --locomotion_cmd_hold_last --stop_on_done
    run_case transition_speed transition "$seed" 1000 1 \
        --locomotion_cmd_speed 0.40 1.00 1.65 0.80 \
        --locomotion_cmd_heading 0 0 0 0 \
        --locomotion_cmd_duration 5 5 5 5 \
        --locomotion_cmd_hold_last --stop_on_done
    run_case transition_coupled transition "$seed" 900 1 \
        --locomotion_cmd_speed 0.60 1.40 1.00 \
        --locomotion_cmd_heading -0.50 0.50 0 \
        --locomotion_cmd_duration 6 6 6 \
        --locomotion_cmd_hold_last --stop_on_done
}

run_recovery_cases() {
    local seed="$1"
    local ids=(
        recovery_pos_forward recovery_pos_backward recovery_pos_left recovery_pos_right
        recovery_vel_forward recovery_vel_backward recovery_vel_left recovery_vel_right
    )
    local pos_f=(0.20 -0.20 0 0 0 0 0 0)
    local pos_l=(0 0 0.15 -0.15 0 0 0 0)
    local vel_f=(0 0 0 0 0.50 -0.50 0 0)
    local vel_l=(0 0 0 0 0 0 0.50 -0.50)
    local i
    for i in "${!ids[@]}"; do
        run_case "${ids[$i]}" recovery "$seed" 600 1 \
            --locomotion_cmd_speed 1.0 \
            --locomotion_cmd_heading 0 \
            --locomotion_cmd_duration 13 \
            --locomotion_cmd_hold_last --stop_on_done \
            --evaluation_ball_perturb_step 200 \
            --evaluation_ball_position_delta "${pos_f[$i]}" "${pos_l[$i]}" \
            --evaluation_ball_velocity_delta "${vel_f[$i]}" "${vel_l[$i]}"
    done
}

run_long_horizon_case() {
    local seed="$1"
    local speeds=(
        0.40 0.80 1.20 1.50 1.00 1.65 0.80 1.20
        1.50 0.60 1.00 1.65 1.20 0.40 1.50 0.80
        1.20 1.65 0.60 1.00 1.50 0.80 1.20 0.40
    )
    local headings=(
        0 0.35 -0.35 0.65 -0.65 0 0.50 -0.50
        0.25 -0.25 0.70 -0.70 0 0.40 -0.40 0.60
        -0.60 0.15 -0.15 0.55 -0.55 0 0.30 -0.30
    )
    local durations=(
        5 5 5 5 5 5 5 5 5 5 5 5
        5 5 5 5 5 5 5 5 5 5 5 5
    )
    run_case long_horizon_120s long-horizon "$seed" 6000 1 \
        --locomotion_cmd_speed "${speeds[@]}" \
        --locomotion_cmd_heading "${headings[@]}" \
        --locomotion_cmd_duration "${durations[@]}" \
        --locomotion_cmd_hold_last --stop_on_done
}

run_initial_condition_grid() {
    local seed="$1"
    local forwards=(0.35 0.45 0.55)
    local laterals=(-0.10 0 0.10)
    local phases=(0 0.125 0.25 0.375 0.50 0.625 0.75 0.875)
    local forward lateral phase f_tag l_tag p_tag case_id
    for forward in "${forwards[@]}"; do
        for lateral in "${laterals[@]}"; do
            for phase in "${phases[@]}"; do
                f_tag="${forward//./p}"
                l_tag="${lateral//-/m}"
                l_tag="${l_tag//./p}"
                p_tag="${phase//./p}"
                case_id="initial_f${f_tag}_l${l_tag}_p${p_tag}"
                run_case "$case_id" initial-robustness "$seed" 600 1 \
                    --locomotion_cmd_speed 1.0 \
                    --locomotion_cmd_heading 0 \
                    --locomotion_cmd_duration 13 \
                    --locomotion_cmd_hold_last --stop_on_done \
                    --evaluation_reference_phase "$phase" \
                    --evaluation_initial_ball_offset "$forward" "$lateral"
            done
        done
    done
}

for seed in "${EVAL_SEEDS[@]}"; do
    if [[ "$PROFILE" == "smoke" ]]; then
        run_smoke_cases "$seed" || break
        continue
    fi

    run_regression_case "$seed" || break
    if [[ "$PROFILE" == "core" ]]; then
        run_core_grid "$seed" || break
    else
        run_paper_grid "$seed" || break
    fi
    run_transition_cases "$seed" || break
    run_recovery_cases "$seed" || break
    run_long_horizon_case "$seed" || break
    if [[ "$PROFILE" == "paper" ]]; then
        run_initial_condition_grid "$seed" || break
    fi
done

{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "total_cases=$TOTAL_CASES"
    echo "passed_cases=$PASSED_CASES"
    echo "skipped_cases=$SKIPPED_CASES"
    echo "failed_cases=$FAILED_COUNT"
    echo "result_dir=$RESULT_DIR"
} | tee "$RESULT_DIR/SUMMARY.txt"

if [[ "$MAKE_ARCHIVE" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    ARCHIVE="${RESULT_DIR}.tar.gz"
    tar -czf "$ARCHIVE" -C "$(dirname "$RESULT_DIR")" "$(basename "$RESULT_DIR")"
    sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
    echo "[INFO] Result bundle: $ARCHIVE"
    echo "[INFO] Bundle checksum: ${ARCHIVE}.sha256"
fi

echo "[INFO] Result directory: $RESULT_DIR"
if [[ "$FAILED_COUNT" -gt 0 ]]; then
    echo "[ERROR] $FAILED_COUNT case process(es) failed. See $FAILED_CASES" >&2
    exit 1
fi
