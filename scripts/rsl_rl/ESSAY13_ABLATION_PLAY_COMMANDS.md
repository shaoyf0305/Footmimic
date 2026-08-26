# Essay13 Ablation Playback Commands

This file records the exact single-rollout playback commands for the five
primary experimental conditions. Every command uses the same evaluation seed,
command sequence, rollout length, camera layout, and diagnostic stride.

The common command sequence is 1.5 m/s at headings
`0 -> +0.65 -> -0.65 rad`, with each segment held for 5 s. Playback lasts
1,800 control steps and resets the physical state after each complete command
sequence. Each run writes one diagnostic `.npz` archive and one dual-view video
to a variant-specific output directory.

## Important budget check

The supplied checkpoint counters are:

| Condition | Checkpoint |
|---|---|
| Full control | `model_42999.pt` |
| No explicit ball velocity | `model_38000.pt` |
| No recovery blending | `model_42999.pt` |
| No Stage-I initialization | `model_19999.pt` |
| No interaction reference | `model_42999.pt` |

`model_38000.pt` for No explicit ball velocity does not have the same checkpoint
counter as the other Stage-I-initialized policies. It can be played and inspected
now, but it should not be treated as the final equal-budget ablation result until
its training manifest and learning log confirm the number of additional PPO
iterations. In particular, inspect `params/ablation_manifest.json` and verify
`max_iterations`, the initialization checkpoint, and its SHA-256 hash.

## Workspace and Singularity machine

The following three runs are stored under
`logs/rsl_rl/g1_dribbling_essay_ablation_formal` inside the Workspace project.

### Full control

```bash
$WORK/run_isaaclab.sh bash -lc '
set -euo pipefail
source ~/isaac_env.sh
cd /workspace/projects/Footmimic

/workspace/isaaclab/isaaclab.sh -p \
    scripts/rsl_rl/play_multi.py \
    --task "Tracking-CG-G1-Dribbling-RNN-control-Ablation-Full" \
    --motion_path motions/master-v2 \
    --experiment_name g1_dribbling_essay_ablation_formal \
    --load_run "2026-08-23_05-53-08_e13_full_seed13" \
    --checkpoint model_42999.pt \
    --num_envs 1 \
    --seed 13 \
    --device cuda:0 \
    --headless \
    --locomotion_cmd_speed 1.5 1.5 1.5 \
    --locomotion_cmd_heading 0 0.65 -0.65 \
    --locomotion_cmd_duration 5 5 5 \
    --locomotion_cmd_reset_on_end \
    --disable_interval_pushes \
    --video_length 1800 \
    --diagnostic \
    --diagnostic_stride 1 \
    --diagnostic_path output/essay13_ablation_play/full_seed13/diagnostic.npz \
    --evaluation_case_id full_seed13_regression \
    --dual_view \
    --cam_layout task_front_side \
    --video_output_dir output/essay13_ablation_play/full_seed13/video
'
```

### No explicit ball velocity

```bash
$WORK/run_isaaclab.sh bash -lc '
set -euo pipefail
source ~/isaac_env.sh
cd /workspace/projects/Footmimic

/workspace/isaaclab/isaaclab.sh -p \
    scripts/rsl_rl/play_multi.py \
    --task "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoBallVelocity" \
    --motion_path motions/master-v2 \
    --experiment_name g1_dribbling_essay_ablation_formal \
    --load_run "2026-08-23_16-04-59_e13_no_ball_velocity_seed13" \
    --checkpoint model_38000.pt \
    --num_envs 1 \
    --seed 13 \
    --device cuda:0 \
    --headless \
    --locomotion_cmd_speed 1.5 1.5 1.5 \
    --locomotion_cmd_heading 0 0.65 -0.65 \
    --locomotion_cmd_duration 5 5 5 \
    --locomotion_cmd_reset_on_end \
    --disable_interval_pushes \
    --video_length 1800 \
    --diagnostic \
    --diagnostic_stride 1 \
    --diagnostic_path output/essay13_ablation_play/no_ball_velocity_seed13/diagnostic.npz \
    --evaluation_case_id no_ball_velocity_seed13_regression \
    --dual_view \
    --cam_layout task_front_side \
    --video_output_dir output/essay13_ablation_play/no_ball_velocity_seed13/video
'
```

### No recovery blending

```bash
$WORK/run_isaaclab.sh bash -lc '
set -euo pipefail
source ~/isaac_env.sh
cd /workspace/projects/Footmimic

/workspace/isaaclab/isaaclab.sh -p \
    scripts/rsl_rl/play_multi.py \
    --task "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoRecovery" \
    --motion_path motions/master-v2 \
    --experiment_name g1_dribbling_essay_ablation_formal \
    --load_run "2026-08-24_15-26-55_e13_no_recovery_seed13" \
    --checkpoint model_42999.pt \
    --num_envs 1 \
    --seed 13 \
    --device cuda:0 \
    --headless \
    --locomotion_cmd_speed 1.5 1.5 1.5 \
    --locomotion_cmd_heading 0 0.65 -0.65 \
    --locomotion_cmd_duration 5 5 5 \
    --locomotion_cmd_reset_on_end \
    --disable_interval_pushes \
    --video_length 1800 \
    --diagnostic \
    --diagnostic_stride 1 \
    --diagnostic_path output/essay13_ablation_play/no_recovery_seed13/diagnostic.npz \
    --evaluation_case_id no_recovery_seed13_regression \
    --dual_view \
    --cam_layout task_front_side \
    --video_output_dir output/essay13_ablation_play/no_recovery_seed13/video
'
```

## Native Conda machine

Activate the Conda environment that is bound to the Ablation repository before
running either command:

```bash
conda activate /home/teambruce/users/syf/conda_envs/isaaclab45_211_ablation
cd /home/teambruce/users/syf/projects/ab/Footmimic
```

The `soccer` package resolved by this environment should point to
`/home/teambruce/users/syf/projects/ab/Footmimic/source/whole_body_tracking`.

### No Stage-I initialization

```bash
/home/teambruce/users/syf/repos/IsaacLab-v2.1.1/isaaclab.sh -p \
    scripts/rsl_rl/play_multi.py \
    --task "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoStage1" \
    --motion_path motions/master-v2 \
    --experiment_name g1_dribbling_essay_ablation_formal \
    --load_run "2026-08-24_04-12-21_e13_no_stage1_seed13" \
    --checkpoint model_19999.pt \
    --num_envs 1 \
    --seed 13 \
    --device cuda:0 \
    --headless \
    --locomotion_cmd_speed 1.5 1.5 1.5 \
    --locomotion_cmd_heading 0 0.65 -0.65 \
    --locomotion_cmd_duration 5 5 5 \
    --locomotion_cmd_reset_on_end \
    --disable_interval_pushes \
    --video_length 1800 \
    --diagnostic \
    --diagnostic_stride 1 \
    --diagnostic_path output/essay13_ablation_play/no_stage1_seed13/diagnostic.npz \
    --evaluation_case_id no_stage1_seed13_regression \
    --dual_view \
    --cam_layout task_front_side \
    --video_output_dir output/essay13_ablation_play/no_stage1_seed13/video
```

### No interaction reference

```bash
/home/teambruce/users/syf/repos/IsaacLab-v2.1.1/isaaclab.sh -p \
    scripts/rsl_rl/play_multi.py \
    --task "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoInteractionReference" \
    --motion_path motions/master-v2 \
    --experiment_name g1_dribbling_essay_ablation_formal \
    --load_run "2026-08-24_15-27-08_e13_no_interaction_reference_seed13" \
    --checkpoint model_42999.pt \
    --num_envs 1 \
    --seed 13 \
    --device cuda:0 \
    --headless \
    --locomotion_cmd_speed 1.5 1.5 1.5 \
    --locomotion_cmd_heading 0 0.65 -0.65 \
    --locomotion_cmd_duration 5 5 5 \
    --locomotion_cmd_reset_on_end \
    --disable_interval_pushes \
    --video_length 1800 \
    --diagnostic \
    --diagnostic_stride 1 \
    --diagnostic_path output/essay13_ablation_play/no_interaction_reference_seed13/diagnostic.npz \
    --evaluation_case_id no_interaction_reference_seed13_regression \
    --dual_view \
    --cam_layout task_front_side \
    --video_output_dir output/essay13_ablation_play/no_interaction_reference_seed13/video
```

## Files to collect

After all five commands finish, collect these directories from the two machines:

```text
output/essay13_ablation_play/full_seed13/
output/essay13_ablation_play/no_ball_velocity_seed13/
output/essay13_ablation_play/no_recovery_seed13/
output/essay13_ablation_play/no_stage1_seed13/
output/essay13_ablation_play/no_interaction_reference_seed13/
```

For each condition, retain the following files:

- `diagnostic.npz`
- the generated dual-view video
- `params/ablation_manifest.json` from the corresponding training run
- the final training log or learning-curve event files

These five playbacks are matched single-rollout diagnostics. They are suitable
for an initial visual and numerical comparison, but the final paper tables still
require the common multi-scenario evaluation battery.
