# Dribble / Control Agent Prompt

Read this before changing control-related code. Follow the constraints below.

## Task IDs

- Tracking-CG-G1-Motion-RNN-mimic: Stage-1 motion style from demo. Preferred warm-start for control.
- Tracking-CG-G1-Motion-RNN-task: Stage-1 with fixed +X / anti-lateral terms. Do not use as control warm-start.
- Tracking-CG-G1-Dribbling-RNN-forward: Stage-2 fixed task +X dribbling.
- Tracking-CG-G1-Dribbling-RNN-follow: Stage-2 root velocity from demo anchor vel.
- Tracking-CG-G1-Dribbling-RNN-control: Stage-2 external locomotion command (speed, heading, duration).

Play entry: scripts/rsl_rl/play_multi.py with dual_view and locomotion_cmd_speed / heading / duration.
Train helper: shell/progressive_dribbling_train.sh with --cg --cg-control.
Stage-1 to control obs expansion (+9) is handled by load_checkpoint_with_obs_expand via MotionOnPolicyRunner.load.

## Goals you must respect

Demo is a motion-style library, not a task tape. Episode lifetime must not equal clip length. Behavior phase must not be forced to demo frame t. Clip end must not reset the whole episode. The episode uses its own clock. Demo supplies gait, touch style, and pose style. Root motion and success come from the command plus ball control.

Ball chase and sustained ball control matter. Do not cut chase or contact rewards just to make turning easier. Short possession time is already a problem.

The geometric conflict between an oblique command and a ball ahead of the robot is intentional. Do not spawn or move the ball onto the command heading to remove that conflict. The policy should learn to turn while keeping the ball.

Do not resume control from a strongly trained forward checkpoint. Prefer Motion-RNN-mimic into Dribbling-RNN-control.

Slalom clips already contain turns, but current control does not use demo yaw as the locomotion target. Mimic strips demo yaw into the task frame and root velocity comes from resampled commands. Having turns in the data does not mean the policy is learning to follow heading commands.

## What play tests already showed

The CLI to polar command to HUD path works. The failure is policy and MDP structure, not wrong play flags.

Speed test (0 to 0.55 to 0): commanded stop still walks forward. Speed command is weak or overridden by demo and chase.

Heading test (0 then large plus or minus angles): robot mostly stays on task +X. Heading following is unreliable.

A later control run followed direction about half the time: straight segments look good, oblique segments are inconsistent. Reward-only tuning is not enough.

Structural cause: control still advances time_steps each step, indexes demo pose and CG labels by that frame, and on clip end calls _resample_command which resets robot and ball. motion_body_pos and related terms still track demo frame by frame, so playback looks like replaying the clip.

Secondary signal cause: soft motion_anchor_lin_vel (old std 0.8) and weak direction terms let the policy ignore oblique commands. Control already zeros forward_velocity. Walking forever along +X comes from ball spawn ahead, chase and CG rewards, and demo gait, not from a leftover +X velocity reward.

## Do not do

Do not align ball spawn or mid-episode ball placement to command heading.

Do not weaken core chase or CG weights to buy turning.

Do not treat more locomotion reward weight as the main fix.

Do not change shared MotionCommand defaults in a way that breaks forward, follow, or Stage-1. Any shared change must be behind a config flag that defaults to the old behavior.

Do not make control follow demo root velocity again.

## Planned changes

Phase A is the main fix: decouple task time from clip time, control only. Add a config flag such as motion_clip_end_resample default True, set False only in control. On clip end, wrap or switch style phase without full episode resample. Episode ends only from episode_length_s and task terminations such as ball_lost and dribbling_no_contact. In play, restart manual polar sequences on episode reset, not on every clip switch. Shared code may change only if the old path remains the default.

Phase B: treat demo as style, not frame tracking. Lower motion_body_pos under control or use a loose phase band. Reduce CG dependence on absolute time_steps; keep foot choice and touch style with contact and distance windows. Drive demo reads from a separate style_phase clock.

Phase C: learn turning under conflict without dropping chase. Keep strong locomotion tracking including near-zero speed and excess-speed penalty. Add a reward that aligns ball xy velocity with command heading when in contact or near the ball. Optionally relax no-contact grace so a turn does not instantly end the episode. Train mimic then new control. Validate with the play tests below.

Phase D acceptance: speed test shows pelvis speed follow stop and go. Heading test shows Robot actual degrees follow plus or minus command with possession not collapsing. Clip looping inside one episode must not wipe the scene.

## Play diagnostics

Speed channel:
locomotion_cmd_speed 0.0 0.55 0.0
locomotion_cmd_heading 0.0 0.0 0.0
locomotion_cmd_duration 5.0 5.0 5.0

Heading channel:
locomotion_cmd_speed 0.5 0.5 0.5
locomotion_cmd_heading 0.0 0.75 -0.75
locomotion_cmd_duration 4.0 6.0 6.0

Compare Loco cmd to Robot actual, hdg_err, and Pelvis v_xy. The track percent field is velocity-direction cosine only. It does not prove heading tracking.

## Older baseline (do not redo)

Already in place: no-contact termination, velocity tracking, world-frame forward terms, CG contact timing prior, orientation reward removed for slalom, distance-based dribble motions, higher friction, rapid-retouch penalty, anti-crab terms with face_ball, task_heading, lateral penalties, and forward_dominance gating.

Reverted: phased touch or chase velocity targets and related terminations and HUD from the v1.7 to v1.13 experiments.

Tooling kept: video_to_npz_pipeline.sh, play_multi dual view and HUD, dual_view_recorder.

Latent AMP or VAE skill libraries come after this control structure work.

## Checklist before you edit

Confirm the change is control-only or behind a flag defaulting off for other tasks.

Prefer Phase A over reward-only edits.

Do not implement ball-to-command alignment.

Do not weaken core chase rewards.

After training, require Test A and Test B plus dual_view HUD before claiming heading works.
