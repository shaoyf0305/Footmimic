# Dribble / Control Notes

This repository supports one two-stage pipeline.

## Active task IDs

- `Tracking-CG-G1-Motion-RNN-mimic`: Stage 1 motion/style mimic pretraining.
- `Tracking-CG-G1-Dribbling-RNN-control`: Stage 2 continuous speed/heading/duration control.

No aliases or historical environment variants are registered.

## Train

Run both stages:

```bash
MIMIC_MOTION_PATH=motions/master-modified \
CONTROL_MOTION_PATH=motions/master-v2 \
    bash shell/progressive_dribbling_train.sh my_run
```

Stage 1 keeps the existing broad mimic/style bank. Stage 2 sees only the single
right-foot `master-v2` clip; the script does not mirror it or mix in another
kick dataset.

Stage 1 and Stage 2 use the same ordered network inputs: 163 actor dimensions
and 295 critic dimensions. Ball position, locomotion speed/heading, and the
simulation ball velocity relative to command heading each use one polar
representation. The unused destination plus redundant Cartesian linear/angular
command inputs have been removed. The checkpoint loader appends the new
velocity inputs with zero model weights and neutral normalizer statistics, so
the preceding 160/292-D Stage 1 and Stage 2 checkpoints can still be
warm-started.

## Play

```bash
python scripts/rsl_rl/play_multi.py \
    --task Tracking-CG-G1-Dribbling-RNN-control \
    --motion_path motions/master-v2 \
    --experiment_name g1_dribbling \
    --load_run <RUN_DIR> \
    --checkpoint model_XXXX.pt \
    --num_envs 1 --device cuda:0
```

Manual command sequences use matching lists of speed, heading, and duration:

```bash
--locomotion_cmd_speed 1.5 1.5 1.5 \
--locomotion_cmd_heading 0.0 0.65 -0.65 \
--locomotion_cmd_duration 5 5 5
```

The active Control policy does not have a separate IDLE/DRIBBLE/STOP observation.
A zero-speed command is therefore a counterfactual evaluation input, not a learned
high-level stop state.

## Diagnostics

Add `--diagnostic` to write a compressed NumPy archive under the checkpoint
run's `diagnostics/` directory. New archives include:

- `reward_term_names`
- `reward_term_weights`
- `reward_term_step_weights`
- `reward_step_dt`
- `reward_terms`
- `step_reward`
- actions, joint tracking, command, ball, contact, manifold, torque, and termination telemetry

The cleaned Control environment records 28 active reward terms. The
`dribbling_ball_velocity_tracking` term rewards command-relative speed and
direction tracking during right-foot possession using a reset-safe 10-step
(0.2 s) ball-velocity EMA. It shares the original `+7.5` positive speed budget
with forward progress (`+2.5` tracking and `+5.0` progress) so speed regulation
cannot hide degraded motion quality behind extra task reward. The existing
ball-speed penalty remains an instantaneous command-relative Huber safety
envelope that retains a gradient at high speed. Three low-use guardrails are
intentionally retained for early training: illegal-body ball contact, vertical
ball bounce, and excessive ankle-contact force. The angular
velocity reward now tracks a bounded yaw-rate target generated from heading
error instead of rewarding zero yaw rate during a commanded turn.

The current Control task accepts only right-ankle ball contact as legal. Both
feet remain in gait tracking, but a left-foot ball touch is an undesired
contact. No mirrored or additional kick data is required.

Current contact truth uses filtered per-link robot contact with a two-control-step
sensor hold. A real right-ankle touch also starts a 30-control-step possession
window shared by progress, proximity, coast, orbit, and gait reward gates. This
restores task coverage between discrete touches without treating ball-ground
friction as robot contact. Diagnostics record both contact channels, possession,
and the ball's net XY contact force; net force remains telemetry only. CG contact
is scored per annotated contact window instead of rewarding easy no-contact matches.

After an episode termination, playback clears the recurrent policy hidden state
while leaving the active manual speed/heading segment unchanged. Thus a physical
scene reset cannot inherit a failed episode's LSTM memory, but command-sequence
timing retains the continuous-Control behavior used by the v5 baseline.

Use `--diagnostic_stride N` to record every Nth control step.

The current MDP inventory and action-quality correction are documented in
[MDP_SUMMARY_08.md](MDP_SUMMARY_08.md). `MDP_SUMMARY_07.md` retains the first
instantaneous speed-feedback implementation, and `MDP_SUMMARY_06.md` retains
the preceding possession/recovery implementation.

## MuJoCo sim2sim

The preserved v4.4 Control export and MuJoCo runner instructions are in
[scripts/mujoco_sim2sim/README.md](scripts/mujoco_sim2sim/README.md).
