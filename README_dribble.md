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

Stage 1 and Stage 2 use the same ordered network inputs: 160 actor dimensions
and 292 critic dimensions. Ball position and locomotion speed/heading each use
one polar representation. The unused destination plus redundant Cartesian
linear/angular command inputs have been removed. The checkpoint loader selects
the corresponding legacy columns by meaning, so older Stage 1 and Stage 2
checkpoints can still be warm-started.

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

The cleaned Control environment records 27 active reward terms. Three low-use
guardrails are intentionally retained for early training: illegal-body ball
contact, vertical ball bounce, and excessive ankle-contact force. The angular
velocity reward now tracks a bounded yaw-rate target generated from heading
error instead of rewarding zero yaw rate during a commanded turn.

The current Control task accepts only right-ankle ball contact as legal. Both
feet remain in gait tracking, but a left-foot ball touch is an undesired
contact. No mirrored or additional kick data is required.

Current reward/termination gates use filtered per-link robot contact with a
two-control-step post-contact hold. Diagnostics separately record this decision
signal, the instantaneous per-link signal, and the ball's net XY contact force;
the net force is telemetry only because it also contains ball-ground friction.
CG contact is scored per annotated contact window instead of rewarding the
large number of easy no-contact/no-contact frame matches.

Use `--diagnostic_stride N` to record every Nth control step.

The current MDP inventory and contact-event update are documented in
[MDP_SUMMARY_05.md](MDP_SUMMARY_05.md). `MDP_SUMMARY_04.md` is retained as the
history of the preceding net-force decision implementation.

## MuJoCo sim2sim

The preserved v4.4 Control export and MuJoCo runner instructions are in
[scripts/mujoco_sim2sim/README.md](scripts/mujoco_sim2sim/README.md).
