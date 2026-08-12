# Dribble / Control Notes

This repository supports one two-stage pipeline.

## Active task IDs

- `Tracking-CG-G1-Motion-RNN-mimic`: Stage 1 motion/style mimic pretraining.
- `Tracking-CG-G1-Dribbling-RNN-control`: Stage 2 continuous speed/heading/duration control.

No aliases or historical environment variants are registered.

## Train

Run both stages:

```bash
DRIBBLE_MOTION_PATH=motions/dribble \
    bash shell/progressive_dribbling_train.sh my_run
```

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
    --motion_path motions/dribble \
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

The cleaned Control environment records 26 active reward terms. Three low-use
guardrails are intentionally retained for early training: illegal-body ball
contact, vertical ball bounce, and excessive ankle-contact force. The angular
velocity reward now tracks a bounded yaw-rate target generated from heading
error instead of rewarding zero yaw rate during a commanded turn.

Use `--diagnostic_stride N` to record every Nth control step.

The authoritative MDP inventory and reward analysis is
[MIMIC_CONTROL_MDP_SUMMARY.md](MIMIC_CONTROL_MDP_SUMMARY.md).

## MuJoCo sim2sim

The preserved v4.4 Control export and MuJoCo runner instructions are in
[scripts/mujoco_sim2sim/README.md](scripts/mujoco_sim2sim/README.md).
