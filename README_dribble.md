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

The cleaned Control environment records 21 active reward terms. Normal control
and ball recovery share one predicted position--velocity state. Inside the
controllable region the pelvis follows the locomotion command; outside it, a
smooth recovery gate blends toward the filtered ball velocity plus a bounded
position correction. The `+7.5` positive ball-speed budget is now one direct,
two-sided velocity-vector tracking term rather than separate progress and
tracking objectives. Its score is softly reduced during recovery, while the
instantaneous command-relative Huber excess penalty remains the high-speed
safety envelope.

The current Control task accepts only right-ankle ball contact as legal. Both
feet remain in gait tracking, but a left-foot ball touch is an undesired
contact. No mirrored or additional kick data is required.

Current contact truth uses filtered per-link robot contact with a two-control-step
sensor hold. There is no reward possession timer. A gentle right-ankle touch is
rewarded only if the combined predicted-position and ball-velocity error is
lower five control steps later; the CG contact window is a soft timing
multiplier on that useful-touch score. Diagnostics record the recovery gate,
predicted ball position, closing speed, blended pelvis target, useful-touch
improvement, both contact channels, and the ball's net XY contact force. Net
force remains telemetry only.

After an episode termination, playback clears the recurrent policy hidden state
while leaving the active manual speed/heading segment unchanged. Thus a physical
scene reset cannot inherit a failed episode's LSTM memory, but command-sequence
timing retains the continuous-Control behavior used by the v5 baseline.

Use `--diagnostic_stride N` to record every Nth control step.

The complete current Stage-1/Stage-2 reward inventory and action-quality
correction are maintained in [MDP_SUMMARY.md](MDP_SUMMARY.md). Future MDP
changes update that living summary directly; the numbered interim summaries
have been removed.

## MuJoCo sim2sim

The preserved v4.4 Control export and MuJoCo runner instructions are in
[scripts/mujoco_sim2sim/README.md](scripts/mujoco_sim2sim/README.md).
