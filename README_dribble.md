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

The cleaned Control environment records 20 active reward terms. Normal control
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
sensor hold. There is no reward possession timer. A new gentle right-ankle
touch receives 25% of the useful-touch score immediately; the remaining 75%
depends on whether combined predicted-position and ball-velocity error is lower
five control steps later. The two parts sum to at most one event score. Useful
touch and rapid-retouch terms return `score / control_dt`, so their configured
weights are frequency-independent per-event returns: up to `+1.0` for a fully
useful touch and `-1.0` for a retouch within 14 steps. The CG contact window
remains a soft timing multiplier.
Diagnostics record the base and improvement components separately, together
with the recovery gate, predicted ball position, closing speed, blended pelvis
target, both contact channels, and the ball's net XY contact force. Net force
remains telemetry only.

After any Control episode termination, playback clears the recurrent policy
hidden state. An ordinary failure preserves the active manual segment, target
heading, and remaining duration; only a completed sequence using `reset_on_end`
restarts from segment one. Robot yaw and the stationary ball are reset in the
current effective command frame, with the ball at a safe `0.48--0.58 m` forward
distance. A manual command installed after environment construction performs
the same robot-and-ball synchronization before the first rollout.

`dribbling_no_contact` is active only during DRIBBLE and always accumulates
positively without a valid right-ankle contact. Its normal rate is `1.0` per
control step, reduced to `0.5` while the nearby ball has a similar XY velocity,
or `0.25` during a command-transition window while the pelvis closes on the
ball. Any valid right-ankle contact clears it. IDLE and STOP do not accumulate
this counter, and `ball_lost` is likewise gated to DRIBBLE.

The 14 arm outputs in Stage 2 are reference-relative residuals while the full
29-D action interface remains unchanged. Zero arm action means the live motion
reference exactly. The correction is projected in the motion-bank PCA tangent,
smoothly bounded to `q_ref ± 0.25 rad`, and filtered as a residual rather than
an absolute target. The policy still observes the final normalized joint target,
so the Stage-1/Stage-2 input layout remains consistent. Diagnostics include the
policy, projected, and executed residuals plus their saturation fraction.

Pre-residual checkpoints require an explicit one-time migration during the
first Stage-2 train or play load:

```text
--migrate_legacy_upper_body_residual
```

This temporary compatibility path preserves the lower-body output, hidden
layers, critic, and observation normalizer while resetting only the 14 legacy
arm output rows. Newly saved checkpoints carry the `reference_residual_v1`
marker. Never pass the migration flag when resuming one of those checkpoints;
the loader rejects repeated migration. Once the retained baselines have all
been converted, the clearly marked temporary loader block and CLI flag can be
removed without changing the runtime residual controller.

Use `--diagnostic_stride N` to record every Nth control step.

The complete current Stage-1/Stage-2 reward inventory and action-quality
correction are maintained in [MDP_SUMMARY.md](MDP_SUMMARY.md). Future MDP
changes update that living summary directly; the numbered interim summaries
have been removed.

## MuJoCo sim2sim

The preserved v4.4 Control export and MuJoCo runner instructions are in
[scripts/mujoco_sim2sim/README.md](scripts/mujoco_sim2sim/README.md).
