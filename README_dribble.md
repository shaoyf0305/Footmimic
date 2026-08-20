# Dribble / Control Notes

This repository supports one two-stage pipeline.

## Active task IDs

- `Tracking-CG-G1-Motion-RNN-mimic`: Stage 1 motion/style mimic pretraining.
- `Tracking-CG-G1-Dribbling-RNN-control`: Stage 2 continuous speed/heading/duration control.

No aliases or historical environment variants are registered.

## Essay 13 baseline and current upper-body candidate

The last frozen full-method baseline is commit
`a589bd71168bd876fe4db93a3d887039c94005a8` (`essay 13`). Its representative
1800-step `e13.npz` rollout has zero failure terminations, `0.219 m/s` filtered
ball-speed MAE, `0.301` predicted-position error, `0.0565 rad` heading MAE, and
one undesired-contact frame. Median right-foot contact interval is `0.86 s`,
and the maximum per-link contact force is `39.7 N`.

The current working tree intentionally diverges from that baseline only in the
Stage-2 upper-body action path. It is a new candidate and must be evaluated
against E13 before being frozen; its results cannot reuse the Essay 13 label.

## Train

Run both stages:

```bash
MIMIC_MOTION_PATH=motions/master-modified \
CONTROL_MOTION_PATH=motions/master-v2 \
    bash shell/progressive_dribbling_train.sh my_run
```

Stage 1 keeps the existing broad mimic/style bank. Stage 2 sees only the single
right-foot `master-v2` clip; the script does not mirror it or mix in another
kick dataset. The script applies `--migrate_legacy_upper_body_residual` exactly
once at the Stage-1-to-Stage-2 boundary, preserving the lower body and shared
network while initializing the 14 arm rows for the direct residual interface.
Checkpoints saved by Stage 2 are v4 and must not use that flag when resumed.

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
- actions, joint tracking, command, ball, contact, torque, and termination telemetry

The cleaned Control environment records 20 active reward terms. Normal control
and ball recovery share one predicted position--velocity state. Inside the
controllable region the pelvis follows the locomotion command; outside it, a
smooth recovery gate blends toward the filtered ball velocity plus a bounded
position correction. The `+7.5` positive ball-speed budget is now one direct,
asymmetric two-sided velocity-vector tracking term rather than separate progress
and tracking objectives. Forward speed receives full credit from `target-0.05`
through `target+0.20 m/s`; its score is softly reduced during recovery. The
instantaneous command-relative Huber excess penalty now begins at
`target+0.30 m/s` and remains the high-speed safety envelope. Stage-2 training
samples speed uniformly over `0.40--1.65 m/s`, placing the common `1.5 m/s`
evaluation command inside rather than at the edge of the training range.

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

Per-link force columns are accepted only when the runtime PhysX filter count
exactly matches the configured body-filter count; the old silent truncation is
removed. A present-but-mismatched matrix fails fast instead of training with a
geometry guess; only runtimes without a matrix use the compatibility fallback.
Diagnostics store both articulation body names and collision-semantic
labels. In particular, `right_knee_link` owns the complete lower-leg cylinder
and is reported as `right_shin_collision`. The first control step after reset is
excluded from contact truth to remove the stale one-frame initialization impulse.
The contact-hold cache is validated against its actual state fields and cleared
inside the reset guard, so its documented two-step hold now persists within an
episode without leaking across episodes.

Diagnostics also store `bounded_upper_body_policy_action`. It is true for the
current Stage 2 controller, whose 14 Gaussian pre-squash arm variables are
bounded by one environment-side tanh, and false for Stage 1. They additionally store
the actor raw mean and squashed action; policy, commanded, and executed arm
residuals; and separate actor-boundary and soft-joint-limit fractions. The old
PCA/manifold telemetry has been removed with the inactive execution path.

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

The frozen implementation applies the initial 50-step grace by absolute episode
age. An early valid touch does not cancel the unused part of that grace. E13
therefore contains one reset-adjacent `2.60 s` no-contact tail even though its
typical interval improved to `0.86 s` median and `1.55 s` P90. Keep this logic
identical across Essay 13 ablations and report median, P90, and reset-inclusive
maximum interval. Changing to an `ever_contact` latch creates a new baseline.

The 14 arm outputs in Stage 2 are reference-relative residuals while the full
29-D action interface remains unchanged. PPO stores ordinary Gaussian
pre-squash variables for every dimension; the Stage-2 action term applies
exactly one tanh to the 14 arm variables. This avoids inverse-tanh likelihood
reconstruction at float32 saturation; the fixed transform's Jacobian cancels
in PPO's new/old ratio. Lower-body actions remain Gaussian. Zero arm action
means the live motion reference exactly. The correction is applied directly as
`q_ref + 0.25 * action`: all 14 arm joints currently use the same `0.25 rad`
margin. PCA tangent projection, orthogonal
residual squashing, the second `±0.25 rad` envelope, and the 1.8 Hz residual
filter have been removed. Simulator soft joint limits and the existing PD
actuator remain physical execution safeguards. The policy still observes the
final normalized absolute joint target, while `action_rate_l2` measures arm
residual changes so reference motion is not counted as policy jitter.
RSL-RL's directly optimized scalar exploration standard deviation is projected
to a `0.05` floor before constructing the Normal distribution; this is a
distribution-validity guard, not another action-space squash. NaN/Inf still
fails immediately with the affected action indices.

Pre-residual checkpoints without an action marker require this one-time flag:

```text
--migrate_legacy_upper_body_residual
```

Essay 11 checkpoints carrying `reference_residual_v1` instead require:

```text
--migrate_bounded_upper_body_policy
```

Both explicit legacy paths preserve the lower-body output, hidden layers,
critic, and observation normalizer while resetting only the 14 incompatible
arm output rows. Essay 12/13 checkpoints carrying
`bounded_reference_residual_v2`, as well as checkpoints from the unstable
`direct_reference_residual_v3` attempt, need no flag. The loader preserves the
lower-body/shared/RNN/critic/normalizer weights, resets only the 14 saturated
arm output rows and their std, and drops the old optimizer once. Newly saved
checkpoints carry `pre_squash_reference_residual_v4` and resume normally with
their optimizer. Never pass either migration flag for v2, v3, or v4.

Use `--diagnostic_stride N` to record every Nth control step.

## Ablation evaluation

Use the same Stage-1 initialization, Stage-2 motion clip, training budget, seed
set, command curriculum, reset/termination, and evaluation sequence for every
variant. Report at least three independent training seeds (prefer five when
compute permits), and evaluate speeds `0.5/1.0/1.5 m/s` with headings
`0/+0.65/-0.65 rad`.

The fixed primary metrics are success rate, filtered ball-speed MAE, predicted
ball-position error, heading MAE, contact-interval median/P90/maximum,
useful/rapid/undesired contacts, maximum per-link force, reference tracking,
action-rate and joint-limit contributions, and upper-body actor boundary
pressure. Read effective weights and contributions from each diagnostic's
`reward_term_weights`, `reward_term_step_weights`, and `reward_terms`; do not
infer them from source configuration alone.

The complete current Stage-1/Stage-2 reward inventory and action-quality
correction are maintained in [MDP_SUMMARY.md](MDP_SUMMARY.md). Future MDP
changes update that living summary directly; the numbered interim summaries
have been removed.

## MuJoCo sim2sim

The preserved v4.4 Control export and MuJoCo runner instructions are in
[scripts/mujoco_sim2sim/README.md](scripts/mujoco_sim2sim/README.md).
