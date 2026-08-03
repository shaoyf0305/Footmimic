# Dribble Control

This note documents the current unified dribbling curriculum, contact semantics,
and playback diagnostics. Legacy tasks are kept only for regression.

## Current tasks and training path

- `Tracking-CG-G1-Motion-RNN-unified-s1-mimic`: S1 pure imitation with the
  aligned reference velocity command.
- `Tracking-CG-G1-Dribbling-RNN-unified-s2-reference`: S2 physical dribbling
  with the same reference command and reference contact prior.
- `Tracking-CG-G1-Dribbling-RNN-unified-s3-task`: S3 free task control with
  sampled task commands and no reference-timed ball/contact-position rewards.

All three use the same strict 163-D actor input and 29-D projected action
interface. Train them with:

```bash
bash shell/progressive_dribbling_train.sh my_run --cg-unified-3stage
```

| Stage | Command | Contact objective |
| --- | --- | --- |
| S1 | Exact task-frame reference velocity | No ball-contact objective; learn motion prior. |
| S2 | Exact task-frame reference velocity | Execute physical dribbling and follow the CG contact prior. |
| S3 | Normal sampled task command | Retain task ball control and generic legal `instep` contacts; do not require the reference foot/contact sequence. |

Resume only in the `S1 -> S2 -> S3` order. The old `unified-mimic` /
`unified-control` pair and other `control` tasks are frozen compatibility
baselines, not interchangeable resume sources for this curriculum.

Use `scripts/rsl_rl/play_multi.py` for playback. Manual
`--locomotion_cmd_speed`, `--locomotion_cmd_heading`, and
`--locomotion_cmd_duration` select manual command mode.

## Contact semantics

`dribble_contact_mode` selects the leg (`right`, `left`, or `both`), while
`dribble_contact_surface` selects `any`, `instep`, `inside_instep`, or
`outside_instep`. The contact region is evaluated from the ball position
relative to the contacted ankle, expressed in the pelvis-yaw frame:

- `instep`: dorsal side (`+Z`), either medial/lateral side.
- `inside_instep`: dorsal and medial side.
- `outside_instep`: dorsal and lateral side.

S2/S3 use `instep`: this enforces a dorsal instep contact without forcing an
inside/outside sequence. The geometry and force threshold are shared by the
reward and diagnostic telemetry. A surface-labelled dataset can additionally
provide `"surface": "inside_instep"` or `"outside_instep"` per contact segment;
review automatic labels before training on them.

## Design constraints

- Treat a demo as a motion/style prior, not a task tape. Episode lifetime and
  task success must not be tied to clip length.
- Keep the conflict between an oblique command and a ball ahead of the robot:
  the policy must turn while retaining the ball, not move the ball to the
  command heading.
- Keep chase and sustained-control incentives when improving turns.
- S1/S2 use the aligned reference command. S3 uses sampled task commands while
  tracking the motion prior; that intentional conflict is the final transfer
  problem to solve.

## Play diagnostics

Run with `--diagnostic --diagnostic_stride 1` (one sample per simulator step).
The `.npz` archive is written to the run's `diagnostics/` directory and uses the
`dribble-v2` schema.

### Stage checks

- **S1/S2 alignment:** `command_mode` must be `reference`.
  `active_command_lin_vel_w`, `effective_command_speed`, and
  `effective_command_heading` are the aligned reference command, not a stale
  manual-command buffer.
- **S3 transfer:** normal playback reports `command_mode=resampled`; supplying
  CLI command segments reports `manual`. At zero requested speed there is no
  defined heading, so `heading_error` and `ball_command_forward_speed` are
  intentionally `NaN`.
- **S2 reference contact:** where `cg_label_available && cg_reference_contact`,
  compare `cg_reference_foot` / `cg_reference_surface` with `contact_foot` and
  `cg_contact_foot_match`. `contact_cg_gate_pass` is the configured CG gate;
  `contact_cg_eligible` additionally requires a current legal touch.
- **S3 contact flexibility:** CG labels are trace data only. Judge S3 by
  possession, task velocity/heading, and generic legal instep contacts rather
  than requiring each touch to match the reference foot sequence.

`contact_cg_eligible` is intentionally not a full reward reconstruction:
reward-specific temporal, new-touch, or directional gates remain in the reward
term trace.

### Contact diagnosis

First filter on `ball_contact`. Then inspect, in order:

1. `contact_body_idx` (index into `contact_body_names`) and `contact_foot`
   (left=`0`, right=`1`, unknown=`-1`).
2. `contact_legal_ankle` and `contact_generic_instep`.
3. `contact_inside_instep` / `contact_outside_instep` when a sided surface is
   expected.
4. `contact_requested_surface_match`, `contact_gentle`, and
   `contact_legal_touch`.

`contact_ball_offset_pelvis_yaw` gives the ball offset in the pelvis-yaw frame;
it is `NaN` when there is no robot-ball contact.

### Action and terminal-transition diagnosis

`policy_action` is the raw network output; `submitted_action` is the action
passed to `env.step` after playback-only constraints; `effective_action` is the
result after action-layer projection, limits, and filtering. `applied_action`
is retained only as a backward-compatible alias of `submitted_action`. The
equivalent trunk fields are `trunk_policy_action`, `trunk_submitted_action`, and
`trunk_effective_action`.

State, reference, contact, policy action, and submitted action are pre-step
samples. Reward, done, effective action, and physical target belong to the
resulting transition. `action_snapshot_available` confirms that the action-layer
snapshot is present. On a terminal transition, `post_step_state_valid` is false
and live post-step state/torque arrays are `NaN`, preventing the next episode's
reset pose from being attributed to the terminal action.

For waist/trunk diagnosis, compare `waist_joint_err`,
`waist_effective_action_step`, `torso_rel_tilt`, `torso_rel_tilt_err`, and
`torso_rel_ang_vel`. The summary also reports legal/generic-instep/CG-eligible
touch rates, CG-foot-match rate, action-snapshot coverage, and valid post-step
state coverage.

## Before changing the task

- Preserve the 163-D/29-D S1/S2/S3 interface.
- Keep changes curriculum-local or behind a default-off compatibility flag.
- Diagnose command mode, contact geometry, and action execution before changing
  rewards or motion data.
