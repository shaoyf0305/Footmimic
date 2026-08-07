# Local-twist dribbling curriculum

The active three-stage path uses one 163-D actor interface and one 29-D action
interface.  Its locomotion command is always the current-pelvis local twist:
`[vx_local, vy_local, wz]`.  There is no required world heading or accumulated
yaw target, which matches real-robot deployment.

Train the complete path with:

```bash
bash shell/progressive_dribbling_train.sh my_run --cg-unified-3stage
```

| Stage | Task ID | Objective |
| --- | --- | --- |
| S1 | `Tracking-CG-G1-Motion-RNN-unified-s1-local-strict` | Strict relative motion/gait imitation and exact reference local twist; no ball objective. |
| S2-A | `Tracking-CG-G1-Dribbling-RNN-unified-s2-local-reference` | One physical touch in a ±0.1 s reference event window. |
| S2-B | `Tracking-CG-G1-Dribbling-RNN-unified-s2-two-contact` | 30% one-touch and 70% adjacent two-touch episodes. |
| S2-C4 | `Tracking-CG-G1-Dribbling-RNN-unified-s2-four-contact` | 20/30/50% one/two/four-touch mixture. |
| S2-C8 | `Tracking-CG-G1-Dribbling-RNN-unified-s2-eight-contact` | 10/20/30/40% one/two/four/eight-touch mixture. |
| S2-full | `Tracking-CG-G1-Dribbling-RNN-unified-s2-full-contact` | 10/20/30/40% one/two/four/full-clip mixture. |
| S3 | `Tracking-CG-G1-Dribbling-RNN-unified-s3-local-task` | Sampled local task twist with generic physical `instep` contacts; no reference ball position, contact time, contact foot, or foot--ball distance reward. |

S1 preserves the raw clip and emits a time-limit `done` on its final frame.
S2 contact-count tasks start 0.3--0.6 s before their first selected event and
end after the last selected contact window; S2-full retains complete-clip
samples. The normal environment reset resets robot, ball, and LSTM state
together. S3 instead loops the same master clip through a
25-frame (0.5 s at 50 Hz) quintic bridge, blending its tail into its start
without resetting the ongoing task scene.

At reset, all three stages retain the selected reference pelvis pose, yaw, and
velocity; they do not canonicalize the robot to a simulation/world `+X` axis.
For S2 the physical ball is initialized at the selected first contact target
with a 5--8 cm planar perturbation. Internally this point is stored as a
pelvis-local offset, so the same reset geometry remains valid under any
simulator yaw. If a motion lacks either a contact label or
`ball_pos_w`, the explicit fallback is 0.45 m ahead in the frame-0 pelvis
local frame. S1 applies no ball reward; it retains this identical spawn only
because the unified 163-D observation contains the ball-relative input.

S3 samples `vx∈[0,1.50] m/s`, `vy∈[-0.50,0.50] m/s`, and
`wz∈[-0.80,0.80] rad/s`, with 10% exact `[0,0,0]` standing commands. For its
first 24,000 environment steps (1,000 PPO updates at 24 rollout steps), the
effective command blends from the S2 reference twist to the sampled task
twist. All root-velocity rewards and policy inputs use that same effective
command.

The legacy `unified-s1-mimic`, `unified-s2-reference`, and `unified-s3-task`
IDs remain frozen compatibility baselines; do not resume them into this local
curriculum.

## Interface and deployment

The final policy input is the 3-D local twist.  A retained 3-D DRIBBLE one-hot
field is only an interface-compatibility placeholder, so the actor remains
163-D; it is not a second command.  The actor does not receive simulated local
linear velocity.  The critic receives that privileged pelvis-local velocity
for value learning.

For one manual command, use `--locomotion_cmd_vx`, `--locomotion_cmd_vy`, and
`--locomotion_cmd_wz`; on the local tasks these are in the current pelvis frame.
For repeatable multi-segment playback, pass
`--locomotion_cmd_plan motions/command_plan_dribble_left_right_local.json`.
The JSON plan uses direct pelvis-local `vx`, `vy`, `wz`, and `duration_s` per
segment; it never exposes a polar command or IDLE/STOP state to the policy.

## Contact semantics

S2 converts each labelled inside/outside event to foot-yaw-local left/right.
Its target region uses forward `[-0.06, 0.14] m`, lateral magnitude
`[0.04, 0.16] m`, and a 4 cm side dead zone. The only S2 ball terms are contact
proximity, one new correct-foot touch, correct-side bonus, and wrong-foot/body
contact penalty. Between events no ball trajectory or velocity is supervised.

Use `--show_s2_contact_regions` during playback to display the frozen reference
foot frame and the inner/outer side boundaries. Combine it with
`--diagnostic --diagnostic_stride 1` to save event id/frame, expected foot,
expected side, reference foot pose, ball offset, and target-region distance.

The cumulative ablation task IDs are
`unified-s2-ablation-motion`, `unified-s2-ablation-time`,
`unified-s2-ablation-foot`, and `unified-s2-ablation-side` (all prefixed by
`Tracking-CG-G1-Dribbling-RNN-`). They respectively add contact timing,
specified-foot gating, and foot-local side gating while keeping the same
single-contact initialization distribution.

All S2 contact-curriculum tasks require valid `dribble_cg_contact`,
`dribble_cg_foot`, `dribble_cg_surface`, and `ball_pos_w` arrays. At this
revision the compatible checked-in bank is `motions/master-single`; passing a
legacy mixed bank fails during environment creation instead of training with
unknown side labels.

`instep` means a dorsal contact region on either foot. S3 does not force either
side or a reference left/right alternation, keeping recovery and consecutive
same-side touches feasible when the task requires them.

## Diagnostic

Run playback with `--diagnostic --diagnostic_stride 1`.  The `dribble-v4`
archive records `reference_twist_local`, `active_twist_local`,
`actual_twist_local`, `twist_local_error`, and `twist_blend_alpha` alongside
contact and action traces. It also records the virtual style phase, its source
frame pair, and the seam blend factor; these seam fields are active only in
S3. `ball_spawn_source`, `ball_spawn_reference_contact_frame`, and
`ball_spawn_reference_local` record whether reset used the labelled first
contact point or the deliberate local-front fallback. For S1/S2, check local twist error and reference
foot/contact agreement.  For S3, judge local twist tracking, possession,
generic legal instep touches, and ball progress; do not score CG-foot match as
an S3 requirement.
