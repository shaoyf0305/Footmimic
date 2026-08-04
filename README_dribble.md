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
| S2 | `Tracking-CG-G1-Dribbling-RNN-unified-s2-local-reference` | Same reference local twist with a physical ball, reference foot--ball distance, and CG contact prior. |
| S3 | `Tracking-CG-G1-Dribbling-RNN-unified-s3-local-task` | Sampled local task twist with generic physical `instep` contacts; no reference ball position, contact time, contact foot, or foot--ball distance reward. |

S3 samples `vx∈[0.25,1.50] m/s`, `vy∈[-0.50,0.50] m/s`, and
`wz∈[-0.80,0.80] rad/s`.  For its first 24,000 environment steps (1,000 PPO
updates at 24 rollout steps), the effective command blends from the S2
reference twist to the sampled task twist.  All root-velocity rewards and
policy inputs use that same effective command.

The legacy `unified-s1-mimic`, `unified-s2-reference`, and `unified-s3-task`
IDs remain frozen compatibility baselines; do not resume them into this local
curriculum.

## Interface and deployment

The final policy input is the 3-D local twist.  A retained 3-D DRIBBLE one-hot
field is only an interface-compatibility placeholder, so the actor remains
163-D; it is not a second command.  The actor does not receive simulated local
linear velocity.  The critic receives that privileged pelvis-local velocity
for value learning.

For manual playback, use `--locomotion_cmd_vx`, `--locomotion_cmd_vy`, and
`--locomotion_cmd_wz`; on the local tasks these are in the current pelvis frame.

## Contact semantics

`instep` means a dorsal contact region on either foot.  `inside_instep` and
`outside_instep` are available for dataset-labelled ablations, but S3 does not
force either side or a reference left/right alternation.  This keeps recovery
and consecutive same-side touches feasible when the task requires them.

## Diagnostic

Run playback with `--diagnostic --diagnostic_stride 1`.  The `dribble-v3`
archive records `reference_twist_local`, `active_twist_local`,
`actual_twist_local`, `twist_local_error`, and `twist_blend_alpha` alongside
contact and action traces.  For S1/S2, check local twist error and reference
foot/contact agreement.  For S3, judge local twist tracking, possession,
generic legal instep touches, and ball progress; do not score CG-foot match as
an S3 requirement.
