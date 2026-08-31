# Reported diagnostic data

This directory contains the compact diagnostic traces used for the quantitative
results in the accompanying paper. The policy checkpoints are not included.

| File | Scope | Evaluation seed |
|---|---|---:|
| `full_method/full_36s_e13.npz` | Formal full method, 1,800-step aggregate scripted trace | 13 (launch setting) |
| `full_method/full_120s_seed42.npz` | Formal full method, uninterrupted 6,000-step trace | 42 |
| `ablations_36s/no_ball_velocity_seed13.npz` | Ball-velocity observation and reward removed | 13 |
| `ablations_36s/no_recovery_seed13.npz` | Recovery target blending and velocity gate removed | 13 |
| `ablations_36s/no_stage1_seed13.npz` | Stage-I initialization removed | 13 |
| `ablations_36s/no_interaction_reference_seed13.npz` | Dense distance and touch-window reference removed | 13 |

The 36-s traces use 50-Hz control and contain two complete 15-s command
sequences followed by the first 6 s of a third sequence. A scheduled environment
reset occurs after each complete 15-s sequence. The 120-s trace runs for 6,000
control steps without an environment reset.

These files are single-seed evidence. They should not be interpreted as
multi-seed estimates of training variance. Integrity hashes are recorded in
`SHA256SUMS`. The older full-method file predates embedded evaluation-seed
metadata, so its seed is recorded from the launch setting rather than the NPZ.

Large raw evaluation archives, videos, and generated output directories are
kept outside the Git submission. The evaluation runners under `scripts/rsl_rl/`
can generate new diagnostics from an explicitly supplied checkpoint.
