# FootMimic

FootMimic trains a Unitree G1 humanoid to perform persistent, command-conditioned
football dribbling in Isaac Lab. The final pipeline uses motion retargeting to
construct one 331-frame reference and then trains two recurrent PPO stages. The
second stage controls ball-speed magnitude and heading while retaining the
reference motion and interaction structure.

This submission contains the formal full method, the four ablations reported in
the paper, deterministic evaluation scenarios, diagnostic export, and the motion
reference required by the experiments.

## Environment

The project is designed for Linux with an Isaac Lab Python environment and a
CUDA-capable GPU. The experiment environment used Isaac Lab 2.1.1. Install the
local extension from the repository root with the Isaac Lab Python launcher:

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e source/whole_body_tracking
```

Check that the tasks register before starting a long run:

```bash
/path/to/IsaacLab/isaaclab.sh -p \
    scripts/rsl_rl/validate_essay13_ablation_configs.py
```

## Reference data

Both training stages use the same file:

```text
motions/master-v2/master_001675_001884_unitree_g1.npz
```

It contains 331 frames at 50 Hz. The reference advances by one frame per control
step and is reused cyclically. The simulated robot and ball state do not reset
when the reference wraps.

The data-preparation pipeline wraps the external GVHMR pose estimator and GMR
retargeter, then converts the retargeted motion into the NPZ schema used here:

```bash
GVHMR_ROOT=/path/to/GVHMR \
GMR_ROOT=/path/to/GMR \
    bash shell/video_to_npz_pipeline.sh --batch <NAME>
```

See `bash shell/video_to_npz_pipeline.sh --help` for environment names, partial
pipeline restarts, and output publication options. The final reference file is
already included, so these external repositories are not required for training.

## Training the full method

Run the two stages from an activated Isaac Lab environment:

```bash
bash shell/progressive_dribbling_train.sh final
```

The defaults use `motions/master-v2` for both stages and 4,096 parallel
environments, matching the reported training setup. The main task IDs are:

| Stage | Task ID |
|---|---|
| Motion-reference initialization | `Tracking-CG-G1-Motion-RNN-mimic` |
| Command-conditioned dribbling | `Tracking-CG-G1-Dribbling-RNN-control` |

The script locates the completed Stage-I run and initializes Stage II from that
checkpoint. It applies the one-time action-interface migration required by the
Stage-II reference-relative arm outputs. Override `NUM_ENVS`,
`STAGE1_ITERATIONS`, `STAGE2_ITERATIONS`, `EXPERIMENT_NAME`, or `MOTION_PATH`
only when deliberately changing the training protocol.

## Playback

```bash
python scripts/rsl_rl/play_multi.py \
    --task Tracking-CG-G1-Dribbling-RNN-control \
    --motion_path motions/master-v2 \
    --experiment_name g1_dribbling \
    --load_run <RUN_DIRECTORY> \
    --checkpoint <CHECKPOINT_FILE> \
    --num_envs 1 \
    --device cuda:0
```

A speed-and-heading sequence is supplied with equal-length argument lists:

```bash
--locomotion_cmd_speed 1.5 1.5 1.5 \
--locomotion_cmd_heading 0.0 0.65 -0.65 \
--locomotion_cmd_duration 5 5 5
```

Add `--diagnostic --diagnostic_path <FILE.npz>` to save command, ball, contact,
reward, reference-tracking, action, and termination telemetry.

## Evaluation and ablations

The formal evaluation runner applies the same command, transition, recovery,
and long-horizon scenarios to every checkpoint:

```bash
bash scripts/rsl_rl/run_essay13_baseline_suite.sh \
    --profile core \
    --load-run <RUN_DIRECTORY> \
    --checkpoint <CHECKPOINT_FILE> \
    --eval-seeds 13 \
    --videos representative
```

The paper reports the scripted 36-s evaluation with seed 13 and the separate
uninterrupted 120-s evaluation with seed 42. The runner accepts a comma-separated
seed list for additional evaluation, but such extra seeds are not part of the
reported evidence.

The five registered ablation conditions are the re-trained full control and the
four paper ablations:

| Variant | Task suffix | Removed factor |
|---|---|---|
| `full` | `Ablation-Full` | none |
| `no_ball_velocity` | `Ablation-NoBallVelocity` | explicit ball-velocity observation and reward |
| `no_recovery` | `Ablation-NoRecovery` | recovery target blending and its velocity gate |
| `no_stage1` | `Ablation-NoStage1` | Stage-I checkpoint initialization |
| `no_interaction_reference` | `Ablation-NoInteractionReference` | dense foot-ball distance and touch-window timing reference |

Training and batch evaluation commands are documented in
[`scripts/rsl_rl/README.md`](scripts/rsl_rl/README.md). The launchers keep the
motion file, network configuration, training budget, command distribution, and
evaluation suite fixed across conditions. The no-Stage-I condition alone starts
from random network weights.

## Reported diagnostics

The compact diagnostic files used by the paper are stored under `paper_data/`.
They contain the 36-s full-method trace, the uninterrupted 120-s trace, and the
four 36-s ablation traces. Their provenance, scope, and checksums are documented
in [`paper_data/README.md`](paper_data/README.md). Large evaluation archives,
videos, checkpoints, and generated outputs are intentionally excluded from Git.

## Repository layout

```text
motions/master-v2/                    formal 331-frame reference
paper_data/                            compact diagnostics reported in the paper
source/whole_body_tracking/soccer/    Isaac Lab extension and task implementation
scripts/rsl_rl/                       training, playback, validation, and evaluation
shell/                                two-stage training and reference utilities
```

Checkpoints and generated outputs are intentionally excluded by `.gitignore`.
Supply checkpoints under `logs/rsl_rl/<experiment>/<run>/` or override the run
and checkpoint arguments explicitly.

The repository is distributed under the terms in [`LICENSE.md`](LICENSE.md).
