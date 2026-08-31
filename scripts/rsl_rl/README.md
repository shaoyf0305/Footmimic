# Training and evaluation commands

Run these commands inside the Isaac Lab environment or container. The shell
runners can also re-enter the project container when `WORK/run_isaaclab.sh` is
available and the repository is mounted at `/workspace/projects/Footmimic`.

## Validate the formal configurations

Validation checks task registration, verifies that `Ablation-Full` is
configuration-equivalent to the main Stage-II task, and confirms that every
ablation changes only its declared fields.

```bash
/workspace/isaaclab/isaaclab.sh -p \
    scripts/rsl_rl/validate_essay13_ablation_configs.py
```

## Train the ablations

All formal conditions except `no_stage1` must start from the same Stage-I
checkpoint. The migration mode depends on the checkpoint metadata. Use `none`
for a checkpoint that already uses the current bounded residual action,
`legacy-residual` for a pre-residual Stage-I checkpoint, or `bounded-policy` for
an older unbounded reference-residual checkpoint.

```bash
bash scripts/rsl_rl/run_essay13_ablation_training.sh \
    --stage1-load-run <STAGE1_RUN> \
    --stage1-checkpoint <STAGE1_CHECKPOINT> \
    --stage1-migration <none|legacy-residual|bounded-policy> \
    --seeds 13 \
    --num-envs 4096 \
    --max-iterations 100000
```

The default variant list is:

```text
full,no_ball_velocity,no_recovery,no_stage1,no_interaction_reference
```

Use `--variants` to run a subset. For example, the random-initialization
condition does not accept Stage-I resume arguments:

```bash
bash scripts/rsl_rl/run_essay13_ablation_training.sh \
    --variants no_stage1 \
    --seeds 13 \
    --num-envs 4096 \
    --max-iterations 100000
```

`--seeds` accepts a comma-separated list when additional independent training
runs are desired. The supplied paper evidence is single-seed evidence.

Each run writes `params/ablation_manifest.json`. It records the task, seed,
training budget, Stage-I checkpoint hash when applicable, motion-file hashes,
Git state, and launch arguments.

## Evaluate the full method

The baseline suite supports `smoke`, `core`, and `paper` profiles. Its defaults
target the formal E13 full-method run, but the checkpoint file itself is not
versioned in this repository.

```bash
bash scripts/rsl_rl/run_essay13_baseline_suite.sh \
    --profile core \
    --load-run <RUN_DIRECTORY> \
    --checkpoint <CHECKPOINT_FILE> \
    --eval-seeds 13 \
    --videos representative
```

Seed 13 reproduces the reported 36-s scripted evaluation. Additional evaluation
seeds are supported by the runner but were not reported in the paper.

Use `--profile smoke --dry-run` to validate command generation and paths without
launching simulation.

## Evaluate all ablations

Create a whitespace-separated checkpoint table with exactly the five supported
variant names:

```text
variant                    load_run               checkpoint
full                       <RUN_DIRECTORY>         <CHECKPOINT_FILE>
no_ball_velocity           <RUN_DIRECTORY>         <CHECKPOINT_FILE>
no_recovery                <RUN_DIRECTORY>         <CHECKPOINT_FILE>
no_stage1                  <RUN_DIRECTORY>         <CHECKPOINT_FILE>
no_interaction_reference   <RUN_DIRECTORY>         <CHECKPOINT_FILE>
```

Then run the common suite:

```bash
bash scripts/rsl_rl/run_essay13_ablation_evaluation.sh \
    --checkpoint-table <TABLE_FILE> \
    --profile core \
    --eval-seeds 13 \
    --videos representative
```

The launcher records one manifest row per policy and forwards each checkpoint to
the same evaluation runner. Use the `paper` profile only after the checkpoint
selection rule and common training budget have been fixed.

## Output policy

Training logs, checkpoints, videos, diagnostics, and evaluation archives are
written under `logs/` or `output/` and are not tracked by Git. Preserve the
generated manifests alongside any result reported from a checkpoint.
