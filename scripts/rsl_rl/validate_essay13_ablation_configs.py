"""Validate that each Essay13 ablation changes only its declared factor.

Run through Isaac Lab's Python launcher before starting expensive training.
The script starts Kit headlessly so Isaac Sim extensions are available, but it
does not create a simulation environment or run physics:

    /workspace/isaaclab/isaaclab.sh -p \
        scripts/rsl_rl/validate_essay13_ablation_configs.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports below require the running Isaac Sim application."""

import gymnasium as gym

import soccer.tasks  # noqa: F401 - registers tasks
from soccer.tasks.tracking.config.g1.soccer_dribbling_ablation_env_cfg import (
    G1Essay13AblationFullEnvCfg,
    G1Essay13NoExplicitBallVelocityEnvCfg,
    G1Essay13NoInteractionReferenceEnvCfg,
    G1Essay13NoRecoveryBlendingEnvCfg,
    G1Essay13NoStage1InitializationEnvCfg,
)
from soccer.tasks.tracking.config.g1.soccer_dribbling_env_cfg import (
    G1FlatCGDribblingControlEnvCfg,
)
from soccer.tasks.tracking.mdp import observations_anchor as obs_anchor


TASK_IDS = (
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-Full",
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoBallVelocity",
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoRecovery",
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoStage1",
    "Tracking-CG-G1-Dribbling-RNN-control-Ablation-NoInteractionReference",
)

CLOSED_LOOP_TERMS = (
    "motion_anchor_lin_vel",
    "dribbling_dynamic_proximity",
    "dribbling_ball_velocity_tracking",
    "dribbling_useful_foot_touch",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _config_dict(cfg) -> dict:
    result = cfg.to_dict()
    result.pop("ablation_variant", None)
    result.pop("requires_stage1_initialization", None)
    return result


def _different_paths(left, right, prefix: str = "") -> set[str]:
    """Return differing config leaf paths without requiring values to serialize."""
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(_different_paths(left[key], right[key], path))
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return {prefix}
        differences = set()
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.update(
                _different_paths(left_item, right_item, f"{prefix}[{index}]")
            )
        return differences
    if left is right:
        return set()
    try:
        equal = left == right
        if isinstance(equal, bool) and equal:
            return set()
    except (TypeError, ValueError):
        pass
    return {prefix}


def _assert_only_paths(full, variant, expected_paths: set[str]) -> None:
    actual_paths = _different_paths(_config_dict(full), _config_dict(variant))
    _assert(
        actual_paths == expected_paths,
        f"{variant.ablation_variant} changed unexpected paths. "
        f"Expected {sorted(expected_paths)}, got {sorted(actual_paths)}.",
    )


def _assert_full_matches_baseline() -> None:
    baseline = G1FlatCGDribblingControlEnvCfg()
    full = G1Essay13AblationFullEnvCfg()

    # Metadata fields are deliberately additional.  Every inherited manager
    # configuration must otherwise serialize identically.
    _assert(
        not _different_paths(_config_dict(baseline), _config_dict(full)),
        "Ablation Full is not configuration-equivalent to the Essay13 baseline.",
    )


def _assert_velocity_variant() -> None:
    full = G1Essay13AblationFullEnvCfg()
    no_velocity = G1Essay13NoExplicitBallVelocityEnvCfg()

    for group_name in ("policy", "critic"):
        term = getattr(no_velocity.observations, group_name).anchor_ball_velocity_polar_cmd
        _assert(
            term.func is obs_anchor.zero_anchor_ball_velocity_polar_command,
            f"{no_velocity.ablation_variant} did not zero {group_name} velocity input.",
        )
    _assert(
        no_velocity.rewards.dribbling_ball_velocity_tracking is None,
        "Combined velocity variant kept the velocity reward.",
    )
    observation_paths = {
        "observations.policy.anchor_ball_velocity_polar_cmd.func",
        "observations.critic.anchor_ball_velocity_polar_cmd.func",
    }
    reward_paths = {"rewards.dribbling_ball_velocity_tracking"}
    _assert_only_paths(full, no_velocity, observation_paths | reward_paths)


def _assert_recovery_variant() -> None:
    full = G1Essay13AblationFullEnvCfg()
    cfg = G1Essay13NoRecoveryBlendingEnvCfg()
    for term_name in CLOSED_LOOP_TERMS:
        term = getattr(cfg.rewards, term_name)
        _assert(
            term.params["recovery_target_blending_enabled"] is False,
            f"No-recovery variant left blending active in {term_name}.",
        )
    _assert(
        cfg.rewards.dribbling_ball_velocity_tracking.params[
            "minimum_controllability_gate"
        ]
        == 1.0,
        "No-recovery variant did not neutralize velocity controllability gating.",
    )
    _assert(
        cfg.terminations.ball_lost is not None
        and cfg.terminations.dribbling_no_contact is not None,
        "No-recovery variant changed required failure conditions.",
    )
    expected_paths = {
        f"rewards.{term_name}.params.recovery_target_blending_enabled"
        for term_name in CLOSED_LOOP_TERMS
    }
    expected_paths.add(
        "rewards.dribbling_ball_velocity_tracking.params.minimum_controllability_gate"
    )
    _assert_only_paths(full, cfg, expected_paths)


def _assert_reference_variants() -> None:
    full = G1Essay13AblationFullEnvCfg()
    no_stage1 = G1Essay13NoStage1InitializationEnvCfg()
    no_interaction = G1Essay13NoInteractionReferenceEnvCfg()

    _assert(
        no_stage1.requires_stage1_initialization is False,
        "No-Stage-I variant did not disable checkpoint initialization.",
    )
    _assert(
        no_interaction.rewards.dribbling_cg_foot_ball_distance is None
        and no_interaction.rewards.dribbling_useful_foot_touch.params[
            "off_window_reward_scale"
        ]
        == 1.0,
        "Joint interaction-reference variant is incomplete.",
    )
    dense_path = {"rewards.dribbling_cg_foot_ball_distance"}
    timing_path = {
        "rewards.dribbling_useful_foot_touch.params.off_window_reward_scale"
    }
    _assert_only_paths(full, no_stage1, set())
    _assert_only_paths(full, no_interaction, dense_path | timing_path)


def main() -> None:
    for task_id in TASK_IDS:
        gym.spec(task_id)
    _assert_full_matches_baseline()
    _assert_velocity_variant()
    _assert_recovery_variant()
    _assert_reference_variants()
    print("[PASS] Essay13 ablation registrations and single-factor contracts are valid.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
