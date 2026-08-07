"""Performance-driven curricula for the tracking environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _s2_curriculum_env_ids(
    env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | slice
) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def _s2_curriculum_scalar(command, name: str, *, dtype: torch.dtype) -> torch.Tensor:
    value = getattr(command, name, None)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        value = torch.zeros((), device=command.device, dtype=dtype)
        setattr(command, name, value)
    return value


def _clear_s2_curriculum_window(command) -> None:
    for name in (
        "_s2_curriculum_episode_count",
        "_s2_curriculum_fall_count",
        "_s2_curriculum_contact_success_sum",
        "_s2_curriculum_side_success_sum",
        "_s2_curriculum_side_attempt_count",
        "_s2_curriculum_sequence_success_sum",
        "_s2_curriculum_sequence_episode_count",
    ):
        value = getattr(command, name, None)
        if isinstance(value, torch.Tensor):
            value.zero_()


def s2_contact_level_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    command_name: str = "motion",
    min_evaluation_episodes: int = 1024,
    required_consecutive_passes: int = 2,
    contact_success_threshold: float = 0.80,
    correct_side_threshold: float = 0.75,
    sequence_completion_threshold: float = 0.60,
    max_fall_rate: float = 0.05,
    fall_termination_terms: tuple[str, ...] = ("anchor_pos_z", "anchor_ori"),
) -> dict[str, torch.Tensor]:
    """Advance the single S2 task through 1/2/4/8/full-contact levels.

    Isaac Lab calls curriculum terms before resetting the command manager, so
    the terminal episode metrics are still available here. Statistics are
    evaluated in non-overlapping windows. Two passing windows are required by
    default to avoid promoting on one unusually favorable reset batch.

    The curriculum is global across the vectorized environment and monotonic:
    it never demotes. Environments that finish an older-level episode after a
    promotion are ignored until they have been resampled at the active level.
    """
    command = env.command_manager.get_term(command_name)
    level_tensor = getattr(command, "s2_curriculum_level", None)
    episode_level = getattr(command, "s2_episode_curriculum_level", None)
    episode_contact_count = getattr(command, "s2_episode_contact_count", None)
    if (
        not isinstance(level_tensor, torch.Tensor)
        or level_tensor.numel() != 1
        or not isinstance(episode_level, torch.Tensor)
        or not isinstance(episode_contact_count, torch.Tensor)
    ):
        zero = torch.zeros((), device=env.device)
        return {
            "level": zero,
            "evaluated_level": zero,
            "episodes": zero,
            "contact_success_rate": zero,
            "correct_side_rate": zero,
            "sequence_completion_rate": zero,
            "fall_rate": zero,
            "pass_streak": zero,
        }

    float_stats = (
        "_s2_curriculum_episode_count",
        "_s2_curriculum_fall_count",
        "_s2_curriculum_contact_success_sum",
        "_s2_curriculum_side_success_sum",
        "_s2_curriculum_side_attempt_count",
        "_s2_curriculum_sequence_success_sum",
        "_s2_curriculum_sequence_episode_count",
        "_s2_curriculum_last_contact_success_rate",
        "_s2_curriculum_last_correct_side_rate",
        "_s2_curriculum_last_sequence_completion_rate",
        "_s2_curriculum_last_fall_rate",
        "_s2_curriculum_last_evaluation_count",
    )
    for name in float_stats:
        _s2_curriculum_scalar(command, name, dtype=torch.float32)
    pass_streak = _s2_curriculum_scalar(
        command, "_s2_curriculum_pass_streak", dtype=torch.long
    )
    last_evaluated_level = _s2_curriculum_scalar(
        command, "_s2_curriculum_last_evaluated_level", dtype=torch.long
    )

    ids = _s2_curriculum_env_ids(env, env_ids)
    level = int(level_tensor.item())
    configured_levels = tuple(getattr(command.cfg, "dribble_cg_curriculum_levels", ()))
    max_level = max(0, len(configured_levels) - 1)

    if ids.numel() > 0 and level < max_level:
        episode_steps = env.episode_length_buf[ids]
        valid = (episode_steps > 0) & (episode_level[ids] == level)
        ids = ids[valid]

        if ids.numel() > 0:
            command._s2_curriculum_episode_count.add_(float(ids.numel()))

            fall = torch.zeros(ids.numel(), dtype=torch.bool, device=env.device)
            active_termination_terms = set(env.termination_manager.active_terms)
            for term_name in fall_termination_terms:
                if term_name in active_termination_terms:
                    fall |= env.termination_manager.get_term(term_name)[ids].to(torch.bool)
            command._s2_curriculum_fall_count.add_(fall.to(torch.float32).sum())

            if level == 0:
                success = command.metrics["s2_contact_success_rate"][ids].to(torch.float32)
                command._s2_curriculum_contact_success_sum.add_(success.sum())
                successful_touch = success > 0.0
                if bool(torch.any(successful_touch)):
                    side = command.metrics["s2_correct_side_rate"][ids][successful_touch]
                    command._s2_curriculum_side_success_sum.add_(side.to(torch.float32).sum())
                    command._s2_curriculum_side_attempt_count.add_(
                        successful_touch.to(torch.float32).sum()
                    )
            else:
                required_contacts = (1, 2, 4, 8)[min(level, 3)]
                sequence_episode = episode_contact_count[ids] >= required_contacts
                if bool(torch.any(sequence_episode)):
                    completion = command.metrics[f"s2_complete_{required_contacts}"][ids]
                    completion = completion[sequence_episode].to(torch.float32)
                    command._s2_curriculum_sequence_success_sum.add_(completion.sum())
                    command._s2_curriculum_sequence_episode_count.add_(
                        sequence_episode.to(torch.float32).sum()
                    )

    episode_count = command._s2_curriculum_episode_count
    contact_rate = command._s2_curriculum_contact_success_sum / episode_count.clamp(min=1.0)
    side_rate = (
        command._s2_curriculum_side_success_sum
        / command._s2_curriculum_side_attempt_count.clamp(min=1.0)
    )
    sequence_rate = (
        command._s2_curriculum_sequence_success_sum
        / command._s2_curriculum_sequence_episode_count.clamp(min=1.0)
    )
    fall_rate = command._s2_curriculum_fall_count / episode_count.clamp(min=1.0)

    evaluation_count = (
        episode_count if level == 0 else command._s2_curriculum_sequence_episode_count
    )
    ready = level < max_level and float(evaluation_count.item()) >= max(
        1, int(min_evaluation_episodes)
    )
    if ready:
        if level == 0:
            passed = (
                float(contact_rate.item()) >= float(contact_success_threshold)
                and float(side_rate.item()) >= float(correct_side_threshold)
                and float(command._s2_curriculum_side_attempt_count.item()) > 0.0
                and float(fall_rate.item()) <= float(max_fall_rate)
            )
        else:
            passed = (
                float(sequence_rate.item()) >= float(sequence_completion_threshold)
                and float(fall_rate.item()) <= float(max_fall_rate)
            )

        command._s2_curriculum_last_contact_success_rate.copy_(contact_rate)
        command._s2_curriculum_last_correct_side_rate.copy_(side_rate)
        command._s2_curriculum_last_sequence_completion_rate.copy_(sequence_rate)
        command._s2_curriculum_last_fall_rate.copy_(fall_rate)
        command._s2_curriculum_last_evaluation_count.copy_(evaluation_count)
        last_evaluated_level.fill_(level)

        if passed:
            pass_streak.add_(1)
        else:
            pass_streak.zero_()
        if int(pass_streak.item()) >= max(1, int(required_consecutive_passes)):
            level_tensor.fill_(min(level + 1, max_level))
            pass_streak.zero_()
        _clear_s2_curriculum_window(command)

    # Before the first completed evaluation window, expose the live window
    # rates. Afterwards retain the last complete window so logs do not fall to
    # zero immediately when the accumulators are cleared.
    has_last_window = (
        (command._s2_curriculum_last_evaluation_count > 0.0)
        | (command._s2_curriculum_last_contact_success_rate != 0.0)
        | (command._s2_curriculum_last_correct_side_rate != 0.0)
        | (command._s2_curriculum_last_sequence_completion_rate != 0.0)
        | (command._s2_curriculum_last_fall_rate != 0.0)
    )
    logged_contact_rate = torch.where(
        has_last_window, command._s2_curriculum_last_contact_success_rate, contact_rate
    )
    logged_side_rate = torch.where(
        has_last_window, command._s2_curriculum_last_correct_side_rate, side_rate
    )
    logged_sequence_rate = torch.where(
        has_last_window, command._s2_curriculum_last_sequence_completion_rate, sequence_rate
    )
    logged_fall_rate = torch.where(
        has_last_window, command._s2_curriculum_last_fall_rate, fall_rate
    )
    return {
        "level": level_tensor.to(torch.float32),
        "evaluated_level": last_evaluated_level.to(torch.float32),
        "episodes": torch.where(
            has_last_window,
            command._s2_curriculum_last_evaluation_count,
            evaluation_count,
        ),
        "contact_success_rate": logged_contact_rate,
        "correct_side_rate": logged_side_rate,
        "sequence_completion_rate": logged_sequence_rate,
        "fall_rate": logged_fall_rate,
        "pass_streak": pass_streak.to(torch.float32),
    }
