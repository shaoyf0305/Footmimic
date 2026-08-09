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


def _s2_required_contacts(command, level: int) -> int:
    configured = tuple(
        getattr(command.cfg, "dribble_cg_curriculum_required_contacts", ())
    )
    if configured:
        # Zero explicitly means every selected event in a full-clip episode.
        return max(0, int(configured[min(level, len(configured) - 1)]))
    return (1, 2, 4, 8)[min(level, 3)]


def s2_contact_level_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    command_name: str = "motion",
    min_evaluation_episodes: int = 1024,
    required_consecutive_passes: int = 2,
    contact_success_threshold: float = 0.75,
    correct_side_threshold: float = 0.70,
    sequence_completion_threshold: float = 0.60,
    max_fall_rate: float = 0.05,
    fall_termination_terms: tuple[str, ...] = ("anchor_pos_z", "anchor_ori"),
    event_min_attempts: int = 30,
    event_success_threshold: float = 0.60,
    event_coverage_threshold: float = 0.80,
    plateau_contact_threshold: float = 0.50,
    plateau_windows: int = 4,
    plateau_min_improvement: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Advance S2 from short contact transitions into longer sequences.

    Isaac Lab calls curriculum terms before resetting the command manager, so
    the terminal episode metrics are still available here. Statistics are
    evaluated in non-overlapping windows. Two passing windows are required by
    default to avoid promoting on one unusually favorable reset batch.

    The active schedule evaluates fixed frame-0 prefixes of two, four, eight,
    and then every contact in the full clip. Optional single-touch
    configurations retain the legacy per-event coverage and hard-replay path,
    but active S2 never samples an isolated or interior contact. The curriculum
    remains global and monotonic; old-level episodes finishing after promotion
    are ignored until resampled.
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
            "event_coverage": zero,
            "hard_replay": zero,
            "plateau_streak": zero,
            "pass_streak": zero,
            "level_entry_iteration": zero,
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
        "_s2_curriculum_last_event_coverage",
        "_s2_curriculum_last_evaluation_count",
        "_s2_curriculum_best_contact_success_rate",
    )
    for name in float_stats:
        _s2_curriculum_scalar(command, name, dtype=torch.float32)
    pass_streak = _s2_curriculum_scalar(
        command, "_s2_curriculum_pass_streak", dtype=torch.long
    )
    last_evaluated_level = _s2_curriculum_scalar(
        command, "_s2_curriculum_last_evaluated_level", dtype=torch.long
    )
    plateau_streak = _s2_curriculum_scalar(
        command, "_s2_curriculum_plateau_streak", dtype=torch.long
    )
    hard_replay = _s2_curriculum_scalar(
        command, "_s2_hard_replay_enabled", dtype=torch.long
    )

    ids = _s2_curriculum_env_ids(env, env_ids)
    level = int(level_tensor.item())
    configured_levels = tuple(getattr(command.cfg, "dribble_cg_curriculum_levels", ()))
    max_level = max(0, len(configured_levels) - 1)
    level_entry_history = getattr(
        command, "_s2_curriculum_level_entry_iteration", None
    )
    if (
        not isinstance(level_entry_history, torch.Tensor)
        or level_entry_history.ndim != 1
        or level_entry_history.shape[0] != max_level + 1
    ):
        level_entry_history = torch.full(
            (max_level + 1,), -1, dtype=torch.long, device=env.device
        )
        setattr(command, "_s2_curriculum_level_entry_iteration", level_entry_history)
    training_iteration = getattr(command, "_s2_training_iteration", None)
    if not isinstance(training_iteration, torch.Tensor) or training_iteration.numel() != 1:
        training_iteration = torch.tensor(-1, dtype=torch.long, device=env.device)
        setattr(command, "_s2_training_iteration", training_iteration)
    required_contacts = _s2_required_contacts(command, level)
    single_touch_level = required_contacts == 1
    full_sequence_level = required_contacts == 0

    if ids.numel() > 0 and level < max_level:
        episode_steps = env.episode_length_buf[ids]
        valid = (episode_steps > 0) & (episode_level[ids] == level)
        ids = ids[valid]

        if ids.numel() > 0:
            episode_audit = getattr(command, "s2_episode_curriculum_audit", None)
            if isinstance(episode_audit, torch.Tensor):
                audit_mask = episode_audit[ids].to(torch.bool)
            else:
                audit_mask = torch.ones(ids.numel(), dtype=torch.bool, device=env.device)

            if single_touch_level:
                all_success = command.metrics["s2_contact_success_rate"][ids].to(
                    torch.float32
                )
                episode_event_index = getattr(
                    command, "s2_episode_first_event_index", None
                )

                def _record_event_outcomes(
                    sample_ids: torch.Tensor,
                    success: torch.Tensor,
                    attempt_count: torch.Tensor | None,
                    success_count: torch.Tensor | None,
                ) -> None:
                    if (
                        not isinstance(episode_event_index, torch.Tensor)
                        or not isinstance(attempt_count, torch.Tensor)
                        or not isinstance(success_count, torch.Tensor)
                        or sample_ids.numel() == 0
                    ):
                        return
                    motion_index = command.motion_idx[sample_ids].to(torch.long)
                    event_index = episode_event_index[sample_ids].to(torch.long)
                    valid_event = (
                        (event_index >= 0)
                        & (motion_index >= 0)
                        & (motion_index < attempt_count.shape[0])
                        & (event_index < attempt_count.shape[1])
                    )
                    if not bool(torch.any(valid_event)):
                        return
                    flat_index = (
                        motion_index[valid_event] * attempt_count.shape[1]
                        + event_index[valid_event]
                    )
                    attempt_count.view(-1).scatter_add_(
                        0,
                        flat_index,
                        torch.ones_like(flat_index, dtype=torch.float32),
                    )
                    success_count.view(-1).scatter_add_(
                        0, flat_index, success[valid_event]
                    )

                # All outcomes rank hard events for training. They never feed
                # promotion coverage unless the episode was uniformly drawn.
                _record_event_outcomes(
                    ids,
                    all_success,
                    getattr(command, "_s2_event_attempt_count", None),
                    getattr(command, "_s2_event_success_count", None),
                )

            audit_ids = ids[audit_mask]
            if audit_ids.numel() > 0:
                command._s2_curriculum_episode_count.add_(float(audit_ids.numel()))

                fall = torch.zeros(
                    audit_ids.numel(), dtype=torch.bool, device=env.device
                )
                active_termination_terms = set(env.termination_manager.active_terms)
                for term_name in fall_termination_terms:
                    if term_name in active_termination_terms:
                        fall |= env.termination_manager.get_term(term_name)[audit_ids].to(
                            torch.bool
                        )
                command._s2_curriculum_fall_count.add_(fall.to(torch.float32).sum())

                if single_touch_level:
                    success = command.metrics["s2_contact_success_rate"][audit_ids].to(
                        torch.float32
                    )
                    command._s2_curriculum_contact_success_sum.add_(success.sum())
                    successful_touch = success > 0.0
                    if bool(torch.any(successful_touch)):
                        side = command.metrics["s2_correct_side_rate"][audit_ids][
                            successful_touch
                        ]
                        command._s2_curriculum_side_success_sum.add_(
                            side.to(torch.float32).sum()
                        )
                        command._s2_curriculum_side_attempt_count.add_(
                            successful_touch.to(torch.float32).sum()
                        )
                    _record_event_outcomes(
                        audit_ids,
                        success,
                        getattr(command, "_s2_audit_event_attempt_count", None),
                        getattr(command, "_s2_audit_event_success_count", None),
                    )
                else:
                    if full_sequence_level:
                        sequence_episode = episode_contact_count[audit_ids] > 0
                    else:
                        sequence_episode = (
                            episode_contact_count[audit_ids] >= required_contacts
                        )
                    if bool(torch.any(sequence_episode)):
                        metric_name = (
                            "s2_complete_selected"
                            if full_sequence_level
                            else f"s2_complete_{required_contacts}"
                        )
                        completion = command.metrics[metric_name][audit_ids]
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

    event_coverage = torch.zeros((), device=env.device, dtype=torch.float32)
    event_attempt_count = getattr(command, "_s2_event_attempt_count", None)
    event_success_count = getattr(command, "_s2_event_success_count", None)
    audit_event_attempt_count = getattr(
        command, "_s2_audit_event_attempt_count", None
    )
    audit_event_success_count = getattr(
        command, "_s2_audit_event_success_count", None
    )
    sequence_candidates = getattr(command, "_s2_sequence_candidates", {})
    single_contact_candidates = (
        sequence_candidates.get(1) if isinstance(sequence_candidates, dict) else None
    )
    if (
        single_touch_level
        and isinstance(audit_event_attempt_count, torch.Tensor)
        and isinstance(audit_event_success_count, torch.Tensor)
        and isinstance(single_contact_candidates, torch.Tensor)
    ):
        valid_events = torch.zeros_like(audit_event_attempt_count, dtype=torch.bool)
        valid_events[
            single_contact_candidates[:, 0], single_contact_candidates[:, 1]
        ] = True
        sufficiently_sampled = audit_event_attempt_count >= float(
            max(1, int(event_min_attempts))
        )
        event_success_rate = audit_event_success_count / audit_event_attempt_count.clamp(
            min=1.0
        )
        covered = (
            valid_events
            & sufficiently_sampled
            & (event_success_rate >= float(event_success_threshold))
        )
        event_coverage = covered.to(torch.float32).sum() / valid_events.to(
            torch.float32
        ).sum().clamp(min=1.0)

    evaluation_count = (
        episode_count if single_touch_level else command._s2_curriculum_sequence_episode_count
    )
    ready = level < max_level and float(evaluation_count.item()) >= max(
        1, int(min_evaluation_episodes)
    )
    if ready:
        if single_touch_level:
            passed = (
                float(contact_rate.item()) >= float(contact_success_threshold)
                and float(side_rate.item()) >= float(correct_side_threshold)
                and float(event_coverage.item()) >= float(event_coverage_threshold)
                and float(command._s2_curriculum_side_attempt_count.item()) > 0.0
                and float(fall_rate.item()) <= float(max_fall_rate)
            )

            best_contact_rate = command._s2_curriculum_best_contact_success_rate
            above_bootstrap = float(contact_rate.item()) >= float(
                plateau_contact_threshold
            )
            improved = float(contact_rate.item()) >= float(
                best_contact_rate.item()
            ) + max(0.0, float(plateau_min_improvement))
            if not above_bootstrap:
                plateau_streak.zero_()
            elif improved:
                best_contact_rate.copy_(contact_rate)
                plateau_streak.zero_()
            else:
                plateau_streak.add_(1)
            if (
                int(plateau_streak.item()) >= max(1, int(plateau_windows))
                and float(fall_rate.item()) <= float(max_fall_rate)
            ):
                hard_replay.fill_(1)
                # The sampling distribution has changed. Begin a fresh
                # plateau audit while retaining per-event evidence for ranking.
                best_contact_rate.copy_(contact_rate)
                plateau_streak.zero_()
        else:
            passed = (
                float(sequence_rate.item()) >= float(sequence_completion_threshold)
                and float(fall_rate.item()) <= float(max_fall_rate)
            )

        command._s2_curriculum_last_contact_success_rate.copy_(contact_rate)
        command._s2_curriculum_last_correct_side_rate.copy_(side_rate)
        command._s2_curriculum_last_sequence_completion_rate.copy_(sequence_rate)
        command._s2_curriculum_last_fall_rate.copy_(fall_rate)
        command._s2_curriculum_last_event_coverage.copy_(event_coverage)
        command._s2_curriculum_last_evaluation_count.copy_(evaluation_count)
        last_evaluated_level.fill_(level)

        if passed:
            pass_streak.add_(1)
        else:
            pass_streak.zero_()
        if int(pass_streak.item()) >= max(1, int(required_consecutive_passes)):
            promoted_level = min(level + 1, max_level)
            level_tensor.fill_(promoted_level)
            if int(training_iteration.item()) >= 0:
                level_entry_history[promoted_level] = training_iteration
            pass_streak.zero_()
            plateau_streak.zero_()
            hard_replay.zero_()
            command._s2_curriculum_best_contact_success_rate.zero_()
            if isinstance(event_attempt_count, torch.Tensor):
                event_attempt_count.zero_()
            if isinstance(event_success_count, torch.Tensor):
                event_success_count.zero_()
            if isinstance(audit_event_attempt_count, torch.Tensor):
                audit_event_attempt_count.zero_()
            if isinstance(audit_event_success_count, torch.Tensor):
                audit_event_success_count.zero_()
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
        | (command._s2_curriculum_last_event_coverage != 0.0)
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
    logged_event_coverage = torch.where(
        has_last_window, command._s2_curriculum_last_event_coverage, event_coverage
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
        "event_coverage": logged_event_coverage,
        "hard_replay": hard_replay.to(torch.float32),
        "plateau_streak": plateau_streak.to(torch.float32),
        "pass_streak": pass_streak.to(torch.float32),
        "level_entry_iteration": level_entry_history[level_tensor].to(torch.float32),
    }
