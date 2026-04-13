"""Anchor-based Motion Command with distance-triggered state machine.

This command class merges approach + strike motions into a SINGLE combined
MultiMotionLoader.  Approach clips get file indices [0..N-1] and strike clips
get indices [N..N+M-1].

During each episode the per-env state transitions:

    APPROACH  ──  d ≤ threshold  ──►  STRIKE  ──  motion_finished  ──►  (resample)

On transition, the command simply swaps ``self.motion_idx`` and ``self.time_steps``
to point at the strike range.  **No property overrides are needed** — all
parent properties (joint_pos, body_pos_w, etc.) work unchanged because they
read from ``self.motion`` / ``self.motion_idx`` / ``self.time_steps``.
"""
from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .commands_multi_motion_soccer import (
    MotionCommand,
    MotionCommandCfg,
    MultiMotionLoader,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Per-env state constants.
STATE_APPROACH = 0
STATE_STRIKE = 1


@configclass
class AnchorMotionCommandCfg(MotionCommandCfg):
    """Config for AnchorMotionCommand.

    Adds:
      - ``strike_motion_files``: list of .npz files for the strike bank.
      - ``strike_trigger_distance``: ball distance threshold for A→B switch.
    """

    class_type: type = None  # filled below after class def

    # Strike-bank motion files (populated at runtime like motion_files).
    strike_motion_files: list[str] = MISSING

    # Distance threshold (metres) for APPROACH → STRIKE transition.
    strike_trigger_distance: float = 0.8


class AnchorMotionCommand(MotionCommand):
    """MotionCommand with a distance-triggered APPROACH / STRIKE state machine.

    Implementation strategy: merge approach + strike files into ONE combined
    MultiMotionLoader so all parent properties work without override.

    File index layout in self.motion:
        [0 .. num_approach-1]   = approach clips
        [num_approach .. total] = strike clips
    """

    cfg: AnchorMotionCommandCfg

    def __init__(self, cfg: AnchorMotionCommandCfg, env: ManagerBasedRLEnv):
        # Merge approach + strike files into a single list.
        # Approach files come from cfg.motion_files (set by parent).
        # Strike files come from cfg.strike_motion_files.
        self._num_approach = len(cfg.motion_files)
        self._num_strike = len(cfg.strike_motion_files)
        assert self._num_approach > 0, "motion_files (approach) must not be empty"
        assert self._num_strike > 0, "strike_motion_files must not be empty"

        # Concatenate all files: approach first, then strike.
        combined_files = list(cfg.motion_files) + list(cfg.strike_motion_files)
        original_files = cfg.motion_files

        # Temporarily replace motion_files with the combined list
        # so the parent loads everything into one MultiMotionLoader.
        cfg.motion_files = combined_files

        super().__init__(cfg, env)

        # Restore original files (for serialization / logging).
        cfg.motion_files = original_files

        # Per-env state: 0 = APPROACH, 1 = STRIKE.
        self._state = torch.full(
            (self.num_envs,), STATE_APPROACH, dtype=torch.long, device=self.device,
        )

        # Store approach/strike index ranges.
        self._approach_indices = torch.arange(
            0, self._num_approach, device=self.device, dtype=torch.long,
        )
        self._strike_indices = torch.arange(
            self._num_approach, self._num_approach + self._num_strike,
            device=self.device, dtype=torch.long,
        )

        # Approach file lengths.
        self._approach_file_lengths = self.motion.file_lengths[:self._num_approach]
        # Strike file lengths.
        self._strike_file_lengths = self.motion.file_lengths[self._num_approach:]

        # Per-env strike time counter (separate from self.time_steps).
        self._strike_time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )

    # ------------------------------------------------------------------
    # State machine core
    # ------------------------------------------------------------------

    def _check_state_transition(self):
        """Check if any env should transition from APPROACH → STRIKE."""
        if self.soccer_ball is None:
            return

        # Only consider envs still in APPROACH.
        approach_mask = (self._state == STATE_APPROACH)
        if not torch.any(approach_mask):
            return

        # Compute ball-pelvis distance for approach envs.
        ball_pos_xy = self.soccer_ball.data.root_pos_w[:, :2]
        pelvis_pos_xy = self.robot_pelvis_pos_w[:, :2]
        dist = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)

        # Trigger: distance ≤ threshold AND still in APPROACH.
        trigger = approach_mask & (dist <= self.cfg.strike_trigger_distance)
        if not torch.any(trigger):
            return

        trigger_ids = torch.where(trigger)[0]
        self._transition_to_strike(trigger_ids)

    def _transition_to_strike(self, env_ids: torch.Tensor):
        """Switch specified envs from APPROACH to STRIKE state."""
        self._state[env_ids] = STATE_STRIKE

        # Assign random strike file indices (from combined bank range).
        if self._num_strike > 1:
            local_idx = torch.randint(
                0, self._num_strike, (env_ids.numel(),), device=self.device,
            )
        else:
            local_idx = torch.zeros(env_ids.numel(), dtype=torch.long, device=self.device)

        # Map to combined index range.
        self.motion_idx[env_ids] = self._strike_indices[local_idx]

        # Reset time to frame 0 of strike clip.
        self._strike_time_steps[env_ids] = 0
        self.time_steps[env_ids] = 0
        self.motion_length[env_ids] = self._strike_file_lengths[local_idx]

    # ------------------------------------------------------------------
    # Override: resample (reset to APPROACH)
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        """Reset resampled envs back to APPROACH state."""
        if len(env_ids) == 0:
            return

        ids = self._to_env_id_tensor(env_ids)
        # Reset state machine.
        self._state[ids] = STATE_APPROACH

        # Let parent handle sampling — but restrict to approach indices only.
        # Temporarily limit sampling to approach range.
        super()._resample_command(env_ids)

        # Override: ensure motion_idx is in approach range.
        # The parent's sampling already uses self.motion.num_files which is
        # the combined total. We need to clamp to approach range.
        ids = self._to_env_id_tensor(env_ids)
        if self._num_approach > 1:
            self.motion_idx[ids] = torch.randint(
                0, self._num_approach, (ids.numel(),), device=self.device,
            )
        else:
            self.motion_idx[ids] = 0
        self.motion_length[ids] = self._approach_file_lengths[self.motion_idx[ids]]

    # ------------------------------------------------------------------
    # Override: update (state machine + time stepping)
    # ------------------------------------------------------------------

    def _update_command(self):
        """Main per-step update with state machine logic."""
        self.kick_contact_tracker.begin_step(self)

        # Advance time for ALL envs.
        self.time_steps += 1

        # Check approach → strike transition.
        self._check_state_transition()

        # Handle end-of-motion for APPROACH envs: hold last frame.
        approach_mask = (self._state == STATE_APPROACH)
        approach_ended = approach_mask & (self.time_steps >= self.motion_length)
        self.time_steps[approach_ended] = (
            self.motion_length[approach_ended] - 1
        ).clamp(min=0)

        # Handle end-of-motion for STRIKE envs: resample episode.
        strike_mask = (self._state == STATE_STRIKE)
        strike_ended = strike_mask & (self.time_steps >= self.motion_length)
        resample_ids = torch.where(strike_ended)[0]
        if resample_ids.numel() > 0:
            self._resample_command(resample_ids)

        # Also resample if approach motion naturally ends without triggering strike.
        # (safety net — shouldn't happen if ball placement is close enough)

        # === Rest is identical to parent _update_command ===

        # Update target point each step using current ball position.
        self._update_target_points_from_sim()

        # Continuously refresh pre-kick target until contact occurs.
        if hasattr(self, "kick_contact_tracker"):
            contact_awarded = self.kick_contact_tracker.get_contact_awarded()
            no_contact_mask = ~contact_awarded
            if torch.any(no_contact_mask):
                self.initial_target_point_pos[no_contact_mask] = self.target_point_pos[no_contact_mask]

        from isaaclab.utils.math import yaw_quat, quat_mul, quat_inv, quat_apply

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

        self._update_metrics()

    # ------------------------------------------------------------------
    # Expose state info for rewards / terminations
    # ------------------------------------------------------------------

    @property
    def env_state(self) -> torch.Tensor:
        """Per-env state: 0=APPROACH, 1=STRIKE."""
        return self._state

    @property
    def is_in_strike(self) -> torch.Tensor:
        """Boolean mask: True if env is in STRIKE state."""
        return self._state == STATE_STRIKE


# Backfill the class_type so the configclass can instantiate it.
AnchorMotionCommandCfg.class_type = AnchorMotionCommand
