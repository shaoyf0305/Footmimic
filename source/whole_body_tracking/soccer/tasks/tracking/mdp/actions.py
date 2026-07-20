from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UpperBodyManifoldJointPositionAction(JointPositionAction):
    """Project upper-body targets onto a motion-bank coordination manifold.

    The policy still emits one action per robot joint.  Lower-body targets pass
    through unchanged, while the selected upper-body targets are represented by
    a low-rank PCA model fitted once from every valid frame in the configured
    motion bank.  The model is independent of the current reference clip,
    motion index, and style phase.
    """

    cfg: UpperBodyManifoldJointPositionActionCfg

    def __init__(self, cfg: UpperBodyManifoldJointPositionActionCfg, env: ManagerBasedEnv):
        if cfg.manifold_rank <= 0:
            raise ValueError("manifold_rank must be positive.")
        if cfg.latent_std_limit <= 0.0 or cfg.min_latent_limit < 0.0:
            raise ValueError("latent limits must be non-negative and latent_std_limit must be positive.")
        if cfg.orthogonal_residual_limit < 0.0 or cfg.cutoff_frequency_hz <= 0.0:
            raise ValueError("orthogonal_residual_limit must be non-negative and cutoff_frequency_hz positive.")
        if cfg.reference_target_margin is not None and cfg.reference_target_margin <= 0.0:
            raise ValueError("reference_target_margin must be positive when enabled.")
        if cfg.trunk_pitch_joint_name is not None:
            if cfg.trunk_pitch_cutoff_frequency_hz <= 0.0:
                raise ValueError("trunk_pitch_cutoff_frequency_hz must be positive when enabled.")
            if cfg.trunk_pitch_lower_deviation <= 0.0 or cfg.trunk_pitch_upper_deviation <= 0.0:
                raise ValueError("trunk pitch reference deviations must be positive when enabled.")
        super().__init__(cfg, env)
        self._manifold_env = env

        robot_joint_ids, found_names = self._asset.find_joints(
            cfg.upper_body_joint_names, preserve_order=True
        )
        if len(robot_joint_ids) != len(cfg.upper_body_joint_names):
            raise ValueError(
                "Could not resolve every upper-body manifold joint: "
                f"expected {cfg.upper_body_joint_names}, found {found_names}."
            )

        if isinstance(self._joint_ids, slice):
            controlled_robot_ids = list(range(self._asset.num_joints))
        else:
            controlled_robot_ids = [int(index) for index in self._joint_ids]
        robot_to_action = {robot_id: action_id for action_id, robot_id in enumerate(controlled_robot_ids)}
        try:
            upper_action_ids = [robot_to_action[int(robot_id)] for robot_id in robot_joint_ids]
        except KeyError as exc:
            raise ValueError(
                f"The joint-position action does not control upper-body robot joint id {exc.args[0]}."
            ) from exc

        self._upper_robot_ids = torch.as_tensor(robot_joint_ids, dtype=torch.long, device=self.device)
        self._upper_action_ids = torch.as_tensor(upper_action_ids, dtype=torch.long, device=self.device)
        self._upper_joint_names = tuple(found_names)
        self._manifold_mean: torch.Tensor | None = None
        self._manifold_basis: torch.Tensor | None = None
        self._manifold_latent_limit: torch.Tensor | None = None

        self._trunk_pitch_robot_id: int | None = None
        self._trunk_pitch_action_id: int | None = None
        if cfg.trunk_pitch_joint_name is not None:
            trunk_ids, trunk_names = self._asset.find_joints(
                [cfg.trunk_pitch_joint_name], preserve_order=True
            )
            if len(trunk_ids) != 1:
                raise ValueError(
                    f"Could not resolve trunk pitch joint {cfg.trunk_pitch_joint_name!r}; found {trunk_names}."
                )
            try:
                trunk_action_id = robot_to_action[int(trunk_ids[0])]
            except KeyError as exc:
                raise ValueError(
                    f"The joint-position action does not control trunk pitch joint id {exc.args[0]}."
                ) from exc
            self._trunk_pitch_robot_id = int(trunk_ids[0])
            self._trunk_pitch_action_id = trunk_action_id

        upper_dim = len(upper_action_ids)
        rank = min(int(cfg.manifold_rank), upper_dim)
        self._filtered_latent = torch.zeros(self.num_envs, rank, device=self.device)
        self._filtered_upper_target = torch.zeros(self.num_envs, upper_dim, device=self.device)
        self._filter_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._filtered_trunk_pitch_target = torch.zeros(self.num_envs, device=self.device)
        self._trunk_pitch_filter_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Public diagnostic tensors.  They are populated by process_actions().
        self.manifold_raw_upper_target = torch.zeros_like(self._filtered_upper_target)
        self.manifold_reference_upper_target = torch.zeros_like(self._filtered_upper_target)
        self.manifold_constrained_upper_target = torch.zeros_like(self._filtered_upper_target)
        self.manifold_projected_upper_target = torch.zeros_like(self._filtered_upper_target)
        self.manifold_latent = torch.zeros_like(self._filtered_latent)
        self.manifold_projection_error = torch.zeros(self.num_envs, device=self.device)
        self.manifold_projection_error_after_reference_constraint = torch.zeros(
            self.num_envs, device=self.device
        )
        self.manifold_latent_clip_fraction = torch.zeros(self.num_envs, device=self.device)
        self.manifold_reference_overflow = torch.zeros_like(self._filtered_upper_target)
        self.manifold_reference_clamp_fraction = torch.zeros(self.num_envs, device=self.device)
        self.trunk_pitch_raw_target = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_reference_target = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_soft_target = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_filtered_target = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_reference_overflow = torch.zeros(self.num_envs, device=self.device)
        self.effective_raw_actions = torch.zeros_like(self.raw_actions)
        self.prev_effective_raw_actions = torch.zeros_like(self.raw_actions)

    def _fit_manifold_from_motion_bank(self) -> None:
        """Fit PCA from all valid motion frames without consulting the active phase."""
        command = self._manifold_env.command_manager.get_term(self.cfg.command_name)
        motion = getattr(command, "motion", None)
        joint_pos = getattr(motion, "joint_pos", None)
        if not isinstance(joint_pos, torch.Tensor):
            raise RuntimeError(
                f"Command '{self.cfg.command_name}' does not expose motion.joint_pos for manifold fitting."
            )

        upper_dim = int(self._upper_robot_ids.numel())
        sample_sum = torch.zeros(upper_dim, dtype=torch.float64, device=self.device)
        sample_gram = torch.zeros(upper_dim, upper_dim, dtype=torch.float64, device=self.device)
        sample_count = 0

        if joint_pos.ndim == 2:
            clips = [(joint_pos, int(joint_pos.shape[0]))]
        elif joint_pos.ndim == 3:
            file_lengths = getattr(motion, "file_lengths", None)
            if file_lengths is None:
                raise RuntimeError("Multi-motion manifold fitting requires motion.file_lengths.")
            clips = [
                (joint_pos[index], int(file_lengths[index].item()))
                for index in range(int(joint_pos.shape[0]))
            ]
        else:
            raise ValueError(f"Expected motion.joint_pos rank 2 or 3, got shape {tuple(joint_pos.shape)}.")

        for clip, length in clips:
            if length <= 0:
                continue
            samples = clip[:length, self._upper_robot_ids].to(dtype=torch.float64)
            sample_sum += samples.sum(dim=0)
            sample_gram += samples.transpose(0, 1) @ samples
            sample_count += length

        if sample_count < 2:
            raise RuntimeError("At least two valid motion frames are required to fit the upper-body manifold.")

        mean64 = sample_sum / sample_count
        covariance = (sample_gram - sample_count * torch.outer(mean64, mean64)) / (sample_count - 1)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        rank = min(int(self.cfg.manifold_rank), upper_dim)
        eigenvalues = torch.clamp(eigenvalues[order[:rank]], min=0.0)
        basis = eigenvectors[:, order[:rank]]
        latent_std = torch.sqrt(eigenvalues)
        latent_limit = torch.clamp(
            self.cfg.latent_std_limit * latent_std,
            min=self.cfg.min_latent_limit,
        )

        self._manifold_mean = mean64.to(dtype=self.raw_actions.dtype)
        self._manifold_basis = basis.to(dtype=self.raw_actions.dtype)
        self._manifold_latent_limit = latent_limit.to(dtype=self.raw_actions.dtype)

    def _scale_tensor(self) -> torch.Tensor:
        if isinstance(self._scale, torch.Tensor):
            return self._scale
        return torch.full_like(self.raw_actions, float(self._scale))

    def _offset_tensor(self) -> torch.Tensor:
        if isinstance(self._offset, torch.Tensor):
            return self._offset
        return torch.full_like(self.raw_actions, float(self._offset))

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        if self._manifold_mean is None:
            self._fit_manifold_from_motion_bank()

        assert self._manifold_mean is not None
        assert self._manifold_basis is not None
        assert self._manifold_latent_limit is not None

        action_ids = self._upper_action_ids
        raw_target = self.processed_actions[:, action_ids].clone()

        # Control can use the current style pose as a bounded safety envelope.
        # This is deliberately applied before PCA: a target far outside the
        # motion bank otherwise saturates every latent dimension and collapses
        # the arm into one static pose.  The default remains disabled so the
        # regular tracking and non-control environments retain their original
        # action semantics.
        reference_margin = self.cfg.reference_target_margin
        if reference_margin is None:
            reference_target = torch.zeros_like(raw_target)
            constrained_target = raw_target
            reference_overflow = torch.zeros_like(raw_target)
        else:
            command = self._manifold_env.command_manager.get_term(self.cfg.command_name)
            reference_target = command.joint_pos[:, self._upper_robot_ids]
            reference_overflow = torch.clamp(
                torch.abs(raw_target - reference_target) - reference_margin,
                min=0.0,
            )
            constrained_target = torch.clamp(
                raw_target,
                min=reference_target - reference_margin,
                max=reference_target + reference_margin,
            )

        centered = constrained_target - self._manifold_mean
        raw_latent = centered @ self._manifold_basis
        bounded_latent = torch.clamp(
            raw_latent,
            min=-self._manifold_latent_limit,
            max=self._manifold_latent_limit,
        )
        parallel = bounded_latent @ self._manifold_basis.transpose(0, 1)
        orthogonal = centered - (raw_latent @ self._manifold_basis.transpose(0, 1))
        residual_limit = float(self.cfg.orthogonal_residual_limit)
        bounded_orthogonal = residual_limit * torch.tanh(orthogonal / max(residual_limit, 1.0e-6))
        projected = self._manifold_mean + parallel + bounded_orthogonal

        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._upper_robot_ids]
        projected = torch.clamp(projected, min=soft_limits[..., 0], max=soft_limits[..., 1])

        step_dt = float(getattr(self._manifold_env, "step_dt", 0.02))
        alpha = 1.0 - math.exp(-2.0 * math.pi * float(self.cfg.cutoff_frequency_hz) * step_dt)
        alpha = min(max(alpha, 0.0), 1.0)
        initialized = self._filter_initialized[:, None]
        filtered_latent = torch.where(
            initialized,
            self._filtered_latent + alpha * (bounded_latent - self._filtered_latent),
            bounded_latent,
        )
        previous_target = torch.where(
            initialized,
            self._filtered_upper_target,
            self._asset.data.joint_pos[:, self._upper_robot_ids],
        )
        filtered_target = previous_target + alpha * (projected - previous_target)
        self._filtered_latent[:] = filtered_latent
        self._filtered_upper_target[:] = filtered_target
        self._filter_initialized[:] = True
        self._processed_actions[:, action_ids] = filtered_target

        if self._trunk_pitch_robot_id is not None and self._trunk_pitch_action_id is not None:
            trunk_action_id = self._trunk_pitch_action_id
            trunk_robot_id = self._trunk_pitch_robot_id
            trunk_raw_target = self.processed_actions[:, trunk_action_id].clone()
            command = self._manifold_env.command_manager.get_term(self.cfg.command_name)
            trunk_reference_target = command.joint_pos[:, trunk_robot_id]
            trunk_deviation = trunk_raw_target - trunk_reference_target
            lower_deviation = float(self.cfg.trunk_pitch_lower_deviation)
            upper_deviation = float(self.cfg.trunk_pitch_upper_deviation)
            trunk_soft_target = trunk_reference_target + torch.where(
                trunk_deviation < 0.0,
                lower_deviation * torch.tanh(trunk_deviation / lower_deviation),
                upper_deviation * torch.tanh(trunk_deviation / upper_deviation),
            )
            trunk_alpha = 1.0 - math.exp(
                -2.0 * math.pi * float(self.cfg.trunk_pitch_cutoff_frequency_hz) * step_dt
            )
            trunk_alpha = min(max(trunk_alpha, 0.0), 1.0)
            previous_trunk_target = torch.where(
                self._trunk_pitch_filter_initialized,
                self._filtered_trunk_pitch_target,
                self._asset.data.joint_pos[:, trunk_robot_id],
            )
            filtered_trunk_target = previous_trunk_target + trunk_alpha * (
                trunk_soft_target - previous_trunk_target
            )
            self._filtered_trunk_pitch_target[:] = filtered_trunk_target
            self._trunk_pitch_filter_initialized[:] = True
            self._processed_actions[:, trunk_action_id] = filtered_trunk_target
            self.trunk_pitch_raw_target[:] = trunk_raw_target
            self.trunk_pitch_reference_target[:] = trunk_reference_target
            self.trunk_pitch_soft_target[:] = trunk_soft_target
            self.trunk_pitch_filtered_target[:] = filtered_trunk_target
            self.trunk_pitch_reference_overflow[:] = torch.where(
                trunk_deviation < 0.0,
                torch.clamp(-trunk_deviation - lower_deviation, min=0.0),
                torch.clamp(trunk_deviation - upper_deviation, min=0.0),
            )

        scale = self._scale_tensor()
        offset = self._offset_tensor()
        upper_scale = scale[:, action_ids]
        safe_upper_scale = torch.where(
            torch.abs(upper_scale) < 1.0e-8,
            torch.ones_like(upper_scale),
            upper_scale,
        )
        self.prev_effective_raw_actions[:] = self.effective_raw_actions
        self.effective_raw_actions[:] = self.raw_actions
        self.effective_raw_actions[:, action_ids] = (
            filtered_target - offset[:, action_ids]
        ) / safe_upper_scale
        if self._trunk_pitch_action_id is not None:
            trunk_action_id = self._trunk_pitch_action_id
            trunk_scale = scale[:, trunk_action_id]
            safe_trunk_scale = torch.where(
                torch.abs(trunk_scale) < 1.0e-8,
                torch.ones_like(trunk_scale),
                trunk_scale,
            )
            self.effective_raw_actions[:, trunk_action_id] = (
                self._processed_actions[:, trunk_action_id] - offset[:, trunk_action_id]
            ) / safe_trunk_scale

        self.manifold_raw_upper_target[:] = raw_target
        self.manifold_reference_upper_target[:] = reference_target
        self.manifold_constrained_upper_target[:] = constrained_target
        self.manifold_projected_upper_target[:] = filtered_target
        self.manifold_latent[:] = filtered_latent
        self.manifold_projection_error[:] = torch.mean(torch.abs(filtered_target - raw_target), dim=1)
        self.manifold_projection_error_after_reference_constraint[:] = torch.mean(
            torch.abs(filtered_target - constrained_target), dim=1
        )
        clipped = torch.abs(raw_latent) > self._manifold_latent_limit
        self.manifold_latent_clip_fraction[:] = clipped.float().mean(dim=1)
        self.manifold_reference_overflow[:] = reference_overflow
        self.manifold_reference_clamp_fraction[:] = (reference_overflow > 0.0).float().mean(dim=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._filtered_latent[env_ids] = 0.0
        self._filtered_upper_target[env_ids] = 0.0
        self._filter_initialized[env_ids] = False
        self._filtered_trunk_pitch_target[env_ids] = 0.0
        self._trunk_pitch_filter_initialized[env_ids] = False
        self.trunk_pitch_raw_target[env_ids] = torch.nan
        self.trunk_pitch_reference_target[env_ids] = torch.nan
        self.trunk_pitch_soft_target[env_ids] = torch.nan
        self.trunk_pitch_filtered_target[env_ids] = torch.nan
        self.trunk_pitch_reference_overflow[env_ids] = 0.0
        self.effective_raw_actions[env_ids] = 0.0
        self.prev_effective_raw_actions[env_ids] = 0.0


@configclass
class UpperBodyManifoldJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for coordinated upper-body joint-position targets."""

    class_type: type[ActionTerm] = UpperBodyManifoldJointPositionAction
    upper_body_joint_names: list[str] = MISSING
    command_name: str = "motion"
    manifold_rank: int = 6
    latent_std_limit: float = 3.0
    min_latent_limit: float = 0.03
    orthogonal_residual_limit: float = 0.10
    cutoff_frequency_hz: float = 1.8
    reference_target_margin: float | None = None
    # Control-only pitch stabilizer.  It smooths the policy's deviation around
    # the motion's normal forward-lean pose; it is not a hard pose lock.
    trunk_pitch_joint_name: str | None = None
    trunk_pitch_lower_deviation: float = 0.45
    trunk_pitch_upper_deviation: float = 0.12
    trunk_pitch_cutoff_frequency_hz: float = 1.8
