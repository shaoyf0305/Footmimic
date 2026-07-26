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

    Lower-body targets pass through unchanged.  In legacy mode the policy emits
    one target per robot joint and the selected upper-body targets are projected
    onto a PCA manifold.  Control mode can instead expose the PCA coordinates
    directly: the policy emits only ``manifold_rank`` upper-body latent actions
    and the term decodes them around the current motion reference.  This removes
    the redundant upper-body action null space that otherwise lets raw actions
    grow without changing the executed pose.
    """

    cfg: UpperBodyManifoldJointPositionActionCfg

    @property
    def action_dim(self) -> int:
        """External policy-action dimension.

        ``JointPositionAction`` is initialized with the full robot-joint action
        layout.  Once the controlled joints are known, direct-latent mode
        exposes the lower-body joint actions plus only the PCA latent actions.
        The internal joint target buffers deliberately remain full-sized.
        """
        if getattr(self, "_use_direct_upper_body_latent", False):
            return self._policy_action_dim
        return super().action_dim

    def __init__(self, cfg: UpperBodyManifoldJointPositionActionCfg, env: ManagerBasedEnv):
        if cfg.manifold_rank <= 0:
            raise ValueError("manifold_rank must be positive.")
        if cfg.latent_std_limit <= 0.0 or cfg.min_latent_limit < 0.0:
            raise ValueError("latent limits must be non-negative and latent_std_limit must be positive.")
        if cfg.orthogonal_residual_limit < 0.0 or cfg.cutoff_frequency_hz <= 0.0:
            raise ValueError("orthogonal_residual_limit must be non-negative and cutoff_frequency_hz positive.")
        if cfg.reference_target_margin is not None and cfg.reference_target_margin <= 0.0:
            raise ValueError("reference_target_margin must be positive when enabled.")
        if cfg.upper_body_raw_action_limit is not None and cfg.upper_body_raw_action_limit <= 0.0:
            raise ValueError("upper_body_raw_action_limit must be positive when enabled.")
        if cfg.trunk_stabilized_joint_names:
            if cfg.trunk_stabilized_reference_margin <= 0.0:
                raise ValueError("trunk_stabilized_reference_margin must be positive when enabled.")
            if cfg.trunk_stabilized_cutoff_frequency_hz <= 0.0:
                raise ValueError("trunk_stabilized_cutoff_frequency_hz must be positive when enabled.")
            if cfg.trunk_stabilized_soft_limit_margin < 0.0:
                raise ValueError("trunk_stabilized_soft_limit_margin must be non-negative when enabled.")
            if cfg.trunk_stabilized_turn_start_angle is not None:
                if cfg.trunk_stabilized_turn_start_angle < 0.0:
                    raise ValueError("trunk_stabilized_turn_start_angle must be non-negative when enabled.")
                if cfg.trunk_stabilized_turn_full_angle <= cfg.trunk_stabilized_turn_start_angle:
                    raise ValueError(
                        "trunk_stabilized_turn_full_angle must exceed trunk_stabilized_turn_start_angle."
                    )
                if (
                    cfg.trunk_stabilized_turn_reference_margin <= 0.0
                    or cfg.trunk_stabilized_turn_cutoff_frequency_hz <= 0.0
                ):
                    raise ValueError("turn-relaxed trunk stabilization margin and cutoff must be positive.")
        if cfg.trunk_pitch_joint_name is not None:
            if cfg.trunk_pitch_cutoff_frequency_hz <= 0.0:
                raise ValueError("trunk_pitch_cutoff_frequency_hz must be positive when enabled.")
            if cfg.trunk_pitch_lower_deviation <= 0.0 or cfg.trunk_pitch_upper_deviation <= 0.0:
                raise ValueError("trunk pitch reference deviations must be positive when enabled.")
            if cfg.trunk_pitch_soft_limit_margin < 0.0:
                raise ValueError("trunk_pitch_soft_limit_margin must be non-negative when enabled.")
            if cfg.trunk_pitch_turn_start_angle is not None:
                if cfg.trunk_pitch_turn_start_angle < 0.0:
                    raise ValueError("trunk_pitch_turn_start_angle must be non-negative when enabled.")
                if cfg.trunk_pitch_turn_full_angle <= cfg.trunk_pitch_turn_start_angle:
                    raise ValueError("trunk_pitch_turn_full_angle must exceed trunk_pitch_turn_start_angle.")
                if (
                    cfg.trunk_pitch_turn_lower_deviation <= 0.0
                    or cfg.trunk_pitch_turn_upper_deviation <= 0.0
                    or cfg.trunk_pitch_turn_cutoff_frequency_hz <= 0.0
                ):
                    raise ValueError("turn-relaxed trunk pitch limits and cutoff must be positive.")
        # Keep the parent buffers in the full joint-action layout while it is
        # constructed.  The public action dimension is reduced afterwards.
        self._use_direct_upper_body_latent = False
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
        self._controlled_robot_ids = tuple(controlled_robot_ids)
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
        self._trunk_pitch_turn_pelvis_body_id: int | None = None
        stabilized_robot_ids, stabilized_names = self._asset.find_joints(
            cfg.trunk_stabilized_joint_names, preserve_order=True
        )
        if len(stabilized_robot_ids) != len(cfg.trunk_stabilized_joint_names):
            raise ValueError(
                "Could not resolve every stabilized trunk joint: "
                f"expected {cfg.trunk_stabilized_joint_names}, found {stabilized_names}."
            )
        try:
            stabilized_action_ids = [robot_to_action[int(robot_id)] for robot_id in stabilized_robot_ids]
        except KeyError as exc:
            raise ValueError(
                f"The joint-position action does not control stabilized trunk joint id {exc.args[0]}."
            ) from exc
        self._trunk_stabilized_robot_ids = torch.as_tensor(
            stabilized_robot_ids, dtype=torch.long, device=self.device
        )
        self._trunk_stabilized_action_ids = torch.as_tensor(
            stabilized_action_ids, dtype=torch.long, device=self.device
        )
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
            if cfg.trunk_pitch_turn_start_angle is not None:
                try:
                    self._trunk_pitch_turn_pelvis_body_id = self._asset.body_names.index("pelvis")
                except ValueError as exc:
                    raise ValueError("Turn-relaxed trunk pitch control requires a body named 'pelvis'.") from exc

        upper_dim = len(upper_action_ids)
        rank = min(int(cfg.manifold_rank), upper_dim)
        self._full_action_dim = int(super().action_dim)
        upper_action_id_set = set(upper_action_ids)
        lower_action_ids = [index for index in range(self._full_action_dim) if index not in upper_action_id_set]
        self._lower_action_ids = torch.as_tensor(lower_action_ids, dtype=torch.long, device=self.device)
        self._lower_robot_ids = torch.as_tensor(
            [controlled_robot_ids[index] for index in lower_action_ids], dtype=torch.long, device=self.device
        )
        self._policy_action_dim = len(lower_action_ids) + rank
        self._latent_policy_action_ids = torch.arange(
            len(lower_action_ids), self._policy_action_dim, dtype=torch.long, device=self.device
        )
        self._use_direct_upper_body_latent = bool(cfg.direct_upper_body_latent_action)
        self._filtered_latent = torch.zeros(self.num_envs, rank, device=self.device)
        self._filtered_upper_target = torch.zeros(self.num_envs, upper_dim, device=self.device)
        self._filter_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._filtered_trunk_pitch_target = torch.zeros(self.num_envs, device=self.device)
        self._trunk_pitch_filter_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._filtered_trunk_stabilized_target = torch.zeros(
            self.num_envs, len(stabilized_action_ids), device=self.device
        )
        self._trunk_stabilized_filter_initialized = torch.zeros(
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
        # Amount of the reference-bounded upper target that lies outside the
        # PCA subspace.  This is diagnostic/reward telemetry only; it does not
        # add a second action constraint.
        self.manifold_nullspace_residual = torch.zeros(self.num_envs, device=self.device)
        self.manifold_latent_clip_fraction = torch.zeros(self.num_envs, device=self.device)
        self.manifold_reference_overflow = torch.zeros_like(self._filtered_upper_target)
        self.manifold_reference_clamp_fraction = torch.zeros(self.num_envs, device=self.device)
        self.manifold_policy_latent = torch.zeros(self.num_envs, rank, device=self.device)
        self.trunk_pitch_raw_target = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_reference_target = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_soft_target = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_filtered_target = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_reference_overflow = torch.zeros(self.num_envs, device=self.device)
        self.trunk_pitch_turn_relaxation = torch.zeros(self.num_envs, device=self.device)
        self.trunk_pitch_active_lower_deviation = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_active_upper_deviation = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.trunk_pitch_active_cutoff_frequency_hz = torch.full((self.num_envs,), torch.nan, device=self.device)
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

    @property
    def uses_direct_upper_body_latent(self) -> bool:
        """Whether the external policy emits PCA coordinates instead of arm joints."""
        return self._use_direct_upper_body_latent

    def policy_action_ids_for_robot_joint_ids(self, robot_joint_ids: Sequence[int]) -> list[int]:
        """Return external policy indices for lower-body robot joints.

        Upper-body joints intentionally have no one-to-one policy action in
        direct-latent mode and therefore cannot be queried through this method.
        """
        if not self._use_direct_upper_body_latent:
            robot_to_action = {
                int(robot_id): action_id for action_id, robot_id in enumerate(self._controlled_robot_ids)
            }
        else:
            robot_to_action = {
                int(robot_id): action_id for action_id, robot_id in enumerate(self._lower_robot_ids.tolist())
            }
        try:
            return [robot_to_action[int(robot_id)] for robot_id in robot_joint_ids]
        except KeyError as exc:
            raise ValueError(
                "Upper-body joints are represented by PCA latent actions and do not have individual "
                f"policy-action indices (robot joint id {exc.args[0]})."
            ) from exc

    def _turn_relaxation(self, command) -> torch.Tensor:
        """Return the shared turn/recovery relaxation signal for trunk controls."""
        start_angle = self.cfg.trunk_stabilized_turn_start_angle
        if start_angle is None:
            return torch.zeros(self.num_envs, device=self.device)
        target_heading = getattr(command, "locomotion_cmd_target_heading", command.locomotion_cmd_heading)
        heading_delta = torch.atan2(
            torch.sin(target_heading - command.locomotion_cmd_heading),
            torch.cos(target_heading - command.locomotion_cmd_heading),
        ).abs()
        if self._trunk_pitch_turn_pelvis_body_id is not None:
            pelvis_quat = self._asset.data.body_quat_w[:, self._trunk_pitch_turn_pelvis_body_id]
            pelvis_yaw = torch.atan2(
                2.0 * (pelvis_quat[:, 0] * pelvis_quat[:, 3] + pelvis_quat[:, 1] * pelvis_quat[:, 2]),
                1.0 - 2.0 * (torch.square(pelvis_quat[:, 2]) + torch.square(pelvis_quat[:, 3])),
            )
            tracking_error = torch.atan2(
                torch.sin(command.locomotion_cmd_heading - pelvis_yaw),
                torch.cos(command.locomotion_cmd_heading - pelvis_yaw),
            ).abs()
            heading_delta = torch.maximum(heading_delta, tracking_error)
        return torch.clamp(
            (heading_delta - float(start_angle))
            / (float(self.cfg.trunk_stabilized_turn_full_angle) - float(start_angle)),
            min=0.0,
            max=1.0,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        policy_latent = None
        if self._use_direct_upper_body_latent:
            if actions.shape[1] != self._policy_action_dim:
                raise ValueError(
                    f"Expected {self._policy_action_dim} direct-latent actions, got {actions.shape[1]}. "
                    "This control environment requires a freshly initialized policy head."
                )
            policy_latent = actions[:, self._latent_policy_action_ids]
            full_actions = torch.zeros(
                actions.shape[0], self._full_action_dim, device=actions.device, dtype=actions.dtype
            )
            full_actions[:, self._lower_action_ids] = actions[:, : self._lower_action_ids.numel()]
            actions = full_actions
        # Legacy mode keeps the optional replay-only guard for old policies.
        elif self.cfg.upper_body_raw_action_limit is not None:
            actions = actions.clone()
            limit = float(self.cfg.upper_body_raw_action_limit)
            actions[:, self._upper_action_ids] = torch.clamp(
                actions[:, self._upper_action_ids], min=-limit, max=limit
            )
        super().process_actions(actions)
        if self._manifold_mean is None:
            self._fit_manifold_from_motion_bank()

        assert self._manifold_mean is not None
        assert self._manifold_basis is not None
        assert self._manifold_latent_limit is not None
        step_dt = float(getattr(self._manifold_env, "step_dt", 0.02))

        action_ids = self._upper_action_ids
        reference_margin = self.cfg.reference_target_margin
        command = self._manifold_env.command_manager.get_term(self.cfg.command_name)
        reference_target = command.joint_pos[:, self._upper_robot_ids]
        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._upper_robot_ids]

        if self._use_direct_upper_body_latent:
            assert policy_latent is not None
            # ``tanh`` gives every latent a continuous bounded physical meaning.
            # The policy therefore cannot hide in the old 14-D target-space null
            # directions or produce an unbounded arm target before PCA.
            bounded_latent = self._manifold_latent_limit * torch.tanh(policy_latent)
            alpha = 1.0 - math.exp(-2.0 * math.pi * float(self.cfg.cutoff_frequency_hz) * step_dt)
            alpha = min(max(alpha, 0.0), 1.0)
            filtered_latent = torch.where(
                self._filter_initialized[:, None],
                self._filtered_latent + alpha * (bounded_latent - self._filtered_latent),
                bounded_latent,
            )
            self._filtered_latent[:] = filtered_latent
            self._filter_initialized[:] = True
            upper_raw_target = reference_target + filtered_latent @ self._manifold_basis.transpose(0, 1)
            clipped = torch.abs(torch.tanh(policy_latent)) >= 0.95
            self.manifold_policy_latent[:] = policy_latent
        else:
            upper_raw_target = self.processed_actions[:, action_ids].clone()
            clipped = None

        # The reference envelope remains the physical/style safety guard in
        # both interfaces.  In direct-latent mode it constrains decoded targets,
        # not an arbitrary 14-D policy target.
        if reference_margin is None:
            constrained_target = upper_raw_target
            reference_overflow = torch.zeros_like(upper_raw_target)
        else:
            reference_overflow = torch.clamp(
                torch.abs(upper_raw_target - reference_target) - reference_margin,
                min=0.0,
            )
            constrained_target = torch.clamp(
                upper_raw_target,
                min=reference_target - reference_margin,
                max=reference_target + reference_margin,
            )

        if self._use_direct_upper_body_latent:
            filtered_target = torch.clamp(constrained_target, min=soft_limits[..., 0], max=soft_limits[..., 1])
            nullspace_residual = torch.zeros(self.num_envs, device=self.device)
        else:
            centered = constrained_target - self._manifold_mean
            raw_latent = centered @ self._manifold_basis
            bounded_latent = torch.clamp(
                raw_latent,
                min=-self._manifold_latent_limit,
                max=self._manifold_latent_limit,
            )
            parallel = bounded_latent @ self._manifold_basis.transpose(0, 1)
            orthogonal = centered - (raw_latent @ self._manifold_basis.transpose(0, 1))
            nullspace_residual = torch.mean(torch.abs(orthogonal), dim=1)
            residual_limit = float(self.cfg.orthogonal_residual_limit)
            bounded_orthogonal = residual_limit * torch.tanh(orthogonal / max(residual_limit, 1.0e-6))
            projected = self._manifold_mean + parallel + bounded_orthogonal
            projected = torch.clamp(projected, min=soft_limits[..., 0], max=soft_limits[..., 1])
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
            filtered_target = torch.clamp(filtered_target, min=soft_limits[..., 0], max=soft_limits[..., 1])
            self._filtered_latent[:] = filtered_latent
            self._filter_initialized[:] = True
            clipped = torch.abs(raw_latent) > self._manifold_latent_limit
        self._filtered_upper_target[:] = filtered_target
        self._processed_actions[:, action_ids] = filtered_target

        if self._trunk_stabilized_action_ids.numel() > 0:
            stabilized_action_ids = self._trunk_stabilized_action_ids
            stabilized_robot_ids = self._trunk_stabilized_robot_ids
            command = self._manifold_env.command_manager.get_term(self.cfg.command_name)
            stabilized_raw_target = self.processed_actions[:, stabilized_action_ids].clone()
            soft_limits = self._asset.data.soft_joint_pos_limits[:, stabilized_robot_ids]
            limit_margin = float(self.cfg.trunk_stabilized_soft_limit_margin)
            stabilized_reference_target = torch.clamp(
                command.joint_pos[:, stabilized_robot_ids],
                min=soft_limits[..., 0] + limit_margin,
                max=soft_limits[..., 1] - limit_margin,
            )
            deviation = stabilized_raw_target - stabilized_reference_target
            turn_relaxation = self._turn_relaxation(command)
            envelope = (
                float(self.cfg.trunk_stabilized_reference_margin)
                + turn_relaxation
                * (
                    float(self.cfg.trunk_stabilized_turn_reference_margin)
                    - float(self.cfg.trunk_stabilized_reference_margin)
                )
            )[:, None]
            soft_target = stabilized_reference_target + envelope * torch.tanh(deviation / envelope)
            soft_target = torch.clamp(
                soft_target, min=soft_limits[..., 0] + limit_margin, max=soft_limits[..., 1] - limit_margin
            )
            cutoff = (
                float(self.cfg.trunk_stabilized_cutoff_frequency_hz)
                + turn_relaxation
                * (
                    float(self.cfg.trunk_stabilized_turn_cutoff_frequency_hz)
                    - float(self.cfg.trunk_stabilized_cutoff_frequency_hz)
                )
            )
            alpha = torch.clamp(1.0 - torch.exp(-2.0 * math.pi * cutoff * step_dt), min=0.0, max=1.0)[:, None]
            initialized = self._trunk_stabilized_filter_initialized[:, None]
            previous_target = torch.where(
                initialized,
                self._filtered_trunk_stabilized_target,
                self._asset.data.joint_pos[:, stabilized_robot_ids],
            )
            stabilized_filtered_target = previous_target + alpha * (soft_target - previous_target)
            stabilized_filtered_target = torch.clamp(
                stabilized_filtered_target,
                min=soft_limits[..., 0] + limit_margin,
                max=soft_limits[..., 1] - limit_margin,
            )
            self._filtered_trunk_stabilized_target[:] = stabilized_filtered_target
            self._trunk_stabilized_filter_initialized[:] = True
            self._processed_actions[:, stabilized_action_ids] = stabilized_filtered_target

        if self._trunk_pitch_robot_id is not None and self._trunk_pitch_action_id is not None:
            trunk_action_id = self._trunk_pitch_action_id
            trunk_robot_id = self._trunk_pitch_robot_id
            trunk_raw_target = self.processed_actions[:, trunk_action_id].clone()
            command = self._manifold_env.command_manager.get_term(self.cfg.command_name)
            # The style clips can contain a waist pitch just outside the
            # simulator's soft range.  Treating that value as the centre of a
            # narrow policy envelope makes the policy permanently fight the
            # soft-limit penalty.  Keep the reference inside the same range
            # used by the simulator, with an optional guard band.
            trunk_reference_target = command.joint_pos[:, trunk_robot_id]
            trunk_soft_limits = self._asset.data.soft_joint_pos_limits[:, trunk_robot_id]
            trunk_limit_margin = float(self.cfg.trunk_pitch_soft_limit_margin)
            trunk_reference_lower = trunk_soft_limits[:, 0] + trunk_limit_margin
            trunk_reference_upper = trunk_soft_limits[:, 1] - trunk_limit_margin
            trunk_reference_target = torch.clamp(
                trunk_reference_target,
                min=trunk_reference_lower,
                max=trunk_reference_upper,
            )
            trunk_deviation = trunk_raw_target - trunk_reference_target
            turn_relaxation = torch.zeros(self.num_envs, device=self.device)
            if self.cfg.trunk_pitch_turn_start_angle is not None:
                target_heading = getattr(command, "locomotion_cmd_target_heading", command.locomotion_cmd_heading)
                heading_delta = torch.atan2(
                    torch.sin(target_heading - command.locomotion_cmd_heading),
                    torch.cos(target_heading - command.locomotion_cmd_heading),
                ).abs()
                if self._trunk_pitch_turn_pelvis_body_id is not None:
                    pelvis_quat = self._asset.data.body_quat_w[:, self._trunk_pitch_turn_pelvis_body_id]
                    pelvis_yaw = torch.atan2(
                        2.0 * (pelvis_quat[:, 0] * pelvis_quat[:, 3] + pelvis_quat[:, 1] * pelvis_quat[:, 2]),
                        1.0 - 2.0 * (torch.square(pelvis_quat[:, 2]) + torch.square(pelvis_quat[:, 3])),
                    )
                    tracking_error = torch.atan2(
                        torch.sin(command.locomotion_cmd_heading - pelvis_yaw),
                        torch.cos(command.locomotion_cmd_heading - pelvis_yaw),
                    ).abs()
                    heading_delta = torch.maximum(heading_delta, tracking_error)
                turn_relaxation = torch.clamp(
                    (heading_delta - float(self.cfg.trunk_pitch_turn_start_angle))
                    / (float(self.cfg.trunk_pitch_turn_full_angle) - float(self.cfg.trunk_pitch_turn_start_angle)),
                    min=0.0,
                    max=1.0,
                )
            lower_deviation = (
                float(self.cfg.trunk_pitch_lower_deviation)
                + turn_relaxation
                * (float(self.cfg.trunk_pitch_turn_lower_deviation) - float(self.cfg.trunk_pitch_lower_deviation))
            )
            upper_deviation = (
                float(self.cfg.trunk_pitch_upper_deviation)
                + turn_relaxation
                * (float(self.cfg.trunk_pitch_turn_upper_deviation) - float(self.cfg.trunk_pitch_upper_deviation))
            )
            trunk_soft_target = trunk_reference_target + torch.where(
                trunk_deviation < 0.0,
                lower_deviation * torch.tanh(trunk_deviation / lower_deviation),
                upper_deviation * torch.tanh(trunk_deviation / upper_deviation),
            )
            trunk_soft_target = torch.clamp(
                trunk_soft_target, min=trunk_reference_lower, max=trunk_reference_upper
            )
            trunk_cutoff_frequency_hz = (
                float(self.cfg.trunk_pitch_cutoff_frequency_hz)
                + turn_relaxation
                * (
                    float(self.cfg.trunk_pitch_turn_cutoff_frequency_hz)
                    - float(self.cfg.trunk_pitch_cutoff_frequency_hz)
                )
            )
            trunk_alpha = 1.0 - torch.exp(-2.0 * math.pi * trunk_cutoff_frequency_hz * step_dt)
            trunk_alpha = torch.clamp(trunk_alpha, min=0.0, max=1.0)
            previous_trunk_target = torch.where(
                self._trunk_pitch_filter_initialized,
                self._filtered_trunk_pitch_target,
                self._asset.data.joint_pos[:, trunk_robot_id],
            )
            filtered_trunk_target = previous_trunk_target + trunk_alpha * (
                trunk_soft_target - previous_trunk_target
            )
            filtered_trunk_target = torch.clamp(
                filtered_trunk_target, min=trunk_reference_lower, max=trunk_reference_upper
            )
            self._filtered_trunk_pitch_target[:] = filtered_trunk_target
            self._trunk_pitch_filter_initialized[:] = True
            self._processed_actions[:, trunk_action_id] = filtered_trunk_target
            self.trunk_pitch_raw_target[:] = trunk_raw_target
            self.trunk_pitch_reference_target[:] = trunk_reference_target
            self.trunk_pitch_soft_target[:] = trunk_soft_target
            self.trunk_pitch_filtered_target[:] = filtered_trunk_target
            self.trunk_pitch_turn_relaxation[:] = turn_relaxation
            self.trunk_pitch_active_lower_deviation[:] = lower_deviation
            self.trunk_pitch_active_upper_deviation[:] = upper_deviation
            self.trunk_pitch_active_cutoff_frequency_hz[:] = trunk_cutoff_frequency_hz
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
        if self._trunk_stabilized_action_ids.numel() > 0:
            stabilized_action_ids = self._trunk_stabilized_action_ids
            stabilized_scale = scale[:, stabilized_action_ids]
            safe_stabilized_scale = torch.where(
                torch.abs(stabilized_scale) < 1.0e-8,
                torch.ones_like(stabilized_scale),
                stabilized_scale,
            )
            self.effective_raw_actions[:, stabilized_action_ids] = (
                self._processed_actions[:, stabilized_action_ids] - offset[:, stabilized_action_ids]
            ) / safe_stabilized_scale

        self.manifold_raw_upper_target[:] = upper_raw_target
        self.manifold_reference_upper_target[:] = reference_target
        self.manifold_constrained_upper_target[:] = constrained_target
        self.manifold_projected_upper_target[:] = filtered_target
        self.manifold_latent[:] = filtered_latent
        self.manifold_projection_error[:] = torch.mean(torch.abs(filtered_target - upper_raw_target), dim=1)
        self.manifold_projection_error_after_reference_constraint[:] = torch.mean(
            torch.abs(filtered_target - constrained_target), dim=1
        )
        self.manifold_nullspace_residual[:] = nullspace_residual
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
        self._filtered_trunk_stabilized_target[env_ids] = 0.0
        self._trunk_stabilized_filter_initialized[env_ids] = False
        self.trunk_pitch_raw_target[env_ids] = torch.nan
        self.trunk_pitch_reference_target[env_ids] = torch.nan
        self.trunk_pitch_soft_target[env_ids] = torch.nan
        self.trunk_pitch_filtered_target[env_ids] = torch.nan
        self.trunk_pitch_reference_overflow[env_ids] = 0.0
        self.trunk_pitch_turn_relaxation[env_ids] = 0.0
        self.trunk_pitch_active_lower_deviation[env_ids] = torch.nan
        self.trunk_pitch_active_upper_deviation[env_ids] = torch.nan
        self.trunk_pitch_active_cutoff_frequency_hz[env_ids] = torch.nan
        self.manifold_policy_latent[env_ids] = 0.0
        self.manifold_nullspace_residual[env_ids] = 0.0
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
    # Control-only interface: policy outputs one PCA coordinate per manifold
    # component instead of one raw action per upper-body joint.
    direct_upper_body_latent_action: bool = False
    # Clamp large normalized arm commands before target scaling.  This is
    # intentionally disabled by default to preserve non-control action semantics.
    upper_body_raw_action_limit: float | None = None
    # Reference-relative stabilizer for waist roll/yaw.  Pitch is separate
    # because it needs an asymmetric, turn-aware envelope.
    trunk_stabilized_joint_names: tuple[str, ...] = ()
    trunk_stabilized_reference_margin: float = 0.20
    trunk_stabilized_cutoff_frequency_hz: float = 1.5
    trunk_stabilized_soft_limit_margin: float = 0.0
    trunk_stabilized_turn_start_angle: float | None = None
    trunk_stabilized_turn_full_angle: float = 0.45
    trunk_stabilized_turn_reference_margin: float = 0.45
    trunk_stabilized_turn_cutoff_frequency_hz: float = 4.0
    # Control-only pitch stabilizer.  It smooths the policy's deviation around
    # the motion's normal forward-lean pose; it is not a hard pose lock.
    trunk_pitch_joint_name: str | None = None
    # Keep a style reference strictly inside the simulator's soft joint range.
    # This avoids centring a policy envelope on a pose that the joint-limit
    # reward will continuously penalize.
    trunk_pitch_soft_limit_margin: float = 0.0
    trunk_pitch_lower_deviation: float = 0.45
    trunk_pitch_upper_deviation: float = 0.12
    trunk_pitch_cutoff_frequency_hz: float = 1.8
    # During an unfinished heading transition the robot needs pitch authority
    # to redirect its momentum.  The style envelope then returns smoothly as
    # the effective heading catches the requested heading.
    trunk_pitch_turn_start_angle: float | None = None
    trunk_pitch_turn_full_angle: float = 0.45
    trunk_pitch_turn_lower_deviation: float = 0.65
    trunk_pitch_turn_upper_deviation: float = 0.28
    trunk_pitch_turn_cutoff_frequency_hz: float = 4.0
