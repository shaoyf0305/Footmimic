"""Dribbling motion command with XGen-style demo ball stitching.

Uses per-frame ``ball_pos_w`` from motion for labels and, for unified local
stages, reset placement at the first labelled contact.  The reset position is
represented relative to the frame-0 pelvis yaw, never by a fixed simulation
``+X`` direction. Optional demo kinematic snap is controlled separately by
``dribble_cg_snap_mode``:

- ``full`` (default): every step writes the demo ball pose into simulation.
- ``non_contact_only``: only overwrite the ball when the CG label says
  non-contact, leaving physics during annotated contact segments.
- ``never``: never overwrite the ball; use the labels/rewards only.

Contact / foot / surface masks come from ``dribble_cg_contact``,
``dribble_cg_foot``, and ``dribble_cg_surface`` in ``.npz``.  The legacy
``kick_frame`` / ``kick_end_frame`` / ``kick_leg`` metadata remains a fallback
for contact timing and foot side, but cannot describe an instep surface (see
:class:`MultiMotionLoader`). ``dribble_cg_flow_*`` stores the outgoing
contact-to-contact direction, distance, and duration used by Stage 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

from soccer.tasks.tracking.mdp.task_frame import mimic_anchor_yaw_delta_quat, spawn_ball_ahead_env_local

from .commands_multi_motion_soccer import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DribbleCGMotionCommand(MotionCommand):
    """Soccer motion command + demo ball sync for dribbling CG."""

    def __init__(self, cfg: DribbleCGMotionCommandCfg, env: ManagerBasedRLEnv):
        # ``MotionCommand.__init__`` invokes ``_compute_soccer_ball_positions``.
        # Allocate reset provenance first because dynamic dispatch reaches the
        # overridden first-contact implementation during that base setup.
        self.ball_spawn_reference_contact_frame = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        # 1 = reference first-contact placement; 0 = explicit local-front
        # fallback; -1 = legacy/unknown source.  Keep this state separate from
        # the live ball pose so diagnostics retain reset provenance after the
        # ball has moved under physics.
        self.ball_spawn_source = torch.full(
            (env.num_envs,), -1, dtype=torch.int8, device=env.device
        )
        self.ball_spawn_reference_local = torch.zeros(env.num_envs, 3, device=env.device)
        self.s2_episode_first_contact_frame = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self.s2_episode_contact_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        super().__init__(cfg, env)
        self._build_s2_contact_event_table()
        for metric_name in (
            "s2_contact_success_rate",
            "s2_missed_contact_rate",
            "s2_correct_foot_rate",
            "s2_correct_side_rate",
            "s2_contact_timing_error",
            "s2_contact_timing_abs_error",
            "s2_target_region_distance",
            "s2_wrong_foot_count",
            "s2_wrong_side_count",
            "s2_invalid_body_contact_count",
            "s2_complete_2",
            "s2_complete_4",
            "s2_complete_8",
        ):
            self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
        self._validate_fixed_touch_spec()

    def _build_s2_contact_event_table(self) -> None:
        """Build windowed contact-event labels from the v5.10 CG segments.

        A rising edge in ``dribble_cg_contact`` is the event frame.  Each
        event owns a symmetric time window and carries exactly one expected
        foot and one foot-yaw-local ball side (0=left, 1=right).  The older
        inside/outside labels are converted to local left/right here so every
        downstream consumer uses one unambiguous convention.
        """
        num_motions, max_frames = self.motion.dribble_cg_contact.shape
        self._s2_event_id = torch.full(
            (num_motions, max_frames), -1, dtype=torch.long, device=self.device
        )
        self._s2_event_frame = torch.full_like(self._s2_event_id, -1)
        self._s2_event_foot = torch.full(
            (num_motions, max_frames), -1, dtype=torch.int8, device=self.device
        )
        self._s2_event_side = torch.full_like(self._s2_event_foot, -1)
        self._s2_event_frames_by_motion: list[torch.Tensor] = []

        fps = float(self.motion.fps.reshape(-1)[0])
        half_window = max(
            0, int(round(float(getattr(self.cfg, "dribble_cg_contact_window_seconds", 0.10)) * fps))
        )
        post_grace = max(
            0, int(getattr(self.cfg, "dribble_cg_missed_contact_grace_steps", 3))
        )
        required_tail = half_window + post_grace + 1
        for motion_idx in range(num_motions):
            length = int(self.motion.file_lengths[motion_idx].item())
            contact = self.motion.dribble_cg_contact[motion_idx, :length] > 0
            previous = torch.cat(
                (torch.zeros(1, dtype=torch.bool, device=self.device), contact[:-1])
            )
            starts = torch.nonzero(contact & ~previous, as_tuple=False).squeeze(-1)
            self._s2_event_frames_by_motion.append(starts)
            for event_id, event_frame_tensor in enumerate(starts):
                event_frame = int(event_frame_tensor.item())
                foot = int(self.motion.dribble_cg_foot[motion_idx, event_frame].item())
                surface = int(self.motion.dribble_cg_surface[motion_idx, event_frame].item())
                # In a foot yaw frame +Y is local left.  Inside is +Y for the
                # right foot and -Y for the left foot.
                side = -1
                if foot in (0, 1) and surface in (0, 1):
                    local_left = (foot == 1 and surface == 0) or (foot == 0 and surface == 1)
                    side = 0 if local_left else 1

                start = max(0, event_frame - half_window)
                end = min(length - 1, event_frame + half_window)
                frames = torch.arange(start, end + 1, device=self.device)
                existing_frame = self._s2_event_frame[motion_idx, frames]
                replace = (existing_frame < 0) | (
                    torch.abs(frames - event_frame) < torch.abs(frames - existing_frame)
                )
                target_frames = frames[replace]
                self._s2_event_id[motion_idx, target_frames] = event_id
                self._s2_event_frame[motion_idx, target_frames] = event_frame
                self._s2_event_foot[motion_idx, target_frames] = foot
                self._s2_event_side[motion_idx, target_frames] = side
        self._s2_two_contact_transition_candidates: dict[
            tuple[int, int], torch.Tensor
        ] = {}
        transition_lists: dict[tuple[int, int], list[tuple[int, int]]] = {
            (0, 1): [], (1, 0): []
        }
        for motion_idx, frames in enumerate(self._s2_event_frames_by_motion):
            final_eligible_event = int(self.motion.file_lengths[motion_idx].item()) - 1 - required_tail
            for start_index in range(max(0, int(frames.numel()) - 1)):
                first = int(frames[start_index].item())
                second = int(frames[start_index + 1].item())
                if second > final_eligible_event:
                    continue
                transition = (
                    int(self.motion.dribble_cg_surface[motion_idx, first].item()),
                    int(self.motion.dribble_cg_surface[motion_idx, second].item()),
                )
                if transition in transition_lists:
                    transition_lists[transition].append((motion_idx, start_index))
        for transition, candidates in transition_lists.items():
            if candidates:
                self._s2_two_contact_transition_candidates[transition] = torch.as_tensor(
                    candidates, dtype=torch.long, device=self.device
                )

        max_events = max((int(frames.numel()) for frames in self._s2_event_frames_by_motion), default=0)
        self._s2_event_count_by_motion = torch.as_tensor(
            [int(frames.numel()) for frames in self._s2_event_frames_by_motion],
            dtype=torch.long,
            device=self.device,
        )
        self._s2_event_frames_padded = torch.full(
            (num_motions, max(1, max_events)), -1, dtype=torch.long, device=self.device
        )
        for motion_idx, frames in enumerate(self._s2_event_frames_by_motion):
            self._s2_event_frames_padded[motion_idx, : frames.numel()] = frames
        requested_counts = {
            int(item[0])
            for item in tuple(getattr(self.cfg, "dribble_cg_curriculum_mix", ()))
            if int(item[0]) > 0
        }
        requested_counts.update((1, 2, 4, 8))
        self._s2_sequence_candidates: dict[int, torch.Tensor] = {}
        for count in requested_counts:
            candidates: list[tuple[int, int]] = []
            for motion_idx, frames in enumerate(self._s2_event_frames_by_motion):
                final_eligible_event = (
                    int(self.motion.file_lengths[motion_idx].item()) - 1 - required_tail
                )
                for start_index in range(max(0, int(frames.numel()) - count + 1)):
                    if int(frames[start_index + count - 1].item()) <= final_eligible_event:
                        candidates.append((motion_idx, start_index))
            if candidates:
                self._s2_sequence_candidates[count] = torch.as_tensor(
                    candidates, dtype=torch.long, device=self.device
                )

    def _s2_current_event_label(self, labels: torch.Tensor) -> torch.Tensor:
        frame = torch.minimum(self.time_steps, self.motion.file_lengths[self.motion_idx] - 1)
        return labels[self.motion_idx, frame]

    @property
    def s2_contact_event_id_ref(self) -> torch.Tensor:
        return self._s2_current_event_label(self._s2_event_id).to(torch.long)

    @property
    def s2_contact_event_frame_ref(self) -> torch.Tensor:
        return self._s2_current_event_label(self._s2_event_frame).to(torch.long)

    @property
    def s2_contact_event_foot_ref(self) -> torch.Tensor:
        return self._s2_current_event_label(self._s2_event_foot).to(torch.long)

    @property
    def s2_contact_event_side_ref(self) -> torch.Tensor:
        """Expected ball side in the reference foot yaw frame: left=0/right=1."""
        return self._s2_current_event_label(self._s2_event_side).to(torch.long)

    @property
    def s2_contact_window_ref(self) -> torch.Tensor:
        return self.s2_contact_event_id_ref >= 0

    def s2_contact_reference_foot_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the selected event's frozen reference foot position/yaw.

        The event pose is mapped through the same live yaw alignment used by
        motion imitation.  It therefore remains reference-defined without
        introducing a fixed world-forward frame.
        """
        event_frame = self.s2_contact_event_frame_ref.clamp(min=0)
        event_foot = self.s2_contact_event_foot_ref
        left_index = self.cfg.body_names.index("left_ankle_roll_link")
        right_index = self.cfg.body_names.index("right_ankle_roll_link")
        foot_index = torch.where(
            event_foot == 1,
            torch.full_like(event_foot, right_index),
            torch.full_like(event_foot, left_index),
        )
        raw_pos = self.motion.body_pos_w[self.motion_idx, event_frame, foot_index]
        raw_quat = self.motion.body_quat_w[self.motion_idx, event_frame, foot_index]
        raw_pos = raw_pos + self._env.scene.env_origins

        alignment = self._mimic_target_yaw_delta_quat(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        mapped_origin = self.robot_anchor_pos_w.clone()
        mapped_origin[:, 2] = self.anchor_pos_w[:, 2]
        mapped_pos = mapped_origin + quat_apply(alignment, raw_pos - self.anchor_pos_w)
        mapped_yaw = yaw_quat(quat_mul(alignment, raw_quat))
        return mapped_pos, mapped_yaw

    def _sample_s2_contact_curriculum(self, env_ids: torch.Tensor) -> bool:
        """Sample a contact-count episode; return false when the feature is off."""
        mix = tuple(getattr(self.cfg, "dribble_cg_curriculum_mix", ()))
        if not mix:
            return False
        counts = torch.as_tensor([int(item[0]) for item in mix], device=self.device)
        probabilities = torch.as_tensor(
            [float(item[1]) for item in mix], dtype=torch.float32, device=self.device
        )
        if bool(torch.any(probabilities < 0.0)) or float(probabilities.sum().item()) <= 0.0:
            raise ValueError("dribble_cg_curriculum_mix probabilities must be non-negative with positive sum")
        probabilities = probabilities / probabilities.sum()
        sampled_counts = counts[torch.multinomial(probabilities, env_ids.numel(), replacement=True)]
        fps = float(self.motion.fps.reshape(-1)[0])
        pre_min, pre_max = getattr(self.cfg, "dribble_cg_pre_contact_seconds_range", (0.3, 0.6))
        pre_min_frames = max(0, int(round(float(pre_min) * fps)))
        pre_max_frames = max(pre_min_frames, int(round(float(pre_max) * fps)))
        half_window = max(
            0, int(round(float(getattr(self.cfg, "dribble_cg_contact_window_seconds", 0.10)) * fps))
        )
        post_grace = max(0, int(getattr(self.cfg, "dribble_cg_missed_contact_grace_steps", 3)))

        for requested_count_tensor in torch.unique(sampled_counts):
            requested_count = int(requested_count_tensor.item())
            group_mask = sampled_counts == requested_count
            group_env_ids = env_ids[group_mask]
            group_size = group_env_ids.numel()
            if group_size == 0:
                continue

            selected_motion = torch.empty(group_size, dtype=torch.long, device=self.device)
            sequence_start = torch.zeros(group_size, dtype=torch.long, device=self.device)
            if requested_count == 0:
                eligible_motion = torch.nonzero(
                    self._s2_event_count_by_motion > 0, as_tuple=False
                ).squeeze(-1)
                if eligible_motion.numel() == 0:
                    raise ValueError("No motion contains an S2 contact event")
                selected_motion[:] = eligible_motion[
                    torch.randint(eligible_motion.numel(), (group_size,), device=self.device)
                ]
                actual_count = self._s2_event_count_by_motion[selected_motion]
            elif requested_count == 2 and self._s2_two_contact_transition_candidates:
                transitions = tuple(self._s2_two_contact_transition_candidates.values())
                transition_choice = torch.randint(len(transitions), (group_size,), device=self.device)
                for transition_index, candidates in enumerate(transitions):
                    transition_mask = transition_choice == transition_index
                    count_in_transition = int(transition_mask.sum().item())
                    if count_in_transition == 0:
                        continue
                    sampled = candidates[
                        torch.randint(candidates.shape[0], (count_in_transition,), device=self.device)
                    ]
                    selected_motion[transition_mask] = sampled[:, 0]
                    sequence_start[transition_mask] = sampled[:, 1]
                actual_count = torch.full_like(selected_motion, 2)
            else:
                candidates = self._s2_sequence_candidates.get(requested_count)
                if candidates is None or candidates.numel() == 0:
                    raise ValueError(
                        f"No motion contains a {requested_count}-contact curriculum sequence"
                    )
                sampled = candidates[
                    torch.randint(candidates.shape[0], (group_size,), device=self.device)
                ]
                selected_motion = sampled[:, 0]
                sequence_start = sampled[:, 1]
                actual_count = torch.full_like(selected_motion, requested_count)

            first_event = self._s2_event_frames_padded[selected_motion, sequence_start]
            last_event = self._s2_event_frames_padded[
                selected_motion, sequence_start + actual_count - 1
            ]
            pre_frames = torch.randint(
                pre_min_frames, pre_max_frames + 1, (group_size,), device=self.device
            )
            true_length = self.motion.file_lengths[selected_motion]
            if requested_count == 0:
                start_frame = torch.zeros_like(first_event)
                end_frame = true_length - 1
            else:
                start_frame = torch.clamp(first_event - pre_frames, min=0)
                end_frame = torch.minimum(
                    true_length - 1,
                    last_event + half_window + post_grace + 1,
                )

            self.motion_idx[group_env_ids] = selected_motion
            self.motion_length[group_env_ids] = end_frame + 1
            self.time_steps[group_env_ids] = start_frame
            self.s2_episode_first_contact_frame[group_env_ids] = first_event
            self.s2_episode_contact_count[group_env_ids] = actual_count
        return True

    def _uniform_sampling(self, env_ids: Sequence[int]):
        ids = self._to_env_id_tensor(env_ids)
        if not hasattr(self, "_s2_event_frames_by_motion"):
            super()._uniform_sampling(ids)
            return
        if self._sample_s2_contact_curriculum(ids):
            return
        super()._uniform_sampling(ids)
        self.s2_episode_first_contact_frame[ids] = -1
        self.s2_episode_contact_count[ids] = 0

    def _set_debug_vis_impl(self, debug_vis: bool):
        regions_only = bool(
            getattr(self.cfg, "dribble_cg_s2_debug_regions_only", False)
        )
        if not regions_only:
            super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "s2_contact_event_foot_visualizer"):
                self.s2_contact_event_foot_visualizer = VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(
                        prim_path="/Visuals/S2Contact/reference_foot"
                    )
                )
                self.s2_contact_region_visualizer = VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(
                        prim_path="/Visuals/S2Contact/region_bounds"
                    )
                )
            self.s2_contact_event_foot_visualizer.set_visibility(True)
            self.s2_contact_region_visualizer.set_visibility(True)
        elif hasattr(self, "s2_contact_event_foot_visualizer"):
            self.s2_contact_event_foot_visualizer.set_visibility(False)
            self.s2_contact_region_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        if not bool(getattr(self.cfg, "dribble_cg_s2_debug_regions_only", False)):
            super()._debug_vis_callback(event)
        if not hasattr(self, "s2_contact_event_foot_visualizer"):
            return
        foot_pos, foot_yaw = self.s2_contact_reference_foot_pose_w()
        active = self.s2_contact_window_ref
        side = self.s2_contact_event_side_ref
        hidden_foot_pos = foot_pos.clone()
        hidden_foot_pos[~active, 2] = -100.0
        self.s2_contact_event_foot_visualizer.visualize(hidden_foot_pos, foot_yaw)

        # Four frames mark the target rectangle corners.  +Y is visibly the
        # foot's local left axis, so the expected side can be audited without
        # relying on a world-frame convention.
        side_sign = torch.where(side == 0, torch.ones_like(side), -torch.ones_like(side)).to(foot_pos.dtype)
        local_corners = foot_pos.new_zeros((self.num_envs, 4, 3))
        local_corners[:, 0, 0] = -0.06
        local_corners[:, 1, 0] = -0.06
        local_corners[:, 2, 0] = 0.14
        local_corners[:, 3, 0] = 0.14
        local_corners[:, 0, 1] = side_sign * 0.04
        local_corners[:, 1, 1] = side_sign * 0.16
        local_corners[:, 2, 1] = side_sign * 0.04
        local_corners[:, 3, 1] = side_sign * 0.16
        corner_yaw = foot_yaw[:, None, :].expand(-1, 4, -1)
        corners = foot_pos[:, None, :] + quat_apply(
            corner_yaw.reshape(-1, 4), local_corners.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)
        corners[~active, :, 2] = -100.0
        self.s2_contact_region_visualizer.visualize(
            corners.reshape(-1, 3), corner_yaw.reshape(-1, 4)
        )

    def _validate_fixed_touch_spec(self) -> None:
        """Reject CG clips that conflict with a fixed foot and/or instep region.

        A fixed-touch task cannot mix labels that describe another foot or a
        different contact surface.  Failing during environment creation makes
        the data-contract error explicit; callers must relabel or omit the
        conflicting motion instead of silently changing the learned contact
        graph.
        """
        configured_foot = getattr(self.cfg, "dribble_cg_fixed_touch_foot", None)
        configured_surface = getattr(self.cfg, "dribble_cg_fixed_touch_surface", None)
        require_surface_labels = bool(getattr(self.cfg, "dribble_cg_require_surface_labels", False))
        require_flow_labels = bool(getattr(self.cfg, "dribble_cg_require_flow_labels", False))

        foot_to_id = {"left": 0, "right": 1}
        surface_to_id = {"inside_instep": 0, "outside_instep": 1}

        foot_name: str | None = None
        if configured_foot is not None:
            foot_name = str(configured_foot).lower().strip()
        if foot_name is not None and foot_name not in foot_to_id:
            raise ValueError(
                "dribble_cg_fixed_touch_foot must be 'left', 'right', or None; "
                f"got {configured_foot!r}."
            )

        surface_name: str | None = None
        if configured_surface is not None:
            surface_name = str(configured_surface).lower().strip()
        if surface_name is not None and surface_name not in surface_to_id:
            raise ValueError(
                "dribble_cg_fixed_touch_surface must be 'inside_instep', "
                "'outside_instep', or None; "
                f"got {configured_surface!r}."
            )

        if foot_name is None and surface_name is None and not require_surface_labels and not require_flow_labels:
            return

        expected_foot_id = foot_to_id[foot_name] if foot_name is not None else None
        expected_surface_id = surface_to_id[surface_name] if surface_name is not None else None
        violations: list[str] = []
        for motion_idx, motion_name in enumerate(self.motion.motion_name):
            contact = self.motion.dribble_cg_contact[motion_idx] > 0
            contact_foot = self.motion.dribble_cg_foot[motion_idx]
            contact_surface = self.motion.dribble_cg_surface[motion_idx]
            distance_foot = self.motion.dribble_cg_dist_foot[motion_idx]
            kick_leg_id = int(self.motion_kick_leg[motion_idx].item())
            if require_surface_labels:
                if not bool(self.motion.motion_has_dribble_cg[motion_idx]):
                    violations.append(f"{motion_name}: missing CG contact labels")
                invalid_contact_foot = contact & ~((contact_foot == 0) | (contact_foot == 1))
                invalid_contact_surface = contact & ~(
                    (contact_surface == 0) | (contact_surface == 1)
                )
                if bool(torch.any(invalid_contact_foot)):
                    violations.append(f"{motion_name}: contact frames without a valid foot label")
                if bool(torch.any(invalid_contact_surface)):
                    violations.append(
                        f"{motion_name}: contact frames without a valid instep-surface label"
                    )
            if require_flow_labels:
                if not bool(self.motion.motion_has_dribble_cg[motion_idx]):
                    violations.append(f"{motion_name}: missing CG contact labels")
                elif not bool(self.motion.motion_has_dribble_cg_flow[motion_idx]):
                    violations.append(f"{motion_name}: missing contact-to-contact flow labels")
            if expected_foot_id is not None:
                unknown_contact_foot = contact & (contact_foot < 0)
                wrong_contact_foot = contact & (contact_foot >= 0) & (contact_foot != expected_foot_id)
                wrong_distance_foot = (distance_foot >= 0) & (distance_foot != expected_foot_id)
                wrong_kick_leg = kick_leg_id >= 0 and kick_leg_id != expected_foot_id
                if bool(torch.any(unknown_contact_foot)):
                    violations.append(f"{motion_name}: contact frames without a foot label")
                if bool(torch.any(wrong_contact_foot)):
                    violations.append(f"{motion_name}: contact labels include the other foot")
                if bool(torch.any(wrong_distance_foot)):
                    violations.append(f"{motion_name}: distance labels include the other foot")
                if wrong_kick_leg:
                    violations.append(f"{motion_name}: kick_leg disagrees with fixed {foot_name} foot")

            if expected_surface_id is not None:
                unknown_contact_surface = contact & (contact_surface < 0)
                wrong_contact_surface = contact & (contact_surface >= 0) & (
                    contact_surface != expected_surface_id
                )
                if bool(torch.any(unknown_contact_surface)):
                    violations.append(f"{motion_name}: contact frames without an instep-surface label")
                if bool(torch.any(wrong_contact_surface)):
                    violations.append(f"{motion_name}: contact labels include the other instep surface")
        if violations:
            configured = " ".join(part for part in (foot_name, surface_name) if part is not None)
            task_description = (
                f"Fixed {configured} CG task" if configured else "Labelled CG task"
            )
            raise ValueError(
                f"{task_description} received incompatible motions: " + "; ".join(violations)
            )

    def _demo_ball_world(self, env_ids: torch.Tensor) -> torch.Tensor:
        """World-frame demo ball positions for env_ids (aligned to anchor tracking)."""
        mi = self.motion_idx[env_ids]
        ts = self.time_steps[env_ids]
        env_origins = self._env.scene.env_origins[env_ids]

        mb = self.motion.ball_pos_w[mi, ts] + env_origins
        ma = self.anchor_pos_w[env_ids]
        ra = self.robot_anchor_pos_w[env_ids]
        delta = ra.clone()
        delta[:, 2] = ma[:, 2]

        anchor_quat = self.anchor_quat_w[env_ids]
        robot_quat = self.robot_anchor_quat_w[env_ids]
        dq = mimic_anchor_yaw_delta_quat(
            anchor_quat,
            robot_quat,
            align_task_frame=bool(getattr(self.cfg, "mimic_align_task_frame", False)),
        )
        rel = mb - ma
        return delta + quat_apply(dq, rel)

    def _demo_ball_lin_vel_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Finite-difference demo ball linear velocity in world frame."""
        mi = self.motion_idx[env_ids]
        ts = self.time_steps[env_ids]
        ts_prev = torch.clamp(ts - 1, min=0)
        env_origins = self._env.scene.env_origins[env_ids]

        mb0 = self.motion.ball_pos_w[mi, ts_prev] + env_origins
        mb1 = self.motion.ball_pos_w[mi, ts] + env_origins
        fps = float(self.motion.fps.reshape(-1)[0])
        return (mb1 - mb0) * fps

    def _should_snap_demo_ball(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Per-env bool: write sim ball from demo this step."""
        mi = self.motion_idx[env_ids]
        has_demo = self.motion.motion_has_ball_demo[mi]
        mode = str(getattr(self.cfg, "dribble_cg_snap_mode", "full")).lower().strip()
        if mode == "never":
            return torch.zeros(env_ids.numel(), device=self.device, dtype=torch.bool)
        if mode == "non_contact_only":
            in_ref_contact = self.motion.dribble_cg_contact[mi, self.time_steps[env_ids]] > 0
            return has_demo & ~in_ref_contact
        return has_demo

    def _anchor_local_front_ball_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Fallback ball ahead of the frame-0 reference pelvis in pelvis-local axes."""
        mi = self.motion_idx[env_ids]
        first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index]
        first_anchor_yaw = yaw_quat(self.motion._body_quat_w[mi, 0, self.motion_anchor_body_index])

        distance = float(getattr(self.cfg, "dribble_cg_front_ball_distance", 0.45))
        lateral_offset = float(getattr(self.cfg, "dribble_cg_front_ball_lateral_offset", 0.0))
        height = float(getattr(self.cfg, "dribble_cg_front_ball_height", self._target_height))

        local_offset = first_anchor.new_zeros((env_ids.numel(), 3))
        local_offset[:, 0] = distance
        local_offset[:, 1] = lateral_offset
        ball_pos = first_anchor + quat_apply(first_anchor_yaw, local_offset)
        ball_pos[:, 2] = height
        return ball_pos

    def _legacy_front_ball_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Historical fallback: fixed env/task +X ahead of the start anchor."""
        mi = self.motion_idx[env_ids]
        first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index]
        distance = float(getattr(self.cfg, "dribble_cg_front_ball_distance", 0.45))
        lateral_offset = float(getattr(self.cfg, "dribble_cg_front_ball_lateral_offset", 0.0))
        height = float(getattr(self.cfg, "dribble_cg_front_ball_height", self._target_height))
        return spawn_ball_ahead_env_local(first_anchor, distance, lateral_offset, height)

    def _reference_first_contact_ball_positions(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return first-contact ball positions, validity, and source frame.

        ``ball_pos_w`` is stored in the same reference coordinate system as
        the pelvis.  Its selected point is therefore equivalent to mapping the
        frame-0 pelvis-local offset back through that pelvis yaw; retaining the
        direct value avoids introducing any simulation/world-forward axis.
        """
        mi = self.motion_idx[env_ids]
        selected_frames = self.s2_episode_first_contact_frame[env_ids]
        contact_frames = torch.where(
            selected_frames >= 0,
            selected_frames,
            self.motion.first_dribble_contact_frame[mi],
        )
        valid = (contact_frames >= 0) & self.motion.motion_has_ball_demo[mi]
        safe_frames = contact_frames.clamp(min=0)
        positions = self.motion.ball_pos_w[mi, safe_frames].clone()
        jitter_min = max(0.0, float(getattr(self.cfg, "dribble_cg_ball_spawn_jitter_min", 0.0)))
        jitter_max = max(jitter_min, float(getattr(self.cfg, "dribble_cg_ball_spawn_jitter_max", 0.0)))
        if jitter_max > 0.0:
            radius = jitter_min + torch.rand(env_ids.numel(), device=self.device) * (jitter_max - jitter_min)
            angle = 2.0 * torch.pi * torch.rand(env_ids.numel(), device=self.device)
            positions[:, 0] += radius * torch.cos(angle)
            positions[:, 1] += radius * torch.sin(angle)
        return positions, valid, contact_frames

    def _record_ball_spawn(
        self,
        env_ids: torch.Tensor,
        ball_positions: torch.Tensor,
        contact_frames: torch.Tensor,
        used_reference_contact: torch.Tensor,
    ) -> None:
        """Keep spawn provenance and frame-0 pelvis-local geometry for diagnostics."""
        mi = self.motion_idx[env_ids]
        first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index]
        first_anchor_yaw_inv = quat_inv(
            yaw_quat(self.motion._body_quat_w[mi, 0, self.motion_anchor_body_index])
        )
        self.ball_spawn_reference_contact_frame[env_ids] = torch.where(
            used_reference_contact,
            contact_frames,
            torch.full_like(contact_frames, -1),
        )
        self.ball_spawn_source[env_ids] = used_reference_contact.to(torch.int8)
        self.ball_spawn_reference_local[env_ids] = quat_apply(
            first_anchor_yaw_inv, ball_positions - first_anchor
        )

    def ball_spawn_reference_info(self) -> dict[str, torch.Tensor]:
        """Reset-ball provenance for playback diagnostics.

        ``reference_local`` is the ball position relative to the reference
        frame-0 pelvis, expressed in that pelvis yaw frame.  A source value of
        one means first labelled contact; zero means the explicit local-front
        fallback for a clip missing the needed labels/data.
        """
        return {
            "source": self.ball_spawn_source,
            "reference_contact_frame": self.ball_spawn_reference_contact_frame,
            "reference_local": self.ball_spawn_reference_local,
        }

    def _compute_soccer_ball_positions(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        spawn_mode = str(getattr(self.cfg, "dribble_cg_ball_spawn_mode", "legacy")).lower().strip()
        if spawn_mode in {"reference_first_contact", "first_contact", "contact"}:
            contact_positions, valid, contact_frames = self._reference_first_contact_ball_positions(ids)
            fallback_positions = self._anchor_local_front_ball_positions(ids)
            ball_positions = torch.where(valid.unsqueeze(-1), contact_positions, fallback_positions)
            self.soccer_ball_pos[ids] = ball_positions
            self._record_ball_spawn(ids, ball_positions, contact_frames, valid)
            return

        has_demo = self._should_snap_demo_ball(ids)
        demo_ids = ids[has_demo]
        fallback_ids = ids[~has_demo]

        if fallback_ids.numel() > 0:
            fallback_mode = str(getattr(self.cfg, "dribble_cg_fallback_ball_mode", "arc_endpoint")).lower().strip()
            if fallback_mode == "front":
                self.soccer_ball_pos[fallback_ids] = self._legacy_front_ball_positions(fallback_ids)
            else:
                super()._compute_soccer_ball_positions(fallback_ids)
        if demo_ids.numel() > 0:
            # Historical +X placement is retained for legacy demo-snap tasks.
            self.soccer_ball_pos[demo_ids] = self._legacy_front_ball_positions(demo_ids)

    def _sync_demo_ball_after_step(self):
        """Kinematic sync of sim ball to demo trajectory (subset of envs)."""
        if getattr(self.cfg, "dribble_cg_use_task_frame", True):
            return
        if self.soccer_ball is None:
            return
        if hasattr(self.soccer_ball, "is_initialized") and not self.soccer_ball.is_initialized:
            return
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return

        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        snap = self._should_snap_demo_ball(all_ids)
        if not torch.any(snap):
            return

        ids = all_ids[snap]
        ball_pos_w = self._demo_ball_world(ids)
        ball_quat = ball_pos_w.new_zeros((ids.numel(), 4))
        ball_quat[:, 0] = 1.0
        ball_lin_vel = self._demo_ball_lin_vel_w(ids)
        ball_ang_vel = ball_pos_w.new_zeros((ids.numel(), 3))
        ball_state = torch.cat([ball_pos_w, ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
        self.soccer_ball.write_root_state_to_sim(ball_state, env_ids=ids)

        self.soccer_ball_pos[ids] = ball_pos_w - env_origins[ids]
        self.target_point_pos[ids] = self.soccer_ball_pos[ids]
        if self.target_point_marker is not None:
            self.target_point_marker.visualize(ball_pos_w)

    def _update_command(self):
        super()._update_command()
        self._sync_demo_ball_after_step()


@configclass
class DribbleCGMotionCommandCfg(MotionCommandCfg):
    """Config for :class:`DribbleCGMotionCommand`.

    Extra fields (read via ``getattr`` on older cfg objects default in command):

    dribble_cg_snap_mode:
        ``full`` — always snap to demo when ``ball_pos_w`` exists.
        ``non_contact_only`` — snap only outside annotated contact frames.
    """

    class_type: type = DribbleCGMotionCommand

    dribble_cg_snap_mode: str = "full"
    # This does not select a reset spawn frame; unified local stages disable
    # demo-ball snapping with ``dribble_cg_snap_mode = \"never\"``.
    dribble_cg_use_task_frame: bool = True
    # ``reference_first_contact`` places a physical reset ball at the first
    # labelled ``dribble_cg_contact`` point in ``ball_pos_w``.  ``legacy``
    # preserves the historical demo/fallback branching for older task IDs.
    dribble_cg_ball_spawn_mode: str = "legacy"
    dribble_cg_fallback_ball_mode: str = "arc_endpoint"
    dribble_cg_front_ball_distance: float = 0.45
    dribble_cg_front_ball_lateral_offset: float = 0.0
    dribble_cg_front_ball_height: float = 0.11
    # S2 event curriculum.  Each pair is ``(number_of_adjacent_contacts,
    # probability)``; zero means the complete clip.  An empty tuple preserves
    # the legacy full-clip sampler.
    dribble_cg_curriculum_mix: tuple[tuple[int, float], ...] = ()
    dribble_cg_pre_contact_seconds_range: tuple[float, float] = (0.3, 0.6)
    dribble_cg_contact_window_seconds: float = 0.10
    dribble_cg_missed_contact_grace_steps: int = 3
    dribble_cg_ball_spawn_jitter_min: float = 0.0
    dribble_cg_ball_spawn_jitter_max: float = 0.0
    dribble_cg_s2_debug_regions_only: bool = False
    # ``None`` preserves the historical mixed-foot CG behavior. Unified
    # training sets this to one side and validates every motion at startup.
    dribble_cg_fixed_touch_foot: str | None = None
    # ``None`` preserves legacy foot-only labels.  A fixed inside/outside
    # instep task requires ``dribble_cg_surface`` on every annotated contact
    # frame and validates it at environment creation.
    dribble_cg_fixed_touch_surface: str | None = None
    # S2 uses per-frame inside/outside supervision. Fail early instead of
    # silently treating a legacy foot-only motion as a surface-labelled one.
    dribble_cg_require_surface_labels: bool = False
    # S2 also requires causal contact-to-contact flow labels. The final contact
    # is intentionally unlabeled because it has no known outgoing destination.
    dribble_cg_require_flow_labels: bool = False
