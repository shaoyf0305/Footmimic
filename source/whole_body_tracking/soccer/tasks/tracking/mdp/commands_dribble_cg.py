"""Dribbling motion command with XGen-style demo ball stitching.

Uses per-frame ``ball_pos_w`` from motion for labels; ball spawn / physics use the
task frame (+X forward). Optional demo kinematic snap is disabled when
``dribble_cg_use_task_frame`` is True. Legacy yaw-only
anchor transform as body tracking targets so the ball follows the stitched
interaction trajectory. Optional ``dribble_cg_snap_mode``:

- ``full`` (default): every step writes the demo ball pose into simulation.
- ``non_contact_only``: only overwrite the ball when the CG label says
  non-contact, leaving physics during annotated contact segments.

Contact / foot / surface masks come from ``dribble_cg_contact``,
``dribble_cg_foot``, and ``dribble_cg_surface`` in ``.npz``.  The legacy
``kick_frame`` / ``kick_end_frame`` / ``kick_leg`` metadata remains a fallback
for contact timing and foot side, but cannot describe an instep surface (see
:class:`MultiMotionLoader`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from soccer.tasks.tracking.mdp.task_frame import mimic_anchor_yaw_delta_quat, spawn_ball_ahead_env_local

from .commands_multi_motion_soccer import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DribbleCGMotionCommand(MotionCommand):
    """Soccer motion command + demo ball sync for dribbling CG."""

    def __init__(self, cfg: DribbleCGMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._validate_fixed_touch_spec()

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

        if foot_name is None and surface_name is None:
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
            raise ValueError(
                f"Fixed {configured} CG task received incompatible motions: " + "; ".join(violations)
            )

    def _use_demo_ball(self) -> bool:
        return bool(getattr(self.cfg, "dribble_cg_use_demo_ball", True))

    def get_dribble_demo_ball_goal_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if not self._use_demo_ball():
            mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            mask = self.motion.motion_has_ball_demo[self.motion_idx]
        goal = self._demo_ball_world(env_ids)
        return goal, mask

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
        fps = float(self.motion.fps)
        if isinstance(fps, torch.Tensor):
            fps = float(fps.item())
        return (mb1 - mb0) * fps

    def _should_snap_demo_ball(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Per-env bool: write sim ball from demo this step."""
        if not self._use_demo_ball():
            return torch.zeros(env_ids.numel(), device=self.device, dtype=torch.bool)
        mi = self.motion_idx[env_ids]
        has_demo = self.motion.motion_has_ball_demo[mi]
        mode = str(getattr(self.cfg, "dribble_cg_snap_mode", "full")).lower().strip()
        if mode == "non_contact_only":
            in_ref_contact = self.motion.dribble_cg_contact[mi, self.time_steps[env_ids]] > 0
            return has_demo & ~in_ref_contact
        return has_demo

    def _front_ball_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Env-local ball on task +X ahead of the motion's starting anchor."""
        mi = self.motion_idx[env_ids]
        first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index]

        distance = float(getattr(self.cfg, "dribble_cg_front_ball_distance", 0.45))
        lateral_offset = float(getattr(self.cfg, "dribble_cg_front_ball_lateral_offset", 0.0))
        height = float(getattr(self.cfg, "dribble_cg_front_ball_height", self._target_height))

        return spawn_ball_ahead_env_local(first_anchor, distance, lateral_offset, height)

    def _compute_soccer_ball_positions(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        has_demo = self._should_snap_demo_ball(ids)
        demo_ids = ids[has_demo]
        fallback_ids = ids[~has_demo]

        if fallback_ids.numel() > 0:
            fallback_mode = str(getattr(self.cfg, "dribble_cg_fallback_ball_mode", "arc_endpoint")).lower().strip()
            if fallback_mode == "front":
                self.soccer_ball_pos[fallback_ids] = self._front_ball_positions(fallback_ids)
            else:
                super()._compute_soccer_ball_positions(fallback_ids)
        if demo_ids.numel() > 0:
            # Spawn on task +X; per-step demo snap (_sync_demo_ball_after_step) may still move the ball.
            self.soccer_ball_pos[demo_ids] = self._front_ball_positions(demo_ids)

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
    dribble_cg_use_demo_ball: bool = True
    dribble_cg_use_task_frame: bool = True
    dribble_cg_fallback_ball_mode: str = "arc_endpoint"
    dribble_cg_front_ball_distance: float = 0.45
    dribble_cg_front_ball_lateral_offset: float = 0.0
    dribble_cg_front_ball_height: float = 0.11
    # ``None`` preserves the historical mixed-foot CG behavior. Unified
    # training sets this to one side and validates every motion at startup.
    dribble_cg_fixed_touch_foot: str | None = None
    # ``None`` preserves legacy foot-only labels.  A fixed inside/outside
    # instep task requires ``dribble_cg_surface`` on every annotated contact
    # frame and validates it at environment creation.
    dribble_cg_fixed_touch_surface: str | None = None
