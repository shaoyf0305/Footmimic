"""Dribbling environment configurations for the G1 robot.

Inherits proximity-level tracking and adds dribbling-specific rewards:
  - velocity / proximity gates (anti static exploit); velocity match requires contact
  - unified task frame (+X fwd / +Y lat): ball spawn, velocity, proximity, obs, progress
  - optional dense foot–ball approach when not in contact (disabled for CG variant)
  - pelvis orientation vs motion reference (anti lean-back / arched torso)
  - ball horizontal speed excess penalty; anti-trap / anti-sustained-contact penalties
  - both ankles legal for gentle touches; anti-trap / sustained-contact block 夹球
  - kick–chase–kick: chase reward, rapid-retouch penalty, coast penalty only when ball is close
  - gait foot tracking between touches; reduced close-proximity / foot-hover shaping
  - pelvis anchor + task-frame mimic (strip demo yaw → +X for torso/arms/legs); relaxed leg imitation weights
  - anti-crab: task-frame forward velocity match, heading-gated touches, stronger lateral penalty
  - ``ball_lost`` and tighter ``dribbling_no_contact`` termination
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp import observations_anchor as obs_anchor
from soccer.tasks.tracking.mdp.commands_dribble_cg import DribbleCGMotionCommand
from .soccer_flat_env_cfg import (
    G1FlatMotionStrictPretrainEnvCfg,
    G1FlatProximityEnvCfg,
)


# Control-only coordinated upper-body action subsystem.
_CONTROL_UPPER_BODY_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class G1FlatDribblingEnvCfg(G1FlatProximityEnvCfg):
    """Flat-ground dribbling environment."""

    # Optional override: "both" (default) allows either ankle; single-foot modes remain useful for diagnostics.
    dribble_contact_mode: str = "both"
    # ``any`` preserves the old whole-foot rule. ``instep`` accepts either
    # labelled side. ``inside_instep`` and ``outside_instep`` classify only
    # the signed lateral offset in the contacted foot's yaw frame; no
    # fore/aft or height box is part of the surface definition.
    dribble_contact_surface: str = "any"

    def __post_init__(self):
        super().__post_init__()

        # Ball-on-ground: USD had mu~0.1 and restitution 1.0 → long coast after one kick.
        # Pair higher friction with moderate bounce; add linear damping on the ball body.
        self.scene.terrain.physics_material = self.scene.terrain.physics_material.replace(
            restitution=0.22,
            static_friction=1.0,
            dynamic_friction=0.95,
        )
        self.scene.soccer_ball.spawn = self.scene.soccer_ball.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Slightly lower damping so the ball rolls between discrete touches.
                linear_damping=0.18,
                angular_damping=0.18,
            ),
        )

        # Pelvis anchor + task-frame mimic: strip demo yaw so torso/arms/legs refs face +X
        # together (legacy mode only yaw-aligns to robot pelvis, leaving arms on demo heading).
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.mimic_align_task_frame = True
        _upper_body_track_names = [
            "pelvis",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        # Slightly relax imitation so the policy can deviate toward the ball while
        # keeping torso/gait reference (touch-related rewards provide the main ball signal).
        if hasattr(self.rewards, "motion_body_pos"):
            self.rewards.motion_body_pos.weight = 0.72
        if hasattr(self.rewards, "foot_distance"):
            self.rewards.foot_distance.weight = 0.35
        if hasattr(self.rewards, "motion_foot_pos"):
            # Baseline foot tracking; ``dribbling_gait_foot_tracking`` adds stronger signal
            # between ball touches so the policy does not freeze in a kick-ready stance.
            self.rewards.motion_foot_pos.weight = 0.55
        if hasattr(self.rewards, "motion_body_ori"):
            self.rewards.motion_body_ori.params["body_names"] = _upper_body_track_names
        # Yaw-aligned velocity match for upper body (world-frame demo vel breaks under +X task).
        if hasattr(self.rewards, "motion_body_lin_vel"):
            self.rewards.motion_body_lin_vel = RewTerm(
                func=mdp.motion_relative_body_linear_velocity_error_exp,
                weight=0.3,
                params={
                    "command_name": "motion",
                    "std": 1.0,
                    "body_names": _upper_body_track_names,
                },
            )
        if hasattr(self.rewards, "motion_body_ang_vel"):
            self.rewards.motion_body_ang_vel = RewTerm(
                func=mdp.motion_relative_body_angular_velocity_error_exp,
                weight=0.3,
                params={
                    "command_name": "motion",
                    "std": 3.14,
                    "body_names": _upper_body_track_names,
                },
            )
        # Kick-style frozen proximity is not used for dribbling.
        if hasattr(self.rewards, "target_point_proximity"):
            self.rewards.target_point_proximity = None

        # Stronger upright / anti-arch than generic proximity alone
        if hasattr(self.rewards, "pelvis_orientation"):
            self.rewards.pelvis_orientation.weight = -2.5

        # Drop world-frame yaw tracking from the motion: zig-zag motion ("slalom around
        # cones") would otherwise force the policy to copy the S-shape heading pattern.
        # The face-ball reward below replaces it with a task-aligned heading signal.
        if hasattr(self.rewards, "motion_global_anchor_ori"):
            self.rewards.motion_global_anchor_ori = None

        self.rewards.dribbling_face_ball = RewTerm(
            func=mdp.dribbling_face_ball,
            weight=2.5,
            params={
                "command_name": "motion",
                "min_distance": 0.05,
            },
        )

        # Pelvis yaw vs task +X (``dribbling_face_ball`` already includes heading × ball-ahead).
        self.rewards.task_heading_alignment = RewTerm(
            func=mdp.task_heading_alignment_reward,
            weight=1.8,
            params={"command_name": "motion"},
        )

        self.rewards.forward_velocity = RewTerm(
            func=mdp.forward_velocity_reward,
            weight=3.0,
            params={
                "command_name": "motion",
                "target_speed": 0.55,
                "std": 0.35,
                "velocity_frame": "world",
                "min_forward_dominance": 0.55,
            },
        )
        self.rewards.lateral_velocity_penalty = RewTerm(
            func=mdp.lateral_velocity_penalty,
            weight=-1.6,
            params={
                "command_name": "motion",
                "velocity_frame": "world",
                "lateral_deadzone": 0.05,
                "lateral_scale": 0.24,
            },
        )

        mode = str(self.dribble_contact_mode).lower().strip()
        if mode not in {"right", "left", "both"}:
            raise ValueError(
                f"Unsupported dribble_contact_mode={self.dribble_contact_mode}. "
                "Expected one of: right, left, both."
            )

        surface = str(self.dribble_contact_surface).lower().strip()
        if surface not in {"any", "instep", "inside_instep", "outside_instep"}:
            raise ValueError(
                f"Unsupported dribble_contact_surface={self.dribble_contact_surface}. "
                "Expected one of: any, instep, inside_instep, outside_instep."
            )

        if mode == "right":
            legal_ankles = ["right_ankle_roll_link"]
            other_ankles = ["left_ankle_roll_link"]
        elif mode == "left":
            legal_ankles = ["left_ankle_roll_link"]
            other_ankles = ["right_ankle_roll_link"]
        else:
            legal_ankles = ["right_ankle_roll_link", "left_ankle_roll_link"]
            other_ankles = []

        _foot_cfg = SceneEntityCfg("robot", body_names=legal_ankles)
        _both_feet = ["left_ankle_roll_link", "right_ankle_roll_link"]

        # Legal ankles must appear first for num_ankle_links gating.
        _contact_body_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                *legal_ankles,
                *other_ankles,
                "right_knee_link",
                "left_knee_link",
                "right_wrist_yaw_link",
                "left_wrist_yaw_link",
            ],
        )
        _num_ankle_links = len(legal_ankles)

        self.rewards.dribbling_velocity_tracking = RewTerm(
            func=mdp.dribbling_velocity_tracking,
            weight=5.0,
            params={
                "command_name": "motion",
                "std": 1.0,
                "pelvis_speed_min": 0.35,
                "ball_speed_min": 0.12,
                # Velocity match only within a short window after a real touch (not coasting).
                "require_contact": True,
                "recent_contact_window": 8,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "min_forward_dominance": 0.55,
            },
        )

        self.rewards.dribbling_dynamic_proximity = RewTerm(
            func=mdp.dribbling_dynamic_proximity,
            weight=3.0,
            params={
                "command_name": "motion",
                "near_dist": 0.28,
                "far_dist": 0.72,
                "penalty_std": 0.15,
                "pelvis_speed_min": 0.35,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                # In the "ball in front" corridor, strongly cut proximity without a touch.
                "no_contact_zone_damping": 0.12,
                "zone_lateral_abs_max": 0.14,
            },
        )

        self.rewards.dribbling_stall_no_touch_penalty = RewTerm(
            func=mdp.dribbling_stall_no_touch_penalty,
            weight=-5.5,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "max_xy_dist": 0.52,
                "pelvis_speed_max": 0.16,
            },
        )

        # Non-CG dribbling only: CG variant sets weight=0 (foot–ball distance labels cover contact).
        self.rewards.dribbling_approach_foot_ball = RewTerm(
            func=mdp.dribbling_approach_foot_ball_distance,
            weight=1.2,
            params={
                "command_name": "motion",
                "foot_cfg": _foot_cfg,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "std": 0.25,
                "pelvis_speed_min": 0.12,
            },
        )

        self.rewards.dribbling_pelvis_quat_tracking = RewTerm(
            func=mdp.dribbling_pelvis_quat_tracking_exp,
            weight=2.0,
            params={
                "command_name": "motion",
                "std": 0.45,
            },
        )

        self.rewards.dribbling_ball_speed_excess = RewTerm(
            func=mdp.dribbling_ball_xy_speed_excess_penalty,
            weight=-2.5,
            params={
                # Was 2.8 — no penalty at ~2.5 m/s where the robot loses the ball.
                "speed_cap": 1.35,
                "linear_scale": 1.2,
            },
        )

        self.rewards.dribbling_ball_coast_penalty = RewTerm(
            func=mdp.dribbling_ball_coast_without_contact_penalty,
            weight=-2.2,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "speed_threshold": 0.55,
                "speed_scale": 0.40,
                "max_close_xy_dist": 0.50,
            },
        )

        self.rewards.dribbling_ball_trapped_penalty = RewTerm(
            func=mdp.dribbling_ball_trapped_penalty,
            weight=-8.0,
            params={
                "command_name": "motion",
                "min_forward_x": 0.32,
                "max_ball_height": 0.20,
            },
        )

        self.rewards.dribbling_sustained_contact_penalty = RewTerm(
            func=mdp.dribbling_sustained_contact_penalty,
            weight=-6.0,
            params={
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "max_contact_steps": 3,
            },
        )

        self.rewards.dribbling_ball_bounce_penalty = RewTerm(
            func=mdp.dribbling_ball_bounce_penalty,
            weight=-3.0,
            params={
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "vz_threshold": 0.32,
            },
        )

        self.rewards.dribbling_ball_forward_progress = RewTerm(
            func=mdp.dribbling_ball_forward_progress_reward,
            weight=7.5,
            params={
                "command_name": "motion",
                "min_forward_speed": 0.42,
                "speed_scale": 0.22,
                "pelvis_speed_min": 0.25,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 0.5,
                "require_recent_contact": True,
                "recent_contact_window": 8,
            },
        )

        self.rewards.dribbling_orbiting_penalty = RewTerm(
            func=mdp.dribbling_orbiting_penalty,
            weight=-6.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "orbit_radius_max": 0.9,
                "tangential_deadzone": 0.08,
                "tangential_scale": 0.35,
            },
        )

        self.rewards.dribbling_gait_foot_tracking = RewTerm(
            func=mdp.dribbling_gait_foot_tracking_exp,
            weight=1.4,
            params={
                "command_name": "motion",
                "std": 0.28,
                "foot_body_names": _both_feet,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
            },
        )

        self.rewards.dribbling_chase_ball = RewTerm(
            func=mdp.dribbling_chase_ball_reward,
            weight=2.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "min_ball_ahead": 0.28,
                "max_chase_xy_dist": 1.05,
                "pelvis_forward_speed_min": 0.25,
                "forward_speed_scale": 0.50,
            },
        )

        self.rewards.dribbling_rapid_retouch_penalty = RewTerm(
            func=mdp.dribbling_rapid_retouch_penalty,
            weight=-6.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 22.0,
                "min_steps_between_touches": 32,
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
            },
        )

        self.rewards.dribbling_legal_foot_touch = RewTerm(
            func=mdp.dribbling_legal_foot_touch,
            weight=5.5,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                # Was 22 — hard kicks still counted as "gentle" legal touches.
                "force_threshold": 14.0,
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
                "contact_surface": surface,
                "min_pelvis_heading": 0.55,
            },
        )

        self.rewards.dribbling_micro_contact_filter = RewTerm(
            func=mdp.dribbling_micro_contact_filter,
            weight=-4.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 22.0,
                "max_penalty": 2.0,
                "ema_alpha": 0.35,
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
            },
        )

        self.rewards.dribbling_undesired_contact_penalty = RewTerm(
            func=mdp.dribbling_undesired_contact_penalty,
            weight=-12.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
                "contact_surface": surface,
                # Preserve the strong -12 signal for knees/wrists, but an
                # ankle touch on the wrong instep side is only -3.  Contact
                # timing is separately penalized by CG premature-contact.
                "wrong_surface_penalty": 0.25,
            },
        )

        self.terminations.ball_lost = DoneTerm(
            func=mdp.ball_lost_dribbling,
            params={
                "command_name": "motion",
                "max_distance": 1.0,
                "max_vel_divergence": 2.0,
                "grace_steps": 50,
            },
        )

        self.terminations.dribbling_no_contact = DoneTerm(
            func=mdp.dribbling_no_ball_contact_timeout,
            params={
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "grace_steps": 50,
                "max_steps_without_contact": 50,
            },
        )


@configclass
class G1FlatCGDribblingEnvCfg(G1FlatDribblingEnvCfg):
    """Dribbling with annotated contacts and causal contact-to-contact ball flow.

    Uses :class:`~soccer.tasks.tracking.mdp.commands_dribble_cg.DribbleCGMotionCommand`
    to optionally drive the sim ball along the demo trajectory (anchor-relative),
    and rewards that align sensor contact / foot side with ``dribble_cg_*`` labels.

    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.class_type = DribbleCGMotionCommand
        # This CG variant uses contact labels to teach the touch timing/side,
        # while keeping the ball task simple: spawn the ball in front and let
        # physics move it.
        setattr(self.commands.motion, "dribble_cg_use_task_frame", True)
        setattr(self.commands.motion, "dribble_cg_snap_mode", "never")
        setattr(self.commands.motion, "dribble_cg_fallback_ball_mode", "front")
        setattr(self.commands.motion, "dribble_cg_front_ball_distance", 0.45)
        setattr(self.commands.motion, "dribble_cg_front_ball_lateral_offset", 0.0)
        setattr(self.commands.motion, "dribble_cg_front_ball_height", 0.11)

        self.observations.policy.anchor_ball_polar = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )
        self.observations.critic.anchor_ball_polar = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )

        mode = str(self.dribble_contact_mode).lower().strip()
        surface = str(self.dribble_contact_surface).lower().strip()
        if mode == "right":
            legal_ankles = ["right_ankle_roll_link"]
            other_ankles = ["left_ankle_roll_link"]
        elif mode == "left":
            legal_ankles = ["left_ankle_roll_link"]
            other_ankles = ["right_ankle_roll_link"]
        else:
            legal_ankles = ["right_ankle_roll_link", "left_ankle_roll_link"]
            other_ankles = []
        cg_body_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                *legal_ankles,
                *other_ankles,
                "right_knee_link",
                "left_knee_link",
                "right_wrist_yaw_link",
                "left_wrist_yaw_link",
            ],
        )

        # Event-based CG flow: a correct touch latches its outgoing direction
        # once, then earns a short velocity-band reward and capped progress.
        # Run scripts/dribble/synthesize_dribble_ball_traj.py on labeled motions first.
        cg_flow_params = {
            "command_name": "motion",
            "ball_sensor_name": "soccer_ball_contact",
            "all_body_cfg": cg_body_cfg,
            "num_ankle_links": 2,
            "contact_surface": surface,
            "medial_y_min": 0.018,
            "contact_force_threshold": 1.0,
            "max_touch_force": 20.0,
            "release_window_steps": 8,
            "speed_lower_ratio": 0.7,
            "speed_upper_ratio": 1.6,
            "lateral_speed_std": 0.35,
            "overspeed_std": 0.5,
            "lateral_corridor_std": 0.18,
            "max_progress_rate": 6.0,
        }
        self.rewards.dribbling_cg_flow_release = RewTerm(
            func=mdp.dribbling_cg_flow_release_reward,
            weight=2.0,
            params=dict(cg_flow_params),
        )
        self.rewards.dribbling_cg_flow_progress = RewTerm(
            func=mdp.dribbling_cg_flow_progress_reward,
            weight=0.5,
            params=dict(cg_flow_params),
        )
        self.rewards.dribbling_cg_contact_consistency = RewTerm(
            func=mdp.dribbling_cg_contact_consistency,
            weight=5.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
            },
        )
        self.rewards.dribbling_cg_premature_contact = RewTerm(
            func=mdp.dribbling_cg_premature_contact_penalty,
            weight=-6.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
            },
        )
        self.rewards.dribbling_cg_foot_consistency = RewTerm(
            func=mdp.dribbling_cg_foot_consistency,
            weight=2.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "all_body_cfg": cg_body_cfg,
                "contact_surface": surface,
            },
        )

        # CG: no dense "hover foot on ball" between touches — proximity + contact labels suffice.
        self.rewards.dribbling_approach_foot_ball = None

        # Velocity / forward progress use recent sim contact, not only labeled CG frames.
        self.rewards.dribbling_velocity_tracking.params["cg_gated_contact"] = False
        self.rewards.dribbling_ball_forward_progress.params["cg_gated_contact"] = False
        self.rewards.dribbling_legal_foot_touch.params["cg_gated"] = True
        self.rewards.dribbling_rapid_retouch_penalty.params["cg_gated"] = True
        self.rewards.dribbling_rapid_retouch_penalty.params["min_steps_between_touches"] = 26


def _apply_unified_control_action_contract(cfg) -> None:
    """Install the frozen 29-D action contract used by the next training run.

    The policy keeps one action for each robot joint, so Stage-1 and Stage-2
    share the action head.  Only the selected upper-body targets are projected
    onto the PCA manifold.  A zero residual limit makes this a strict
    projection; requested, projected, limited, and executed targets remain
    available on the action term for diagnostics.
    """
    old_action_cfg = cfg.actions.joint_pos
    cfg.actions.joint_pos = mdp.UpperBodyManifoldJointPositionActionCfg(
        asset_name=old_action_cfg.asset_name,
        joint_names=old_action_cfg.joint_names,
        use_default_offset=old_action_cfg.use_default_offset,
        upper_body_joint_names=_CONTROL_UPPER_BODY_JOINT_NAMES,
        command_name="motion",
        manifold_rank=6,
        latent_std_limit=3.0,
        min_latent_limit=0.03,
        orthogonal_residual_limit=0.0,
        cutoff_frequency_hz=1.8,
        reference_target_margin=0.25,
        direct_upper_body_latent_action=False,
    )
    cfg.actions.joint_pos.scale = old_action_cfg.scale
    cfg.actions.joint_pos.offset = old_action_cfg.offset
    cfg.actions.joint_pos.preserve_order = getattr(old_action_cfg, "preserve_order", False)
    if hasattr(old_action_cfg, "clip"):
        cfg.actions.joint_pos.clip = old_action_cfg.clip

    # Recurrent action feedback and rate regularization must use the command
    # after PCA projection, joint limits, and filtering rather than the raw
    # policy value that may never reach the actuator.
    cfg.observations.policy.actions.func = mdp.effective_joint_action
    cfg.observations.policy.actions.params = {"action_name": "joint_pos"}
    cfg.observations.critic.actions.func = mdp.effective_joint_action
    cfg.observations.critic.actions.params = {"action_name": "joint_pos"}
    cfg.rewards.action_rate_l2.func = mdp.effective_action_rate_l2_clip
    cfg.rewards.action_rate_l2.params = {"action_name": "joint_pos"}
    cfg.rewards.upper_body_reference_overflow = RewTerm(
        func=mdp.upper_body_reference_overflow_penalty,
        weight=-0.05,
        params={"action_name": "joint_pos"},
    )
    cfg.rewards.upper_body_manifold_nullspace = RewTerm(
        func=mdp.upper_body_manifold_nullspace_penalty,
        weight=-0.02,
        params={"action_name": "joint_pos", "scale": 0.10},
    )


def _remove_unified_kick_only_terms(cfg) -> None:
    """Remove ball Cartesian and kick-destination signals from unified dribbling.

    ``target_point_pos`` is the simulated ball position in pelvis-frame
    Cartesian coordinates, which duplicates ``anchor_ball_polar``.  The
    destination signal and the inherited frozen-proximity reward belong to the
    kick task, not continuous dribbling.  Keep the command's internal target
    buffers untouched: generic ball reset/visualization code still owns them,
    but they are no longer policy inputs or learning objectives here.
    """
    for observations in (cfg.observations.policy, cfg.observations.critic):
        observations.target_point_pos = None
        observations.target_destination_pos_local = None

    # Keep the term absent so a later parent-config change cannot reintroduce
    # a kick-style objective into the frozen unified contract.
    if hasattr(cfg.rewards, "target_point_proximity"):
        cfg.rewards.target_point_proximity = None

    # These markers communicate kick-only targets and become misleading when
    # the associated observations/objective are absent.
    cfg.commands.motion.target_point_marker_cfg = None
    cfg.commands.motion.target_destination_marker_cfg = None


def _apply_cg_pretrain_obs(cfg) -> None:
    """``anchor_ball_polar`` on policy/critic — required for CG Stage-2 resume."""
    cfg.observations.policy.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_polar,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_polar,
        params={"command_name": "motion"},
    )


def _disable_reference_ball_contact_terms(cfg) -> None:
    """Remove reference-timed ball and contact-position objectives for S3."""
    for reward_name in (
        "dribbling_cg_flow_release",
        "dribbling_cg_flow_progress",
        "dribbling_cg_contact_consistency",
        "dribbling_cg_premature_contact",
        "dribbling_cg_foot_consistency",
    ):
        if hasattr(cfg.rewards, reward_name):
            setattr(cfg.rewards, reward_name, None)

    # Keep reference gait only between touches. At contact, the task owns the
    # foot, surface, time, and ball position.
    if hasattr(cfg.rewards, "motion_foot_pos"):
        cfg.rewards.motion_foot_pos = None
    if hasattr(cfg.rewards, "dribbling_legal_foot_touch"):
        cfg.rewards.dribbling_legal_foot_touch.params["cg_gated"] = False
    if hasattr(cfg.rewards, "dribbling_rapid_retouch_penalty"):
        cfg.rewards.dribbling_rapid_retouch_penalty.params["cg_gated"] = False


def _apply_unified_local_twist_observations(cfg) -> None:
    """Install the local-twist part of the 163-D unified input contract.

    Layout is deliberately preserved as ``154-D proprioception + 3-D ball +
    3-D compatibility state + 3-D twist``.  The final three policy values are
    therefore always ``[vx_local, vy_local, wz]``.  The state field remains a
    fixed DRIBBLE one-hot for S1/S2/S3; it avoids a checkpoint shape change but
    does not encode a second, competing task command.
    """
    cfg.observations.policy.motion_ref_ang_vel = ObsTerm(
        func=mdp.motion_locomotion_ang_vel_command_local,
        params={"command_name": "motion"},
    )
    # Critic-only, simulator-derived pelvis-local linear velocity.  Do not add
    # this observation to policy: it is privileged state unavailable on hardware.
    cfg.observations.critic.base_lin_vel = ObsTerm(
        func=mdp.robot_anchor_lin_vel_local,
        params={"command_name": "motion"},
    )
    cfg.observations.policy.motion_locomotion_polar_cmd = None
    cfg.observations.critic.motion_locomotion_polar_cmd = None
    cfg.observations.policy.motion_locomotion_task_state = ObsTerm(
        func=mdp.motion_locomotion_task_state,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_task_state = ObsTerm(
        func=mdp.motion_locomotion_task_state,
        params={"command_name": "motion"},
    )
    cfg.observations.policy.motion_locomotion_twist_cmd = ObsTerm(
        func=mdp.motion_locomotion_twist_command_local,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_twist_cmd = ObsTerm(
        func=mdp.motion_locomotion_twist_command_local,
        params={"command_name": "motion"},
    )


def _apply_unified_local_163_contract(cfg) -> None:
    """Shared 163-D/29-D deployment-oriented interface for local S1/S2/S3."""
    _apply_cg_pretrain_obs(cfg)
    cfg.observations.policy.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_pelvis_local_polar,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_pelvis_local_polar,
        params={"command_name": "motion"},
    )
    _remove_unified_kick_only_terms(cfg)
    _apply_unified_control_action_contract(cfg)
    _apply_unified_local_twist_observations(cfg)


def _apply_local_twist_command_mode(cfg, mode: str) -> None:
    """Make reference and task commands use one current-pelvis convention."""
    cfg.commands.motion.locomotion_command_frame = "pelvis_local"
    cfg.commands.motion.locomotion_command_mode = mode
    cfg.commands.motion.locomotion_task_state_enabled = False
    cfg.commands.motion.locomotion_task_state_sequence = ("dribble",)
    cfg.commands.motion.mimic_align_task_frame = False
    cfg.commands.motion.mimic_align_locomotion_heading = False


def _apply_reference_local_reset(cfg) -> None:
    """Reset a local-reference stage at the unmodified demo start state.

    A local twist does not need a simulation-wide forward axis.  Keep the
    reference yaw and velocities, and make any physical ball placement depend
    only on the reference pelvis-local geometry.  Simulation domain
    randomization events remain enabled; this only removes reset-state noise
    that would otherwise desynchronise a strict reference episode.
    """
    cfg.commands.motion.reset_face_task_forward = False
    cfg.commands.motion.reset_zero_velocity = False
    cfg.commands.motion.pose_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.commands.motion.velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.commands.motion.joint_position_range = (0.0, 0.0)


def _apply_local_twist_velocity_rewards(
    cfg, *, lin_weight: float, lin_std: float, yaw_weight: float, yaw_std: float
) -> None:
    """Replace world/task-frame root tracking with the active pelvis-local twist."""
    for reward_name in ("forward_velocity", "lateral_velocity_penalty", "task_heading_alignment"):
        if hasattr(cfg.rewards, reward_name):
            setattr(cfg.rewards, reward_name, None)
    if hasattr(cfg.rewards, "motion_global_anchor_pos"):
        cfg.rewards.motion_global_anchor_pos = None
    if hasattr(cfg.rewards, "motion_global_anchor_ori"):
        cfg.rewards.motion_global_anchor_ori = None
    cfg.rewards.motion_anchor_lin_vel = RewTerm(
        func=mdp.motion_anchor_local_lin_vel_tracking_exp,
        weight=lin_weight,
        params={"command_name": "motion", "std": lin_std},
    )
    cfg.rewards.motion_anchor_ang_vel = RewTerm(
        func=mdp.motion_anchor_local_ang_vel_tracking_exp,
        weight=yaw_weight,
        params={"command_name": "motion", "std": yaw_std},
    )
    if hasattr(cfg.rewards, "dribbling_velocity_tracking"):
        cfg.rewards.dribbling_velocity_tracking = None


def _apply_local_ball_objectives(cfg, *, disable_pelvis_quat_tracking: bool) -> None:
    """Remove fixed-world-forward geometry from a local-twist dribbling stage."""
    cfg.rewards.dribbling_dynamic_proximity.func = mdp.dribbling_pelvis_local_dynamic_proximity
    cfg.rewards.dribbling_ball_forward_progress.func = mdp.dribbling_command_ball_progress_reward
    cfg.rewards.dribbling_ball_trapped_penalty.func = mdp.dribbling_pelvis_local_ball_trapped_penalty
    cfg.rewards.dribbling_chase_ball.func = mdp.dribbling_command_chase_ball_reward
    cfg.rewards.dribbling_face_ball.func = mdp.dribbling_pelvis_local_face_ball
    cfg.rewards.dribbling_legal_foot_touch.params["min_pelvis_heading"] = 0.0
    if disable_pelvis_quat_tracking and hasattr(cfg.rewards, "dribbling_pelvis_quat_tracking"):
        cfg.rewards.dribbling_pelvis_quat_tracking = None


def _apply_local_task_ball_objectives(cfg) -> None:
    """S3 task version: local geometry with no reference-pelvis orientation prior."""
    _apply_local_ball_objectives(cfg, disable_pelvis_quat_tracking=True)


_S2_REWARD_NAMES = frozenset(
    {
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_anchor_lin_vel",
        "motion_anchor_ang_vel",
        "action_rate_l2",
        "waist_action_rate_l2",
        "joint_limit",
        "undesired_contacts",
        "pelvis_orientation",
        "upper_body_reference_overflow",
        "upper_body_manifold_nullspace",
        "s2_windowed_foot_tracking",
        "s2_approach_progress",
        "s2_contact_proximity",
        "s2_immediate_contact_bonus",
        "s2_next_touch_release",
        "s2_surface_style",
        "s2_nonfoot_ball_contact",
        "s2_wrong_foot_contact",
        "s2_premature_contact",
    }
)


def _prune_s2_reward_manager(cfg) -> None:
    """Remove every inherited reward outside the final S2 contract."""
    for reward_name in dir(cfg.rewards):
        if reward_name.startswith("_") or reward_name in _S2_REWARD_NAMES:
            continue
        reward_cfg = getattr(cfg.rewards, reward_name)
        if isinstance(reward_cfg, RewTerm):
            # ``None`` removes the term from RewardManager. A zero weight would
            # keep dead configuration and runtime bookkeeping alive.
            setattr(cfg.rewards, reward_name, None)
    missing = [
        reward_name
        for reward_name in sorted(_S2_REWARD_NAMES)
        if not isinstance(getattr(cfg.rewards, reward_name, None), RewTerm)
    ]
    if missing:
        raise RuntimeError(f"S2 reward contract is incomplete: {missing}")


@configclass
class G1FlatMotionCGPretrainUnifiedS1LocalStrictEnvCfg(G1FlatMotionStrictPretrainEnvCfg):
    """S1 local strict: exact demo local twist plus strict relative motion tracking.

    The reset still starts from the exact sampled demonstration state and the
    full pose/gait imitation prior remains active.  Root supervision is local
    ``[vx, vy, wz]`` rather than a non-deployable absolute world path/yaw.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_local_twist_command_mode(self, "reference")
        # The 163-D contract includes a ball-relative input.  Reuse the S2/S3
        # first-contact spawn without introducing any ball objective in S1.
        self.commands.motion.class_type = DribbleCGMotionCommand
        setattr(self.commands.motion, "dribble_cg_use_task_frame", False)
        setattr(self.commands.motion, "dribble_cg_snap_mode", "never")
        setattr(self.commands.motion, "dribble_cg_ball_spawn_mode", "reference_first_contact")
        # S1 is raw strict imitation: preserve every demo frame and end the
        # episode at the source boundary.  ``time_out=True`` resets LSTM
        # memory while retaining normal value bootstrapping at a time limit.
        self.commands.motion.motion_clip_end_resample = True
        self.commands.motion.motion_clip_end_terminate = True
        self.commands.motion.motion_cyclic_blend_frames = 0
        self.terminations.motion_clip_end = DoneTerm(
            func=mdp.motion_clip_finished,
            params={"command_name": "motion"},
            time_out=True,
        )
        _apply_local_twist_velocity_rewards(self, lin_weight=5.0, lin_std=0.45, yaw_weight=2.0, yaw_std=0.80)
        _apply_unified_local_163_contract(self)


@configclass
class G1FlatCGDribblingUnifiedS2LocalReferenceEnvCfg(G1FlatCGDribblingEnvCfg):
    """S2: learn frame-0 prefixes from two contacts through the full clip."""

    dribble_contact_mode: str = "both"
    dribble_contact_surface: str = "instep"
    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),
        ((4, 1.0),),
        ((8, 1.0),),
        ((0, 1.0),),
        ((0, 1.0),),
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_local_twist_command_mode(self, "reference")
        _apply_reference_local_reset(self)
        # S2 remains an exact demo/contact episode.  Ball, robot, and LSTM
        # all reset together at the final reference frame; no synthetic
        # bridge or unlabelled contact interval is introduced here.
        self.commands.motion.motion_clip_end_resample = True
        self.commands.motion.motion_clip_end_terminate = True
        self.commands.motion.motion_cyclic_blend_frames = 0
        self.terminations.motion_clip_end = DoneTerm(
            func=mdp.motion_clip_finished,
            params={"command_name": "motion"},
            time_out=True,
        )
        _apply_local_twist_velocity_rewards(self, lin_weight=5.0, lin_std=0.60, yaw_weight=1.0, yaw_std=1.50)
        _apply_unified_local_163_contract(self)

        # S2 keeps normal two-foot imitation outside contact windows. During
        # contact it preserves the support-foot target and nearly releases
        # only the labelled contact foot so physics can determine its final
        # ball-relative placement.
        self.rewards.s2_windowed_foot_tracking = RewTerm(
            func=mdp.dribbling_s2_windowed_foot_tracking_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.30,
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "contact_foot_scale": 0.10,
                "contact_foot_release_seconds": 0.30,
            },
        )

        # Contact-dependent terms share one event/release state.  The target
        # body list starts with both legal ankle links; later entries are
        # explicit non-foot penalties.
        ankle_names = ["right_ankle_roll_link", "left_ankle_roll_link"]
        num_ankle_links = len(ankle_names)
        s2_contact_body_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                *ankle_names,
                *(
                    body_name
                    for body_name in self.commands.motion.body_names
                    if body_name not in ankle_names
                ),
            ],
        )
        event_params = {
            "command_name": "motion",
            "ball_sensor_name": "soccer_ball_contact",
            "all_body_cfg": s2_contact_body_cfg,
            "num_ankle_links": num_ankle_links,
            "require_expected_foot": True,
            "target_side_enabled": True,
            "side_deadzone": 0.04,
            "proximity_contact_distance_max": 0.25,
            "target_region_std": 0.12,
            "proximity_approach_seconds": 0.30,
            "proximity_approach_min_weight": 0.20,
            "missed_contact_grace_steps": 3,
        }
        self.rewards.s2_approach_progress = RewTerm(
            func=mdp.dribbling_s2_approach_progress,
            weight=2.0,
            params={
                **dict(event_params),
                "approach_distance_std": 0.40,
                "approach_progress_max": 4.0,
            },
        )
        self.rewards.s2_contact_proximity = RewTerm(
            func=mdp.dribbling_s2_contact_proximity,
            # Dense side-aware guidance bridges the released reference foot to
            # the sparse physical event. It is disabled immediately after a
            # side-valid contact succeeds.
            weight=1.0,
            params=dict(event_params),
        )
        # Correct time and foot define contact success.  The source
        # inside/outside label is deliberately only a style preference.
        self.rewards.s2_immediate_contact_bonus = RewTerm(
            func=mdp.dribbling_s2_immediate_contact_bonus,
            weight=5.0,
            params=dict(event_params),
        )
        self.rewards.s2_next_touch_release = RewTerm(
            func=mdp.dribbling_s2_next_touch_release,
            # 12 gives a first-release maximum return of 1.92 over eight
            # 20-ms steps.  It is deliberately below the earlier aggressive
            # proposal of 20 while remaining large enough to shape touch 1.
            weight=12.0,
            params={
                **dict(event_params),
                "release_window_steps": 8,
                "release_speed_lower_ratio": 0.65,
                "release_speed_upper_ratio": 1.35,
                "release_lateral_speed_std": 0.35,
                "release_overspeed_decay_ratio": 0.35,
                "release_velocity_outlier_scale": 0.75,
                "release_separation_min": 0.14,
                "release_separation_full": 0.22,
                "release_quality_threshold": 0.50,
                "release_min_target_distance": 0.05,
            },
        )
        self.rewards.s2_surface_style = RewTerm(
            func=mdp.dribbling_s2_surface_style,
            weight=1.0,
            params=dict(event_params),
        )
        self.rewards.s2_nonfoot_ball_contact = RewTerm(
            func=mdp.dribbling_s2_nonfoot_ball_contact_penalty,
            weight=-5.0,
            params=dict(event_params),
        )
        self.rewards.s2_wrong_foot_contact = RewTerm(
            func=mdp.dribbling_s2_wrong_foot_contact_penalty,
            weight=-1.0,
            params=dict(event_params),
        )
        self.rewards.s2_premature_contact = RewTerm(
            func=mdp.dribbling_s2_premature_contact_penalty,
            weight=-0.5,
            params=dict(event_params),
        )

        # The inherited dribbling/CG classes define many objectives for other
        # tasks. S2 exposes only its exact 22-term contract to RewardManager.
        _prune_s2_reward_manager(self)

        # Ball-lost is distance-only and debounced for 0.2 s.  Reference
        # ankle/wrist height and fixed no-touch terminations are removed.
        self.terminations.ball_lost.params.update(
            {
                "max_distance": 1.10,
                "max_vel_divergence": 0.0,
                "grace_steps": 10,
                "max_consecutive_steps": 10,
            }
        )
        self.terminations.dribbling_no_contact = None
        if hasattr(self.terminations, "ee_body_pos"):
            self.terminations.ee_body_pos = None
        # A missed event ends only the short one/two-contact curriculum. Longer
        # sequences keep running so the policy observes downstream physical
        # state, while their consecutive-success streak is still reset.
        self.terminations.missed_valid_contact = DoneTerm(
            func=mdp.dribbling_missed_valid_contact,
            params={
                **dict(event_params),
                "max_required_contact_count": 2,
            },
        )
        self.terminations.invalid_acquisition_start = DoneTerm(
            func=mdp.dribbling_s2_invalid_acquisition_start,
            params={
                **dict(event_params),
                "acquisition_min_history_steps": 20,
            },
        )

        # The ball always follows physics. Only the side-aware event region,
        # event approach/contact/release objectives and split penalties supervise it. Its
        # reset location is the selected first event's reference target,
        # represented in the reference pelvis-local frame rather than task +X.
        setattr(self.commands.motion, "dribble_cg_use_task_frame", False)
        setattr(self.commands.motion, "dribble_cg_snap_mode", "never")
        setattr(self.commands.motion, "dribble_cg_ball_spawn_mode", "reference_first_contact")
        setattr(self.commands.motion, "dribble_cg_curriculum_levels", self.s2_curriculum_levels)
        setattr(
            self.commands.motion,
            "dribble_cg_curriculum_required_contacts",
            # Zero means that every selected event in the full clip must be
            # completed consecutively, rather than accepting an eight-touch
            # prefix as completion of a longer reference.
            (2, 4, 8, 0, 0),
        )
        setattr(self.commands.motion, "dribble_cg_curriculum_start_level", 0)
        setattr(self.commands.motion, "dribble_cg_contact_window_seconds", 0.10)
        setattr(self.commands.motion, "dribble_cg_missed_contact_grace_steps", 3)
        setattr(self.commands.motion, "dribble_cg_release_window_steps", 8)
        setattr(self.commands.motion, "dribble_cg_acquisition_initial_probability", 0.25)
        setattr(self.commands.motion, "dribble_cg_acquisition_min_history_steps", 20)
        setattr(self.commands.motion, "dribble_cg_acquisition_max_history_steps", 30)
        # Per level: (exact-reference probability, maximum initial radius).
        # Levels 0--3 preserve the exact frame-0 reference state. Only the
        # final full-clip level perturbs the ball once at reset; every later
        # error still arises from ball physics and the preceding touches.
        setattr(
            self.commands.motion,
            "dribble_cg_curriculum_ball_spawn_jitter",
            (
                (1.00, 0.00),
                (1.00, 0.00),
                (1.00, 0.00),
                (1.00, 0.00),
                (0.20, 0.06),
            ),
        )
        setattr(
            self.commands.motion,
            "dribble_cg_ball_spawn_safe_forward_range",
            (-0.04, 0.12),
        )
        setattr(
            self.commands.motion,
            "dribble_cg_ball_spawn_safe_side_range",
            (0.06, 0.14),
        )
        setattr(self.commands.motion, "dribble_cg_ball_spawn_jitter_min", 0.0)
        setattr(self.commands.motion, "dribble_cg_ball_spawn_jitter_max", 0.0)
        setattr(self.commands.motion, "dribble_cg_require_surface_labels", True)
        setattr(self.commands.motion, "dribble_cg_require_flow_labels", False)

        # Isaac Lab runs this term before CommandManager.reset(), allowing it
        # to evaluate the just-finished episode and promote the sampler before
        # the next episode is drawn.
        self.curriculum.s2_contact_levels = CurrTerm(
            func=mdp.s2_contact_level_curriculum,
            params={
                "command_name": "motion",
                "min_evaluation_episodes": 1024,
                "required_consecutive_passes": 2,
                "sequence_completion_threshold": 0.60,
                "max_fall_rate": 0.05,
                "fall_termination_terms": ("anchor_pos_z", "anchor_ori"),
            },
        )


@configclass
class G1FlatCGDribblingUnifiedS3LocalTaskEnvCfg(G1FlatCGDribblingEnvCfg):
    """S3 local task: sampled local twist and flexible physical instep control."""

    dribble_contact_mode: str = "both"
    dribble_contact_surface: str = "instep"

    def __post_init__(self):
        super().__post_init__()
        _apply_local_twist_command_mode(self, "resampled")
        _apply_reference_local_reset(self)
        self.commands.motion.motion_clip_end_resample = False
        self.commands.motion.motion_clip_end_terminate = False
        self.commands.motion.motion_cyclic_blend_frames = 25
        self.commands.motion.locomotion_cmd_lin_vel_range = {
            "x": (0.0, 1.50),
            "y": (-0.50, 0.50),
            "z": (0.0, 0.0),
        }
        self.commands.motion.locomotion_cmd_ang_vel_range = {
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.80, 0.80),
        }
        self.commands.motion.locomotion_cmd_duration_range = (1.5, 3.0)
        # Exact rest commands are required for a playback plan to start/end
        # cleanly without reintroducing the old IDLE/STOP state command.
        self.commands.motion.locomotion_cmd_stationary_probability = 0.10
        # 24 env steps per rollout * 1000 PPO updates.  All local S3 reward
        # terms see the same blended command, preventing a hidden objective
        # change during the S2 -> S3 distribution transition.
        self.commands.motion.locomotion_twist_reference_blend_env_steps = 24_000
        _apply_local_twist_velocity_rewards(self, lin_weight=5.0, lin_std=0.65, yaw_weight=1.0, yaw_std=1.50)
        _apply_unified_local_163_contract(self)
        _disable_reference_ball_contact_terms(self)
        _apply_local_task_ball_objectives(self)

        # Playback command plans may request a full reset after their final
        # segment.  The underlying flag is always false during resampled
        # training, so this term is inert for PPO.
        self.terminations.locomotion_manual_sequence_end = DoneTerm(
            func=mdp.locomotion_manual_sequence_finished,
            params={"command_name": "motion"},
        )

        setattr(self.commands.motion, "dribble_cg_use_task_frame", False)
        setattr(self.commands.motion, "dribble_cg_snap_mode", "never")
        setattr(self.commands.motion, "dribble_cg_ball_spawn_mode", "reference_first_contact")
        setattr(self.commands.motion, "dribble_cg_fixed_touch_foot", None)
        setattr(self.commands.motion, "dribble_cg_fixed_touch_surface", None)
