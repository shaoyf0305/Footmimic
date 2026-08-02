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
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp import observations_anchor as obs_anchor
from soccer.tasks.tracking.mdp.commands_dribble_cg import DribbleCGMotionCommand
from .soccer_flat_env_cfg import (
    G1FlatMotionPretrainEnvCfg,
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

    # Optional override: "both" (default) allows either ankle; single-foot modes kept for ablations.
    dribble_contact_mode: str = "both"
    # ``any`` preserves the old whole-foot rule.  ``inside_instep`` and
    # ``outside_instep`` use the closest ankle link's local frame, requiring
    # a dorsal/medial or dorsal/lateral ball contact rather than just a foot.
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
            self.rewards.target_point_proximity.weight = 0.0

        # Stronger upright / anti-arch than generic proximity alone
        if hasattr(self.rewards, "pelvis_orientation"):
            self.rewards.pelvis_orientation.weight = -2.5

        # Drop world-frame yaw tracking from the motion: zig-zag motion ("slalom around
        # cones") would otherwise force the policy to copy the S-shape heading pattern.
        # The face-ball reward below replaces it with a task-aligned heading signal.
        if hasattr(self.rewards, "motion_global_anchor_ori"):
            self.rewards.motion_global_anchor_ori.weight = 0.0

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
        if surface not in {"any", "inside_instep", "outside_instep"}:
            raise ValueError(
                f"Unsupported dribble_contact_surface={self.dribble_contact_surface}. "
                "Expected one of: any, inside_instep, outside_instep."
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
    """Dribbling with annotated contact graph + stitched demo ``ball_pos_w``.

    Uses :class:`~soccer.tasks.tracking.mdp.commands_dribble_cg.DribbleCGMotionCommand`
    to optionally drive the sim ball along the demo trajectory (anchor-relative),
    and rewards that align sensor contact / foot side with ``dribble_cg_*`` labels.

    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.class_type = DribbleCGMotionCommand
        # This CG variant uses contact labels to teach the touch timing/side, while
        # keeping the ball task simple: spawn the ball in front and let physics move it.
        setattr(self.commands.motion, "dribble_cg_use_demo_ball", False)
        setattr(self.commands.motion, "dribble_cg_use_task_frame", True)
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

        self.rewards.dribbling_cg_demo_ball_tracking = RewTerm(
            func=mdp.dribbling_cg_demo_ball_tracking_exp,
            weight=4.0,
            params={"command_name": "motion", "std": 0.32},
        )
        # Continuous CG: demo foot–ball distance (from synthesized ball_pos_w).
        # Run scripts/dribble/synthesize_dribble_ball_traj.py on labeled motions first.
        self.rewards.dribbling_cg_foot_ball_distance = RewTerm(
            func=mdp.dribbling_cg_foot_ball_distance_exp,
            weight=3.5,
            params={
                "command_name": "motion",
                "std": 0.22,
                "use_xy_only": True,
            },
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
        self.rewards.dribbling_approach_foot_ball.weight = 0.0

        # Velocity / forward progress use recent sim contact, not only labeled CG frames.
        self.rewards.dribbling_velocity_tracking.params["cg_gated_contact"] = False
        self.rewards.dribbling_ball_forward_progress.params["cg_gated_contact"] = False
        self.rewards.dribbling_legal_foot_touch.params["cg_gated"] = True
        self.rewards.dribbling_rapid_retouch_penalty.params["cg_gated"] = True
        self.rewards.dribbling_rapid_retouch_penalty.params["min_steps_between_touches"] = 26


def _apply_dribbling_locomotion_velocity_terms(
    cfg,
    *,
    include_cartesian_command_observations: bool = True,
) -> None:
    """Shared follow/control mechanics, optionally exposing legacy Cartesian command obs."""
    cfg.rewards.forward_velocity.weight = 0.0
    cfg.rewards.lateral_velocity_penalty.weight = 0.0
    cfg.rewards.task_heading_alignment.weight = 0.0
    if hasattr(cfg.rewards, "dribbling_face_ball"):
        cfg.rewards.dribbling_face_ball.weight = 1.0

    cfg.rewards.motion_anchor_lin_vel = RewTerm(
        func=mdp.motion_anchor_lin_vel_tracking_exp,
        weight=5.0,
        params={"command_name": "motion", "std": 0.8},
    )
    cfg.rewards.motion_anchor_ang_vel = RewTerm(
        func=mdp.motion_anchor_ang_vel_tracking_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 2.0},
    )

    # In follow/control the velocity target comes from motion_anchor_lin_vel (above).
    # dribbling_velocity_tracking rewards task +X sync specifically, which conflicts
    # with arbitrary-heading follow/control — zero it out.
    if hasattr(cfg.rewards, "dribbling_velocity_tracking"):
        cfg.rewards.dribbling_velocity_tracking.weight = 0.0

    if include_cartesian_command_observations:
        cfg.observations.policy.motion_anchor_lin_vel_cmd = ObsTerm(
            func=mdp.motion_anchor_lin_vel_command,
            params={"command_name": "motion"},
        )
        cfg.observations.critic.motion_anchor_lin_vel_cmd = ObsTerm(
            func=mdp.motion_anchor_lin_vel_command,
            params={"command_name": "motion"},
        )
        cfg.observations.policy.motion_anchor_ang_vel_cmd = ObsTerm(
            func=mdp.motion_anchor_ang_vel_command,
            params={"command_name": "motion"},
        )
        cfg.observations.critic.motion_anchor_ang_vel_cmd = ObsTerm(
            func=mdp.motion_anchor_ang_vel_command,
            params={"command_name": "motion"},
        )


def _apply_dribbling_control_velocity_terms(cfg) -> None:
    """Control: external speed/heading/duration — ref provides pose/gait/CG only, not root vel."""
    _apply_dribbling_locomotion_velocity_terms(cfg)
    cfg.commands.motion.locomotion_command_mode = "resampled"
    # The control heading is also the intended facing direction (see
    # ``dribbling_command_face_ball``).  Rotate the style pose into that frame
    # so the upper-body mimic reward does not hold arms and wrists at world +X
    # during an oblique turn.
    cfg.commands.motion.mimic_align_locomotion_heading = True
    # A control episode owns the task clock.  Demo time is a looping style
    # phase and must not reset the robot, ball, or locomotion command.
    cfg.commands.motion.motion_clip_end_resample = False
    # 1.5 m/s is a normal control target, not an out-of-distribution play-only
    # command.  Long holds force the policy to keep control across turns.
    cfg.commands.motion.locomotion_cmd_speed_range = (0.40, 1.50)
    cfg.commands.motion.locomotion_cmd_heading_range = (-0.75, 0.75)
    cfg.commands.motion.locomotion_cmd_duration_range = (3.0, 6.0)
    cfg.commands.motion.locomotion_cmd_wz_range = (0.0, 0.0)
    # Keep command transitions within the distribution used by the good
    # control replay: heading slews at 0.85 rad/s while speed brakes into a
    # turn and recovers afterwards.  The policy observes this effective
    # command, rather than an instantaneous 0 -> +/-0.65 rad jump.
    cfg.commands.motion.locomotion_cmd_smoothing_enabled = True
    cfg.commands.motion.locomotion_cmd_heading_rate_limit = 0.85
    cfg.commands.motion.locomotion_cmd_accel_limit = 1.4
    cfg.commands.motion.locomotion_cmd_decel_limit = 2.4
    cfg.commands.motion.locomotion_cmd_turn_slowdown_angle = 0.55
    cfg.commands.motion.locomotion_cmd_turn_min_speed_scale = 0.60

    # The legacy task_heading_alignment term faces world +X and therefore
    # cannot supervise arbitrary-heading control.  Use the smoothed effective
    # heading instead: it remains feasible during a direction reversal while
    # giving PPO a direct signal to recover pelvis yaw before the ball drifts
    # sideways.
    cfg.rewards.locomotion_heading_tracking = RewTerm(
        func=mdp.locomotion_heading_tracking_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.45,
            "min_command_speed": 0.25,
        },
    )

    # A turn or recovery necessarily departs from the instantaneous demo pose.
    # Keep these references as soft rewards, but never end a control episode
    # because of a style-tracking height/orientation error.  Real falls and
    # ball-task failures remain active terminations.
    for term_name in ("ee_body_pos", "anchor_pos_z", "anchor_ori"):
        if hasattr(cfg.terminations, term_name):
            setattr(cfg.terminations, term_name, None)

    # Rotate every forward-geometry dribbling term into the active command
    # frame.  The legacy functions keep their fixed +X semantics for forward
    # and follow; only control receives these replacements.
    cfg.rewards.dribbling_dynamic_proximity.func = mdp.dribbling_command_dynamic_proximity
    cfg.rewards.dribbling_ball_forward_progress.func = mdp.dribbling_command_ball_progress_reward
    cfg.rewards.dribbling_ball_trapped_penalty.func = mdp.dribbling_command_ball_trapped_penalty
    cfg.rewards.dribbling_chase_ball.func = mdp.dribbling_command_chase_ball_reward
    cfg.rewards.dribbling_face_ball.func = mdp.dribbling_command_face_ball
    cfg.rewards.dribbling_ball_forward_progress.params.update(
        {
            "command_speed_ratio": 0.50,
            "lateral_ratio_max": 0.70,
        }
    )
    # The legacy legal-touch gate checks pelvis heading against world +X.
    # Command-frame face-ball shaping above now supplies the heading signal.
    cfg.rewards.dribbling_legal_foot_touch.params["min_pelvis_heading"] = 0.0

    # CG labels remain a weak cadence/style prior only.  Their per-frame touch
    # timing must not compete with a command-triggered turn or ball recovery.
    cfg.rewards.dribbling_legal_foot_touch.params["cg_gated"] = False
    cfg.rewards.dribbling_rapid_retouch_penalty.params["cg_gated"] = False
    if hasattr(cfg.rewards, "dribbling_cg_contact_consistency"):
        cfg.rewards.dribbling_cg_contact_consistency.weight = 1.0
    if hasattr(cfg.rewards, "dribbling_cg_premature_contact"):
        cfg.rewards.dribbling_cg_premature_contact.weight = 0.0
    if hasattr(cfg.rewards, "dribbling_cg_foot_consistency"):
        cfg.rewards.dribbling_cg_foot_consistency.weight = 0.5

    # During a labeled touch, retain only the opposite (support) ankle's roll
    # style.  The touching ankle remains unrestricted, so this small cosmetic
    # correction cannot trade away reachability or ball-control recovery.
    cfg.rewards.dribbling_support_ankle_roll = RewTerm(
        func=mdp.dribbling_support_ankle_roll_tracking_exp,
        weight=0.25,
        params={
            "command_name": "motion",
            "std": 0.24,
            "deadzone": 0.10,
            "error_cap": 0.35,
        },
    )

    # Keep the no-contact termination, but give an actual command-change
    # recovery attempt a bounded extra window.  The counter only slows while
    # the ball is recoverable and the pelvis is closing distance.
    cfg.terminations.dribbling_no_contact.params.update(
        {
            "recovery_window_steps": 75,
            "recovery_max_distance": 0.85,
            "recovery_min_closing_speed": 0.05,
            "recovery_counter_increment": 0.25,
        }
    )

    # Playback-only opt-in: ``--locomotion_cmd_reset_on_end`` sets a flag on a
    # manual polar sequence.  Consume it in ordinary control too, so the final
    # segment resets the robot, ball, and sequence back to segment zero.
    # Resampled training commands never set this flag, so training is inert.
    cfg.terminations.locomotion_manual_sequence_end = DoneTerm(
        func=mdp.locomotion_manual_sequence_finished,
        params={"command_name": "motion"},
    )

    # Do not pull linear/angular vel from demo bodies — velocity comes from the command only.
    if hasattr(cfg.rewards, "motion_body_lin_vel"):
        cfg.rewards.motion_body_lin_vel.weight = 0.0
    if hasattr(cfg.rewards, "motion_body_ang_vel"):
        cfg.rewards.motion_body_ang_vel.weight = 0.0

    # The reference is a style phase, not a task tape.  Keep a gentle whole
    # body posture prior, but soften it in control so a phase wrap and an
    # actual turn cannot produce large wrist/arm corrections.  This does not
    # touch the CG foot/contact rewards (which are independent of the arms).
    if hasattr(cfg.rewards, "motion_body_pos"):
        cfg.rewards.motion_body_pos.weight = 0.45
    if hasattr(cfg.rewards, "motion_body_ori"):
        cfg.rewards.motion_body_ori.weight = 0.35

    # The policy retains the v3.9 29-D joint-action interface.  Upper-body
    # targets are constrained and then projected onto the motion-bank PCA
    # manifold; lower-body actions pass through unchanged.
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
        orthogonal_residual_limit=0.10,
        cutoff_frequency_hz=1.8,
        # Keep the control policy inside the current style pose's local arm
        # envelope.  This is control-only: the base config defaults to None.
        # It prevents saturated PCA latents from locking an arm in a raised pose.
        reference_target_margin=0.25,
    )
    cfg.actions.joint_pos.scale = old_action_cfg.scale
    cfg.actions.joint_pos.offset = old_action_cfg.offset
    cfg.actions.joint_pos.preserve_order = getattr(old_action_cfg, "preserve_order", False)
    if hasattr(old_action_cfg, "clip"):
        cfg.actions.joint_pos.clip = old_action_cfg.clip

    # Feed back the effective joint command after manifold projection.
    cfg.observations.policy.actions.func = mdp.effective_joint_action
    cfg.observations.policy.actions.params = {"action_name": "joint_pos"}
    cfg.observations.critic.actions.func = mdp.effective_joint_action
    cfg.observations.critic.actions.params = {"action_name": "joint_pos"}
    cfg.rewards.action_rate_l2.func = mdp.effective_action_rate_l2_clip
    cfg.rewards.action_rate_l2.params = {"action_name": "joint_pos"}
    # Keep the light v3.9 pre-constraint overflow signal.  It is diagnostic and
    # trainable, but does not add another action transform.
    cfg.rewards.upper_body_reference_overflow = RewTerm(
        func=mdp.upper_body_reference_overflow_penalty,
        weight=-0.05,
        params={"action_name": "joint_pos"},
    )
    # The sole post-v3.9 intervention: make PCA null-space effort visible to
    # PPO without adding another action transform or reducing ball reach.
    cfg.rewards.upper_body_manifold_nullspace = RewTerm(
        func=mdp.upper_body_manifold_nullspace_penalty,
        weight=-0.02,
        params={"action_name": "joint_pos", "scale": 0.10},
    )

    cfg.observations.policy.motion_locomotion_polar_cmd = ObsTerm(
        func=mdp.motion_locomotion_polar_command,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_polar_cmd = ObsTerm(
        func=mdp.motion_locomotion_polar_command,
        params={"command_name": "motion"},
    )


def _apply_dribbling_full_control_velocity_terms(
    cfg,
    *,
    include_cartesian_command_observations: bool = True,
) -> None:
    """Current full Control: stateful external command + start/dribble/stop task."""
    _apply_dribbling_locomotion_velocity_terms(
        cfg,
        include_cartesian_command_observations=include_cartesian_command_observations,
    )
    cfg.commands.motion.locomotion_command_mode = "resampled"
    # The task input is distinct from speed: both IDLE and STOP command zero
    # velocity, but only STOP asks the policy to settle the ball after a run.
    # Each episode begins with a genuine stationary wait, then trains the
    # complete start -> dribble -> stop loop rather than isolated gait clips.
    cfg.commands.motion.locomotion_task_state_enabled = True
    cfg.commands.motion.locomotion_task_state_sequence = ("idle", "dribble", "stop")
    # Explicit play/evaluation sequences restart from IDLE after either a
    # failure or a successful STOP reset, instead of remaining at their last
    # held segment.
    cfg.commands.motion.locomotion_task_state_restart_manual_sequence_on_reset = True
    cfg.commands.motion.locomotion_task_idle_duration_range = (1.0, 2.5)
    cfg.commands.motion.locomotion_task_dribble_duration_range = (3.0, 6.0)
    cfg.commands.motion.locomotion_task_stop_duration_range = (2.0, 3.5)
    # The longest IDLE -> DRIBBLE -> STOP schedule is 12 s; leave margin for
    # a complete stop phase instead of truncating it at the generic 10 s cap.
    cfg.episode_length_s = 15.0
    # The control heading is also the intended facing direction (see
    # ``dribbling_command_face_ball``).  Rotate the style pose into that frame
    # so the upper-body mimic reward does not hold arms and wrists at world +X
    # during an oblique turn.
    cfg.commands.motion.mimic_align_locomotion_heading = True
    # A control episode owns the task clock.  Demo time is a looping style
    # phase and must not reset the robot, ball, or locomotion command.
    cfg.commands.motion.motion_clip_end_resample = False
    # 1.5 m/s is a normal control target, not an out-of-distribution play-only
    # command.  Long holds force the policy to keep control across turns.
    # The complete training command range now includes exactly zero.  Positive
    # speeds remain a separate DRIBBLE-only range, while IDLE/STOP generate
    # stable zero-speed targets with explicit state semantics.
    cfg.commands.motion.locomotion_cmd_speed_range = (0.0, 1.50)
    cfg.commands.motion.locomotion_cmd_dribble_speed_range = (0.40, 1.50)
    cfg.commands.motion.locomotion_cmd_heading_range = (-0.75, 0.75)
    cfg.commands.motion.locomotion_cmd_duration_range = (3.0, 6.0)
    cfg.commands.motion.locomotion_cmd_wz_range = (0.0, 0.0)
    # Keep command transitions within the distribution used by the good
    # control replay: heading slews at 0.85 rad/s while speed brakes into a
    # turn and recovers afterwards.  The policy observes this effective
    # command, rather than an instantaneous 0 -> +/-0.65 rad jump.
    cfg.commands.motion.locomotion_cmd_smoothing_enabled = True
    cfg.commands.motion.locomotion_cmd_heading_rate_limit = 0.85
    cfg.commands.motion.locomotion_cmd_accel_limit = 1.4
    cfg.commands.motion.locomotion_cmd_decel_limit = 2.4
    cfg.commands.motion.locomotion_cmd_turn_slowdown_angle = 0.55
    cfg.commands.motion.locomotion_cmd_turn_min_speed_scale = 0.60

    # The legacy task_heading_alignment term faces world +X and therefore
    # cannot supervise arbitrary-heading control.  Use the smoothed effective
    # heading instead: it remains feasible during a direction reversal while
    # giving PPO a direct signal to recover pelvis yaw before the ball drifts
    # sideways.
    cfg.rewards.locomotion_heading_tracking = RewTerm(
        func=mdp.locomotion_heading_tracking_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.45,
            "min_command_speed": 0.25,
        },
    )

    # A turn or recovery necessarily departs from the instantaneous demo pose.
    # Keep these references as soft rewards, but never end a control episode
    # because of a style-tracking height/orientation error.  Real falls and
    # ball-task failures remain active terminations.
    for term_name in ("ee_body_pos", "anchor_pos_z", "anchor_ori"):
        if hasattr(cfg.terminations, term_name):
            setattr(cfg.terminations, term_name, None)

    # Rotate every forward-geometry dribbling term into the active command
    # frame.  The legacy functions keep their fixed +X semantics for forward
    # and follow; only control receives these replacements.
    cfg.rewards.dribbling_dynamic_proximity.func = mdp.dribbling_command_dynamic_proximity
    cfg.rewards.dribbling_ball_forward_progress.func = mdp.dribbling_command_ball_progress_reward
    cfg.rewards.dribbling_ball_trapped_penalty.func = mdp.dribbling_command_ball_trapped_penalty
    cfg.rewards.dribbling_chase_ball.func = mdp.dribbling_command_chase_ball_reward
    cfg.rewards.dribbling_idle_stand = RewTerm(
        func=mdp.dribbling_idle_stand_reward,
        weight=2.5,
        params={
            "command_name": "motion",
            "linear_speed_std": 0.10,
            "angular_speed_std": 0.35,
        },
    )
    cfg.rewards.dribbling_stop_settle = RewTerm(
        func=mdp.dribbling_stop_settle_reward,
        weight=6.0,
        params={
            "command_name": "motion",
            "pelvis_speed_std": 0.10,
            "pelvis_angular_speed_std": 0.35,
            "settled_pelvis_speed": 0.05,
            "settled_pelvis_angular_speed": 0.20,
        },
    )
    cfg.rewards.dribbling_face_ball.func = mdp.dribbling_command_face_ball
    cfg.rewards.dribbling_ball_forward_progress.params.update(
        {
            "command_speed_ratio": 0.50,
            "lateral_ratio_max": 0.70,
        }
    )
    # The legacy legal-touch gate checks pelvis heading against world +X.
    # Command-frame face-ball shaping above now supplies the heading signal.
    cfg.rewards.dribbling_legal_foot_touch.params["min_pelvis_heading"] = 0.0

    # CG labels remain a weak cadence/style prior only.  Their per-frame touch
    # timing must not compete with a command-triggered turn or ball recovery.
    cfg.rewards.dribbling_legal_foot_touch.params["cg_gated"] = False
    cfg.rewards.dribbling_rapid_retouch_penalty.params["cg_gated"] = False
    if hasattr(cfg.rewards, "dribbling_cg_contact_consistency"):
        cfg.rewards.dribbling_cg_contact_consistency.weight = 1.0
    if hasattr(cfg.rewards, "dribbling_cg_premature_contact"):
        cfg.rewards.dribbling_cg_premature_contact.weight = 0.0
    if hasattr(cfg.rewards, "dribbling_cg_foot_consistency"):
        cfg.rewards.dribbling_cg_foot_consistency.weight = 0.5

    # Moving gait and contact-graph priors are meaningful only while carrying
    # the ball.  Leaving them active in IDLE/STOP would make a zero-speed
    # request conflict with the walking reference clip.
    if hasattr(cfg.rewards, "motion_body_pos"):
        cfg.rewards.motion_body_pos.params["active_task_states"] = (mdp.TASK_STATE_DRIBBLE,)
    if hasattr(cfg.rewards, "motion_foot_pos"):
        cfg.rewards.motion_foot_pos.params["active_task_states"] = (mdp.TASK_STATE_DRIBBLE,)
    if hasattr(cfg.rewards, "dribbling_gait_foot_tracking"):
        cfg.rewards.dribbling_gait_foot_tracking.params["active_task_states"] = (mdp.TASK_STATE_DRIBBLE,)
    if hasattr(cfg.rewards, "dribbling_stall_no_touch_penalty"):
        cfg.rewards.dribbling_stall_no_touch_penalty.params["active_task_states"] = (mdp.TASK_STATE_DRIBBLE,)
    if hasattr(cfg.rewards, "dribbling_cg_foot_ball_distance"):
        cfg.rewards.dribbling_cg_foot_ball_distance.params["active_task_states"] = (mdp.TASK_STATE_DRIBBLE,)
    if hasattr(cfg.rewards, "dribbling_cg_contact_consistency"):
        cfg.rewards.dribbling_cg_contact_consistency.params["active_task_states"] = (mdp.TASK_STATE_DRIBBLE,)
    if hasattr(cfg.rewards, "dribbling_cg_foot_consistency"):
        cfg.rewards.dribbling_cg_foot_consistency.params["active_task_states"] = (mdp.TASK_STATE_DRIBBLE,)

    # During a labeled touch, retain only the opposite (support) ankle's roll
    # style.  The touching ankle remains unrestricted, so this small cosmetic
    # correction cannot trade away reachability or ball-control recovery.
    cfg.rewards.dribbling_support_ankle_roll = RewTerm(
        func=mdp.dribbling_support_ankle_roll_tracking_exp,
        weight=0.25,
        params={
            "command_name": "motion",
            "std": 0.24,
            "deadzone": 0.10,
            "error_cap": 0.35,
            "active_task_states": (mdp.TASK_STATE_DRIBBLE,),
        },
    )

    # Once STOP is requested, the ball is intentionally no longer a task
    # target.  It may roll away while the robot stabilizes, so ball-loss is a
    # DRIBBLE-only failure condition.
    cfg.terminations.ball_lost.params["active_task_states"] = (mdp.TASK_STATE_DRIBBLE,)

    # Keep the no-contact termination, but give an actual command-change
    # recovery attempt a bounded extra window.  The counter only slows while
    # the ball is recoverable and the pelvis is closing distance.
    cfg.terminations.dribbling_no_contact.params.update(
        {
            "recovery_window_steps": 75,
            "recovery_max_distance": 0.85,
            "recovery_min_closing_speed": 0.05,
            "recovery_counter_increment": 0.25,
            # A stationary wait or a deliberate settle must not accrue the
            # running-only no-contact failure counter.
            "active_task_states": (mdp.TASK_STATE_DRIBBLE,),
        }
    )
    # Completing a stable STOP is a success terminal, not an indefinitely
    # held command.  The next episode is reset into IDLE by the command term.
    cfg.terminations.dribbling_stop_success = DoneTerm(
        func=mdp.dribbling_stop_success,
        params={
            "command_name": "motion",
            "min_settle_duration_s": 0.5,
            "settled_pelvis_speed": 0.05,
            "settled_pelvis_angular_speed": 0.20,
        },
    )
    # Playback-only opt-in: a manual command sequence can end its final STOP
    # segment with a full scene reset instead of looping or holding it. The
    # command flag is false during resampled training, so this term is inert.
    cfg.terminations.locomotion_manual_sequence_end = DoneTerm(
        func=mdp.locomotion_manual_sequence_finished,
        params={"command_name": "motion"},
    )

    # STOP is a robot-stability endpoint, not a ball-control objective.  Gate
    # every dribble-specific shaping term to DRIBBLE so a freely rolling ball
    # neither improves nor harms return during IDLE/STOP.
    dribble_only_reward_names = (
        "dribbling_face_ball",
        "dribbling_velocity_tracking",
        "dribbling_dynamic_proximity",
        "dribbling_stall_no_touch_penalty",
        "dribbling_approach_foot_ball",
        "dribbling_pelvis_quat_tracking",
        "dribbling_ball_speed_excess",
        "dribbling_ball_coast_penalty",
        "dribbling_ball_trapped_penalty",
        "dribbling_sustained_contact_penalty",
        "dribbling_ball_bounce_penalty",
        "dribbling_ball_forward_progress",
        "dribbling_orbiting_penalty",
        "dribbling_gait_foot_tracking",
        "dribbling_chase_ball",
        "dribbling_rapid_retouch_penalty",
        "dribbling_legal_foot_touch",
        "dribbling_micro_contact_filter",
        "dribbling_undesired_contact_penalty",
        "dribbling_phase_graph_alignment",
        "dribbling_cg_demo_ball_tracking",
        "dribbling_cg_foot_ball_distance",
        "dribbling_cg_contact_consistency",
        "dribbling_cg_premature_contact",
        "dribbling_cg_foot_consistency",
        "dribbling_support_ankle_roll",
    )
    for reward_name in dribble_only_reward_names:
        if not hasattr(cfg.rewards, reward_name):
            continue
        term = getattr(cfg.rewards, reward_name)
        original_func = term.func
        original_params = dict(term.params or {})
        term.func = mdp.task_state_gated_reward
        term.params = {
            "reward_func": original_func,
            "reward_params": original_params,
            "command_name": "motion",
            "active_task_states": (mdp.TASK_STATE_DRIBBLE,),
        }

    # Do not pull linear/angular vel from demo bodies — velocity comes from the command only.
    if hasattr(cfg.rewards, "motion_body_lin_vel"):
        cfg.rewards.motion_body_lin_vel.weight = 0.0
    if hasattr(cfg.rewards, "motion_body_ang_vel"):
        cfg.rewards.motion_body_ang_vel.weight = 0.0

    # The reference is a style phase, not a task tape.  Keep a gentle whole
    # body posture prior, but soften it in control so a phase wrap and an
    # actual turn cannot produce large wrist/arm corrections.  This does not
    # touch the CG foot/contact rewards (which are independent of the arms).
    if hasattr(cfg.rewards, "motion_body_pos"):
        cfg.rewards.motion_body_pos.weight = 0.45
    if hasattr(cfg.rewards, "motion_body_ori"):
        cfg.rewards.motion_body_ori.weight = 0.35

    # The policy retains the v3.9 29-D joint-action interface.  Upper-body
    # targets are constrained and then projected onto the motion-bank PCA
    # manifold; lower-body actions pass through unchanged.
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
        orthogonal_residual_limit=0.10,
        cutoff_frequency_hz=1.8,
        # Keep the control policy inside the current style pose's local arm
        # envelope.  This is control-only: the base config defaults to None.
        # It prevents saturated PCA latents from locking an arm in a raised pose.
        reference_target_margin=0.25,
    )
    cfg.actions.joint_pos.scale = old_action_cfg.scale
    cfg.actions.joint_pos.offset = old_action_cfg.offset
    cfg.actions.joint_pos.preserve_order = getattr(old_action_cfg, "preserve_order", False)
    if hasattr(old_action_cfg, "clip"):
        cfg.actions.joint_pos.clip = old_action_cfg.clip

    # Feed back the effective joint command after manifold projection.
    cfg.observations.policy.actions.func = mdp.effective_joint_action
    cfg.observations.policy.actions.params = {"action_name": "joint_pos"}
    cfg.observations.critic.actions.func = mdp.effective_joint_action
    cfg.observations.critic.actions.params = {"action_name": "joint_pos"}
    cfg.rewards.action_rate_l2.func = mdp.effective_action_rate_l2_clip
    cfg.rewards.action_rate_l2.params = {"action_name": "joint_pos"}
    # Keep the light v3.9 pre-constraint overflow signal.  It is diagnostic and
    # trainable, but does not add another action transform.
    cfg.rewards.upper_body_reference_overflow = RewTerm(
        func=mdp.upper_body_reference_overflow_penalty,
        weight=-0.05,
        params={"action_name": "joint_pos"},
    )
    # The sole post-v3.9 intervention: make PCA null-space effort visible to
    # PPO without adding another action transform or reducing ball reach.
    cfg.rewards.upper_body_manifold_nullspace = RewTerm(
        func=mdp.upper_body_manifold_nullspace_penalty,
        weight=-0.02,
        params={"action_name": "joint_pos", "scale": 0.10},
    )

    cfg.observations.policy.motion_locomotion_polar_cmd = ObsTerm(
        func=mdp.motion_locomotion_polar_command,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_polar_cmd = ObsTerm(
        func=mdp.motion_locomotion_polar_command,
        params={"command_name": "motion"},
    )
    cfg.observations.policy.motion_locomotion_task_state = ObsTerm(
        func=mdp.motion_locomotion_task_state,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_task_state = ObsTerm(
        func=mdp.motion_locomotion_task_state,
        params={"command_name": "motion"},
    )


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


def _apply_unified_control_command_observations(cfg) -> None:
    """Use one polar locomotion command plus the explicit task-state input.

    The complete Stage-2 policy command is
    ``[speed, cos(heading), sin(heading), idle, dribble, stop]``.  Cartesian
    velocity terms are intentionally absent: they duplicate the same planar
    command and made checkpoint resume depend on an observation expansion.
    """
    cfg.observations.policy.motion_locomotion_polar_cmd = ObsTerm(
        func=mdp.motion_locomotion_polar_command,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_polar_cmd = ObsTerm(
        func=mdp.motion_locomotion_polar_command,
        params={"command_name": "motion"},
    )
    cfg.observations.policy.motion_locomotion_task_state = ObsTerm(
        func=mdp.motion_locomotion_task_state,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_task_state = ObsTerm(
        func=mdp.motion_locomotion_task_state,
        params={"command_name": "motion"},
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

    # G1FlatDribblingEnvCfg already assigns zero weight.  Removing the term
    # entirely prevents a later parent-config change from reintroducing a
    # kick-style objective into the frozen unified contract.
    if hasattr(cfg.rewards, "target_point_proximity"):
        cfg.rewards.target_point_proximity = None

    # These markers communicate kick-only targets and become misleading when
    # the associated observations/objective are absent.
    cfg.commands.motion.target_point_marker_cfg = None
    cfg.commands.motion.target_destination_marker_cfg = None


@configclass
class G1FlatCGDribblingControlEnvCfg(G1FlatCGDribblingEnvCfg):
    """CG Stage-2 **control**: preserved v4.4 continuous external-command recipe.

    Reference motion teaches **pose / gait / CG touch timing** only. Locomotion is driven by
    a sampled ``(speed, heading, duration)`` command — not ``anchor_lin_vel_w`` from the clip.
    Kept unchanged as a frozen verification baseline.

    Gym id: ``Tracking-CG-G1-Dribbling-RNN-control``.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_dribbling_control_velocity_terms(self)


@configclass
class G1FlatCGDribblingFullControlEnvCfg(G1FlatCGDribblingEnvCfg):
    """CG Stage-2 full-control: current stateful start/dribble/stop recipe.

    This freezes the pre-split Control behavior behind its own ID.  It adds an
    IDLE/DRIBBLE/STOP input, zero-speed state handling, state-gated rewards and
    terminations, and a STOP success condition.  Its policy and critic inputs
    grow by 12 dimensions from CG Stage-1: lin/ang velocity, polar command,
    and one-hot task state.

    Gym id: ``Tracking-CG-G1-Dribbling-RNN-full-control``.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_dribbling_full_control_velocity_terms(self)


@configclass
class G1FlatCGDribblingUnifiedControlEnvCfg(G1FlatCGDribblingEnvCfg):
    """Stateful CG Stage-2 control with the frozen polar-only interface.

    This is a sibling of the frozen ``full-control`` baseline.  It carries the
    same state machine but never creates legacy Cartesian command observations.
    """

    dribble_contact_mode: str = "right"
    # Current datasets only label contact foot.  Keep the optional instep
    # classifier disabled until a surface-labelled dataset is available.
    dribble_contact_surface: str = "any"

    def __post_init__(self):
        super().__post_init__()
        _apply_dribbling_full_control_velocity_terms(
            self,
            include_cartesian_command_observations=False,
        )
        _remove_unified_kick_only_terms(self)
        _apply_unified_control_action_contract(self)
        # Always enforce the configured foot.  A non-``any`` surface activates
        # the additional label check only when a surface-labelled dataset is
        # deliberately supplied.
        setattr(self.commands.motion, "dribble_cg_fixed_touch_foot", self.dribble_contact_mode)
        setattr(
            self.commands.motion,
            "dribble_cg_fixed_touch_surface",
            None if self.dribble_contact_surface == "any" else self.dribble_contact_surface,
        )
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


def _apply_unified_stage1_command_inputs(cfg) -> None:
    """Make pure imitation expose the exact Stage-2 task-command layout.

    Stage-1 does not train a locomotion task: it observes a fixed zero-speed,
    zero-heading polar command ``[0, 1, 0]`` and the IDLE one-hot default.  The
    values are explicit and stable, while the tensor layout is identical to
    the Stage-2 unified-control environment, so resume has no padded inputs.
    """
    cfg.commands.motion.locomotion_command_mode = "manual"
    cfg.commands.motion.locomotion_manual_lin_vel = (0.0, 0.0, 0.0)
    cfg.commands.motion.locomotion_manual_ang_vel = (0.0, 0.0, 0.0)
    cfg.commands.motion.locomotion_task_state_enabled = True
    cfg.commands.motion.locomotion_task_state_sequence = ("idle",)
    _apply_unified_control_command_observations(cfg)


@configclass
class G1FlatMotionCGPretrainUnifiedMimicEnvCfg(G1FlatMotionPretrainEnvCfg):
    """Pure-mimic Stage-1 with the frozen unified Stage-2 input interface.

    Gym id: ``Tracking-CG-G1-Motion-RNN-unified-mimic``.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_cg_pretrain_obs(self)
        _remove_unified_kick_only_terms(self)
        _apply_unified_stage1_command_inputs(self)


@configclass
class G1FlatMotionCGPretrainStrictEnvCfg(G1FlatMotionStrictPretrainEnvCfg):
    """Strict CG Stage-1 (``Tracking-CG-G1-Motion-RNN-strict``).

    Tracks the raw demonstration root path, yaw, pose and velocity without the
    task-frame transform or reset/domain perturbations used by the other Stage-1
    recipes.  ``anchor_ball_polar`` is retained on both observation groups so its
    policy layout matches the other CG Stage-1 environments.  Stage-2 adds command
    inputs on resume; only full-control additionally adds the task-state input.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_cg_pretrain_obs(self)


@configclass
class G1FlatMotionCGPretrainMimicEnvCfg(G1FlatMotionPretrainEnvCfg):
    """Legacy Stage-1 CG mimic pretrain.

    This keeps the historical observation layout for frozen ``control`` and
    ``full-control`` comparisons.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_cg_pretrain_obs(self)
