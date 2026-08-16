"""Active Stage-2 G1 dribbling control configuration."""

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp.commands_dribble_cg import DribbleCGMotionCommand

from .soccer_flat_env_cfg import G1_BALL_CONTACT_BODY_NAMES, G1FlatMotionEnvCfg, SOCCER_BALL_RADIUS


_CONTROL_TRACK_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "right_hip_roll_link",
    "right_knee_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

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
class G1FlatCGDribblingControlEnvCfg(G1FlatMotionEnvCfg):
    """Active Stage-2 speed/heading/duration dribbling controller."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.physics_material = self.scene.terrain.physics_material.replace(
            restitution=0.22,
            static_friction=1.0,
            dynamic_friction=0.95,
        )
        self.scene.soccer_ball.spawn = self.scene.soccer_ball.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.18,
                angular_damping=0.18,
            ),
        )

        # CG provides pose/touch style; locomotion comes from the control command.
        self.commands.motion.class_type = DribbleCGMotionCommand
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.mimic_align_task_frame = True
        self.commands.motion.mimic_align_locomotion_heading = True
        self.commands.motion.motion_clip_end_resample = False
        self.commands.motion.curve_offset_range = {
            "radius": (-0.25, 0.25),
            "lateral_spawn_jitter": 0.12,
            "height": SOCCER_BALL_RADIUS,
        }
        setattr(self.commands.motion, "dribble_cg_use_demo_ball", False)
        setattr(self.commands.motion, "dribble_cg_use_task_frame", True)
        setattr(self.commands.motion, "dribble_cg_fallback_ball_mode", "front")
        setattr(self.commands.motion, "dribble_cg_front_ball_distance", 0.45)
        setattr(self.commands.motion, "dribble_cg_front_ball_lateral_offset", 0.0)
        setattr(self.commands.motion, "dribble_cg_front_ball_height", SOCCER_BALL_RADIUS)

        self.commands.motion.locomotion_command_mode = "resampled"
        self.commands.motion.locomotion_cmd_speed_range = (0.40, 1.50)
        self.commands.motion.locomotion_cmd_heading_range = (-0.75, 0.75)
        self.commands.motion.locomotion_cmd_duration_range = (3.0, 6.0)
        self.commands.motion.locomotion_cmd_wz_range = (0.0, 0.0)
        self.commands.motion.locomotion_cmd_smoothing_enabled = True
        self.commands.motion.locomotion_cmd_heading_rate_limit = 0.85
        self.commands.motion.locomotion_cmd_accel_limit = 1.4
        self.commands.motion.locomotion_cmd_decel_limit = 2.4
        self.commands.motion.locomotion_cmd_turn_slowdown_angle = 0.55
        self.commands.motion.locomotion_cmd_turn_min_speed_scale = 0.60

        # The active motion bank contains right-foot ball contacts only.  Both
        # feet remain in gait/mimic tracking, but only the right ankle is a
        # legal ball-contact link; a left-foot touch is an undesired contact.
        legal_ankles = ["right_ankle_roll_link"]
        both_feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_cfg = SceneEntityCfg("robot", body_names=both_feet)
        other_contact_bodies = [
            body_name for body_name in G1_BALL_CONTACT_BODY_NAMES if body_name not in legal_ankles
        ]
        contact_body_cfg = SceneEntityCfg(
            "robot",
            body_names=[*legal_ankles, *other_contact_bodies],
        )
        num_ankle_links = len(legal_ankles)

        # Stage-2 keeps position style only; its orientation contribution was 0.17%.
        self.rewards.motion_body_ori = None
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=0.45,
            params={
                "command_name": "motion",
                "std": 0.3,
                "body_names": _CONTROL_TRACK_BODY_NAMES,
            },
        )
        self.rewards.motion_foot_pos = RewTerm(
            func=mdp.motion_relative_foot_position_error_exp,
            weight=0.55,
            params={
                "command_name": "motion",
                "std": 0.3,
                "foot_body_names": both_feet,
            },
        )
        self.rewards.foot_distance = RewTerm(
            func=mdp.foot_distance,
            weight=0.35,
            params={"threshold": 0.24, "std": 0.5, "foot_cfg": foot_cfg},
        )
        self.rewards.motion_anchor_lin_vel = RewTerm(
            func=mdp.motion_anchor_lin_vel_tracking_exp,
            weight=5.0,
            params={"command_name": "motion", "std": 0.8},
        )
        self.rewards.motion_anchor_ang_vel = RewTerm(
            func=mdp.motion_anchor_ang_vel_tracking_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 1.0,
                "heading_error_gain": 2.0,
                "max_yaw_rate": 1.2,
            },
        )
        self.rewards.locomotion_heading_tracking = RewTerm(
            func=mdp.locomotion_heading_tracking_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.45, "min_command_speed": 0.25},
        )
        self.rewards.dribbling_face_ball = RewTerm(
            func=mdp.dribbling_command_face_ball,
            weight=1.0,
            params={"command_name": "motion", "min_distance": 0.05},
        )
        self.rewards.dribbling_dynamic_proximity = RewTerm(
            func=mdp.dribbling_command_dynamic_proximity,
            weight=3.0,
            params={
                "command_name": "motion",
                "near_dist": 0.28,
                "far_dist": 0.72,
                "penalty_std": 0.15,
                "pelvis_speed_min": 0.35,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "no_contact_zone_damping": 0.12,
                "zone_lateral_abs_max": 0.14,
            },
        )
        self.rewards.dribbling_ball_too_close_penalty = RewTerm(
            func=mdp.dribbling_ball_too_close_penalty,
            weight=-8.0,
            params={
                "command_name": "motion",
                "min_forward_dist": 0.28,
                "full_penalty_dist": 0.14,
            },
        )
        self.rewards.dribbling_pelvis_quat_tracking = RewTerm(
            func=mdp.dribbling_pelvis_quat_tracking_exp,
            weight=2.0,
            params={"command_name": "motion", "std": 0.45},
        )
        self.rewards.dribbling_ball_speed_excess = RewTerm(
            func=mdp.dribbling_ball_xy_speed_excess_penalty,
            weight=-2.5,
            params={"speed_cap": 1.35, "linear_scale": 1.2},
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
                "recent_contact_grace_steps": 8,
            },
        )
        self.rewards.dribbling_sustained_contact_penalty = RewTerm(
            func=mdp.dribbling_sustained_contact_penalty,
            weight=-6.0,
            params={
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "ema_window_steps": 20,
                "duty_threshold": 0.25,
                "full_penalty_duty": 0.60,
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
            func=mdp.dribbling_command_ball_progress_reward,
            weight=7.5,
            params={
                "command_name": "motion",
                "min_forward_speed": 0.42,
                "command_speed_ratio": 0.50,
                "speed_scale": 0.22,
                "lateral_ratio_max": 0.70,
                "pelvis_speed_min": 0.25,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
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
                "foot_body_names": both_feet,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
            },
        )
        self.rewards.dribbling_chase_ball = RewTerm(
            func=mdp.dribbling_command_chase_ball_reward,
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
                "min_steps_between_touches": 26,
                "all_body_cfg": contact_body_cfg,
                "num_ankle_links": num_ankle_links,
            },
        )
        self.rewards.dribbling_legal_foot_touch = RewTerm(
            func=mdp.dribbling_legal_foot_touch,
            weight=5.5,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 14.0,
                "all_body_cfg": contact_body_cfg,
                "num_ankle_links": num_ankle_links,
                "min_pelvis_heading": 0.0,
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
                "all_body_cfg": contact_body_cfg,
                "num_ankle_links": num_ankle_links,
            },
        )
        self.rewards.dribbling_undesired_contact_penalty = RewTerm(
            func=mdp.dribbling_undesired_contact_penalty,
            weight=-12.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "all_body_cfg": contact_body_cfg,
                "num_ankle_links": num_ankle_links,
            },
        )
        self.rewards.dribbling_cg_foot_ball_distance = RewTerm(
            func=mdp.dribbling_cg_foot_ball_distance_exp,
            weight=3.5,
            params={"command_name": "motion", "std": 0.22, "use_xy_only": True},
        )
        self.rewards.dribbling_cg_contact_consistency = RewTerm(
            func=mdp.dribbling_cg_contact_consistency,
            weight=1.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "contact_window_tolerance_steps": 2,
                "premature_contact_penalty": 1.0,
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
                "recovery_window_steps": 75,
                "recovery_max_distance": 0.85,
                "recovery_min_closing_speed": 0.05,
                "recovery_counter_increment": 0.25,
            },
        )
        self.terminations.locomotion_manual_sequence_end = DoneTerm(
            func=mdp.locomotion_manual_sequence_finished,
            params={"command_name": "motion"},
        )

        # Preserve the 29-D policy interface while constraining upper-body targets.
        old_action_cfg = self.actions.joint_pos
        self.actions.joint_pos = mdp.UpperBodyManifoldJointPositionActionCfg(
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
            reference_target_margin=0.25,
        )
        self.actions.joint_pos.scale = old_action_cfg.scale
        self.actions.joint_pos.offset = old_action_cfg.offset
        self.actions.joint_pos.preserve_order = getattr(old_action_cfg, "preserve_order", False)
        if hasattr(old_action_cfg, "clip"):
            self.actions.joint_pos.clip = old_action_cfg.clip

        self.rewards.action_rate_l2.func = mdp.effective_action_rate_l2_clip
        self.rewards.action_rate_l2.params = {"action_name": "joint_pos"}
        self.rewards.upper_body_reference_overflow = RewTerm(
            func=mdp.upper_body_reference_overflow_penalty,
            weight=-0.05,
            params={"action_name": "joint_pos"},
        )
