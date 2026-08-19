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
        # Reset into a collision-safe command-aligned scene.  The previous
        # inherited +/-0.25 m radius range could place a ground ball inside a
        # randomized leg pose, and a failure during a turn respawned the ball
        # along task +X instead of the active command heading.
        self.commands.motion.soccer_ball_start_ahead_distance = 0.53
        self.commands.motion.soccer_ball_spawn_align_locomotion_heading = True
        self.commands.motion.reset_face_locomotion_heading = True
        self.commands.motion.curve_offset_range = {
            "radius": (-0.05, 0.05),
            "lateral_spawn_jitter": 0.08,
            "height": SOCCER_BALL_RADIUS,
        }
        setattr(self.commands.motion, "dribble_cg_use_demo_ball", False)
        setattr(self.commands.motion, "dribble_cg_use_task_frame", True)
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

        # One physical position--velocity state drives normal control and
        # recovery.  Keep these parameters identical for every closed-loop
        # reward so the cached recovery gate is evaluated exactly once per
        # control step.
        closed_loop_params = {
            "desired_forward_offset": 0.45,
            "prediction_horizon": 0.20,
            "position_forward_std": 0.22,
            "position_lateral_std": 0.18,
            "recovery_forward_half_width": 0.16,
            "recovery_lateral_half_width": 0.12,
            "recovery_transition_width": 0.16,
            "recovery_gate_filter_steps": 4,
            "position_correction_gain": 1.5,
            "max_position_correction_speed": 0.45,
            "max_recovery_target_speed": 2.20,
            "velocity_ema_window_steps": 10,
        }

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
            func=mdp.dribbling_closed_loop_pelvis_velocity_tracking_exp,
            weight=5.0,
            params={"command_name": "motion", "std": 0.8, **closed_loop_params},
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
        self.rewards.dribbling_dynamic_proximity = RewTerm(
            func=mdp.dribbling_command_dynamic_proximity,
            weight=3.0,
            params={
                "command_name": "motion",
                **closed_loop_params,
            },
        )
        self.rewards.dribbling_ball_too_close_penalty = RewTerm(
            func=mdp.dribbling_ball_too_close_penalty,
            weight=-8.0,
            params={
                "command_name": "motion",
                "min_xy_dist": 0.28,
                "full_penalty_dist": 0.14,
            },
        )
        self.rewards.dribbling_pelvis_quat_tracking = RewTerm(
            func=mdp.dribbling_pelvis_quat_tracking_exp,
            weight=2.0,
            params={"command_name": "motion", "std": 0.45},
        )
        self.rewards.dribbling_ball_velocity_tracking = RewTerm(
            func=mdp.dribbling_command_ball_velocity_tracking_reward,
            weight=7.5,
            params={
                "command_name": "motion",
                "absolute_tolerance": 0.10,
                "relative_tolerance": 0.05,
                "speed_error_std": 0.30,
                "lateral_speed_std": 0.35,
                "minimum_controllability_gate": 0.10,
                **closed_loop_params,
            },
        )
        self.rewards.dribbling_ball_speed_excess = RewTerm(
            func=mdp.dribbling_ball_xy_speed_excess_penalty,
            weight=-2.5,
            params={
                "command_name": "motion",
                "speed_margin": 0.20,
                "min_speed_cap": 0.35,
                "huber_scale": 0.45,
                "max_penalty": 6.0,
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
        self.rewards.dribbling_rapid_retouch_penalty = RewTerm(
            func=mdp.dribbling_rapid_retouch_penalty,
            # Event-normalized: one too-fast retouch contributes -1.0 return.
            weight=-1.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 14.0,
                "min_steps_between_touches": 14,
                "all_body_cfg": contact_body_cfg,
                "num_ankle_links": num_ankle_links,
            },
        )
        self.rewards.dribbling_useful_foot_touch = RewTerm(
            func=mdp.dribbling_useful_foot_touch,
            # Event-normalized: one fully useful touch contributes +1.0 return.
            weight=1.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 14.0,
                "all_body_cfg": contact_body_cfg,
                "num_ankle_links": num_ankle_links,
                "evaluation_delay_steps": 5,
                # Give a neutral gentle touch immediate credit while reserving
                # most of the unchanged unit score budget for better control.
                "base_touch_score": 0.25,
                "improvement_scale": 0.50,
                "contact_window_tolerance_steps": 2,
                "off_window_reward_scale": 0.35,
                **closed_loop_params,
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
            params={
                "command_name": "motion",
                "std": 0.22,
                "use_xy_only": True,
                "max_reference_distance": 0.55,
            },
        )

        self.terminations.ball_lost = DoneTerm(
            func=mdp.ball_lost_dribbling,
            params={
                "command_name": "motion",
                "max_distance": 1.0,
                "max_vel_divergence": 2.0,
                "grace_steps": 50,
                "active_task_states": (mdp.TASK_STATE_DRIBBLE,),
            },
        )
        self.terminations.dribbling_no_contact = DoneTerm(
            func=mdp.dribbling_no_ball_contact_timeout,
            params={
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "contact_body_names": ("right_ankle_roll_link",),
                "active_task_states": (mdp.TASK_STATE_DRIBBLE,),
                "grace_steps": 50,
                "max_steps_without_contact": 50,
                "recovery_window_steps": 75,
                "recovery_max_distance": 0.85,
                "recovery_min_closing_speed": 0.05,
                "recovery_counter_increment": 0.25,
                # A controllable ball gets two seconds to make a light touch;
                # the counter still grows and can never be restored by coasting.
                "proximity_recovery_max_distance": 0.75,
                "proximity_recovery_max_relative_speed": 0.9,
                "proximity_recovery_counter_increment": 0.5,
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
            reference_relative_upper_body_residual=True,
            upper_body_policy_action_is_bounded=True,
        )
        self.actions.joint_pos.scale = old_action_cfg.scale
        self.actions.joint_pos.offset = old_action_cfg.offset
        self.actions.joint_pos.preserve_order = getattr(old_action_cfg, "preserve_order", False)
        if hasattr(old_action_cfg, "clip"):
            self.actions.joint_pos.clip = old_action_cfg.clip

        self.rewards.action_rate_l2.func = mdp.effective_action_rate_l2_clip
        self.rewards.action_rate_l2.params = {"action_name": "joint_pos"}
