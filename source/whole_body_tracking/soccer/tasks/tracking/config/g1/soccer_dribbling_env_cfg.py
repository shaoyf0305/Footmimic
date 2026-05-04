"""Dribbling environment configurations for the G1 robot.

Inherits proximity-level tracking and adds dribbling-specific rewards:
  - gated velocity / proximity (anti static exploit)
  - dense foot–ball approach when not in contact
  - pelvis orientation vs motion reference (anti lean-back / arched torso)
  - ball horizontal speed excess penalty
  - ankle-based gentle touch / hard-hit EMA / non-ankle contact penalty (no ``kick_leg``)
  - ``ball_lost`` and ``dribbling_no_contact`` terminations
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from soccer.tasks.tracking import mdp
from .soccer_flat_env_cfg import G1FlatProximityEnvCfg


@configclass
class G1FlatDribblingEnvCfg(G1FlatProximityEnvCfg):
    """Flat-ground dribbling environment."""

    def __post_init__(self):
        super().__post_init__()

        # Stronger upright / anti-arch than generic proximity alone
        if hasattr(self.rewards, "pelvis_orientation"):
            self.rewards.pelvis_orientation.weight = -2.5

        _foot_cfg = SceneEntityCfg(
            "robot",
            body_names=["right_ankle_roll_link", "left_ankle_roll_link"],
        )

        # Ankles **must** be the first ``num_ankle_links`` entries for contact logic.
        _contact_body_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                "right_ankle_roll_link",
                "left_ankle_roll_link",
                "right_knee_link",
                "left_knee_link",
                "right_wrist_yaw_link",
                "left_wrist_yaw_link",
            ],
        )
        _num_ankle_links = 2

        self.rewards.dribbling_velocity_tracking = RewTerm(
            func=mdp.dribbling_velocity_tracking,
            weight=5.0,
            params={
                "command_name": "motion",
                "std": 1.0,
                "pelvis_speed_min": 0.12,
                "ball_speed_min": 0.0,
                "require_contact": False,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
            },
        )

        self.rewards.dribbling_dynamic_proximity = RewTerm(
            func=mdp.dribbling_dynamic_proximity,
            weight=5.0,
            params={
                "command_name": "motion",
                "near_dist": 0.2,
                "far_dist": 0.5,
                "penalty_std": 0.15,
                "pelvis_speed_min": 0.12,
            },
        )

        self.rewards.dribbling_approach_foot_ball = RewTerm(
            func=mdp.dribbling_approach_foot_ball_distance,
            weight=5.0,
            params={
                "command_name": "motion",
                "foot_cfg": _foot_cfg,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "std": 0.22,
                "pelvis_speed_min": 0.08,
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
            weight=-1.5,
            params={
                "speed_cap": 4.2,
                "linear_scale": 1.5,
            },
        )

        self.rewards.dribbling_orbiting_penalty = RewTerm(
            func=mdp.dribbling_orbiting_penalty,
            weight=-4.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "orbit_radius_max": 0.9,
                "tangential_deadzone": 0.08,
                "tangential_scale": 0.35,
            },
        )

        self.rewards.dribbling_legal_foot_touch = RewTerm(
            func=mdp.dribbling_legal_foot_touch,
            weight=5.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 22.0,
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
            },
        )

        self.rewards.dribbling_micro_contact_filter = RewTerm(
            func=mdp.dribbling_micro_contact_filter,
            weight=-4.5,
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
                "max_steps_without_contact": 75,
            },
        )


@configclass
class G1TerrainDribblingAnkleDisturbEnvCfg(G1FlatDribblingEnvCfg):
    """Flat dribbling with ankle disturbances (stage-1 style pretrain for dribble)."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.motion_foot_pos.weight = 0.0

        self.events.ankle_torque_disturbance = EventTerm(
            func=mdp.apply_random_ankle_torque,
            mode="interval",
            interval_range_s=(0.1, 0.3),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_ankle_pitch_joint",
                        "left_ankle_roll_joint",
                        "right_ankle_pitch_joint",
                        "right_ankle_roll_joint",
                    ],
                ),
                "torque_range": (-15.0, 15.0),
            },
        )
