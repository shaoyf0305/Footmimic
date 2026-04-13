"""Dribbling environment configurations for the G1 robot.

This module defines standalone dribbling task environments that inherit
from the proximity-level configuration but replace all kick-specific
rewards with dribbling-specific ones:
  - velocity tracking (ball velocity aligned with pelvis velocity)
  - dynamic proximity (ball kept in a [0.2m, 0.5m] safe zone ahead)
  - micro-contact filter (penalise foot-ball impacts > 20 N)
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from soccer.tasks.tracking import mdp
from .soccer_flat_env_cfg import G1FlatProximityEnvCfg


@configclass
class G1FlatDribblingEnvCfg(G1FlatProximityEnvCfg):
    """Flat-ground dribbling environment.

    Inherits the scene (soccer ball, contact sensors), motion tracking command,
    and basic locomotion / proximity rewards from ``G1FlatProximityEnvCfg``.
    On top of that, three dribbling-specific reward terms are added, and a
    ``ball_lost`` termination prevents the robot from ignoring the ball.
    """

    def __post_init__(self):
        super().__post_init__()

        # ── Dribbling Rewards ────────────────────────────────────────

        # 1) Velocity Tracking — ball vel should match pelvis vel
        self.rewards.dribbling_velocity_tracking = RewTerm(
            func=mdp.dribbling_velocity_tracking,
            weight=5.0,
            params={
                "command_name": "motion",
                "std": 1.0,
            },
        )

        # 2) Dynamic Proximity — ball stays in safe zone [0.2m, 0.5m]
        self.rewards.dribbling_dynamic_proximity = RewTerm(
            func=mdp.dribbling_dynamic_proximity,
            weight=5.0,
            params={
                "command_name": "motion",
                "near_dist": 0.2,
                "far_dist": 0.5,
                "penalty_std": 0.15,
            },
        )

        # ── Contact Body Whitelist ────────────────────────────────────
        # Shared body cfg listing all candidate bodies for proximity-based
        # identification of which body caused the ball contact.
        _contact_body_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                "right_ankle_roll_link",   # index 0 — the ONLY legal dribbling foot
                "left_ankle_roll_link",     # index 1 — forbidden
                "right_knee_link",          # index 2 — forbidden
                "left_knee_link",           # index 3 — forbidden
                "right_wrist_yaw_link",     # index 4 — forbidden
                "left_wrist_yaw_link",      # index 5 — forbidden
            ],
        )

        # 3a) Legal Foot Gentle Touch — positive reward for correct foot
        self.rewards.dribbling_legal_foot_touch = RewTerm(
            func=mdp.dribbling_legal_foot_touch,
            weight=2.0,  # positive reward for correct gentle touch
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 20.0,
                "all_body_cfg": _contact_body_cfg,
            },
        )

        # 3b) Micro-Contact Filter — moderate EMA penalty for legal foot hard kicks
        self.rewards.dribbling_micro_contact_filter = RewTerm(
            func=mdp.dribbling_micro_contact_filter,
            weight=-5.0,  # moderate penalty for hard kicks with correct foot
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 20.0,
                "max_penalty": 2.0,
                "ema_alpha": 0.4,
                "all_body_cfg": _contact_body_cfg,
            },
        )

        # 3c) Undesired Contact — severe penalty for wrong body touching ball
        self.rewards.dribbling_undesired_contact_penalty = RewTerm(
            func=mdp.dribbling_undesired_contact_penalty,
            weight=-10.0,  # severe instant penalty
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "all_body_cfg": _contact_body_cfg,
            },
        )

        # ── Ball Lost Termination ────────────────────────────────────
        self.terminations.ball_lost = DoneTerm(
            func=mdp.ball_lost_dribbling,
            params={
                "command_name": "motion",
                "max_distance": 1.0,          # ball > 1m away → episode over
                "max_vel_divergence": 2.0,     # ball-pelvis vel diff > 2 m/s → over
                "grace_steps": 50,             # 1 second warm-up (50 Hz)
            },
        )


@configclass
class G1TerrainDribblingAnkleDisturbEnvCfg(G1FlatDribblingEnvCfg):
    """Stage 1 terrain locomotion with ankle disturbances for dribbling.

    Key differences from vanilla terrain training:
    1. **Ankle tracking reward zeroed** — ``motion_foot_pos`` weight = 0
       so the robot doesn't try to imitate noisy ankle trajectories.
    2. **Random ankle torques** — extreme random torques are injected into
       both ankle pitch/roll joints, forcing the robot to stabilise through
       its trunk core and support leg rather than relying on ankle precision.
    3. **Terrain** — uses rough terrain from ``G1TerrainEnvCfg`` for
       additional balance robustness.
    """

    def __post_init__(self):
        super().__post_init__()

        # ── Zero out ankle foot tracking reward ─────────────────────
        self.rewards.motion_foot_pos.weight = 0.0

        # ── Inject random ankle torque disturbances ─────────────────
        self.events.ankle_torque_disturbance = EventTerm(
            func=mdp.apply_random_ankle_torque,
            mode="interval",
            interval_range_s=(0.1, 0.3),  # high frequency disturbances
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
                "torque_range": (-15.0, 15.0),  # extreme random torques
            },
        )
