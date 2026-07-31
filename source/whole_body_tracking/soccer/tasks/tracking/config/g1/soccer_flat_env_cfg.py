import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg

from soccer.assets import ASSET_DIR
from soccer.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, MySceneCfg, CurriculumCfg
from .flat_env_cfg import G1FlatEnvCfg

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains import TerrainGeneratorCfg

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from isaaclab.managers import TerminationTermCfg as DoneTerm

SOCCER_BALL_RADIUS = 0.11

SOCCER_ASSET_PATH = f"{ASSET_DIR}/soccer/soccer.usda"


def _apply_soccer_obs(cfg):
    cfg.observations.policy.target_point_pos = ObsTerm(
        func=mdp.constant_target_point_pos,
        params={"command_name": "motion"},
    )

    cfg.observations.critic.target_point_pos = ObsTerm(
        func=mdp.constant_target_point_pos,
        params={"command_name": "motion"},
    )

    cfg.observations.policy.target_destination_pos_local = ObsTerm(
        func=mdp.target_destination_pos_local,
        params={"command_name": "motion"},
    )

    cfg.observations.critic.target_destination_pos_local = ObsTerm(
        func=mdp.target_destination_pos_local,
        params={"command_name": "motion"},
    )


def _apply_soccer_scene(cfg):
    cfg.scene.soccer_ball = cfg.scene.soccer_ball.replace(prim_path="{ENV_REGEX_NS}/SoccerBall")
    cfg.scene.soccer_ball.init_state.pos = (0.0, 0.0, SOCCER_BALL_RADIUS)

    cfg.commands.motion.target_point_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/TargetPoint",
        markers={
            "target_sphere": sim_utils.SphereCfg(
                radius=0.11,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    cfg.commands.motion.target_destination_marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/PostKickTarget",
        markers={
            "destination_sphere": sim_utils.SphereCfg(
                radius=0.11,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )

## Scene configuration

@configclass
class G1FlatSoccerSceneCfg(MySceneCfg):
    def __post_init__(self):
        super().__post_init__()
        # Keep parent terrain material settings and explicitly set restitution.
        self.terrain.physics_material = self.terrain.physics_material.replace(restitution=0.8)

    soccer_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCCER_ASSET_PATH,
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7, 0.0, SOCCER_BALL_RADIUS),
        ),
    )
    soccer_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        history_length=3,
        track_air_time=False,
        force_threshold=0.0,
        debug_vis=False,
    )
    

## Environment configuration

@configclass
class G1TerrainEnvCfg(G1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = mdp.commands_multi_motion_soccer.MotionCommand
        self.terminations.anchor_pos_z = DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25},  # Slightly larger threshold for robustness.
        )
        self.terminations.anchor_ori = DoneTerm(
            func=mdp.bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
        )
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25, # 0.75, # 0.25,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
            curriculum=False,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1., noise_range=(-0.02, 0.02), noise_step=0.02, border_width=0.0
                )
            },
        )

        # ground terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=GRAVEL_TERRAINS_CFG
        )


@configclass
class G1TerrainMotionEnvCfg(G1TerrainEnvCfg):
    scene: G1FlatSoccerSceneCfg = G1FlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.sampling_strategy = "adaptive"
        _apply_soccer_obs(self)
        _apply_soccer_scene(self)


@configclass
class G1PAiDOriginalStage1EnvCfg(G1TerrainMotionEnvCfg):
    """Released PAiD Stage-I motion-skill acquisition environment.

    This deliberately retains the original 160-D actor / 292-D critic layout:
    Cartesian ball and goal observations are present, but no later CG polar
    observation is appended.  It also keeps torso anchoring, yaw-only reference
    alignment, motion-phase adaptive sampling, rough terrain, reset noise, and
    the original domain randomization and full-body tracking objective.
    """

    def __post_init__(self):
        super().__post_init__()

        # Freeze the PAiD Stage-I command semantics instead of inheriting newer
        # flat/task-frame pretrain defaults.
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.mimic_align_task_frame = False
        self.commands.motion.mimic_align_locomotion_heading = False
        self.commands.motion.motion_clip_end_resample = True
        self.commands.motion.sampling_strategy = "adaptive"
        self.commands.motion.soccer_ball_spawn_mode = "paid_original"
        self.commands.motion.reset_face_task_forward = False
        self.commands.motion.reset_zero_velocity = False


# Stage-1 body groups: legs/torso vs arms (upper-body ori is de-emphasised under task-frame slalom).
_STAGE1_LOCOMOTION_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "right_hip_roll_link",
    "right_knee_link",
    "torso_link",
]
_STAGE1_UPPER_BODY_NAMES = [
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
_STAGE1_LEG_VEL_BODY_NAMES = [
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
]


def _apply_stage1_strict_mimic_pretrain(cfg) -> None:
    """Stage-1 raw-reference motion tracking with no task-frame or reset perturbations.

    This is the strict replay recipe: reward all of the original global anchor,
    relative-body pose, and global-body velocity terms in the demonstration frame.
    In particular, a clip's root path and yaw evolution are preserved; this helper
    deliberately does *not* strip yaw into the task +X frame.

    The standard action/limit/contact regularizers remain so the reference is
    reproduced by a physically feasible robot rather than by an unconstrained pose
    player.  Domain randomization and reset noise are disabled here because they
    would make the initial robot state differ from the reference being imitated.
    """
    cfg.commands.motion.anchor_body_name = "pelvis"
    cfg.commands.motion.mimic_align_task_frame = False
    cfg.commands.motion.mimic_align_locomotion_heading = False

    # Restore the full tracking objective from TrackingEnvCfg explicitly.  Keeping
    # these assignments here makes strict independent of changes to sibling recipes.
    if hasattr(cfg.rewards, "motion_global_anchor_pos"):
        cfg.rewards.motion_global_anchor_pos.weight = 1.0
    if hasattr(cfg.rewards, "motion_global_anchor_ori"):
        cfg.rewards.motion_global_anchor_ori.weight = 1.0
    if hasattr(cfg.rewards, "motion_body_pos"):
        cfg.rewards.motion_body_pos.weight = 1.0
        cfg.rewards.motion_body_pos.params.pop("body_names", None)
    if hasattr(cfg.rewards, "motion_body_ori"):
        cfg.rewards.motion_body_ori.weight = 1.0
        cfg.rewards.motion_body_ori.params.pop("body_names", None)
    if hasattr(cfg.rewards, "motion_body_lin_vel"):
        cfg.rewards.motion_body_lin_vel.weight = 1.0
    if hasattr(cfg.rewards, "motion_body_ang_vel"):
        cfg.rewards.motion_body_ang_vel.weight = 1.0

    # Reset to the exact reference state.  G1FlatMotionEnvCfg enables these
    # robustness-oriented settings for its task-oriented children, so strict must
    # override every one of them after ``super().__post_init__``.
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
    cfg.commands.motion.curve_offset_range = {
        "radius": (0.0, 0.0),
        "lateral_spawn_jitter": 0.0,
        "height": SOCCER_BALL_RADIUS,
    }

    # No random material/COM/default-joint offsets or interval pushes in strict
    # Stage-1.  The ball remains in the scene solely to preserve the CG observation
    # layout; no ball/task reward is introduced by this configuration.
    cfg.events.physics_material = None
    cfg.events.add_joint_default_pos = None
    cfg.events.base_com = None
    cfg.events.push_robot = None


def _apply_stage1_mimic_pretrain(cfg) -> None:
    """Stage-1 dribble pretrain: layered mimic + demo-root velocity (no lateral/heading task terms).

    Matches Stage-2 anchor convention (pelvis + task-frame yaw strip). Locomotion follows
    the reference anchor velocity (including slalom lateral); upper-body orientation is
    soft so arms are not twisted by task-frame slalom refs.
    """
    cfg.commands.motion.anchor_body_name = "pelvis"
    cfg.commands.motion.mimic_align_task_frame = True

    if hasattr(cfg.rewards, "motion_global_anchor_pos"):
        cfg.rewards.motion_global_anchor_pos.weight = 0.0
    if hasattr(cfg.rewards, "motion_global_anchor_ori"):
        cfg.rewards.motion_global_anchor_ori.weight = 0.0
    if hasattr(cfg.rewards, "motion_body_lin_vel"):
        cfg.rewards.motion_body_lin_vel.weight = 0.0
    if hasattr(cfg.rewards, "motion_body_ang_vel"):
        cfg.rewards.motion_body_ang_vel.weight = 0.0

    if hasattr(cfg.rewards, "motion_body_pos"):
        cfg.rewards.motion_body_pos.weight = 1.0
        cfg.rewards.motion_body_pos.params["body_names"] = _STAGE1_LOCOMOTION_BODY_NAMES
    if hasattr(cfg.rewards, "motion_body_ori"):
        cfg.rewards.motion_body_ori.weight = 1.0
        cfg.rewards.motion_body_ori.params["body_names"] = _STAGE1_LOCOMOTION_BODY_NAMES

    cfg.rewards.motion_upper_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.85,
        params={"command_name": "motion", "std": 0.3, "body_names": _STAGE1_UPPER_BODY_NAMES},
    )
    cfg.rewards.motion_upper_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.35,
        params={"command_name": "motion", "std": 0.4, "body_names": _STAGE1_UPPER_BODY_NAMES},
    )
    cfg.rewards.motion_leg_lin_vel = RewTerm(
        func=mdp.motion_relative_body_linear_velocity_error_exp,
        weight=0.4,
        params={"command_name": "motion", "std": 1.0, "body_names": _STAGE1_LEG_VEL_BODY_NAMES},
    )
    cfg.rewards.motion_leg_ang_vel = RewTerm(
        func=mdp.motion_relative_body_angular_velocity_error_exp,
        weight=0.4,
        params={"command_name": "motion", "std": 3.14, "body_names": _STAGE1_LEG_VEL_BODY_NAMES},
    )
    cfg.rewards.motion_anchor_lin_vel = RewTerm(
        func=mdp.motion_anchor_lin_vel_tracking_exp,
        weight=2.2,
        params={"command_name": "motion", "std": 1.0},
    )
    cfg.rewards.motion_anchor_pos_z = RewTerm(
        func=mdp.motion_anchor_pos_z_error_exp,
        weight=0.6,
        params={"command_name": "motion", "std": 0.15},
    )
    cfg.rewards.motion_foot_pos = RewTerm(
        func=mdp.motion_relative_foot_position_error_exp,
        weight=0.7,
        params={
            "command_name": "motion",
            "std": 0.3,
            "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        },
    )

    if hasattr(cfg.terminations, "ee_body_pos"):
        cfg.terminations.ee_body_pos.params["grace_steps_after_resample"] = 20
    if hasattr(cfg.terminations, "anchor_pos_z"):
        cfg.terminations.anchor_pos_z.params["threshold"] = 0.32


# Upper-body mimic set used by the task-frame Stage-1 variant (v2 CG pretrain style).
_STAGE1_TASK_UPPER_BODY_NAMES = [
    "pelvis",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


def _apply_stage1_task_pretrain(cfg) -> None:
    """Stage-1 pretrain with task +X velocity / heading (legacy v2 CG style).

    Keeps default full-body ``motion_body_pos`` from :class:`TrackingEnvCfg`, restricts
    orientation mimic to the upper body, and adds forward-speed / anti-lateral task terms
    instead of layered leg-vs-arm mimic splits.
    """
    cfg.commands.motion.anchor_body_name = "pelvis"
    cfg.commands.motion.mimic_align_task_frame = True

    if hasattr(cfg.rewards, "motion_global_anchor_pos"):
        cfg.rewards.motion_global_anchor_pos.weight = 0.0
    if hasattr(cfg.rewards, "motion_global_anchor_ori"):
        cfg.rewards.motion_global_anchor_ori.weight = 0.0

    if hasattr(cfg.rewards, "motion_body_ori"):
        cfg.rewards.motion_body_ori.params["body_names"] = _STAGE1_TASK_UPPER_BODY_NAMES

    cfg.rewards.motion_body_lin_vel = RewTerm(
        func=mdp.motion_relative_body_linear_velocity_error_exp,
        weight=0.3,
        params={
            "command_name": "motion",
            "std": 1.0,
            "body_names": _STAGE1_TASK_UPPER_BODY_NAMES,
        },
    )
    cfg.rewards.motion_body_ang_vel = RewTerm(
        func=mdp.motion_relative_body_angular_velocity_error_exp,
        weight=0.3,
        params={
            "command_name": "motion",
            "std": 3.14,
            "body_names": _STAGE1_TASK_UPPER_BODY_NAMES,
        },
    )

    cfg.rewards.forward_velocity = RewTerm(
        func=mdp.forward_velocity_reward,
        weight=3.0,
        params={
            "command_name": "motion",
            "target_speed": 0.8,
            "std": 0.4,
            "velocity_frame": "world",
            "min_forward_dominance": 0.55,
        },
    )
    cfg.rewards.lateral_velocity_penalty = RewTerm(
        func=mdp.lateral_velocity_penalty,
        weight=-0.8,
        params={
            "command_name": "motion",
            "velocity_frame": "world",
            "lateral_deadzone": 0.06,
            "lateral_scale": 0.28,
        },
    )
    cfg.rewards.task_heading_alignment = RewTerm(
        func=mdp.task_heading_alignment_reward,
        weight=1.5,
        params={"command_name": "motion"},
    )


@configclass
class G1FlatMotionEnvCfg(G1FlatEnvCfg):
    scene: G1FlatSoccerSceneCfg = G1FlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.class_type = mdp.commands_multi_motion_soccer.MotionCommand
        self.commands.motion.sampling_strategy = "uniform"
        # Flat motion + ball scene defaults (Stage-1/Stage-2 siblings override rewards separately).
        self.commands.motion.soccer_ball_spawn_mode = "start_ahead"
        self.commands.motion.soccer_ball_start_ahead_distance = 0.45
        self.commands.motion.reset_face_task_forward = True
        self.commands.motion.reset_zero_velocity = True
        self.commands.motion.curve_offset_range = {
            "radius": (0.0, 0.0),
            "lateral_spawn_jitter": 0.05,
            "height": SOCCER_BALL_RADIUS,
        }
        _apply_soccer_obs(self)
        _apply_soccer_scene(self)


@configclass
class G1FlatMotionPretrainEnvCfg(G1FlatMotionEnvCfg):
    """Stage-1 flat motion pretrain (``Tracking-Flat-G1-Motion-RNN-v0``).

    Sibling of :class:`G1FlatProximityEnvCfg` under :class:`G1FlatMotionEnvCfg`:
    shared scene/commands live on the base; mimic-pretrain rewards/terminations
    are applied here only.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_stage1_mimic_pretrain(self)


@configclass
class G1FlatMotionStrictPretrainEnvCfg(G1FlatMotionEnvCfg):
    """Stage-1 raw-reference tracking for ``*-Motion-RNN-strict``.

    This sibling of ``G1FlatMotionPretrainEnvCfg`` intentionally does not inherit its
    task-frame layered-mimic changes.  It is the dedicated strict Stage-1 recipe.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_stage1_strict_mimic_pretrain(self)


@configclass
class G1FlatMotionTaskPretrainEnvCfg(G1FlatMotionEnvCfg):
    """Stage-1 flat motion pretrain with task +X velocity / heading (``*-Motion-RNN-task``).

    Sibling of :class:`G1FlatMotionPretrainEnvCfg`: same scene/commands, but uses
    :func:`_apply_stage1_task_pretrain` instead of layered mimic-only rewards.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_stage1_task_pretrain(self)


@configclass
class G1FlatProximityEnvCfg(G1FlatMotionEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.foot_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        )

        self.waist_cfg = SceneEntityCfg(
            "robot",
            joint_names=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],
        )

        self.commands.motion.curve_offset_range = {
            "radius": (-0.25, 0.25),
            "lateral_spawn_jitter": 0.12,
            "height": SOCCER_BALL_RADIUS,
        }


        self.rewards.foot_distance = RewTerm(
            func=mdp.foot_distance,
            weight=0.2,
            params={
                "threshold": 0.24,
                "std": 0.5,
                "foot_cfg": self.foot_cfg,
            },
        )

        # self.rewards.feet_slip_penalty = RewTerm(
        #     func=mdp.feet_slip_penalty,
        #     weight=-1.0,
        #     params={
        #         "foot_cfg": self.foot_cfg,
        #         "slip_force_threshold": 5.0,
        #     },
        # )

        self.rewards.target_point_proximity = RewTerm(
            func=mdp.target_point_proximity,
            weight=1.0,
            params={
                "std": 4.0,
                "command_name": "motion",
            },
        )

        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.motion_global_anchor_position_error_exp,
            # weight=0.5,
            weight=0.0,
            params={"command_name": "motion", "std": 0.3},
        )

        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        )

        self.rewards.waist_action_rate_l2 = RewTerm(
            func=mdp.waist_action_rate_l2_clip,
            weight=-2.5e-1,
            params={
                "waist_cfg": self.waist_cfg,
            },
        )

        self.rewards.pelvis_orientation = RewTerm(
            func=mdp.pelvis_orientation,
            weight=-1.0,
            params={"command_name": "motion",},
        )

        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.3,
                "body_names" : [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    # "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    # "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        self.rewards.motion_body_ori = RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.4,
                "body_names": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    # "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    # "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        self.rewards.motion_foot_pos = RewTerm(
            func=mdp.motion_relative_foot_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3,
                    "foot_body_names" : [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
            },
        )




@configclass
class G1FlatKickEnvCfg(G1FlatProximityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Segmented Contact Graph: terminate if ball is touched outside the
        # annotated kick window.  DISABLED: ball placement is too close for
        # Phase 1 enforcement — robot can't avoid contact during approach.
        # Re-enable after implementing ball distance curriculum.
        # self.terminations.contact_phase = DoneTerm(
        #     func=mdp.contact_phase_violation,
        #     params={
        #         "command_name": "motion",
        #         "ball_sensor_name": "soccer_ball_contact",
        #     },
        # )

        self.rewards.target_point_contact = RewTerm(
            func=mdp.target_point_contact,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.sideways_kick = RewTerm(
            func=mdp.sideways_kick,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        
        self.rewards.ball_velocity_direction_alignment = RewTerm(
            func=mdp.ball_velocity_direction_alignment,
            weight=30.0,
            params={
                "command_name": "motion",
                "std": 0.8,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.ball_speed_reward = RewTerm(
            func=mdp.ball_speed_reward,
            weight=10.0,
            params={
                "command_name": "motion",
                # "target_speed": 4.0,
                "std": 1.2,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
            },
        )

        self.rewards.ball_z_speed_penalty_reward = RewTerm(
            func=mdp.ball_z_speed_penalty_reward,
            weight=-0.0,
            params={
                "command_name": "motion",
                "std": 3,
                "velocity_threshold": 0.5,
            },
        )

@configclass
class G1FlatKickMovingEnvCfg(G1FlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Initial soccer-ball linear velocity configuration.
        self.commands.motion.enable_soccer_ball_init_vel = True  # Enable sampling of initial ball velocity.
        self.commands.motion.soccer_ball_init_lin_vel_range = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (0.0, 0.0),
        }


@configclass
class G1FlatSoccerBlindEnvCfg(G1FlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Custom blind-zone range: the ball is invisible when (x, y) distance is outside [min, max].
        self.commands.motion.blind_distance_min_range = (0.2, 0.8)  # Minimum distance sampling range.
        self.commands.motion.blind_distance_max_range = (1.8, 2.5)  # Maximum distance sampling range.
        
        self.observations.policy.target_point_pos = ObsTerm(
            func=mdp.blind_zone_target_point_pos,
            params={"command_name": "motion"},
        )

        self.observations.critic.target_point_pos = ObsTerm(
            func=mdp.blind_zone_target_point_pos,
            params={"command_name": "motion"},
        )


@configclass
class G1FlatSuperSoccerEnvCfg(G1FlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        self.observations.policy.motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        self.observations.policy.body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        self.observations.policy.body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        self.observations.policy.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)


        self.observations.critic.projected_gravity = ObsTerm(func=mdp.projected_gravity)
        self.observations.critic.motion_ref_ang_vel = ObsTerm(func=mdp.motion_anchor_ang_vel, params={"command_name": "motion"})




@configclass
class G1FlatSoccerStudentEnvCfg(G1FlatKickEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        student_obs = self.observations.policy.copy()
        student_obs.target_point_pos = ObsTerm(
            func=mdp.target_point_pos_first_frame,
            params={"command_name": "motion"},
        )
        self.observations.StudentPolicyCfg = student_obs

        student_obs.target_destination_pos_local = ObsTerm(
            func=mdp.target_destination_pos_local_first_frame,
            params={"command_name": "motion"},
        )
