import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg

from soccer.assets import ASSET_DIR
from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp import observations_anchor as obs_anchor
from soccer.tasks.tracking.tracking_env_cfg import MySceneCfg
from .flat_env_cfg import G1FlatEnvCfg


SOCCER_BALL_RADIUS = 0.11

SOCCER_ASSET_PATH = f"{ASSET_DIR}/soccer/soccer.usda"

# Explicit one-to-many filters let the ball sensor expose per-link force
# matrices.  Rewards fall back to the legacy net-force/nearest-body path when
# an older IsaacLab runtime does not provide ``force_matrix_w``.
G1_BALL_CONTACT_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


def _apply_shared_dribble_inputs(cfg) -> None:
    """Give Stage 1 and Stage 2 the same ordered network input interface."""
    cfg.observations.policy.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_polar,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_polar,
        params={"command_name": "motion"},
    )
    cfg.observations.policy.motion_locomotion_polar_cmd = ObsTerm(
        func=mdp.motion_locomotion_polar_command,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_polar_cmd = ObsTerm(
        func=mdp.motion_locomotion_polar_command,
        params={"command_name": "motion"},
    )
    cfg.observations.policy.anchor_ball_velocity_polar_cmd = ObsTerm(
        func=obs_anchor.anchor_ball_velocity_polar_command,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.anchor_ball_velocity_polar_cmd = ObsTerm(
        func=obs_anchor.anchor_ball_velocity_polar_command,
        params={"command_name": "motion"},
    )

    # Both stages expose the normalized joint command that the action term
    # actually executes. The Stage-1 action falls back to its raw action.
    cfg.observations.policy.actions.func = mdp.effective_joint_action
    cfg.observations.policy.actions.params = {"action_name": "joint_pos"}
    cfg.observations.critic.actions.func = mdp.effective_joint_action
    cfg.observations.critic.actions.params = {"action_name": "joint_pos"}


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
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in G1_BALL_CONTACT_BODY_NAMES
        ],
        history_length=3,
        track_air_time=False,
        force_threshold=0.0,
        debug_vis=False,
    )
    

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


def _apply_stage1_mimic_pretrain(cfg) -> None:
    """Stage-1 dribble pretrain: layered mimic + demo-root velocity (no lateral/heading task terms).

    Matches Stage-2 anchor convention (pelvis + task-frame yaw strip). Locomotion follows
    the reference anchor velocity (including slalom lateral); upper-body orientation is
    soft so arms are not twisted by task-frame slalom refs.
    """
    cfg.commands.motion.anchor_body_name = "pelvis"
    cfg.commands.motion.mimic_align_task_frame = True

    cfg.rewards.motion_body_pos.params["body_names"] = _STAGE1_LOCOMOTION_BODY_NAMES
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

    cfg.terminations.anchor_pos_z = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.32},
    )
    cfg.terminations.anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    cfg.terminations.ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "grace_steps_after_resample": 20,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
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
        _apply_shared_dribble_inputs(self)
        _apply_soccer_scene(self)


@configclass
class G1FlatMotionPretrainEnvCfg(G1FlatMotionEnvCfg):
    """Stage-1 mimic with the exact same network input layout as Stage 2."""

    def __post_init__(self):
        super().__post_init__()
        _apply_stage1_mimic_pretrain(self)
