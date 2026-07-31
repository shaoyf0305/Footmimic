#!/usr/bin/env python3
"""Run the Footmimic v4.4 Stage-2 control policy in MuJoCo."""

from __future__ import annotations

import argparse
import dataclasses
import math
import tempfile
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from v44_common import (
    ACTION_DIM,
    ACTION_SCALE_ISAAC,
    BALL_MASS,
    BALL_RADIUS,
    CONTROL_DT,
    DEFAULT_JOINT_POS_ISAAC,
    DEFAULT_MJCF,
    DEFAULT_MOTION_BANK,
    ISAACLAB_TO_MUJOCO_REINDEX,
    ISAAC_BALL_ANGULAR_DAMPING,
    ISAAC_BALL_GROUND_EFFECTIVE_FRICTION,
    ISAAC_BALL_LINEAR_DAMPING,
    ISAAC_BALL_ROBOT_EFFECTIVE_FRICTION,
    ISAAC_FOOT_GROUND_EFFECTIVE_FRICTION,
    JOINT_DAMPING_MJ,
    JOINT_EFFORT_LIMIT_MJ,
    JOINT_STIFFNESS_MJ,
    MUJOCO_JOINT_NAMES,
    MUJOCO_TO_ISAACLAB_REINDEX,
    OBS_DIM,
    PHYSICS_DT,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    UPPER_BODY_ISAACLAB_IDS,
    quat_apply_inverse,
    strip_yaw,
)


@dataclasses.dataclass
class MotionClip:
    path: Path
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_ang_vel_w: np.ndarray

    @property
    def length(self) -> int:
        return int(self.joint_pos.shape[0])


def load_motion(path: Path) -> MotionClip:
    if not path.is_file():
        raise FileNotFoundError(f"Motion not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = (
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_ang_vel_w",
        )
        missing = [name for name in required if name not in data.files]
        if missing:
            raise KeyError(f"{path} is missing motion arrays: {missing}")
        clip = MotionClip(
            path=path,
            fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
            joint_pos=np.asarray(data["joint_pos"], dtype=np.float32),
            joint_vel=np.asarray(data["joint_vel"], dtype=np.float32),
            body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float32),
            body_quat_w=np.asarray(data["body_quat_w"], dtype=np.float32),
            body_ang_vel_w=np.asarray(data["body_ang_vel_w"], dtype=np.float32),
        )
    if clip.joint_pos.ndim != 2 or clip.joint_pos.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected motion joint_pos [T,{ACTION_DIM}], got {clip.joint_pos.shape}")
    if clip.joint_vel.shape != clip.joint_pos.shape:
        raise ValueError(
            f"joint_vel shape {clip.joint_vel.shape} does not match joint_pos {clip.joint_pos.shape}"
        )
    if clip.body_pos_w.ndim != 3 or clip.body_pos_w.shape[1] < 1:
        raise ValueError(f"Expected body_pos_w [T,B,3], got {clip.body_pos_w.shape}")
    return clip


def resolve_motion(motion_bank: Path, selector: str | None) -> Path:
    if selector is not None:
        candidate = Path(selector)
        if candidate.is_file():
            return candidate.resolve()
        candidate = motion_bank / selector
        if candidate.suffix != ".npz":
            candidate = candidate.with_suffix(".npz")
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Could not resolve --motion {selector!r} in {motion_bank}")
    files = sorted(motion_bank.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in motion bank: {motion_bank}")
    return files[0].resolve()


def fit_upper_body_manifold(
    motion_bank: Path,
    *,
    rank: int = 6,
    latent_std_limit: float = 3.0,
    min_latent_limit: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Reproduce UpperBodyManifoldJointPositionAction._fit_manifold_from_motion_bank."""
    files = sorted(motion_bank.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in motion bank: {motion_bank}")
    upper_dim = int(UPPER_BODY_ISAACLAB_IDS.size)
    sample_sum = np.zeros(upper_dim, dtype=np.float64)
    sample_gram = np.zeros((upper_dim, upper_dim), dtype=np.float64)
    sample_count = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            samples = np.asarray(data["joint_pos"], dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected {path} joint_pos [T,{ACTION_DIM}], got {samples.shape}")
        upper = samples[:, UPPER_BODY_ISAACLAB_IDS]
        sample_sum += upper.sum(axis=0)
        sample_gram += upper.T @ upper
        sample_count += int(upper.shape[0])
    if sample_count < 2:
        raise ValueError("At least two motion frames are required to fit the upper-body manifold.")

    mean = sample_sum / sample_count
    covariance = (sample_gram - sample_count * np.outer(mean, mean)) / (sample_count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][: min(rank, upper_dim)]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order]
    latent_limit = np.maximum(latent_std_limit * np.sqrt(eigenvalues), min_latent_limit)
    return (
        mean.astype(np.float32),
        basis.astype(np.float32),
        latent_limit.astype(np.float32),
        sample_count,
    )


class OnnxActor:
    def __init__(self, path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required. Activate the footmimic-mj environment.") from exc
        if not path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {path}")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_names = [value.name for value in self.session.get_inputs()]
        self.output_names = [value.name for value in self.session.get_outputs()]
        if "obs" not in self.input_names:
            raise ValueError(f"ONNX inputs must include 'obs', got {self.input_names}")
        self.h = np.zeros((RNN_NUM_LAYERS, 1, RNN_HIDDEN_DIM), dtype=np.float32)
        self.c = np.zeros_like(self.h)
        obs_input = next(value for value in self.session.get_inputs() if value.name == "obs")
        obs_dim = obs_input.shape[-1]
        if isinstance(obs_dim, int) and obs_dim != OBS_DIM:
            raise ValueError(f"Policy expects {obs_dim} observations; v4.4 control requires {OBS_DIM}.")

    def reset(self) -> None:
        self.h.fill(0.0)
        self.c.fill(0.0)

    def step(self, obs: np.ndarray, style_step: int) -> np.ndarray:
        feeds: dict[str, np.ndarray] = {"obs": obs.astype(np.float32).reshape(1, OBS_DIM)}
        if "h_in" in self.input_names and "c_in" in self.input_names:
            feeds["h_in"] = self.h
            feeds["c_in"] = self.c
        if "time_step" in self.input_names:
            feeds["time_step"] = np.asarray([[style_step]], dtype=np.float32)
        outputs = self.session.run(None, feeds)
        mapped = dict(zip(self.output_names, outputs, strict=False))
        if "h_out" in mapped and "c_out" in mapped:
            self.h = np.asarray(mapped["h_out"], dtype=np.float32)
            self.c = np.asarray(mapped["c_out"], dtype=np.float32)
        actions = np.asarray(mapped.get("actions", outputs[0]), dtype=np.float32).reshape(-1)
        if actions.shape != (ACTION_DIM,):
            raise ValueError(f"Policy returned actions with shape {actions.shape}; expected ({ACTION_DIM},)")
        return actions


class V44ActionProcessor:
    """Exact legacy 29-D upper-body manifold processing used by v4.4 control."""

    def __init__(
        self,
        mean: np.ndarray,
        basis: np.ndarray,
        latent_limit: np.ndarray,
        soft_limits_isaac: np.ndarray,
        control_dt: float,
    ):
        self.mean = mean
        self.basis = basis
        self.latent_limit = latent_limit
        self.soft_limits = soft_limits_isaac
        self.alpha = float(1.0 - math.exp(-2.0 * math.pi * 1.8 * control_dt))
        self.filtered_target = np.zeros(UPPER_BODY_ISAACLAB_IDS.size, dtype=np.float32)
        self.initialized = False

    def reset(self) -> None:
        self.filtered_target.fill(0.0)
        self.initialized = False

    def process(
        self,
        raw_action: np.ndarray,
        reference_joint_pos: np.ndarray,
        actual_joint_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = DEFAULT_JOINT_POS_ISAAC + ACTION_SCALE_ISAAC * raw_action
        upper_ids = UPPER_BODY_ISAACLAB_IDS
        raw_upper_target = target[upper_ids].copy()
        reference_upper = reference_joint_pos[upper_ids]
        constrained = np.clip(
            raw_upper_target,
            reference_upper - 0.25,
            reference_upper + 0.25,
        )
        centered = constrained - self.mean
        raw_latent = centered @ self.basis
        bounded_latent = np.clip(raw_latent, -self.latent_limit, self.latent_limit)
        parallel = bounded_latent @ self.basis.T
        orthogonal = centered - raw_latent @ self.basis.T
        bounded_orthogonal = 0.10 * np.tanh(orthogonal / 0.10)
        projected = self.mean + parallel + bounded_orthogonal
        upper_limits = self.soft_limits[upper_ids]
        projected = np.clip(projected, upper_limits[:, 0], upper_limits[:, 1])

        previous = self.filtered_target if self.initialized else actual_joint_pos[upper_ids]
        filtered = previous + self.alpha * (projected - previous)
        filtered = np.clip(filtered, upper_limits[:, 0], upper_limits[:, 1]).astype(np.float32)
        self.filtered_target[:] = filtered
        self.initialized = True
        target[upper_ids] = filtered

        effective_action = raw_action.copy()
        effective_action[upper_ids] = (
            filtered - DEFAULT_JOINT_POS_ISAAC[upper_ids]
        ) / ACTION_SCALE_ISAAC[upper_ids]
        return target.astype(np.float32), effective_action.astype(np.float32)


def _find_robot_worldbody(root: ET.Element) -> ET.Element:
    for worldbody in root.findall("worldbody"):
        if worldbody.find("./body[@name='pelvis']") is not None:
            return worldbody
    raise ValueError("MJCF does not contain a pelvis body in any worldbody.")


def make_scene_xml(
    base_xml: Path,
    output_xml: Path,
    physics_dt: float,
    ball_mass: float,
    isaac_aligned: bool = False,
) -> None:
    tree = ET.parse(base_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        mesh_dir = (base_xml.parent / compiler.attrib.get("meshdir", ".")).resolve()
        compiler.set("meshdir", str(mesh_dir))

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{physics_dt:.9g}")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")

    if isaac_aligned:
        # The Isaac Lab articulation is generated from main.urdf, whose joints
        # do not define Coulomb friction. The Unitree MJCF instead adds
        # frictionloss=0.1 to every joint through its g1 default class.
        for joint in root.findall(".//joint[@frictionloss]"):
            joint.set("frictionloss", "0")

    for worldbody in root.findall("worldbody"):
        floor = worldbody.find("./geom[@name='floor']")
        if floor is not None:
            floor.set("friction", "0.95 0.005 0.0001")

    worldbody = _find_robot_worldbody(root)
    ball = ET.SubElement(
        worldbody,
        "body",
        {"name": "soccer_ball", "pos": f"0 0 {BALL_RADIUS}"},
    )
    ET.SubElement(
        ball,
        "joint",
        {
            "name": "soccer_ball_joint",
            "type": "free",
            # PhysX rigid-body damping is a velocity decay rate. A MuJoCo
            # free-joint damping value is instead a generalized force
            # coefficient, so the aligned profile applies the equivalent
            # mass/inertia-scaled forces explicitly before every mj_step.
            "damping": "0" if isaac_aligned else "0.18",
        },
    )
    ET.SubElement(
        ball,
        "geom",
        {
            "name": "soccer_ball_geom",
            "type": "sphere",
            "size": f"{BALL_RADIUS}",
            "mass": f"{ball_mass}",
            "friction": "0.62 0.005 0.0001",
            "solref": "0.02 1",
            "solimp": "0.9 0.95 0.01",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "0.95 0.95 0.95 1",
        },
    )
    goal = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "destination_marker",
            "mocap": "true",
            "pos": f"0 -5 {BALL_RADIUS}",
        },
    )
    ET.SubElement(
        goal,
        "geom",
        {
            "name": "destination_marker_geom",
            "type": "sphere",
            "size": "0.07",
            "rgba": "1 0.1 0.1 0.8",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    if isaac_aligned:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")

        foot_friction = ISAAC_FOOT_GROUND_EFFECTIVE_FRICTION
        for pair in contact.findall("pair"):
            # The base MJCF contains only the 14 explicit foot-floor pairs.
            pair.set(
                "friction",
                f"{foot_friction} {foot_friction} 0.005 0.0001 0.0001",
            )

        common_contact = {
            "solref": "0.02 1",
            "solimp": "0.9 0.95 0.01",
        }
        ball_ground_friction = ISAAC_BALL_GROUND_EFFECTIVE_FRICTION
        ET.SubElement(
            contact,
            "pair",
            {
                "name": "soccer_ball_floor",
                "geom1": "soccer_ball_geom",
                "geom2": "floor",
                "friction": (
                    f"{ball_ground_friction} {ball_ground_friction} "
                    "0.005 0.0001 0.0001"
                ),
                **common_contact,
            },
        )

        # PhysX uses multiply combine mode for robot-ball contacts as well.
        # Add explicit MuJoCo pairs because its default same-priority mixing
        # takes the larger coefficient instead of multiplying coefficients.
        robot_collision_names: list[str] = []
        tree_collision_names = {
            "left_thigh",
            "left_shin",
            "right_thigh",
            "right_shin",
        }
        for geom in root.findall(".//geom"):
            name = geom.get("name", "")
            if not name or name in {
                "floor",
                "soccer_ball_geom",
                "destination_marker_geom",
            }:
                continue
            if (
                geom.get("class") == "collision"
                or "collision" in name
                or name in tree_collision_names
            ):
                robot_collision_names.append(name)

        ball_robot_friction = ISAAC_BALL_ROBOT_EFFECTIVE_FRICTION
        for index, geom_name in enumerate(robot_collision_names):
            ET.SubElement(
                contact,
                "pair",
                {
                    "name": f"soccer_ball_robot_{index}",
                    "geom1": "soccer_ball_geom",
                    "geom2": geom_name,
                    "friction": (
                        f"{ball_robot_friction} {ball_robot_friction} "
                        "0.005 0.0001 0.0001"
                    ),
                    **common_contact,
                },
            )

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


class G1Mujoco:
    def __init__(self, xml_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.joint_qpos_addr = np.asarray(
            [self.model.jnt_qposadr[self._joint_id(name)] for name in MUJOCO_JOINT_NAMES],
            dtype=np.int32,
        )
        self.joint_dof_addr = np.asarray(
            [self.model.jnt_dofadr[self._joint_id(name)] for name in MUJOCO_JOINT_NAMES],
            dtype=np.int32,
        )
        self.actuator_ids = np.asarray(
            [self._actuator_id(name) for name in MUJOCO_JOINT_NAMES],
            dtype=np.int32,
        )
        self.base_joint_id = self._joint_id("floating_base_joint")
        self.base_qpos_addr = int(self.model.jnt_qposadr[self.base_joint_id])
        self.base_dof_addr = int(self.model.jnt_dofadr[self.base_joint_id])
        self.pelvis_body_id = self._body_id("pelvis")
        self.ball_body_id = self._body_id("soccer_ball")
        self.ball_joint_id = self._joint_id("soccer_ball_joint")
        self.ball_qpos_addr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_dof_addr = int(self.model.jnt_dofadr[self.ball_joint_id])
        self.destination_body_id = self._body_id("destination_marker")
        self.destination_mocap_id = int(self.model.body_mocapid[self.destination_body_id])
        self.gyro_sensor_adr = self._sensor_address("base_gyro")
        self.soft_limits_isaac = self._soft_limits_isaac()

    def _named_id(self, object_type, name: str) -> int:
        value = mujoco.mj_name2id(self.model, object_type, name)
        if value < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return int(value)

    def _joint_id(self, name: str) -> int:
        return self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name)

    def _body_id(self, name: str) -> int:
        return self._named_id(mujoco.mjtObj.mjOBJ_BODY, name)

    def _actuator_id(self, name: str) -> int:
        return self._named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _sensor_address(self, name: str) -> int:
        sensor_id = self._named_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        if int(self.model.sensor_dim[sensor_id]) != 3:
            raise ValueError(f"Sensor {name} must have dimension 3.")
        return int(self.model.sensor_adr[sensor_id])

    def _soft_limits_isaac(self) -> np.ndarray:
        limits_mj = np.empty((ACTION_DIM, 2), dtype=np.float32)
        for row, name in enumerate(MUJOCO_JOINT_NAMES):
            joint_id = self._joint_id(name)
            low, high = self.model.jnt_range[joint_id]
            midpoint = 0.5 * (low + high)
            half_range = 0.5 * (high - low) * 0.9
            limits_mj[row] = (midpoint - half_range, midpoint + half_range)
        return limits_mj[MUJOCO_TO_ISAACLAB_REINDEX]

    @property
    def joint_pos_mj(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos_addr].copy()

    @property
    def joint_vel_mj(self) -> np.ndarray:
        return self.data.qvel[self.joint_dof_addr].copy()

    @property
    def joint_pos_isaac(self) -> np.ndarray:
        return self.joint_pos_mj[MUJOCO_TO_ISAACLAB_REINDEX]

    @property
    def joint_vel_isaac(self) -> np.ndarray:
        return self.joint_vel_mj[MUJOCO_TO_ISAACLAB_REINDEX]

    @property
    def pelvis_pos(self) -> np.ndarray:
        return self.data.xpos[self.pelvis_body_id].copy()

    @property
    def pelvis_quat(self) -> np.ndarray:
        return self.data.xquat[self.pelvis_body_id].copy()

    @property
    def base_ang_vel_body(self) -> np.ndarray:
        start = self.gyro_sensor_adr
        return self.data.sensordata[start : start + 3].copy()

    @property
    def ball_pos(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id].copy()

    def reset(
        self,
        motion: MotionClip,
        start_frame: int,
        ball_forward: float,
        ball_lateral: float,
        destination: np.ndarray,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        frame = int(start_frame % motion.length)
        root_pos = motion.body_pos_w[frame, 0].astype(np.float64).copy()
        root_pos[:2] = 0.0
        root_quat = strip_yaw(motion.body_quat_w[frame, 0])
        base_qpos = self.base_qpos_addr
        self.data.qpos[base_qpos : base_qpos + 3] = root_pos
        self.data.qpos[base_qpos + 3 : base_qpos + 7] = root_quat
        # The Stage-2 task resets root/joint velocities to zero.
        self.data.qvel[self.base_dof_addr : self.base_dof_addr + 6] = 0.0
        self.data.qpos[self.joint_qpos_addr] = motion.joint_pos[frame][
            ISAACLAB_TO_MUJOCO_REINDEX
        ]
        self.data.qvel[self.joint_dof_addr] = 0.0

        ball_xyz = np.asarray(
            [root_pos[0] + ball_forward, root_pos[1] + ball_lateral, BALL_RADIUS],
            dtype=np.float64,
        )
        self.data.qpos[self.ball_qpos_addr : self.ball_qpos_addr + 3] = ball_xyz
        self.data.qpos[self.ball_qpos_addr + 3 : self.ball_qpos_addr + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.data.qvel[self.ball_dof_addr : self.ball_dof_addr + 6] = 0.0
        self.data.mocap_pos[self.destination_mocap_id] = destination
        self.data.mocap_quat[self.destination_mocap_id] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def set_pd_target_isaac(self, target_isaac: np.ndarray) -> np.ndarray:
        target_mj = target_isaac[ISAACLAB_TO_MUJOCO_REINDEX]
        torque = (
            JOINT_STIFFNESS_MJ * (target_mj - self.joint_pos_mj)
            - JOINT_DAMPING_MJ * self.joint_vel_mj
        )
        torque = np.clip(torque, -JOINT_EFFORT_LIMIT_MJ, JOINT_EFFORT_LIMIT_MJ)
        self.data.ctrl[self.actuator_ids] = torque
        return torque

    def apply_physx_ball_damping(
        self,
        linear_damping: float,
        angular_damping: float,
    ) -> None:
        """Apply PhysX-style velocity damping as equivalent MuJoCo forces.

        PhysX damping coefficients are velocity decay rates:
        ``dv/dt=-d*v`` and ``dw/dt=-d*w``. MuJoCo free-joint damping is a
        generalized force coefficient, so the matching force/torque must be
        scaled by the ball mass and inertia respectively.
        """
        dof = self.ball_dof_addr
        mass = float(self.model.body_mass[self.ball_body_id])
        # The soccer ball is isotropic; every principal inertia is identical.
        inertia = float(self.model.body_inertia[self.ball_body_id, 0])
        self.data.qfrc_applied[dof : dof + 3] = (
            -mass * linear_damping * self.data.qvel[dof : dof + 3]
        )
        self.data.qfrc_applied[dof + 3 : dof + 6] = (
            -inertia * angular_damping * self.data.qvel[dof + 3 : dof + 6]
        )


def build_observation(
    env: G1Mujoco,
    reference_joint_pos: np.ndarray,
    reference_joint_vel: np.ndarray,
    reference_anchor_ang_vel: np.ndarray,
    effective_last_action: np.ndarray,
    destination: np.ndarray,
    speed: float,
    heading: float,
    yaw_rate: float,
) -> np.ndarray:
    pelvis_quat = env.pelvis_quat
    projected_gravity = quat_apply_inverse(
        pelvis_quat,
        np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
    )
    joint_pos_rel = env.joint_pos_isaac - DEFAULT_JOINT_POS_ISAAC
    ball_delta_world = env.ball_pos - env.pelvis_pos
    target_point_body = quat_apply_inverse(pelvis_quat, ball_delta_world)
    destination_body = quat_apply_inverse(pelvis_quat, destination - env.pelvis_pos)
    distance = max(float(np.linalg.norm(ball_delta_world[:2])), 1.0e-4)
    ball_polar = np.asarray(
        [
            distance,
            ball_delta_world[0] / distance,
            ball_delta_world[1] / distance,
        ],
        dtype=np.float32,
    )
    lin_command = np.asarray(
        [speed * math.cos(heading), speed * math.sin(heading), 0.0],
        dtype=np.float32,
    )
    ang_command = np.asarray([0.0, 0.0, yaw_rate], dtype=np.float32)
    polar_command = np.asarray(
        [speed, math.cos(heading), math.sin(heading)],
        dtype=np.float32,
    )
    observation = np.concatenate(
        [
            reference_joint_pos,
            reference_joint_vel,
            projected_gravity,
            reference_anchor_ang_vel,
            env.base_ang_vel_body,
            joint_pos_rel,
            env.joint_vel_isaac,
            effective_last_action,
            target_point_body,
            destination_body,
            ball_polar,
            lin_command,
            ang_command,
            polar_command,
        ]
    ).astype(np.float32)
    if observation.shape != (OBS_DIM,):
        raise RuntimeError(f"Internal observation layout is {observation.shape}, expected ({OBS_DIM},)")
    if not np.all(np.isfinite(observation)):
        raise FloatingPointError("Observation contains NaN or Inf.")
    return observation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuJoCo sim2sim runner for Tracking-CG-G1-Dribbling-RNN-control (v4.4)."
    )
    parser.add_argument("--policy", type=Path, required=True, help="Exported ONNX policy.")
    parser.add_argument("--motion-bank", type=Path, default=DEFAULT_MOTION_BANK)
    parser.add_argument(
        "--motion",
        type=str,
        default=None,
        help="Motion filename/path; default is the first .npz in --motion-bank.",
    )
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--speed", type=float, default=0.55)
    parser.add_argument("--heading", type=float, default=0.0, help="Absolute task-frame heading in radians.")
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--ball-forward", type=float, default=0.45)
    parser.add_argument("--ball-lateral", type=float, default=0.0)
    parser.add_argument(
        "--destination",
        type=float,
        nargs=2,
        default=(0.0, -5.0),
        metavar=("X", "Y"),
        help="Destination observation in task/world XY; training center was (0,-5).",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--sim-time",
        type=float,
        default=20.0,
        help="Seconds between manual resets. In --render mode, 0 runs indefinitely.",
    )
    parser.add_argument("--control-dt", type=float, default=CONTROL_DT)
    parser.add_argument("--physics-dt", type=float, default=PHYSICS_DT)
    parser.add_argument("--ball-mass", type=float, default=BALL_MASS)
    parser.add_argument(
        "--isaac-aligned",
        action="store_true",
        help=(
            "Use the deterministic Isaac-aligned joint, contact, and ball dynamics. "
            "Omit this flag to retain the original MJCF behavior."
        ),
    )
    parser.add_argument("--render", action="store_true", help="Open passive MuJoCo viewer.")
    parser.add_argument(
        "--start-paused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start the rendered simulation paused; press Space to run.",
    )
    parser.add_argument(
        "--track-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make the rendered camera follow the pelvis.",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace rendered simulation in real time.",
    )
    parser.add_argument("--log-interval", type=float, default=1.0)
    parser.add_argument(
        "--stop-on-fall",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fall-height", type=float, default=0.35)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.control_dt <= 0.0 or args.physics_dt <= 0.0:
        raise ValueError("--control-dt and --physics-dt must be positive.")
    ratio = args.control_dt / args.physics_dt
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("--control-dt must be an integer multiple of --physics-dt.")
    if not math.isclose(args.control_dt, CONTROL_DT, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"v4.4 was trained at control_dt={CONTROL_DT}; got {args.control_dt}.")
    if args.speed < 0.0:
        raise ValueError("--speed must be non-negative.")
    if args.ball_mass <= 0.0:
        raise ValueError("--ball-mass must be positive.")
    if args.sim_time < 0.0:
        raise ValueError("--sim-time must be non-negative.")
    if not args.render and args.sim_time == 0.0:
        raise ValueError("--sim-time 0 is only valid with --render.")
    if args.log_interval <= 0.0:
        raise ValueError("--log-interval must be positive.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    policy_path = args.policy.resolve()
    motion_bank = args.motion_bank.resolve()
    mjcf_path = args.mjcf.resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

    motion_path = resolve_motion(motion_bank, args.motion)
    motion = load_motion(motion_path)
    mean, basis, latent_limit, sample_count = fit_upper_body_manifold(motion_bank)
    policy = OnnxActor(policy_path)
    destination = np.asarray(
        [args.destination[0], args.destination[1], BALL_RADIUS],
        dtype=np.float64,
    )

    with tempfile.TemporaryDirectory(prefix="footmimic_v44_mujoco_") as temp_dir:
        generated_xml = Path(temp_dir) / "g1_v44_dribble.xml"
        make_scene_xml(
            mjcf_path,
            generated_xml,
            args.physics_dt,
            args.ball_mass,
            isaac_aligned=args.isaac_aligned,
        )
        env = G1Mujoco(generated_xml)
        processor = V44ActionProcessor(
            mean,
            basis,
            latent_limit,
            env.soft_limits_isaac,
            args.control_dt,
        )
        env.reset(
            motion,
            args.start_frame,
            args.ball_forward,
            args.ball_lateral,
            destination,
        )
        policy.reset()
        processor.reset()

        print(f"[INFO] policy: {policy_path}")
        print(f"[INFO] motion: {motion_path.name} ({motion.length} frames @ {motion.fps:g} Hz)")
        print(f"[INFO] PCA: {sample_count} frames from {motion_bank}, rank={basis.shape[1]}")
        print(
            f"[INFO] dt: physics={args.physics_dt:g}s, control={args.control_dt:g}s, "
            f"substeps={round(args.control_dt / args.physics_dt)}"
        )
        print(
            f"[INFO] command: speed={args.speed:g} m/s, heading={args.heading:g} rad, "
            f"ball=({args.ball_forward:g},{args.ball_lateral:g}) m"
        )
        print(
            "[INFO] dynamics: "
            + (
                "isaac-aligned "
                f"(joint frictionloss=0, foot-ground mu={ISAAC_FOOT_GROUND_EFFECTIVE_FRICTION:g}, "
                f"ball-ground mu={ISAAC_BALL_GROUND_EFFECTIVE_FRICTION:g}, "
                f"ball-robot mu={ISAAC_BALL_ROBOT_EFFECTIVE_FRICTION:g})"
                if args.isaac_aligned
                else "legacy MJCF (joint frictionloss=0.1)"
            )
        )

        interactive = {
            "paused": bool(args.render and args.start_paused),
            "reset_requested": False,
        }

        def key_callback(keycode: int) -> None:
            if keycode == ord(" "):
                interactive["paused"] = not interactive["paused"]
                state = "PAUSED" if interactive["paused"] else "RUNNING"
                print(f"[KEY] {state}")
            elif keycode in (ord("R"), ord("r")):
                interactive["reset_requested"] = True

        viewer = None
        if args.render:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(
                env.model,
                env.data,
                key_callback=key_callback,
            )
            if args.track_camera:
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = env.pelvis_body_id
                    viewer.cam.distance = 3.0
                    viewer.cam.azimuth = 135.0
                    viewer.cam.elevation = -20.0
            print("[CONTROLS] Space: run/pause | R: reset and pause | close window: quit")
            if interactive["paused"]:
                print("[INFO] Viewer started PAUSED. Press Space to begin.")

        control_steps = (
            int(math.ceil(args.sim_time / args.control_dt))
            if args.sim_time > 0.0
            else None
        )
        physics_steps = int(round(args.control_dt / args.physics_dt))
        effective_last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        log_every = max(1, int(round(args.log_interval / args.control_dt)))
        initial_pelvis_z = float(env.pelvis_pos[2])
        completed_steps = 0
        control_step = 0
        try:
            while True:
                if viewer is not None and not viewer.is_running():
                    break
                if interactive["reset_requested"]:
                    env.reset(
                        motion,
                        args.start_frame,
                        args.ball_forward,
                        args.ball_lateral,
                        destination,
                    )
                    policy.reset()
                    processor.reset()
                    effective_last_action.fill(0.0)
                    initial_pelvis_z = float(env.pelvis_pos[2])
                    control_step = 0
                    interactive["reset_requested"] = False
                    interactive["paused"] = True
                    print("[RESET] Robot, ball, motion phase and LSTM were reset. Press Space to run.")
                    if viewer is not None:
                        viewer.sync()
                    continue
                if interactive["paused"]:
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(0.01)
                        continue
                    break
                if control_steps is not None and control_step >= control_steps:
                    if viewer is not None:
                        interactive["paused"] = True
                        print(
                            f"[INFO] Reached --sim-time {args.sim_time:g}s; paused without closing. "
                            "Press R to reset."
                        )
                        continue
                    break

                tick = time.perf_counter()
                # Control owns the episode clock; the motion is only a looping style phase.
                elapsed_style_steps = int(
                    math.floor(control_step * args.control_dt * motion.fps + 1.0e-8)
                )
                style_step = (args.start_frame + elapsed_style_steps) % motion.length
                q_ref = motion.joint_pos[style_step]
                qd_ref = motion.joint_vel[style_step]
                # The configured anchor is pelvis, index 0 in master-modified.
                ref_ang_vel = motion.body_ang_vel_w[style_step, 0]
                obs = build_observation(
                    env,
                    q_ref,
                    qd_ref,
                    ref_ang_vel,
                    effective_last_action,
                    destination,
                    args.speed,
                    args.heading,
                    args.yaw_rate,
                )
                raw_action = policy.step(obs, style_step)
                target, effective_last_action = processor.process(
                    raw_action,
                    q_ref,
                    env.joint_pos_isaac,
                )
                if not np.all(np.isfinite(target)):
                    raise FloatingPointError("Processed action target contains NaN or Inf.")

                for _ in range(physics_steps):
                    if args.isaac_aligned:
                        env.apply_physx_ball_damping(
                            ISAAC_BALL_LINEAR_DAMPING,
                            ISAAC_BALL_ANGULAR_DAMPING,
                        )
                    env.set_pd_target_isaac(target)
                    mujoco.mj_step(env.model, env.data)
                completed_steps += 1

                if viewer is not None:
                    viewer.sync()
                    if args.realtime:
                        remaining = args.control_dt - (time.perf_counter() - tick)
                        if remaining > 0.0:
                            time.sleep(remaining)

                if control_step % log_every == 0:
                    ball_delta = env.ball_pos - env.pelvis_pos
                    print(
                        f"[STEP {control_step:05d}] t={env.data.time:6.2f} "
                        f"pelvis_z={env.pelvis_pos[2]:.3f} "
                        f"ball_xy=({ball_delta[0]:+.3f},{ball_delta[1]:+.3f}) "
                        f"|action|={np.linalg.norm(raw_action):.3f}"
                    )
                if args.stop_on_fall and env.pelvis_pos[2] < args.fall_height:
                    message = (
                        f"[WARN] Fall detected: pelvis_z={env.pelvis_pos[2]:.3f} "
                        f"(initial {initial_pelvis_z:.3f})."
                    )
                    if viewer is not None:
                        interactive["paused"] = True
                        print(f"{message} Paused; press R to reset.")
                    else:
                        print(f"{message} Stopped.")
                        break
                control_step += 1
        finally:
            if viewer is not None:
                viewer.close()

    print(
        f"[DONE] {completed_steps} control steps, "
        f"{completed_steps * args.control_dt:.3f} simulated seconds."
    )


if __name__ == "__main__":
    main()
