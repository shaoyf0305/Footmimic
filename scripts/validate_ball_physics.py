"""Validate the instantiated soccer-ball physics used by the dribbling task.

The script deliberately does *not* run a learned policy.  It constructs the
same ball asset, ground material, damping, and physics timestep as the
dribbling configuration, then performs controlled measurements:

* USD/runtime inspection (rigid body, collision API, material binding, mass),
* drop test (measured coefficient of restitution),
* sliding test (initial effective kinetic friction),
* pure-rolling test (roll-out distance and speed decay),
* a repeatable kinematic-striker contact test, and
* timestep sensitivity over one or more physics timesteps.

The output directory contains ``summary.json`` (the report to inspect first)
and ``samples.csv`` (raw position/velocity traces).  The measured contact
values, rather than the authored USD values, should be reported externally.

Examples
--------
  python scripts/validate_ball_physics.py --headless
  python scripts/validate_ball_physics.py --headless --dt_sweep 0.005 0.0025 0.001
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Measure the effective physics of the soccer ball in Isaac Lab.")
parser.add_argument(
    "--dt_sweep",
    type=float,
    nargs="+",
    default=[0.005],
    metavar="DT",
    help="Physics timestep(s) to test in seconds (default: 0.005).",
)
parser.add_argument("--drop_height", type=float, default=1.0, help="Ball-centre drop height above the resting height (m).")
parser.add_argument("--slide_speed", type=float, default=1.5, help="Initial speed for the sliding test (m/s).")
parser.add_argument("--roll_speed", type=float, default=1.5, help="Initial speed for the rolling test (m/s).")
parser.add_argument("--roll_duration", type=float, default=3.0, help="Rolling-test duration (s).")
parser.add_argument("--output_dir", type=Path, default=None, help="Output directory (default: logs/ball_physics_validation/<timestamp>).")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import SOCCER_ASSET_PATH, SOCCER_BALL_RADIUS


# These match G1FlatCGDribblingControlEnvCfg. Keep the values explicit so the
# report remains meaningful even when this script is run outside an RL task.
BALL_MASS_KG = 0.40
GROUND_STATIC_FRICTION = 1.00
GROUND_DYNAMIC_FRICTION = 0.95
GROUND_RESTITUTION = 0.22
BALL_LINEAR_DAMPING = 0.18
BALL_ANGULAR_DAMPING = 0.18
STRIKER_STATIC_FRICTION = 1.00
STRIKER_DYNAMIC_FRICTION = 0.95
STRIKER_RESTITUTION = 0.00
GRAVITY = 9.81


@configclass
class BallValidationSceneCfg(InteractiveSceneCfg):
    """Minimal scene: the task's ball/ground plus a controlled rigid striker."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=GROUND_STATIC_FRICTION,
                dynamic_friction=GROUND_DYNAMIC_FRICTION,
                restitution=GROUND_RESTITUTION,
            )
        ),
    )

    soccer_ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCCER_ASSET_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=BALL_LINEAR_DAMPING,
                angular_damping=BALL_ANGULAR_DAMPING,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SOCCER_BALL_RADIUS)),
    )
    soccer_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        history_length=3,
        track_air_time=False,
        force_threshold=0.0,
        debug_vis=False,
    )

    # This is a deliberately simple and repeatable proxy, not a claim that it
    # has the G1 foot's exact material.  Its material values are logged in the
    # report and can be changed after the G1 foot material is identified.
    striker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ValidationStriker",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.26, 0.14),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=STRIKER_STATIC_FRICTION,
                dynamic_friction=STRIKER_DYNAMIC_FRICTION,
                restitution=STRIKER_RESTITUTION,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 10.0, 10.0)),
    )


def _json_value(value: Any) -> Any:
    """Convert USD/NumPy/Torch values to a JSON-safe representation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _stage_report() -> dict[str, Any]:
    """Inspect the spawned USD prims after PhysX/Isaac Lab has constructed them."""
    report: dict[str, Any] = {}
    try:
        import omni.usd
        from pxr import UsdShade

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/SoccerBall")
        geom = stage.GetPrimAtPath("/World/envs/env_0/SoccerBall/BallGeom")
        report["ball_root_path"] = str(root.GetPath()) if root and root.IsValid() else None
        report["ball_geom_path"] = str(geom.GetPath()) if geom and geom.IsValid() else None
        report["ball_root_applied_schemas"] = list(root.GetAppliedSchemas()) if root and root.IsValid() else []
        report["ball_geom_applied_schemas"] = list(geom.GetAppliedSchemas()) if geom and geom.IsValid() else []
        report["ball_geom_properties"] = {}
        if geom and geom.IsValid():
            for name in (
                "radius",
                "physics:mass",
                "physics:staticFriction",
                "physics:dynamicFriction",
                "physics:restitution",
                "physics:collisionEnabled",
            ):
                attr = geom.GetAttribute(name)
                report["ball_geom_properties"][name] = _json_value(attr.Get()) if attr and attr.IsValid() else None
            try:
                material, relationship = UsdShade.MaterialBindingAPI(geom).ComputeBoundMaterial()
                report["bound_material_path"] = str(material.GetPath()) if material and material.GetPrim().IsValid() else None
                report["material_binding_relationship"] = str(relationship.GetPath()) if relationship and relationship.IsValid() else None
                report["bound_material_applied_schemas"] = (
                    list(material.GetPrim().GetAppliedSchemas()) if material and material.GetPrim().IsValid() else []
                )
                report["bound_material_physics_properties"] = {}
                if material and material.GetPrim().IsValid():
                    for name in ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution"):
                        attr = material.GetPrim().GetAttribute(name)
                        report["bound_material_physics_properties"][name] = (
                            _json_value(attr.Get()) if attr and attr.IsValid() else None
                        )
            except Exception as exc:  # USD bindings vary slightly between Isaac versions.
                report["material_binding_error"] = repr(exc)
    except Exception as exc:
        report["stage_inspection_error"] = repr(exc)
    return report


def _runtime_mass_kg(ball: RigidObject) -> float | None:
    """Read the mass from the instantiated PhysX view, if the API exposes it."""
    try:
        masses = ball.root_physx_view.get_masses().detach().cpu().numpy().reshape(-1)
        return float(masses[0])
    except Exception:
        return None


class BallPhysicsValidator:
    def __init__(self, sim: SimulationContext, scene: InteractiveScene, output_dir: Path):
        self.sim = sim
        self.scene = scene
        self.ball: RigidObject = scene["soccer_ball"]
        self.striker: RigidObject = scene["striker"]
        self.ball_contact = scene["soccer_ball_contact"]
        self.output_dir = output_dir
        self.samples: list[dict[str, float | str]] = []

    @property
    def dt(self) -> float:
        return float(self.sim.get_physics_dt())

    def _state(self, position: tuple[float, float, float], linear_velocity=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0)) -> torch.Tensor:
        state = torch.zeros((1, 13), device=self.sim.device, dtype=torch.float32)
        state[0, :3] = torch.tensor(position, device=self.sim.device)
        state[0, 3] = 1.0  # Isaac Lab root quaternion convention: w, x, y, z.
        state[0, 7:10] = torch.tensor(linear_velocity, device=self.sim.device)
        state[0, 10:13] = torch.tensor(angular_velocity, device=self.sim.device)
        return state

    def _set_state(
        self,
        ball_position: tuple[float, float, float],
        ball_linear_velocity=(0.0, 0.0, 0.0),
        ball_angular_velocity=(0.0, 0.0, 0.0),
        striker_position=(10.0, 10.0, 10.0),
        striker_linear_velocity=(0.0, 0.0, 0.0),
    ) -> None:
        self.ball.write_root_state_to_sim(self._state(ball_position, ball_linear_velocity, ball_angular_velocity))
        self.striker.write_root_state_to_sim(self._state(striker_position, striker_linear_velocity))
        self.scene.write_data_to_sim()
        # One step clears stale contact manifolds after a teleport, then the
        # explicit state is written again so the test starts from the request.
        self.sim.step()
        self.scene.update(self.dt)
        self.ball.write_root_state_to_sim(self._state(ball_position, ball_linear_velocity, ball_angular_velocity))
        self.striker.write_root_state_to_sim(self._state(striker_position, striker_linear_velocity))
        self.scene.write_data_to_sim()

    def _step_and_record(self, test: str, t: float, dt: float) -> dict[str, float]:
        self.sim.step()
        self.scene.update(dt)
        pos = self.ball.data.root_pos_w[0].detach().cpu().numpy()
        lin = self.ball.data.root_lin_vel_w[0].detach().cpu().numpy()
        ang = self.ball.data.root_ang_vel_w[0].detach().cpu().numpy()
        contact_force = 0.0
        try:
            net_forces = self.ball_contact.data.net_forces_w
            contact_force = float(torch.linalg.vector_norm(net_forces, dim=-1).max().item())
        except Exception:
            # Keep the mechanical tests usable on Isaac versions exposing a
            # different contact-sensor tensor, and report the missing values.
            contact_force = float("nan")
        row = {
            "test": test,
            "dt_s": float(dt),
            "time_s": float(t),
            "x_m": float(pos[0]),
            "y_m": float(pos[1]),
            "z_m": float(pos[2]),
            "vx_mps": float(lin[0]),
            "vy_mps": float(lin[1]),
            "vz_mps": float(lin[2]),
            "wx_radps": float(ang[0]),
            "wy_radps": float(ang[1]),
            "wz_radps": float(ang[2]),
            "contact_force_N": contact_force,
        }
        self.samples.append(row)
        return {key: float(value) for key, value in row.items() if key not in {"test"}}

    def _trace(self, test: str, duration_s: float) -> list[dict[str, float]]:
        steps = max(1, int(math.ceil(duration_s / self.dt)))
        return [self._step_and_record(test, (index + 1) * self.dt, self.dt) for index in range(steps)]

    def drop_test(self, drop_height: float) -> dict[str, Any]:
        start_z = SOCCER_BALL_RADIUS + float(drop_height)
        self._set_state((0.0, 0.0, start_z))
        trace = self._trace("drop", 2.5)
        z = np.asarray([row["z_m"] for row in trace])
        vz = np.asarray([row["vz_mps"] for row in trace])
        contact = np.flatnonzero((z <= SOCCER_BALL_RADIUS + 0.006) & (vz > 0.0))
        result: dict[str, Any] = {"drop_height_m": float(drop_height), "impact_detected": bool(contact.size)}
        if not contact.size:
            return result
        impact_i = int(contact[0])
        incoming_speed = float(abs(np.min(vz[: impact_i + 1])))
        outgoing_speed = float(np.max(vz[impact_i : min(len(vz), impact_i + max(3, int(0.15 / self.dt)))]))
        after = z[impact_i:]
        rebound_height = float(max(0.0, np.max(after) - SOCCER_BALL_RADIUS))
        result.update(
            {
                "incoming_speed_mps": incoming_speed,
                "outgoing_speed_mps": outgoing_speed,
                "measured_restitution_velocity_ratio": outgoing_speed / incoming_speed if incoming_speed > 1e-6 else None,
                "rebound_height_m": rebound_height,
                "measured_restitution_height_ratio": math.sqrt(rebound_height / drop_height) if drop_height > 0 else None,
                "peak_contact_force_N": float(np.nanmax([row["contact_force_N"] for row in trace])),
            }
        )
        return result

    def sliding_test(self, speed: float) -> dict[str, Any]:
        # Begin at the solved resting height.  Starting above the ground mixes
        # the landing impulse into a friction measurement.
        self._set_state((0.0, 0.0, SOCCER_BALL_RADIUS), ball_linear_velocity=(speed, 0.0, 0.0))
        trace = self._trace("slide", 0.35)
        t = np.asarray([row["time_s"] for row in trace])
        vx = np.asarray([row["vx_mps"] for row in trace])
        # Restrict to the first 0.12 s: later motion increasingly becomes rolling,
        # which is not a Coulomb sliding-friction measurement.
        keep = (t <= min(0.12, t[-1])) & (vx > 0.05)
        result: dict[str, Any] = {"initial_speed_mps": float(speed), "fit_samples": int(np.count_nonzero(keep))}
        if np.count_nonzero(keep) >= 3:
            slope, _ = np.polyfit(t[keep], vx[keep], 1)
            result["initial_deceleration_mps2"] = float(-slope)
            result["effective_kinetic_friction_estimate"] = float(max(0.0, -slope / GRAVITY))
        return result

    def rolling_test(self, speed: float, duration_s: float) -> dict[str, Any]:
        # For +x rolling on a z-up plane, omega_y = +v/r gives no-slip rolling:
        # v_contact,x = v_com,x - omega_y * r = 0.  The opposite sign spins
        # backwards and measures a braking skid rather than free rolling.
        self._set_state(
            (0.0, 0.0, SOCCER_BALL_RADIUS),
            ball_linear_velocity=(speed, 0.0, 0.0),
            ball_angular_velocity=(0.0, speed / SOCCER_BALL_RADIUS, 0.0),
        )
        trace = self._trace("roll", duration_s)
        t = np.asarray([row["time_s"] for row in trace])
        x = np.asarray([row["x_m"] for row in trace])
        vx = np.asarray([row["vx_mps"] for row in trace])
        keep = vx > 0.08
        result: dict[str, Any] = {
            "initial_speed_mps": float(speed),
            "duration_s": float(duration_s),
            "distance_m": float(x[-1]),
            "final_speed_mps": float(vx[-1]),
            "fit_samples": int(np.count_nonzero(keep)),
        }
        if np.count_nonzero(keep) >= 3:
            slope, _ = np.polyfit(t[keep], np.log(vx[keep]), 1)
            result["fitted_speed_decay_rate_per_s"] = float(-slope)
        return result

    def striker_test(self) -> dict[str, Any]:
        ball_x = 0.40
        striker_x = 0.00
        striker_speed = 1.0
        self._set_state(
            (ball_x, 0.0, SOCCER_BALL_RADIUS),
            striker_position=(striker_x, 0.0, SOCCER_BALL_RADIUS),
            striker_linear_velocity=(striker_speed, 0.0, 0.0),
        )
        trace: list[dict[str, float]] = []
        hit = False
        for index in range(max(1, int(math.ceil(0.7 / self.dt)))):
            # Kinematic bodies move only when their pose target is written to
            # PhysX.  A velocity in their root state alone does not advance the
            # transform, which previously left this striker stationary.
            t = index * self.dt
            self.striker.write_root_state_to_sim(
                self._state(
                    (striker_x + striker_speed * t, 0.0, SOCCER_BALL_RADIUS),
                    linear_velocity=(striker_speed, 0.0, 0.0),
                )
            )
            self.scene.write_data_to_sim()
            trace.append(self._step_and_record("striker", (index + 1) * self.dt, self.dt))
            if trace[-1]["vx_mps"] > 0.05:
                hit = True
                break
        # Remove the kinematic cube immediately after first contact, then measure
        # the ball's post-impact velocity without continued pushing.
        self.striker.write_root_state_to_sim(self._state((10.0, 10.0, 10.0)))
        self.scene.write_data_to_sim()
        trace.extend(self._trace("striker", 0.20))
        vx = np.asarray([row["vx_mps"] for row in trace])
        return {
            "striker_speed_mps": striker_speed,
            "contact_detected_from_ball_motion": hit,
            "peak_ball_speed_mps": float(np.max(vx)),
            "ball_speed_after_0p2s_mps": float(vx[-1]),
            "peak_contact_force_N": float(np.nanmax([row["contact_force_N"] for row in trace])),
        }

    def run_at_dt(self, dt: float) -> dict[str, Any]:
        if abs(self.dt - float(dt)) > 1e-12:
            if not hasattr(self.sim, "set_physics_dt"):
                raise RuntimeError(
                    "This Isaac Lab version does not expose SimulationContext.set_physics_dt(); "
                    "run one invocation per --dt_sweep value instead."
                )
            self.sim.set_physics_dt(float(dt))
        actual_dt = self.dt
        print(f"[ball-validation]   drop test ({2.5 / actual_dt:.0f} physics steps)", flush=True)
        drop = self.drop_test(args_cli.drop_height)
        print(f"[ball-validation]   sliding test ({0.35 / actual_dt:.0f} physics steps)", flush=True)
        slide = self.sliding_test(args_cli.slide_speed)
        print(
            f"[ball-validation]   rolling test ({args_cli.roll_duration / actual_dt:.0f} physics steps)",
            flush=True,
        )
        roll = self.rolling_test(args_cli.roll_speed, args_cli.roll_duration)
        print(f"[ball-validation]   striker test (up to {0.9 / actual_dt:.0f} physics steps)", flush=True)
        striker = self.striker_test()
        print(f"[ball-validation]   completed dt={actual_dt:.6f} s", flush=True)
        return {
            "physics_dt_s": actual_dt,
            "drop": drop,
            "slide": slide,
            "roll": roll,
            "striker": striker,
        }

    def write_samples(self) -> Path:
        path = self.output_dir / "samples.csv"
        fields = [
            "test", "dt_s", "time_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "wx_radps",
            "wy_radps", "wz_radps", "contact_force_N",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)
        return path


def main() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args_cli.output_dir or Path("logs") / "ball_physics_validation" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = float(args_cli.dt_sweep[0])
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(BallValidationSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    scene.update(sim.get_physics_dt())

    validator = BallPhysicsValidator(sim, scene, output_dir)
    report: dict[str, Any] = {
        "purpose": "Effective ball-physics validation; no learned policy was used.",
        "asset": {
            "usd_path": SOCCER_ASSET_PATH,
            "configured_radius_m": SOCCER_BALL_RADIUS,
            "configured_mass_kg": BALL_MASS_KG,
            "runtime_mass_kg": _runtime_mass_kg(validator.ball),
            "configured_ball_linear_damping": BALL_LINEAR_DAMPING,
            "configured_ball_angular_damping": BALL_ANGULAR_DAMPING,
        },
        "configured_ground_material": {
            "static_friction": GROUND_STATIC_FRICTION,
            "dynamic_friction": GROUND_DYNAMIC_FRICTION,
            "restitution": GROUND_RESTITUTION,
            "friction_combine_mode": "multiply",
            "restitution_combine_mode": "multiply",
        },
        "configured_striker_material": {
            "static_friction": STRIKER_STATIC_FRICTION,
            "dynamic_friction": STRIKER_DYNAMIC_FRICTION,
            "restitution": STRIKER_RESTITUTION,
        },
        "runtime_usd_inspection": _stage_report(),
        "tests_by_timestep": {},
    }
    for dt in args_cli.dt_sweep:
        key = f"dt_{float(dt):.7f}"
        print(f"[ball-validation] running controlled tests at dt={float(dt):.6f} s")
        report["tests_by_timestep"][key] = validator.run_at_dt(float(dt))

    report["raw_samples_csv"] = str(validator.write_samples())
    report_path = output_dir / "summary.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(f"[ball-validation] wrote {report_path}")
    print(f"[ball-validation] wrote {report['raw_samples_csv']}")
    return report_path


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
