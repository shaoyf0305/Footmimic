#!/usr/bin/env python3
"""Remove vertical root drift by keeping one G1 sole in contact with the ground.

The motion stores ankle-roll link poses rather than a sole point. For every
frame this script transforms the G1 foot collision-capsule endpoints into world
space, finds the lowest point of either foot, and applies one common Z
translation to every reference body. Relative body poses and all horizontal
motion are therefore unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


# Endpoints of the seven G1 ankle-roll collision capsules. Their centre lines
# are at local z=-0.025 m and the largest capsule radius is 0.01 m.
_FOOT_CAPSULE_ENDPOINTS = np.asarray(
    [
        [0.100, -0.026, -0.025],
        [0.050, -0.026, -0.025],
        [-0.044, -0.018, -0.025],
        [0.123, -0.018, -0.025],
        [-0.052, -0.010, -0.025],
        [0.130, -0.010, -0.025],
        [-0.054, 0.000, -0.025],
        [0.132, 0.000, -0.025],
        [-0.052, 0.010, -0.025],
        [0.130, 0.010, -0.025],
        [-0.044, 0.018, -0.025],
        [0.123, 0.018, -0.025],
        [0.100, 0.026, -0.025],
        [0.050, 0.026, -0.025],
    ],
    dtype=np.float64,
)
_FOOT_CAPSULE_RADIUS = 0.01


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by normalized quaternions ``q`` in wxyz order."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)
    w = q[..., :1]
    qv = q[..., 1:]
    return v + 2.0 * (w * np.cross(qv, v) + np.cross(qv, np.cross(qv, v)))


def _sole_height(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    foot_indices: tuple[int, int],
) -> np.ndarray:
    """Return the lowest collision-geometry height of either foot per frame."""
    heights = []
    points = _FOOT_CAPSULE_ENDPOINTS[None, :, :]
    for foot_index in foot_indices:
        quat = body_quat_w[:, foot_index, None, :]
        pos = body_pos_w[:, foot_index, None, :]
        endpoints_w = pos + _quat_rotate_wxyz(quat, points)
        heights.append(np.min(endpoints_w[..., 2], axis=1) - _FOOT_CAPSULE_RADIUS)
    return np.minimum(heights[0], heights[1])


def ground_motion(
    source: Path,
    target: Path,
    *,
    left_foot_index: int,
    right_foot_index: int,
    ground_height: float,
) -> None:
    with np.load(source, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}

    body_pos_src = np.asarray(payload["body_pos_w"])
    body_pos = np.asarray(body_pos_src, dtype=np.float64).copy()
    body_quat = np.asarray(payload["body_quat_w"], dtype=np.float64)
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError(f"Unexpected body_pos_w shape in {source}: {body_pos.shape}")

    sole_before = _sole_height(
        body_pos,
        body_quat,
        (left_foot_index, right_foot_index),
    )
    z_offset = np.asarray(ground_height - sole_before, dtype=np.float64)
    body_pos[..., 2] += z_offset[:, None]
    payload["body_pos_w"] = body_pos.astype(body_pos_src.dtype)

    # A time-varying common translation contributes the same vertical velocity
    # to every rigid body. Keep relative velocities and leave all XY velocities
    # (including the pelvis forward speed) untouched.
    if "body_lin_vel_w" in payload and len(z_offset) > 1:
        fps = float(np.asarray(payload.get("fps", [50.0])).reshape(-1)[0])
        edge_order = 2 if len(z_offset) > 2 else 1
        offset_vel = np.gradient(z_offset, 1.0 / fps, edge_order=edge_order)
        body_vel_src = np.asarray(payload["body_lin_vel_w"])
        body_vel = np.asarray(body_vel_src, dtype=np.float64).copy()
        body_vel[..., 2] += offset_vel[:, None]
        payload["body_lin_vel_w"] = body_vel.astype(body_vel_src.dtype)

    previous_offset = np.asarray(
        payload.get("ref_grounding_z_offset", np.zeros_like(z_offset)),
        dtype=np.float64,
    ).reshape(-1)
    if previous_offset.size != z_offset.size:
        previous_offset = np.zeros_like(z_offset)
    payload["ref_grounding_z_offset"] = (previous_offset + z_offset).astype(np.float32)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.grounding_tmp.npz")
    np.savez(tmp, **payload)
    tmp.replace(target)

    grounded = _sole_height(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        body_quat,
        (left_foot_index, right_foot_index),
    )
    print(
        f"[OK] {target}: frames={len(z_offset)}, "
        f"sole before={sole_before.min():.3f}..{sole_before.max():.3f} m, "
        f"shift={z_offset.min():.3f}..{z_offset.max():.3f} m, "
        f"grounded max error={np.max(np.abs(grounded - ground_height)):.2e} m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_path", type=Path, help="Input motion .npz")
    parser.add_argument("--output", type=Path, default=None, help="Output .npz (default: edit input)")
    parser.add_argument("--left_foot_index", type=int, default=18)
    parser.add_argument("--right_foot_index", type=int, default=19)
    parser.add_argument("--ground_height", type=float, default=0.0)
    args = parser.parse_args()

    source = args.motion_path.resolve()
    target = args.output.resolve() if args.output is not None else source
    if target != source:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source = target
    ground_motion(
        source,
        target,
        left_foot_index=args.left_foot_index,
        right_foot_index=args.right_foot_index,
        ground_height=args.ground_height,
    )


if __name__ == "__main__":
    main()
