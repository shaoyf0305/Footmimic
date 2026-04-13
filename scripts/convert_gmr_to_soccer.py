"""Convert GMR output .pkl to HumanoidSoccer compatible .pkl format.

Usage:
    # Basic conversion (no yaw normalization):
    python scripts/convert_gmr_to_soccer.py \\
        --input motions/pkl/hmr4d_1_unitree_g1.pkl \\
        --output motions/pkl/hmr4d_1_unitree_g1_compatible.pkl

    # With yaw normalization (rotate to match MoCap -90° convention):
    python scripts/convert_gmr_to_soccer.py \\
        --input motions/pkl/hmr4d_1_unitree_g1.pkl \\
        --output motions/pkl/hmr4d_1_unitree_g1_compatible.pkl \\
        --normalize_yaw

    # With custom target yaw (e.g. face +X direction = 0°):
    python scripts/convert_gmr_to_soccer.py \\
        --input motions/pkl/hmr4d_1_unitree_g1.pkl \\
        --output motions/pkl/hmr4d_1_unitree_g1_compatible.pkl \\
        --normalize_yaw --target_yaw 0.0
"""

import argparse
import math
import numpy as np
import joblib


def quat_yaw_xyzw(q: np.ndarray) -> float:
    """Extract yaw (Z-axis rotation) from a single XYZW quaternion."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for XYZW quaternions. Supports batched b: [T, 4]."""
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], axis=-1).astype(np.float32)


def normalize_yaw(root_pos: np.ndarray, root_rot: np.ndarray,
                  target_yaw_deg: float = -90.0) -> tuple[np.ndarray, np.ndarray]:
    """Rotate root trajectory so initial yaw matches target_yaw_deg.

    Args:
        root_pos: [T, 3] root translations
        root_rot: [T, 4] root rotations in XYZW format
        target_yaw_deg: desired initial yaw in degrees (default: -90°, facing -Y)

    Returns:
        (rotated_pos, rotated_rot) with same shapes
    """
    current_yaw = quat_yaw_xyzw(root_rot[0])
    target_yaw = math.radians(target_yaw_deg)
    delta_yaw = target_yaw - current_yaw

    print(f"[NORMALIZE] current_yaw={math.degrees(current_yaw):.1f}°  "
          f"target_yaw={target_yaw_deg:.1f}°  delta={math.degrees(delta_yaw):.1f}°")

    # Build delta rotation quaternion (pure yaw around Z, XYZW convention).
    half = delta_yaw / 2.0
    q_delta = np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float32)  # XYZW

    # Rotate root orientations: q_new = q_delta * q_old
    rotated_rot = quat_mul_xyzw(q_delta, root_rot)

    # Rotate root positions around the initial position (pivot).
    pivot = root_pos[0, :2].copy()
    cos_d, sin_d = math.cos(delta_yaw), math.sin(delta_yaw)
    rot2d = np.array([[cos_d, -sin_d], [sin_d, cos_d]], dtype=np.float32)

    rotated_pos = root_pos.copy()
    xy_centered = rotated_pos[:, :2] - pivot  # [T, 2]
    rotated_pos[:, :2] = (rot2d @ xy_centered.T).T + pivot

    return rotated_pos, rotated_rot


def main():
    parser = argparse.ArgumentParser(
        description="Convert GMR .pkl to HumanoidSoccer compatible .pkl format.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input GMR .pkl file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output compatible .pkl file")
    parser.add_argument("--normalize_yaw", action="store_true", default=False,
                        help="Rotate trajectory so initial yaw matches --target_yaw")
    parser.add_argument("--target_yaw", type=float, default=-90.0,
                        help="Target initial yaw in degrees (default: -90, facing -Y)")
    args = parser.parse_args()

    gmr_data = joblib.load(args.input)

    root_pos = gmr_data['root_pos']
    root_rot = gmr_data['root_rot']  # XYZW
    dof = gmr_data['dof_pos']
    fps = gmr_data['fps']

    print(f"[INFO] Loaded {args.input}: {root_pos.shape[0]} frames, fps={fps}")

    if args.normalize_yaw:
        root_pos, root_rot = normalize_yaw(root_pos, root_rot, args.target_yaw)

    hsoccer_data = {
        'gmr_motion': {
            'fps': fps,
            'root_trans_offset': root_pos,
            'root_rot': root_rot,
            'dof': dof,
        }
    }
    joblib.dump(hsoccer_data, args.output)
    print(f"[INFO] Saved to {args.output}")


if __name__ == "__main__":
    main()
