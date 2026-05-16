#!/usr/bin/env python3
"""Synthesize ``ball_pos_w`` and per-frame foot–ball distance from CG 0/1 labels.

Pipeline (XGen-style, football simplification):

1. **Contact segments** (``dribble_cg_contact==1``, foot from ``dribble_cg_foot``):
   Place the ball at the labeled ankle + a fixed offset in horizontal yaw
   (``p_ball = p_foot + R_yaw @ phi``).

2. **Between segments**: linear interpolation in XY, fixed ground height.

3. **Before first / after last segment**: interpolate from a front spawn point
   (motion frame-0 anchor + forward distance) to the first/last contact ball pose.

4. **``dribble_cg_foot_ball_dist[t]``** = 3D distance from the labeled foot to the
   synthesized ball at frame ``t`` (meters). Frames without a foot label are ``-1``.

Writes ``ball_pos_w`` and ``dribble_cg_foot_ball_dist`` into each ``.npz``.
Keeps existing ``dribble_cg_contact`` / ``dribble_cg_foot`` for compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _collect_npz_files(motion_path: Path) -> list[Path]:
    if motion_path.is_file() and motion_path.suffix == ".npz":
        return [motion_path]
    if motion_path.is_dir():
        files = sorted(motion_path.glob("*.npz"))
        if files:
            return files
    raise ValueError(f"No .npz files found at: {motion_path}")


def _yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Extract yaw (rad) from quaternions ``[..., 4]`` in wxyz order."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny, cosy)


def _yaw_rotate_xy(yaw: np.ndarray, vec_xy: np.ndarray) -> np.ndarray:
    """Rotate 2D offsets by per-frame yaw. ``yaw`` (T,), ``vec_xy`` (2,) -> (T, 2)."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    x, y = vec_xy[0], vec_xy[1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def _contact_segments(contact: np.ndarray, foot: np.ndarray) -> list[tuple[int, int, int]]:
    """Return list of (start, end, foot_id) for contiguous contact runs."""
    n = int(contact.size)
    segs: list[tuple[int, int, int]] = []
    i = 0
    while i < n:
        if contact[i] <= 0:
            i += 1
            continue
        j = i + 1
        while j < n and contact[j] > 0:
            j += 1
        votes = foot[i:j]
        votes = votes[votes >= 0]
        if votes.size == 0:
            fid = 1
        else:
            fid = int(np.bincount(votes.astype(np.int64)).argmax())
        segs.append((i, j - 1, fid))
        i = j
    return segs


def synthesize_ball_trajectory(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    cg_contact: np.ndarray,
    cg_foot: np.ndarray,
    *,
    left_foot_index: int,
    right_foot_index: int,
    anchor_body_index: int,
    ball_radius: float,
    foot_offset_x: float,
    foot_offset_y: float,
    init_forward_dist: float,
    use_foot_yaw: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ball_pos_w [T,3], foot_ball_dist [T])``."""
    T = int(body_pos_w.shape[0])
    ball = np.zeros((T, 3), dtype=np.float32)
    dist = np.full(T, -1.0, dtype=np.float32)

    segs = _contact_segments(cg_contact, cg_foot)
    phi_xy = np.array([foot_offset_x, foot_offset_y], dtype=np.float64)

    def _foot_idx(fid: int) -> int:
        return left_foot_index if fid == 0 else right_foot_index

    def _place_contact_frame(t: int, fid: int) -> None:
        fi = _foot_idx(fid)
        foot_p = body_pos_w[t, fi]
        if use_foot_yaw and body_quat_w is not None:
            yaw = _yaw_from_quat_wxyz(body_quat_w[t, fi])
        else:
            yaw = _yaw_from_quat_wxyz(body_quat_w[t, anchor_body_index])
        off_xy = _yaw_rotate_xy(np.asarray([yaw]), phi_xy)[0]
        ball[t, 0] = foot_p[0] + off_xy[0]
        ball[t, 1] = foot_p[1] + off_xy[1]
        ball[t, 2] = ball_radius

    # Contact frames: anchor placement
    for s, e, fid in segs:
        for t in range(s, e + 1):
            _place_contact_frame(t, fid)

    # Between segments: XY lerp
    for k in range(len(segs) - 1):
        s0, e0, _ = segs[k]
        s1, _, _ = segs[k + 1]
        if s1 <= e0 + 1:
            continue
        p0 = ball[e0, :2]
        p1 = ball[s1, :2]
        gap = s1 - e0
        for j, t in enumerate(range(e0 + 1, s1)):
            alpha = float(j) / float(gap)
            ball[t, :2] = (1.0 - alpha) * p0 + alpha * p1
            ball[t, 2] = ball_radius

    # Before first contact: from spawn in front of frame-0 anchor
    if segs:
        s0, _, fid0 = segs[0]
        anchor0 = body_pos_w[0, anchor_body_index]
        yaw0 = _yaw_from_quat_wxyz(body_quat_w[0, anchor_body_index])
        spawn_xy = anchor0[:2] + _yaw_rotate_xy(np.asarray([yaw0]), np.array([init_forward_dist, 0.0]))[0]
        if s0 > 0:
            p1 = ball[s0, :2]
            for t in range(s0):
                alpha = float(t + 1) / float(s0 + 1)
                ball[t, :2] = (1.0 - alpha) * spawn_xy + alpha * p1
                ball[t, 2] = ball_radius
    else:
        # No contact labels: single spawn in front of anchor
        anchor0 = body_pos_w[0, anchor_body_index]
        yaw0 = _yaw_from_quat_wxyz(body_quat_w[0, anchor_body_index])
        off = _yaw_rotate_xy(np.asarray([yaw0]), np.array([init_forward_dist, 0.0]))[0]
        ball[:, 0] = anchor0[0] + off[0]
        ball[:, 1] = anchor0[1] + off[1]
        ball[:, 2] = ball_radius

    # After last contact: hold last pose (rolling can be added later)
    if segs:
        _, e_last, _ = segs[-1]
        if e_last < T - 1:
            ball[e_last + 1 :] = ball[e_last]

    # Per-frame foot–ball distance (demo kinematics, not sim). Use XY only: ankle
    # height is ~0.7 m while the ball sits on the ground — 3D norm is misleading.
    for t in range(T):
        fid = int(cg_foot[t])
        if fid < 0:
            continue
        fi = _foot_idx(fid)
        dxy = body_pos_w[t, fi, :2] - ball[t, :2]
        dist[t] = float(np.linalg.norm(dxy))

    return ball, dist


def _npz_replace(npz_path: Path, updates: dict[str, np.ndarray]) -> None:
    with np.load(npz_path, allow_pickle=True) as old:
        payload = {k: old[k] for k in old.files}
    payload.update(updates)
    np.savez(npz_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize ball_pos_w + foot-ball distance from CG labels")
    parser.add_argument("--motion_path", type=str, required=True, help="Directory or single .npz")
    parser.add_argument("--ball_radius", type=float, default=0.11)
    parser.add_argument("--foot_offset_x", type=float, default=0.12, help="Ball ahead of foot (m) in yaw frame")
    parser.add_argument("--foot_offset_y", type=float, default=0.0, help="Lateral offset (m) in yaw frame")
    parser.add_argument("--init_forward_dist", type=float, default=0.45, help="Pre-contact spawn distance (m)")
    parser.add_argument("--left_foot_index", type=int, default=3)
    parser.add_argument("--right_foot_index", type=int, default=6)
    parser.add_argument("--anchor_body_index", type=int, default=7, help="torso_link index in body_pos_w")
    parser.add_argument(
        "--use_foot_yaw",
        action="store_true",
        help="Use foot quaternion yaw for offset (default: anchor/torso yaw)",
    )
    args = parser.parse_args()

    files = _collect_npz_files(Path(args.motion_path))
    updated = 0
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            if "body_pos_w" not in d.files or "body_quat_w" not in d.files:
                print(f"[SKIP] {f.name}: missing body_pos_w / body_quat_w")
                continue
            body_pos = np.asarray(d["body_pos_w"], dtype=np.float32)
            body_quat = np.asarray(d["body_quat_w"], dtype=np.float32)
            T = int(body_pos.shape[0])
            if "dribble_cg_contact" in d.files:
                cg_contact = np.asarray(d["dribble_cg_contact"], dtype=np.int8).reshape(-1)[:T]
            else:
                print(f"[SKIP] {f.name}: no dribble_cg_contact")
                continue
            if "dribble_cg_foot" in d.files:
                cg_foot = np.asarray(d["dribble_cg_foot"], dtype=np.int8).reshape(-1)[:T]
            else:
                cg_foot = np.full(T, -1, dtype=np.int8)

        ball, dist = synthesize_ball_trajectory(
            body_pos,
            body_quat,
            cg_contact,
            cg_foot,
            left_foot_index=args.left_foot_index,
            right_foot_index=args.right_foot_index,
            anchor_body_index=args.anchor_body_index,
            ball_radius=args.ball_radius,
            foot_offset_x=args.foot_offset_x,
            foot_offset_y=args.foot_offset_y,
            init_forward_dist=args.init_forward_dist,
            use_foot_yaw=args.use_foot_yaw,
        )

        _npz_replace(
            f,
            {
                "ball_pos_w": ball.astype(np.float32),
                "dribble_cg_foot_ball_dist": dist.astype(np.float32),
            },
        )
        n_labeled = int(np.sum(dist >= 0))
        d_med = float(np.median(dist[dist >= 0])) if n_labeled > 0 else float("nan")
        print(f"[OK] {f.name}: ball_pos_w {ball.shape}, foot_ball_dist labeled_frames={n_labeled}, median={d_med:.3f}m")
        updated += 1

    print(f"[DONE] Updated {updated}/{len(files)} files.")


if __name__ == "__main__":
    main()
