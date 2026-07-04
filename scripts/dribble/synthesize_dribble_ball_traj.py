#!/usr/bin/env python3
"""Synthesize ``ball_pos_w`` and per-frame foot–ball distance from CG 0/1 labels.

Pipeline (XGen-style, football simplification):

1. **Contact segments** (``dribble_cg_contact==1``, foot from ``dribble_cg_foot``):
   Place the ball at the labeled ankle + a fixed offset in horizontal yaw
   (``p_ball = p_foot + R_yaw @ phi``).

2. **``traj_mode=hybrid``** (recommended): **contact** frames adhere to the foot
   (ground foot bodies + optional medial/inward offset); **gaps** linearly interpolate
   in XY between contact endpoints; pre-first-contact approaches the first touch.

   ``foot_follow``: ball tracks a stepping foot every frame. ``lerp``: legacy gaps +
   anchor spawn (avoid for master data).

4. **``dribble_cg_foot_ball_dist[t]``** = XY distance from the reference foot to the
   synthesized ball at frame ``t`` (meters). Computed on **every** frame where a
   stitched ball trajectory exists (contact + approach gaps + tail), not only on
   annotated contact frames.

5. **``dribble_cg_dist_foot[t]``** = which foot the ball was placed against (0 left,
   1 right). ``dribble_cg_foot`` (contact annotation) is left unchanged.

**Foot body indices:** ``pkl_to_npz`` stores all ~30 G1 bodies. Indices 3/6 match the
14-link *tracking subset*, not ground feet — on full clips index 3 often equals
pelvis. Use ``--auto_foot_indices`` (default) to pick the lowest, most mobile foot
links (``LL_FOOT`` / ``LR_FOOT`` class).

Writes ``ball_pos_w``, ``dribble_cg_foot_ball_dist``, and ``dribble_cg_dist_foot``
into each ``.npz``.
"""

from __future__ import annotations

import argparse
import shutil
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


def _resolve_anchor_body_index(num_bodies: int, anchor_body_index: int) -> int:
    if anchor_body_index >= 0:
        return anchor_body_index
    return 0 if num_bodies >= 20 else 7


def resolve_foot_body_indices(
    body_pos_w: np.ndarray,
    *,
    left_foot_index: int = -1,
    right_foot_index: int = -1,
) -> tuple[int, int]:
    """Return ``(left_body_idx, right_body_idx)`` into ``body_pos_w``.

    For the 14-body tracking subset (from some exporters), ankles are at 3/6.
    For full G1 clips (~30 bodies from ``pkl_to_npz``), pick the two ground foot
    links by median height + horizontal travel (``LL_FOOT`` / ``LR_FOOT``).
    """
    num_bodies = int(body_pos_w.shape[1])
    if left_foot_index >= 0 and right_foot_index >= 0:
        return left_foot_index, right_foot_index

    if num_bodies == 14:
        return 3, 6

    # Full G1 from pkl_to_npz: 30 rigid bodies; ground feet are LL_FOOT/LR_FOOT (~18/19).
    if num_bodies >= 20 and 18 < num_bodies and 19 < num_bodies:
        mean_y_18 = float(np.mean(body_pos_w[:, 18, 1]))
        mean_y_19 = float(np.mean(body_pos_w[:, 19, 1]))
        if mean_y_18 <= mean_y_19:
            return 18, 19
        return 19, 18

    med_z = np.median(body_pos_w[:, :, 2], axis=0)
    travel = np.ptp(body_pos_w[:, :, :2], axis=0)
    travel = np.hypot(travel[:, 0], travel[:, 1])

    candidates = [
        i
        for i in range(num_bodies)
        if med_z[i] < 0.42 and travel[i] > 1.5
    ]
    if len(candidates) < 2:
        order = np.argsort(med_z)
        candidates = order[: min(6, num_bodies)].tolist()

    candidates = sorted(candidates, key=lambda i: (-travel[i], med_z[i]))
    shortlist = candidates[:4]
    if len(shortlist) < 2:
        shortlist = candidates[:2]

    mean_y = {i: float(np.mean(body_pos_w[:, i, 1])) for i in shortlist}
    left_idx = min(shortlist, key=lambda i: mean_y[i])
    right_idx = max(shortlist, key=lambda i: mean_y[i])
    if left_idx == right_idx and len(shortlist) >= 2:
        left_idx, right_idx = shortlist[0], shortlist[1]
    return left_idx, right_idx


def _forward_foot_id(
    t: int,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    anchor_body_index: int,
    left_foot_index: int,
    right_foot_index: int,
) -> int:
    """Pick the foot most ahead in anchor yaw (for frame 0 / cold start)."""
    yaw = float(_yaw_from_quat_wxyz(body_quat_w[t, anchor_body_index]))
    c, s = np.cos(-yaw), np.sin(-yaw)

    def forward_x(i: int) -> float:
        x, y = body_pos_w[t, i, 0], body_pos_w[t, i, 1]
        return c * x - s * y

    lf = forward_x(left_foot_index)
    rf = forward_x(right_foot_index)
    return 0 if lf >= rf else 1


def _foot_id_for_frame(
    t: int,
    cg_contact: np.ndarray,
    cg_foot: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    ball: np.ndarray,
    *,
    left_foot_index: int,
    right_foot_index: int,
    anchor_body_index: int,
) -> int:
    """Choose left (0) or right (1) foot from actual body kinematics."""
    lp = body_pos_w[t, left_foot_index, :2]
    rp = body_pos_w[t, right_foot_index, :2]

    if t > 0:
        bprev = ball[t - 1, :2]
        dl = float(np.linalg.norm(lp - bprev))
        dr = float(np.linalg.norm(rp - bprev))
        closest = 0 if dl <= dr else 1
    else:
        closest = _forward_foot_id(
            t, body_pos_w, body_quat_w, anchor_body_index, left_foot_index, right_foot_index
        )

    if cg_contact[t] > 0 and cg_foot[t] >= 0:
        labeled = int(cg_foot[t])
        if t > 0:
            d_lab = dl if labeled == 0 else dr
            d_oth = dr if labeled == 0 else dl
            if d_oth + 0.04 < d_lab:
                return closest
        return labeled
    return closest


def _inward_xy(
    t: int,
    fid: int,
    body_pos_w: np.ndarray,
    left_foot_index: int,
    right_foot_index: int,
    magnitude: float,
) -> np.ndarray:
    """Unit inward (medial) offset in XY: left foot → toward right, vice versa."""
    if magnitude <= 0.0:
        return np.zeros(2, dtype=np.float64)
    lp = body_pos_w[t, left_foot_index, :2]
    rp = body_pos_w[t, right_foot_index, :2]
    foot = lp if fid == 0 else rp
    other = rp if fid == 0 else lp
    inward = other - foot
    norm = float(np.linalg.norm(inward))
    if norm < 1e-6:
        return np.zeros(2, dtype=np.float64)
    return magnitude * (inward / norm)


def _ref_foot_per_frame(
    T: int,
    segs: list[tuple[int, int, int]],
) -> np.ndarray:
    """Per-frame foot id for foot–ball distance (separate from contact labels).

    Contact frames use the segment foot. Gaps between touches use the *next*
    segment foot (approach). Before the first / after the last touch use the
    first / last segment foot.
    """
    ref = np.full(T, -1, dtype=np.int8)
    if not segs:
        return ref

    for s, e, fid in segs:
        ref[s : e + 1] = fid

    s0, _, fid0 = segs[0]
    if s0 > 0:
        ref[:s0] = fid0

    for k in range(len(segs) - 1):
        _, e0, _ = segs[k]
        s1, _, fid1 = segs[k + 1]
        for t in range(e0 + 1, s1):
            ref[t] = fid1

    _, e_last, fid_last = segs[-1]
    if e_last < T - 1:
        ref[e_last + 1 :] = fid_last

    return ref


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
    foot_inner_offset: float,
    init_forward_dist: float,
    use_foot_yaw: bool,
    traj_mode: str = "hybrid",
    auto_foot_indices: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return ``(ball_pos_w [T,3], foot_ball_dist [T], dist_foot [T], L_idx, R_idx)``."""
    T = int(body_pos_w.shape[0])
    num_bodies = int(body_pos_w.shape[1])
    ball = np.zeros((T, 3), dtype=np.float32)
    dist = np.full(T, -1.0, dtype=np.float32)
    dist_foot = np.full(T, -1, dtype=np.int8)

    if auto_foot_indices and (left_foot_index < 0 or right_foot_index < 0):
        left_foot_index, right_foot_index = resolve_foot_body_indices(
            body_pos_w,
            left_foot_index=left_foot_index,
            right_foot_index=right_foot_index,
        )
    anchor_body_index = _resolve_anchor_body_index(num_bodies, anchor_body_index)

    segs = _contact_segments(cg_contact, cg_foot)
    ref_foot = _ref_foot_per_frame(T, segs)
    mode = str(traj_mode).lower().strip()
    placement_foot = np.full(T, -1, dtype=np.int8)

    def _foot_idx(fid: int) -> int:
        return left_foot_index if fid == 0 else right_foot_index

    def _place_contact_frame(t: int, fid: int) -> None:
        """Adhere ball to foot: forward offset in foot yaw + medial (inward) shift."""
        fi = _foot_idx(fid)
        foot_p = body_pos_w[t, fi, :2]
        if use_foot_yaw and body_quat_w is not None:
            yaw = _yaw_from_quat_wxyz(body_quat_w[t, fi])
        else:
            yaw = _yaw_from_quat_wxyz(body_quat_w[t, anchor_body_index])
        phi_fwd = np.array([foot_offset_x, foot_offset_y], dtype=np.float64)
        off_fwd = _yaw_rotate_xy(np.asarray([yaw]), phi_fwd)[0]
        off_in = _inward_xy(t, fid, body_pos_w, left_foot_index, right_foot_index, foot_inner_offset)
        ball_xy = foot_p + off_fwd + off_in
        ball[t, 0] = ball_xy[0]
        ball[t, 1] = ball_xy[1]
        ball[t, 2] = ball_radius
        placement_foot[t] = fid

    def _lerp_gap(t0: int, t1: int, p0: np.ndarray, p1: np.ndarray, fid: int) -> None:
        """Linear XY blend for frames ``t0+1 .. t1-1`` (exclusive endpoints)."""
        if t1 <= t0 + 1:
            return
        gap = t1 - t0
        for j, t in enumerate(range(t0 + 1, t1)):
            alpha = float(j + 1) / float(gap)
            ball[t, :2] = (1.0 - alpha) * p0 + alpha * p1
            ball[t, 2] = ball_radius
            placement_foot[t] = fid

    def _fill_hybrid_gaps() -> None:
        if not segs:
            return

        for s, e, fid in segs:
            for t in range(s, e + 1):
                seg_fid = int(cg_foot[t]) if cg_foot[t] >= 0 else fid
                _place_contact_frame(t, seg_fid)

        for k in range(len(segs) - 1):
            s0, e0, _ = segs[k]
            s1, _, fid1 = segs[k + 1]
            _lerp_gap(e0, s1, ball[e0, :2], ball[s1, :2], fid1)

        s0, _, fid0 = segs[0]
        if s0 > 0:
            _place_contact_frame(0, fid0)
            p_start = ball[0, :2].copy()
            _lerp_gap(-1, s0, p_start, ball[s0, :2], fid0)
            ball[0, :2] = p_start
            placement_foot[0] = fid0

        _, e_last, fid_last = segs[-1]
        if e_last < T - 1:
            ball[e_last + 1 :, :2] = ball[e_last, :2]
            ball[e_last + 1 :, 2] = ball_radius
            placement_foot[e_last + 1 :] = fid_last

    def _spawn_no_contact() -> None:
        for t in range(T):
            fid = _forward_foot_id(
                t, body_pos_w, body_quat_w, anchor_body_index, left_foot_index, right_foot_index
            )
            _place_contact_frame(t, fid)

    if mode == "hybrid":
        if segs:
            _fill_hybrid_gaps()
        else:
            _spawn_no_contact()
    elif mode == "foot_follow":
        if segs:
            for t in range(T):
                fid = _foot_id_for_frame(
                    t,
                    cg_contact,
                    cg_foot,
                    body_pos_w,
                    body_quat_w,
                    ball,
                    left_foot_index=left_foot_index,
                    right_foot_index=right_foot_index,
                    anchor_body_index=anchor_body_index,
                )
                _place_contact_frame(t, fid)
        else:
            _spawn_no_contact()
    elif mode == "lerp":
        for s, e, fid in segs:
            for t in range(s, e + 1):
                _place_contact_frame(t, fid)

        for k in range(len(segs) - 1):
            s0, e0, _ = segs[k]
            s1, _, fid1 = segs[k + 1]
            _lerp_gap(e0, s1, ball[e0, :2], ball[s1, :2], fid1)

        if segs:
            s0, _, fid0 = segs[0]
            anchor0 = body_pos_w[0, anchor_body_index]
            yaw0 = _yaw_from_quat_wxyz(body_quat_w[0, anchor_body_index])
            spawn_xy = anchor0[:2] + _yaw_rotate_xy(np.asarray([yaw0]), np.array([init_forward_dist, 0.0]))[0]
            if s0 > 0:
                _lerp_gap(-1, s0, spawn_xy, ball[s0, :2], fid0)
                ball[0, :2] = spawn_xy
                placement_foot[0] = fid0
        else:
            _spawn_no_contact()

        if segs:
            _, e_last, fid_last = segs[-1]
            if e_last < T - 1:
                ball[e_last + 1 :, :2] = ball[e_last, :2]
                ball[e_last + 1 :, 2] = ball_radius
                placement_foot[e_last + 1 :] = fid_last
    else:
        raise ValueError(f"Unsupported traj_mode={traj_mode!r}; use 'hybrid', 'foot_follow', or 'lerp'.")

    for t in range(T):
        fid = int(placement_foot[t])
        if fid < 0:
            fid = int(ref_foot[t])
        if fid < 0:
            continue
        fi = _foot_idx(fid)
        dxy = body_pos_w[t, fi, :2] - ball[t, :2]
        dist[t] = float(np.linalg.norm(dxy))
        dist_foot[t] = fid

    return ball, dist, dist_foot, left_foot_index, right_foot_index


def _npz_replace(npz_path: Path, updates: dict[str, np.ndarray]) -> None:
    with np.load(npz_path, allow_pickle=True) as old:
        payload = {k: old[k] for k in old.files}
    payload.update(updates)
    np.savez(npz_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize ball_pos_w + foot-ball distance from CG labels")
    parser.add_argument("--motion_path", type=str, required=True, help="Directory or single .npz")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Write updated npz copies here (source files are left unchanged).",
    )
    parser.add_argument("--ball_radius", type=float, default=0.11)
    parser.add_argument("--foot_offset_x", type=float, default=0.06, help="Ball ahead of foot (m) in foot yaw frame")
    parser.add_argument("--foot_offset_y", type=float, default=0.0, help="Extra lateral offset (m) in foot yaw frame")
    parser.add_argument(
        "--foot_inner_offset",
        type=float,
        default=0.035,
        help="Medial shift (m) toward the other foot during contact adhere (0=disable).",
    )
    parser.add_argument("--init_forward_dist", type=float, default=0.45, help="Pre-contact spawn distance (m, lerp mode only)")
    parser.add_argument("--left_foot_index", type=int, default=-1, help="Left foot body index (-1 = auto)")
    parser.add_argument("--right_foot_index", type=int, default=-1, help="Right foot body index (-1 = auto)")
    parser.add_argument("--anchor_body_index", type=int, default=-1, help="Anchor body for yaw fallback (-1 = pelvis/torso auto)")
    parser.add_argument(
        "--no_auto_foot_indices",
        action="store_true",
        help="Use fixed --left_foot_index/--right_foot_index (default 3/6) instead of auto-detect.",
    )
    parser.add_argument(
        "--traj_mode",
        type=str,
        default="hybrid",
        choices=("hybrid", "foot_follow", "lerp"),
        help="hybrid: contact adhere + gap lerp; foot_follow: always track foot; lerp: legacy.",
    )
    parser.add_argument(
        "--no_foot_yaw",
        action="store_true",
        help="Use anchor/torso yaw for offset (default: foot quaternion yaw).",
    )
    args = parser.parse_args()

    src_root = Path(args.motion_path)
    out_root = Path(args.output_dir) if args.output_dir else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    files = _collect_npz_files(src_root)
    updated = 0
    for f in files:
        target = (out_root / f.name) if out_root is not None else f
        if out_root is not None:
            shutil.copy2(f, target)

        with np.load(target, allow_pickle=True) as d:
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

        li = 3 if args.no_auto_foot_indices and args.left_foot_index < 0 else args.left_foot_index
        ri = 6 if args.no_auto_foot_indices and args.right_foot_index < 0 else args.right_foot_index

        ball, dist, dist_foot, li, ri = synthesize_ball_trajectory(
            body_pos,
            body_quat,
            cg_contact,
            cg_foot,
            left_foot_index=li,
            right_foot_index=ri,
            anchor_body_index=args.anchor_body_index,
            ball_radius=args.ball_radius,
            foot_offset_x=args.foot_offset_x,
            foot_offset_y=args.foot_offset_y,
            foot_inner_offset=args.foot_inner_offset,
            init_forward_dist=args.init_forward_dist,
            use_foot_yaw=not args.no_foot_yaw,
            traj_mode=args.traj_mode,
            auto_foot_indices=not args.no_auto_foot_indices,
        )

        _npz_replace(
            target,
            {
                "ball_pos_w": ball.astype(np.float32),
                "dribble_cg_foot_ball_dist": dist.astype(np.float32),
                "dribble_cg_dist_foot": dist_foot.astype(np.int8),
            },
        )
        n_labeled = int(np.sum(dist >= 0))
        n_contact = int(np.sum(cg_contact > 0))
        d_med = float(np.median(dist[dist >= 0])) if n_labeled > 0 else float("nan")
        print(
            f"[OK] {target.name}: feet L/R={li}/{ri} ball {ball.shape}, "
            f"dist_frames={n_labeled} (contact={n_contact}), median={d_med:.3f}m"
        )
        updated += 1

    print(f"[DONE] Updated {updated}/{len(files)} files.")


if __name__ == "__main__":
    main()
