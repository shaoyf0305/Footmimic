#!/usr/bin/env python3
"""Build motions/dribble-distance-modified from dribble-distance.

- seg1, seg3: copy as-is
- seg4: split into 3 equal parts
- seg5, seg6: split into 2 equal parts each
- seg7: excluded
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "motions" / "dribble-distance"
OUT_DIR = REPO_ROOT / "motions" / "dribble-distance-modified"

TRAJ_KEYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "ball_pos_w",
    "dribble_cg_contact",
    "dribble_cg_foot",
    "dribble_cg_foot_ball_dist",
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def slice_payload(payload: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key in TRAJ_KEYS and hasattr(value, "shape") and len(value.shape) >= 1:
            out[key] = value[start:end]
        else:
            out[key] = value

    if "kick_frame" in out:
        kf = int(np.asarray(out["kick_frame"]).flat[0])
        if kf >= 0:
            if start <= kf < end:
                out["kick_frame"] = np.array(kf - start, dtype=np.int32)
            else:
                out["kick_frame"] = np.array(-1, dtype=np.int32)
    if "kick_end_frame" in out:
        kef = int(np.asarray(out["kick_end_frame"]).flat[0])
        if kef >= 0:
            if start <= kef < end:
                out["kick_end_frame"] = np.array(kef - start, dtype=np.int32)
            else:
                out["kick_end_frame"] = np.array(-1, dtype=np.int32)

    return out


def save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    frames = payload["joint_pos"].shape[0]
    print(f"  saved {path.name}  ({frames} frames)")


def equal_splits(num_frames: int, n_parts: int) -> list[tuple[int, int]]:
    bounds = [round(i * num_frames / n_parts) for i in range(n_parts + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(n_parts)]


def copy_clip(stem: str) -> None:
    src = SRC_DIR / f"{stem}.npz"
    dst = OUT_DIR / f"{stem}.npz"
    shutil.copy2(src, dst)
    frames = int(np.load(dst)["joint_pos"].shape[0])
    print(f"  copied {dst.name}  ({frames} frames)")


def split_clip(stem: str, n_parts: int) -> None:
    payload = load_npz(SRC_DIR / f"{stem}.npz")
    total = int(payload["joint_pos"].shape[0])
    print(f"[INFO] {stem}: {total} frames → {n_parts} parts")
    for i, (start, end) in enumerate(equal_splits(total, n_parts), start=1):
        part = slice_payload(payload, start, end)
        save_npz(OUT_DIR / f"{stem}_part{i}.npz", part)


def main() -> None:
    if not SRC_DIR.is_dir():
        raise SystemExit(f"Source not found: {SRC_DIR}")

    if OUT_DIR.exists():
        for f in OUT_DIR.glob("*.npz"):
            f.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output: {OUT_DIR}")
    print("[INFO] Copy seg1, seg3")
    copy_clip("FAST-seg1_unitree_g1")
    copy_clip("FAST-seg3_unitree_g1")

    print("[INFO] Split seg4 → 3")
    split_clip("FAST-seg4_unitree_g1", 3)

    print("[INFO] Split seg5 → 2")
    split_clip("FAST-seg5_unitree_g1", 2)

    print("[INFO] Split seg6 → 2")
    split_clip("FAST-seg6_unitree_g1", 2)

    print("[INFO] seg7 excluded")
    clips = sorted(OUT_DIR.glob("*.npz"))
    print(f"[DONE] {len(clips)} clips in {OUT_DIR}")
    for p in clips:
        t = int(np.load(p)["joint_pos"].shape[0])
        print(f"  - {p.name} ({t} frames)")


if __name__ == "__main__":
    main()
