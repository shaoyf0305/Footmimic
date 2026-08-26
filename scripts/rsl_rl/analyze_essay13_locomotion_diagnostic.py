"""Summarize physical-vs-ghost locomotion diagnostics for the frozen policy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _scalar_text(value: np.ndarray, default: str) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return default
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _case_metrics(path: Path, warmup_steps: int) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        total_steps = int(len(data["step"]))
        start = int(warmup_steps) if total_steps > int(warmup_steps) else 0
        window = slice(start, total_steps)

        command = np.asarray(data["effective_command_speed"], dtype=float)[window]
        if "pelvis_command_forward_speed" in data:
            pelvis_forward = np.asarray(
                data["pelvis_command_forward_speed"], dtype=float
            )[window]
        else:
            # Backward-compatible fallback for zero-heading diagnostics.
            pelvis_forward = np.asarray(data["pelvis_xy_speed"], dtype=float)[window]

        ball_forward = np.asarray(
            data["ball_filtered_command_forward_speed"], dtype=float
        )[window]
        done = np.asarray(data["done"], dtype=bool)
        termination = np.asarray(data["termination_reason"]).astype(str)
        ball_failure_mask = np.zeros(termination.shape, dtype=bool)
        for label in ("ball_lost", "dribbling_no_contact", "no_contact"):
            ball_failure_mask |= np.char.find(termination, label) >= 0
        ball_failure_count = int(np.count_nonzero(ball_failure_mask))
        termination_reasons = ",".join(sorted(set(termination[termination != ""])))

        target = float(np.nanmean(command))
        pelvis_mean = float(np.nanmean(pelvis_forward))
        ball_mean = float(np.nanmean(ball_forward))
        return {
            "case_id": _scalar_text(data.get("evaluation_case_id", np.asarray("")), path.parent.name),
            "mode": _scalar_text(data.get("evaluation_ball_mode", np.asarray("physical")), "physical"),
            "seed": int(np.asarray(data.get("evaluation_seed", np.asarray(-1))).reshape(-1)[0]),
            "steps": total_steps,
            "target_speed": target,
            "pelvis_forward_speed": pelvis_mean,
            "pelvis_mae": float(np.nanmean(np.abs(pelvis_forward - command))),
            "pelvis_response_ratio": pelvis_mean / target if target > 1.0e-6 else np.nan,
            "ball_forward_speed": ball_mean,
            "ball_minus_pelvis": ball_mean - pelvis_mean,
            "done_count": int(np.count_nonzero(done)),
            "ball_failures": ball_failure_count,
            "termination_reasons": termination_reasons,
        }


def _paired_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Pair physical and pelvis-locked cases by seed and steady target speed."""
    groups: dict[tuple[int, float], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (int(row["seed"]), round(float(row["target_speed"]), 3))
        groups.setdefault(key, {})[str(row["mode"])] = row

    paired: list[dict[str, object]] = []
    for (seed, target), modes in sorted(groups.items()):
        physical = modes.get("physical")
        pelvis_locked = modes.get("pelvis_locked")
        if physical is None or pelvis_locked is None:
            continue
        physical_speed = float(physical["pelvis_forward_speed"])
        locked_speed = float(pelvis_locked["pelvis_forward_speed"])
        paired.append(
            {
                "seed": seed,
                "target_speed": target,
                "physical_pelvis_speed": physical_speed,
                "pelvis_locked_pelvis_speed": locked_speed,
                "pelvis_locked_minus_physical": locked_speed - physical_speed,
                "physical_response_ratio": float(physical["pelvis_response_ratio"]),
                "pelvis_locked_response_ratio": float(
                    pelvis_locked["pelvis_response_ratio"]
                ),
                "physical_done_count": int(physical["done_count"]),
                "pelvis_locked_done_count": int(pelvis_locked["done_count"]),
            }
        )
    return paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=75,
        help="Discard this many 50-Hz control steps before computing metrics.",
    )
    args = parser.parse_args()

    diagnostic_paths = sorted(args.result_dir.glob("*/diagnostic.npz"))
    if not diagnostic_paths:
        raise SystemExit(f"No diagnostic.npz files found under {args.result_dir}")

    rows = [_case_metrics(path, args.warmup_steps) for path in diagnostic_paths]
    output_path = args.result_dir / "locomotion_summary.tsv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    paired_rows = _paired_metrics(rows)
    paired_output_path = args.result_dir / "locomotion_pair_summary.tsv"
    if paired_rows:
        with paired_output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(paired_rows[0]), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(paired_rows)

    print("mode\tseed\ttarget\tpelvis\tratio\tball\tball-pelvis\tdones\treasons")
    for row in sorted(rows, key=lambda value: (str(value["mode"]), float(value["target_speed"]), int(value["seed"]))):
        print(
            f"{row['mode']}\t{row['seed']}\t{row['target_speed']:.3f}\t"
            f"{row['pelvis_forward_speed']:.3f}\t{row['pelvis_response_ratio']:.3f}\t"
            f"{row['ball_forward_speed']:.3f}\t{row['ball_minus_pelvis']:+.3f}\t"
            f"{row['done_count']}\t{row['termination_reasons'] or '-'}"
        )
    print(output_path)
    if paired_rows:
        print("\nseed\ttarget\tphysical-pelvis\tlocked-pelvis\tlocked-physical")
        for row in paired_rows:
            print(
                f"{row['seed']}\t{row['target_speed']:.3f}\t"
                f"{row['physical_pelvis_speed']:.3f}\t"
                f"{row['pelvis_locked_pelvis_speed']:.3f}\t"
                f"{row['pelvis_locked_minus_physical']:+.3f}"
            )
        print(paired_output_path)


if __name__ == "__main__":
    main()
