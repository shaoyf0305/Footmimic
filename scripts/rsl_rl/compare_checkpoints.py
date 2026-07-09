"""Quick checkpoint diff: compare two .pt files and show L2 change per layer.

Usage:
    python scripts/rsl_rl/compare_checkpoints.py \
        logs/rsl_rl/g1_dribbling/2026-06-13_18-29-34_v1.20/model_59000.pt \
        logs/rsl_rl/g1_dribbling/<new_run>/model_XXXX.pt
"""
import sys
import torch


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    a = torch.load(path_a, map_location="cpu", weights_only=False)["model_state_dict"]
    b = torch.load(path_b, map_location="cpu", weights_only=False)["model_state_dict"]

    print(f"\nComparing:\n  A: {path_a}\n  B: {path_b}\n")

    keys_a = set(a.keys())
    keys_b = set(b.keys())
    only_in_b = keys_b - keys_a
    if only_in_b:
        print(f"Keys only in B (new obs dims added): {only_in_b}\n")

    total_params = 0
    total_changed = 0.0
    rows = []
    for key in sorted(keys_a & keys_b):
        ta = a[key].float()
        tb = b[key].float()
        if ta.shape != tb.shape:
            # obs-expanded layer: compare shared part only
            shared = ta.shape[1] if ta.ndim == 2 else ta.shape[0]
            if ta.ndim == 2:
                diff = (tb[:, :shared] - ta).norm().item()
            else:
                diff = (tb[:shared] - ta).norm().item()
            rows.append((diff, key, str(tuple(ta.shape)), str(tuple(tb.shape)), "EXPANDED"))
        else:
            diff = (tb - ta).norm().item()
            rows.append((diff, key, str(tuple(ta.shape)), str(tuple(tb.shape)), ""))
        total_params += ta.numel()
        total_changed += diff

    # Sort by absolute change descending
    rows.sort(reverse=True)

    print(f"{'L2 diff':>12}  {'key':<50}  {'shape_A':>18} -> {'shape_B':<18} note")
    print("-" * 115)
    for diff, key, sa, sb, note in rows:
        print(f"{diff:12.4f}  {key:<50}  {sa:>18} -> {sb:<18} {note}")

    print(f"\nTotal param count (A): {total_params:,}")
    print(f"Sum of L2 norms of all diffs: {total_changed:.4f}")
    print()
    # Highlight the new-obs input layers specifically
    rnn_input_layers = [k for k in keys_a if "weight_ih" in k]
    if rnn_input_layers:
        print("=== RNN input weight changes (these carry the obs-expansion) ===")
        for key in sorted(rnn_input_layers):
            ta = a[key].float()
            tb = b[key].float()
            shared = ta.shape[1] if ta.ndim == 2 else ta.shape[0]
            if ta.shape != tb.shape:
                diff_shared = (tb[:, :shared] - ta).norm().item()
                new_cols = tb[:, shared:].norm().item()
                print(f"  {key}: shared L2 diff={diff_shared:.4f}  new-col L2={new_cols:.4f}  (new-cols should grow from 0)")
            else:
                diff = (tb - ta).norm().item()
                print(f"  {key}: L2 diff={diff:.4f}")


if __name__ == "__main__":
    main()
