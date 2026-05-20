"""
Download TotalSegmentator model weights via its built-in download helper.

Fast single-model (default):
  python download_totalseg_weights.py

Full multi-model pipeline:
  python download_totalseg_weights.py --full

Custom weights directory:
  TOTALSEG_WEIGHTS_PATH=/my/dir python download_totalseg_weights.py
"""
import argparse
import sys
from pathlib import Path

# Make sure TotalSegmentator package is importable
TOTALSEG_SRC = Path(__file__).parent.parent / "TotalSegmentator"
if TOTALSEG_SRC.exists():
    sys.path.insert(0, str(TOTALSEG_SRC))

from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.config import get_weights_dir


def main():
    parser = argparse.ArgumentParser(description="Download TotalSegmentator weights")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also download full multi-part models (291-295 for CT, 850-851 for MR)",
    )
    args = parser.parse_args()

    weights_dir = get_weights_dir()
    print(f"Weights directory: {weights_dir}")

    # Fast all-in-one 3mm models — needed for Step 3 single-model inference
    fast_models = {
        297: "total CT (3mm fast)",
        852: "total_mr MRI (3mm fast)",
    }

    # Full multi-part models — needed for Step 4 multi-model pipeline
    full_models = {
        291: "total CT — organs",
        292: "total CT — vertebrae",
        293: "total CT — cardiac",
        294: "total CT — muscles",
        295: "total CT — ribs",
        850: "total_mr MRI — organs",
        851: "total_mr MRI — muscles",
    }

    targets = dict(fast_models)
    if args.full:
        targets.update(full_models)

    for task_id, description in targets.items():
        print(f"\n[{task_id}] {description}")
        download_pretrained_weights(task_id)
        print(f"  -> done")

    print("\nAll requested weights downloaded.")
    print(f"Location: {weights_dir}")


if __name__ == "__main__":
    main()
