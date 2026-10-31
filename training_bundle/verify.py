#!/usr/bin/env python
"""Check the trained model and report the numbers to send back."""
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "pid"
    w = HERE / "runs" / name / "weights" / "best.pt"
    if not w.exists():
        sys.exit(f"No model at {w}. Has training finished?")

    csv = HERE / "runs" / name / "results.csv"
    if csv.exists():
        rows = [r.split(",") for r in csv.read_text().strip().split("\n")]
        head = [h.strip() for h in rows[0]]
        try:
            i50 = next(i for i, h in enumerate(head) if "mAP50(B)" in h)
            i95 = next(i for i, h in enumerate(head) if "mAP50-95(B)" in h)
            best50 = max(float(r[i50]) for r in rows[1:])
            best95 = max(float(r[i95]) for r in rows[1:])
            print(f"epochs completed : {len(rows)-1}")
            print(f"best mAP50       : {best50:.4f}")
            print(f"best mAP50-95    : {best95:.4f}")
            if best50 >= 0.90:
                print("\nExcellent — send it back.")
            elif best50 >= 0.85:
                print("\nGood — send it back.")
            else:
                print("\nLower than hoped. Send it anyway and mention the number;")
                print("more epochs or a larger model may help.")
        except StopIteration:
            print("(could not parse results.csv — send it anyway)")

    print(f"\nmodel size: {w.stat().st_size/1e6:.1f} MB")
    print("\nSEND BACK:")
    print(f"  {w.relative_to(HERE)}")
    if csv.exists():
        print(f"  {csv.relative_to(HERE)}")


if __name__ == "__main__":
    main()
