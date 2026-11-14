#!/usr/bin/env python3
"""Initial register block for a fixed matmul compute budget, without HLS search.

Among exact factorizations, minimize input values consumed per reduction step
(rows + columns). This chooses 8x8 for 64 lanes. Storage feasibility, achieved
II and physical resources require subsequent compiler/HLS/FPGA checks. This
does not select fusion/reuse or claim to find the fastest workload-specific tile.
"""
import argparse
import math


def balanced_block(lanes):
    if not 1 <= lanes <= 256:
        raise ValueError("matmul compute budget must be in 1..256 lanes")
    rows = max(r for r in range(1, math.isqrt(lanes) + 1) if lanes % r == 0)
    return rows, lanes // rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lanes", type=int)
    args = parser.parse_args()
    try:
        rows, cols = balanced_block(args.lanes)
    except ValueError as error:
        parser.error(str(error))
    print(f"{rows},{cols}")
