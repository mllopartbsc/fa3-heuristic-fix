#!/usr/bin/env python3
"""
CI-Style Before/After Benchmark Report

Outputs exact before/after kernel timings in a machine-readable format suitable
for FA3's CI or maintainer hardware verification. Reads from main_results.json
(produced by experiments/main_results.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.track_config import add_results_dir_argument, add_track_argument


def main():
    parser = argparse.ArgumentParser(
        description="Generate CI-style before/after benchmark report"
    )
    add_track_argument(parser)
    add_results_dir_argument(parser)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    results_dir = args.results_dir or (Path("results") / args.track)
    main_path = results_dir / "main_results.json"

    if not main_path.exists():
        print(f"ERROR: {main_path} not found. Run main_results first:", file=sys.stderr)
        print("  python3 experiments/main_results.py --track", args.track, file=sys.stderr)
        sys.exit(1)

    with open(main_path) as f:
        data = json.load(f)

    report = {
        "format": "fa3_heuristic_fix_ci_v1",
        "device": data.get("device", "unknown"),
        "sm_count": data.get("sm_count"),
        "track": args.track,
        "before_after": [],
    }

    for row in data.get("results", []):
        report["before_after"].append({
            "L_K": row["L_K"],
            "H_KV": row["H_KV"],
            "B": row.get("B", 1),
            "tiles": row.get("tiles"),
            "regime": row.get("regime", ""),
            "baseline_us": row["baseline_median_us"],
            "patched_us": row["fix_median_us"],
            "speedup": row["speedup"],
            "splits_baseline": row.get("splits_base"),
            "splits_patched": row.get("splits_fix"),
        })

    out = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out)
        print(f"Report written to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
