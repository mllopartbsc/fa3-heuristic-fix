#!/usr/bin/env python3
"""
Generate reviewer-facing LaTeX tables from track-scoped result JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.track_config import (
    TRACK_UPSTREAM_PATCH,
    add_track_argument,
    add_results_dir_argument,
    artifacts_dir_for_track,
)


def _load_json(path: Path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def _candidate_label(track: str) -> str:
    return "Upstream Patch" if track == TRACK_UPSTREAM_PATCH else "Latest-Stack Tuned"


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results")
    add_track_argument(parser)
    add_results_dir_argument(parser)
    parser.add_argument("--output-tex", default=None, type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir or (Path("results") / args.track)
    output_tex = args.output_tex or (artifacts_dir_for_track(args.track) / "tables.tex")
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    main_res = _load_json(results_dir / "main_results.json")
    boundary = _load_json(results_dir / "boundary_sweep.json")

    with open(output_tex, "w") as f:
        f.write("% Reviewer-facing tables generated from track-scoped results\n")
        f.write(f"% track: {args.track}\n\n")

        if main_res and "results" in main_res:
            label = _candidate_label(args.track)
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write(f"\\caption{{Main Results ({label})}}\n")
            f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
            f.write("$L_K$ & $H_{KV}$ & Baseline (\\si{\\us}) & Candidate (\\si{\\us}) & Speedup \\\\\n\\midrule\n")
            for row in main_res["results"]:
                lk = row["L_K"]
                hkv = row["H_KV"]
                base = f"{row['baseline_median_us']:.2f}"
                cand = f"{row['fix_median_us']:.2f}"
                spd = f"{row['speedup']:.2f}\\times"
                sig = "*" if row.get("significant", False) else ""
                f.write(f"{lk} & {hkv} & {base} & {cand} & {spd}{sig} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        if boundary and "results" in boundary:
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\caption{Boundary Sweep Summary}\n")
            f.write("\\begin{tabular}{rrrrr}\n\\toprule\n")
            f.write("$L_K$ & nBlk & Baseline (\\si{\\us}) & Candidate (\\si{\\us}) & Speedup \\\\\n\\midrule\n")
            for row in boundary["results"]:
                f.write(
                    f"{row['L_K']} & {row['nblk']} & "
                    f"{row['baseline_us']:.2f} & {row['fix_us']:.2f} & "
                    f"{row['speedup']:.2f}\\times \\\\\n"
                )
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"Tables written to {output_tex}")


if __name__ == "__main__":
    main()
