#!/usr/bin/env python3
"""
Generate LaTeX Tables from JSON Results
═══════════════════════════════════════
Reads the output JSON files in results/ and generates a single
LaTeX file containing Tables 3, 5, 6, 8, 9 from the paper.

Usage:
  python3 scripts/generate_tables.py --results-dir results/ --output-tex results/tables.tex
"""

import argparse
import json
import os
import sys

def _load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results")
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output-tex", default="results/tables.tex")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_files = []

    # Table 5: Main Results
    main_res = _load_json(os.path.join(results_dir, "main_results.json"))
    if main_res and "results" in main_res:
        with open(args.output_tex, "w") as f:
            f.write("% Table 5: Main Results (Generated)\n")
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
            f.write("$L_K$ & $H_{KV}$ & Baseline (\\si{\\us}) & Fix (\\si{\\us}) & Speedup \\\\\n\\midrule\n")
            for r in main_res["results"]:
                lk = r["L_K"]
                hkv = r["H_KV"]
                base = f"{r['baseline_median_us']:.1f}"
                fix = f"{r['fix_median_us']:.1f}"
                spd = f"{r['speedup']:.2f}\\times"
                sig = "*" if r.get("significant", False) else ""
                f.write(f"{lk} & {hkv} & {base} & {fix} & {spd}{sig} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    # This is a simplified stub. In a full reproducible workflow, this script
    # formats all the resulting JSON files into the exact LaTeX tables used in
    # the conference submission.
    print(f"Tables written to {args.output_tex} (simplified output)")

if __name__ == "__main__":
    main()
