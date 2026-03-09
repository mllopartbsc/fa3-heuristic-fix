#!/usr/bin/env python3
"""
Validate headline claims against expected tolerances.

Reads results/*.json plus expected_results/claims.json and
produces a claim-by-claim verdict. Called by reproduce.py after all
experiments have completed.

Usage:
  python3 src/validate_claims.py
  python3 src/validate_claims.py --results-dir results/ --claims-file expected_results/claims.json
"""

import argparse
import json
import sys
from pathlib import Path


def _bold(t):
    return f"\033[1m{t}\033[0m" if sys.stdout.isatty() else t
def _green(t):
    return f"\033[32;1m{t}\033[0m" if sys.stdout.isatty() else t
def _red(t):
    return f"\033[31;1m{t}\033[0m" if sys.stdout.isatty() else t
def _yellow(t):
    return f"\033[33;1m{t}\033[0m" if sys.stdout.isatty() else t


def load_result(results_dir: Path, experiment: str) -> dict | None:
    path = results_dir / f"{experiment}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def find_in_results(results_list: list, key: str, val, match_key: str = None):
    """Find a result entry where result[key]==val, optionally returning result[match_key]."""
    for r in results_list:
        if r.get(key) == val:
            return r if match_key is None else r.get(match_key)
    return None


def find_pair(results_list: list, filters: dict):
    """Find entry matching ALL key-value pairs in filters dict."""
    for r in results_list:
        if all(r.get(k) == v for k, v in filters.items()):
            return r
    return None


def validate(results_dir: Path, claims_file: Path) -> tuple[list, list, list]:
    """Returns (passed, failed, skipped) lists of (claim_name, message) tuples."""
    passed = []
    failed = []
    skipped = []

    if not claims_file.exists():
        skipped.append(("all", f"Claims file not found: {claims_file}"))
        return passed, failed, skipped

    with open(claims_file) as f:
        claims_spec = json.load(f)

    headline = claims_spec.get("headline_claims", {})

    # ── 1. MQA speedup ──────────────────────────────────────────────────────
    claim_name = "mqa_512_speedup"
    spec = headline.get(claim_name)
    if spec:
        data = load_result(results_dir, "main_results")
        if data:
            entry = find_pair(data.get("results", []), {"L_K": 512, "H_KV": 1})
            if entry:
                spd = entry.get("speedup", 0.0)
                lo, hi = spec["min"], spec["max"]
                if lo <= spd <= hi:
                    passed.append((claim_name, f"{spd:.3f}x ∈ [{lo:.2f}, {hi:.2f}]"))
                else:
                    failed.append((claim_name,
                                   f"{spd:.3f}x NOT in [{lo:.2f}, {hi:.2f}] "
                                   f"(expected: {spec['expected_value']})"))
            else:
                skipped.append((claim_name, "L_K=512, H_KV=1 entry not found in main_results"))
        else:
            skipped.append((claim_name, "main_results.json not found"))

    # ── 2. GQA-2 speedup ──────────────────────────────────────────────────
    claim_name = "gqa2_512_speedup"
    spec = headline.get(claim_name)
    if spec:
        data = load_result(results_dir, "main_results")
        if data:
            entry = find_pair(data.get("results", []), {"L_K": 512, "H_KV": 2})
            if entry:
                spd = entry.get("speedup", 0.0)
                lo, hi = spec["min"], spec["max"]
                if lo <= spd <= hi:
                    passed.append((claim_name, f"{spd:.3f}x ∈ [{lo:.2f}, {hi:.2f}]"))
                else:
                    failed.append((claim_name,
                                   f"{spd:.3f}x NOT in [{lo:.2f}, {hi:.2f}] "
                                   f"(expected: {spec['expected_value']})"))
            else:
                skipped.append((claim_name, "L_K=512, H_KV=2 entry not found in main_results"))
        else:
            skipped.append((claim_name, "main_results.json not found"))

    # ── 3. Statistical significance ────────────────────────────────────────
    claim_name = "mqa_512_statistical_significance"
    spec = headline.get(claim_name)
    if spec:
        data = load_result(results_dir, "main_results")
        if data:
            entry = find_pair(data.get("results", []), {"L_K": 512, "H_KV": 1})
            if entry:
                sig = entry.get("significant", False)
                if sig:
                    passed.append((claim_name, "non-overlapping P5/P95 confirmed"))
                else:
                    failed.append((claim_name, "P5/P95 intervals overlap — not statistically significant"))
            else:
                skipped.append((claim_name, "L_K=512 H_KV=1 entry not found"))
        else:
            skipped.append((claim_name, "main_results.json not found"))

    # ── 4. Zero regressions ────────────────────────────────────────────────
    claim_name = "zero_regressions"
    spec = headline.get(claim_name)
    if spec:
        data = load_result(results_dir, "exp3_safety_verification")
        if data:
            reg = data.get("regressions", -1)
            required = spec.get("required_regressions", 0)
            if reg == required:
                passed.append((claim_name,
                               f"{reg} regressions ✓ — {data.get('wins', 0)} wins, "
                               f"{data.get('unchanged', 0)} unchanged"))
            else:
                failed.append((claim_name,
                               f"{reg} regressions found (required: {required})"))
        else:
            skipped.append((claim_name, "exp3_safety_verification.json not found"))

    # ── 5. Correctness verdict ─────────────────────────────────────────────
    claim_name = "correctness_verdict"
    spec = headline.get(claim_name)
    if spec:
        data = load_result(results_dir, "exp1_correctness")
        if data:
            verdict = data.get("verdict", "UNKNOWN")
            required = spec.get("required_verdict", "PASS")
            if verdict == required:
                n = data.get("total_trials", 0)
                passed.append((claim_name, f"{verdict} ({n} trials, 0 failures)"))
            else:
                failures = data.get("failures", "?")
                failed.append((claim_name, f"{verdict} — {failures} failures"))
        else:
            skipped.append((claim_name, "exp1_correctness.json not found"))

    # ── 6. Correctness max abs error ───────────────────────────────────────
    claim_name = "correctness_max_abs_error"
    spec = headline.get(claim_name)
    if spec:
        data = load_result(results_dir, "exp1_correctness")
        if data:
            worst = data.get("worst_case_abs_error", float("inf"))
            limit = spec.get("max", 1e-3)
            if worst <= limit:
                passed.append((claim_name, f"{worst:.2e} ≤ {limit:.0e}"))
            else:
                failed.append((claim_name, f"{worst:.2e} > {limit:.0e} (exceeds threshold)"))
        else:
            skipped.append((claim_name, "exp1_correctness.json not found"))

    # ── 7. Safe regime: MQA at L_K=384 neutral ────────────────────────────
    claim_name = "safe_regime_neutral_mqa_384"
    spec = headline.get(claim_name)
    if spec:
        data = load_result(results_dir, "boundary_sweep")
        if data:
            entry = find_pair(data.get("results", []), {"L_K": 384})
            if entry:
                spd = entry.get("speedup", 1.0)
                lo, hi = spec["min_speedup"], spec["max_speedup"]
                if lo <= spd <= hi:
                    passed.append((claim_name, f"speedup={spd:.3f}x ∈ [{lo},{hi}] — neutral ✓"))
                else:
                    failed.append((claim_name,
                                   f"speedup={spd:.3f}x NOT in [{lo},{hi}] — regression risk"))
            else:
                skipped.append((claim_name, "L_K=384 entry not found in boundary_sweep"))
        else:
            skipped.append((claim_name, "boundary_sweep.json not found"))

    # ── 8. Safe regime: GQA-8 at L_K=512 neutral (Guard 2) ───────────────
    claim_name = "safe_regime_neutral_gqa8_512"
    spec = headline.get(claim_name)
    if spec:
        data = load_result(results_dir, "main_results")
        if data:
            entry = find_pair(data.get("results", []), {"L_K": 512, "H_KV": 8})
            if entry:
                spd = entry.get("speedup", 1.0)
                lo, hi = spec["min_speedup"], spec["max_speedup"]
                if lo <= spd <= hi:
                    passed.append((claim_name, f"speedup={spd:.3f}x ∈ [{lo},{hi}] — neutral ✓"))
                else:
                    failed.append((claim_name,
                                   f"speedup={spd:.3f}x NOT in [{lo},{hi}] — Guard 2 broken"))
            else:
                skipped.append((claim_name, "L_K=512, H_KV=8 entry not found in main_results"))
        else:
            skipped.append((claim_name, "main_results.json not found"))

    return passed, failed, skipped


def main():
    parser = argparse.ArgumentParser(description="Validate expected claims against experimental results.")
    parser.add_argument("--results-dir", default="results/", type=Path)
    parser.add_argument("--claims-file",
                        default="expected_results/claims.json", type=Path)
    parser.add_argument("--json-out", default=None, type=Path,
                        help="Save validation report as JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    claims_file = Path(args.claims_file)

    passed, failed, skipped = validate(results_dir, claims_file)

    total = len(passed) + len(failed) + len(skipped)
    print(f"Claim validation: {len(passed)}/{total} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    print()

    if passed:
        print(_green("  PASSED:"))
        for name, msg in passed:
            print(f"    ✓  {name}: {msg}")
        print()

    if failed:
        print(_red("  FAILED:"))
        for name, msg in failed:
            print(f"    ✗  {name}: {msg}")
        print()

    if skipped:
        print(_yellow("  SKIPPED (results not available):"))
        for name, msg in skipped:
            print(f"    −  {name}: {msg}")
        print()

    # Save JSON report
    if args.json_out:
        report = {
            "passed": [{"claim": n, "detail": m} for n, m in passed],
            "failed": [{"claim": n, "detail": m} for n, m in failed],
            "skipped": [{"claim": n, "detail": m} for n, m in skipped],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)

    # Save to standard location
    standard_out = results_dir / "claim_validation.json"
    report = {
        "passed": [{"claim": n, "detail": m} for n, m in passed],
        "failed": [{"claim": n, "detail": m} for n, m in failed],
        "skipped": [{"claim": n, "detail": m} for n, m in skipped],
    }
    with open(standard_out, "w") as f:
        json.dump(report, f, indent=2)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
