#!/usr/bin/env python3
"""
Experiment 3: Safety Contract Verification (160-Configuration Sweep)
════════════════════════════════════════════════════════════════════
Paper reference: Section 4.5, Tables 4 (Policy Matrix), 7 (Safety Contract)

Sweeps 160 configurations: B ∈ {1,2,4,8} × H_KV ∈ {1,2,4,8,32} × L_K ∈ {128,...,8192}
Verifies zero regressions for the tile-aware fix (Policy C).

Protocol:
  - Configs where fix changes splits: A/B interleaved measurement
  - Configs where fix is a no-op: verified by heuristic logic (unchanged splits)
  - Regression threshold: speedup < 0.97x
  - Win threshold: speedup > 1.03x

Output: results/exp3_safety_verification.json
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.heuristics_reference import baseline_num_splits, latest_stack_tuned_num_splits, _ceil_div
from src.bench_utils import (
    build_flash_kwargs,
    make_decode_tensors,
    measure_ab_interleaved,
    DEFAULT_WARMUPS,
    DEFAULT_SAMPLE_ITERS,
)


def run(quick: bool = False):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name} ({sm_count} SMs)")
    print()

    ab_rounds = 5 if quick else 20
    ab_iters_per_round = 100 if quick else 500

    # Build the 160-configuration matrix
    B_values = [1, 2, 4, 8]
    H_values = [1, 2, 4, 8, 32]
    L_values = [128, 256, 384, 512, 1024, 2048, 4096, 8192]

    configs = [(B, H, L) for B in B_values for H in H_values for L in L_values]
    assert len(configs) == 160, f"Expected 160 configs, got {len(configs)}"
    print(f"Testing {len(configs)} configurations (4 × 5 × 8 = 160)")
    print()

    wins = 0
    regressions = 0
    neutrals = 0
    unchanged = 0
    results = []

    header = (f"{'Config':<22} {'tiles':<6} {'nBlk':<5} {'Base(s)':<8} "
              f"{'Fix(s)':<7} {'Base us':<10} {'Fix us':<10} {'Speedup':<9} {'Result'}")
    print(header)
    print("-" * 110)

    for B, H_KV, L_K in configs:
        tiles = B * H_KV
        nblk = _ceil_div(L_K, 128)
        hq = max(H_KV, 64)  # GQA: at least 64 query heads

        s_base = baseline_num_splits(b=B, hkv=H_KV, lq=1, lk=L_K, d=128, num_sms=sm_count)
        s_fix = latest_stack_tuned_num_splits(b=B, hkv=H_KV, lq=1, lk=L_K, d=128, num_sms=sm_count)

        tag = f"B={B},H={H_KV},L={L_K}"

        if s_base == s_fix:
            unchanged += 1
            results.append({
                "B": B, "H_KV": H_KV, "L_K": L_K,
                "tiles": tiles, "nblk": nblk,
                "splits_base": s_base, "splits_fix": s_fix,
                "result": "UNCHANGED",
            })
            print(f"{tag:<22} {tiles:<6} {nblk:<5} s={s_base:<5} s={s_fix:<4} "
                  f"{'—':>10} {'—':>10} {'—':>9} UNCHANGED")
            continue

        # Splits differ — measure with A/B interleaving
        q, k_cache, v_cache, cs, rcos, rsin, k, v = make_decode_tensors(
            b=B, lk=L_K, hq=hq, hkv=H_KV,
        )
        base_kwargs = build_flash_kwargs(
            batch=B, lk=L_K, hq=hq, hkv=H_KV, d=128,
            cache_seqlens=cs, num_splits=s_base, pack_gqa=None,
        )
        fix_kwargs = build_flash_kwargs(
            batch=B, lk=L_K, hq=hq, hkv=H_KV, d=128,
            cache_seqlens=cs, num_splits=s_fix, pack_gqa=None,
        )

        try:
            ab = measure_ab_interleaved(
                q=q, k_cache=k_cache, v_cache=v_cache,
                cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
                k=k, v=v, splits_a=s_base, splits_b=s_fix,
                flash_kwargs_a=base_kwargs, flash_kwargs_b=fix_kwargs,
                warmups=DEFAULT_WARMUPS, rounds=ab_rounds, iters_per_round=ab_iters_per_round,
                sample_iters=DEFAULT_SAMPLE_ITERS,
            )
            lat_b = ab["a_median"]
            lat_f = ab["b_median"]
            spd = lat_b / lat_f if lat_f > 0 else 1.0

            if spd > 1.03:
                result = "WIN"
                wins += 1
            elif spd < 0.97:
                result = "REGRESSION"
                regressions += 1
            else:
                result = "NEUTRAL"
                neutrals += 1

            results.append({
                "B": B, "H_KV": H_KV, "L_K": L_K,
                "tiles": tiles, "nblk": nblk,
                "splits_base": s_base, "splits_fix": s_fix,
                "baseline_us": round(lat_b, 2),
                "fix_us": round(lat_f, 2),
                "speedup": round(spd, 4),
                "result": result,
            })

            print(f"{tag:<22} {tiles:<6} {nblk:<5} s={s_base:<5} s={s_fix:<4} "
                  f"{lat_b:8.2f}us {lat_f:8.2f}us {spd:7.3f}x  {result}")

        except Exception as e:
            results.append({
                "B": B, "H_KV": H_KV, "L_K": L_K,
                "tiles": tiles, "nblk": nblk,
                "splits_base": s_base, "splits_fix": s_fix,
                "result": "ERROR", "error": str(e),
            })
            print(f"{tag:<22} {tiles:<6} {nblk:<5} ERROR: {e}")

    print()
    print(f"Summary: {wins} WIN, {regressions} REGRESSION, {neutrals} NEUTRAL, {unchanged} UNCHANGED")
    print(f"Total: {len(configs)} configurations")
    if regressions == 0:
        print(f"*** ZERO REGRESSIONS across {len(configs)} configurations ***")
    else:
        print(f"*** {regressions} REGRESSIONS FOUND ***")

    os.makedirs("results", exist_ok=True)
    output = {
        "experiment": "exp3_safety_verification",
        "paper_reference": "Section 4.5, Tables 4, 7",
        "device": device_name,
        "sm_count": sm_count,
        "total_configs": len(configs),
        "wins": wins,
        "regressions": regressions,
        "neutrals": neutrals,
        "unchanged": unchanged,
        "verdict": "PASS (zero regressions)" if regressions == 0 else "FAIL",
        "results": results,
    }
    with open("results/exp3_safety_verification.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/exp3_safety_verification.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 3: Safety Contract Verification")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer A/B rounds per config")
    args = parser.parse_args()
    run(quick=args.quick)
