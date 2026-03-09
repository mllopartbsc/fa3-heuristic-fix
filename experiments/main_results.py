#!/usr/bin/env python3
"""
Main Results: Kernel-Level Latency
════════════════════════════════════════════

A/B interleaved kernel-level benchmarks with CUDA Graph capture.
Reports median, P5, P95 for headline configurations.

Expected results (H100 SXM5):
  - MQA (H_KV=1) at L_K=512: 1.12x speedup, non-overlapping P5/P95
  - GQA-2 (H_KV=2) at L_K=512: 1.15x speedup
  - All other configs: 1.00x (neutral)

Output: results/main_results.json
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.heuristics_reference import baseline_num_splits, latest_stack_tuned_num_splits
from src.bench_utils import (
    build_flash_kwargs,
    make_decode_tensors,
    measure_kernel_us_detailed,
    measure_ab_interleaved,
    DEFAULT_WARMUPS,
    DEFAULT_SAMPLE_ITERS,
)


def run(quick: bool = False):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name} ({sm_count} SMs)")
    print(f"Protocol: CUDA Graphs, A/B interleaved, {'2K' if quick else '10K'} iters, 200 warmups")
    print()

    # In quick mode use fewer rounds to cut runtime ~5x
    ab_rounds = 5 if quick else 20
    ab_iters_per_round = 100 if quick else 500

    configs = [
        # Safe regime
        {"B": 1, "H_KV": 8, "L_K": 128, "regime": "safe"},
        {"B": 1, "H_KV": 8, "L_K": 256, "regime": "safe"},
        {"B": 1, "H_KV": 1, "L_K": 384, "regime": "safe"},
        {"B": 1, "H_KV": 8, "L_K": 384, "regime": "safe"},
        {"B": 1, "H_KV": 8, "L_K": 512, "regime": "safe"},
        # Win regime
        {"B": 1, "H_KV": 1, "L_K": 512, "regime": "win"},
        {"B": 1, "H_KV": 2, "L_K": 512, "regime": "win"},
        # Long context
        {"B": 1, "H_KV": 8, "L_K": 2048, "regime": "long"},
        {"B": 1, "H_KV": 8, "L_K": 4096, "regime": "long"},
    ]

    results = []

    header = (f"{'L_K':<6} {'H_KV':<6} {'Tiles':<6} "
              f"{'Base (us)':<14} {'Fix (us)':<14} {'Speedup':<9} {'Regime'}")
    print(header)
    print("-" * 80)

    for cfg in configs:
        B, H_KV, L_K = cfg["B"], cfg["H_KV"], cfg["L_K"]
        tiles = B * H_KV
        hq = max(H_KV, 64)

        s_base = baseline_num_splits(b=B, hkv=H_KV, lq=1, lk=L_K, d=128, num_sms=sm_count)
        s_fix = latest_stack_tuned_num_splits(b=B, hkv=H_KV, lq=1, lk=L_K, d=128, num_sms=sm_count)

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

        if s_base != s_fix:
            # A/B interleaved for changed configs
            ab = measure_ab_interleaved(
                q=q, k_cache=k_cache, v_cache=v_cache,
                cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
                k=k, v=v, splits_a=s_base, splits_b=s_fix,
                flash_kwargs_a=base_kwargs, flash_kwargs_b=fix_kwargs,
                warmups=DEFAULT_WARMUPS, rounds=ab_rounds, iters_per_round=ab_iters_per_round,
                sample_iters=DEFAULT_SAMPLE_ITERS,
            )
            entry = {
                **cfg, "tiles": tiles,
                "splits_base": s_base, "splits_fix": s_fix,
                "baseline_median_us": round(ab["a_median"], 2),
                "fix_median_us": round(ab["b_median"], 2),
                "baseline_p5": round(ab["a_p5"], 2),
                "baseline_p95": round(ab["a_p95"], 2),
                "fix_p5": round(ab["b_p5"], 2),
                "fix_p95": round(ab["b_p95"], 2),
                "speedup": round(ab["speedup"], 4),
                "significant": ab["b_p95"] < ab["a_p5"],
            }
            base_str = f"{ab['a_median']:7.2f} ± {ab['a_p95']-ab['a_median']:.2f}"
            fix_str = f"{ab['b_median']:7.2f} ± {ab['b_p95']-ab['b_median']:.2f}"
            spd_str = f"{ab['speedup']:.3f}x"
        else:
            det = measure_kernel_us_detailed(
                q=q, k_cache=k_cache, v_cache=v_cache,
                cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
                k=k, v=v, num_splits=s_base,
                flash_kwargs=base_kwargs,
            )
            entry = {
                **cfg, "tiles": tiles,
                "splits_base": s_base, "splits_fix": s_fix,
                "baseline_median_us": round(det["median"], 2),
                "fix_median_us": round(det["median"], 2),
                "speedup": 1.0,
                "significant": False,
            }
            base_str = f"{det['median']:7.2f}"
            fix_str = f"{det['median']:7.2f}"
            spd_str = "1.00x"

        results.append(entry)
        print(f"{L_K:<6} {H_KV:<6} {tiles:<6} {base_str:<14} {fix_str:<14} {spd_str:<9} {cfg['regime']}")

    os.makedirs("results", exist_ok=True)
    output = {
        "experiment": "main_results",
        "device": device_name,
        "sm_count": sm_count,
        "protocol": "CUDA Graphs, A/B interleaved, precomputed scheduler metadata, 10K iters, 200 warmups",
        "results": results,
    }
    with open("results/main_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/main_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Main Results: kernel-level A/B benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer rounds (less statistically precise)")
    args = parser.parse_args()
    run(quick=args.quick)
