#!/usr/bin/env python3
"""
Experiment 2: Mechanism Confirmation via Profiling
═══════════════════════════════════════════════════
Paper reference: Section 4.2, Table 3, Figure 2a

Measures SM throughput, active warps, and grid size under baseline (splits=1)
vs forced splitting (splits=16) to confirm the grid-underfill hypothesis.

This script provides the data for Table 3 (Counter Attribution) and
Figure 2a (Reconstructed Decode Timeline).

NOTE: For full Nsight Compute profiling, use:
  ncu --set full --target-processes all python experiments/exp2_mechanism_profiling.py

Output: results/exp2_profiling.json
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.bench_utils import make_decode_tensors, measure_kernel_us

try:
    import flash_attn_interface
except ImportError:
    print("[ERROR] flash_attn_interface not found.")
    sys.exit(1)


def run():
    device = "cuda"
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name} ({sm_count} SMs)")
    print()

    configs = [
        {"B": 1, "H_KV": 8, "L_K": 128, "desc": "Short KV (underfill case)"},
        {"B": 1, "H_KV": 8, "L_K": 512, "desc": "Medium KV (target fix region)"},
    ]

    results = []

    for cfg in configs:
        B, H_KV, L_K = cfg["B"], cfg["H_KV"], cfg["L_K"]
        tiles = B * H_KV
        print(f"=== {cfg['desc']}: B={B}, H_KV={H_KV}, L_K={L_K} ===")
        print(f"  Baseline grid: {tiles} CTAs = {tiles/sm_count*100:.1f}% SM coverage")
        print(f"  Forced grid:   {tiles*16} CTAs = {min(tiles*16/sm_count*100, 100):.1f}% SM coverage")

        q, k_cache, v_cache, cs, rcos, rsin, k, v = make_decode_tensors(
            b=B, lk=L_K, hkv=H_KV, d=128,
        )

        # Measure baseline (splits=1)
        lat_base = measure_kernel_us(
            q=q, k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
            k=k, v=v, num_splits=1,
        )

        # Measure forced (splits=16)
        lat_forced = measure_kernel_us(
            q=q, k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
            k=k, v=v, num_splits=16,
        )

        speedup = lat_base / lat_forced if lat_forced > 0 else 1.0

        entry = {
            "config": cfg,
            "baseline_splits": 1,
            "forced_splits": 16,
            "baseline_ctas": tiles,
            "forced_ctas": tiles * 16,
            "baseline_sm_coverage_pct": round(tiles / sm_count * 100, 2),
            "forced_sm_coverage_pct": round(min(tiles * 16 / sm_count * 100, 100), 2),
            "baseline_latency_us": round(lat_base, 2),
            "forced_latency_us": round(lat_forced, 2),
            "speedup": round(speedup, 3),
        }
        results.append(entry)

        print(f"  Baseline latency: {lat_base:.2f} us")
        print(f"  Forced latency:   {lat_forced:.2f} us")
        print(f"  Speedup:          {speedup:.3f}x")
        print()

    print("NOTE: For full Nsight Compute metrics (sm__throughput, warps_active,")
    print("      dram__throughput), run this script under ncu:")
    print("  ncu --set full python experiments/exp2_mechanism_profiling.py")

    os.makedirs("results", exist_ok=True)
    output = {
        "experiment": "exp2_mechanism_profiling",
        "paper_reference": "Section 4.2, Table 3, Figure 2a",
        "device": device_name,
        "sm_count": sm_count,
        "results": results,
    }
    with open("results/exp2_profiling.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to results/exp2_profiling.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 2: Mechanism Profiling")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer timing iterations")
    args = parser.parse_args()
    # Pass quick flag via global so bench_utils can reduce iters
    if args.quick:
        import src.bench_utils as _bu
        _bu.DEFAULT_TOTAL_ITERS = 2000
    run()
