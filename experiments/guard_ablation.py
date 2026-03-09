#!/usr/bin/env python3
"""
Guard Ablation Study
══════════════════════════════

Analyzes the necessity of the proposed two-guard design by comparing the
performance of four heuristic policies:
  1. Base (splits=1 unconditionally inside boundary)
  2. No Shortcut (efficiency loop runs unconditionally) -> regresses L_K=256
  3. Relaxed (guard lowered to nBlk<=2)               -> misses win at L_K=512
  4. Fix (the proposed two-guard design)              -> optimal across all

Output: results/guard_ablation.json
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.heuristics_reference import (
    baseline_num_splits,
    latest_stack_tuned_num_splits,
    no_shortcut_num_splits,
    relaxed_num_splits,
    _ceil_div,
)
from src.bench_utils import (
    build_flash_kwargs,
    make_decode_tensors,
    measure_kernel_us,
    DEFAULT_WARMUPS,
    DEFAULT_TOTAL_ITERS,
    DEFAULT_SAMPLE_ITERS,
)


def run(quick: bool = False):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)

    # For table 8, we look at B=1, H_KV=8 across three specific L_K values
    configs = [
        {"L_K": 256, "desc": "nblk=2 (Short)"},
        {"L_K": 512, "desc": "nblk=4 (Medium)"},
        {"L_K": 1024, "desc": "nblk=8 (Long)"},
    ]

    total_iters = 2000 if quick else 10000

    results = []
    print(f"Device: {device_name} ({sm_count} SMs)\n")
    header = f"{'L_K':<6} {'Base':<12} {'NoShortcut':<12} {'Relaxed':<12} {'Fix':<12}"
    print(header)
    print("-" * 60)

    for cfg in configs:
        B, H_KV, L_K = 1, 8, cfg["L_K"]
        hq = max(H_KV, 64)

        # Get policy decisions (num_splits)
        s_base = baseline_num_splits(b=B, hkv=H_KV, lq=1, lk=L_K, d=128, num_sms=sm_count)
        s_noshrt = no_shortcut_num_splits(b=B, hkv=H_KV, lq=1, lk=L_K, d=128, num_sms=sm_count)
        s_relax = relaxed_num_splits(b=B, hkv=H_KV, lq=1, lk=L_K, d=128, num_sms=sm_count)
        s_fix = latest_stack_tuned_num_splits(b=B, hkv=H_KV, lq=1, lk=L_K, d=128, num_sms=sm_count)

        # We need to measure latency for unique split values
        unique_splits = set([s_base, s_noshrt, s_relax, s_fix])
        latencies = {}

        q, k_cache, v_cache, cs, rcos, rsin, k, v = make_decode_tensors(
            b=B, lk=L_K, hq=hq, hkv=H_KV,
        )

        for s in unique_splits:
            kwargs = build_flash_kwargs(
                batch=B, lk=L_K, hq=hq, hkv=H_KV, d=128,
                cache_seqlens=cs, num_splits=s, pack_gqa=None,
            )
            lat = measure_kernel_us(
                q=q, k_cache=k_cache, v_cache=v_cache,
                cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
                k=k, v=v, num_splits=s, flash_kwargs=kwargs,
                warmups=DEFAULT_WARMUPS, total_iters=total_iters, sample_iters=DEFAULT_SAMPLE_ITERS
            )
            latencies[s] = lat

        lat_base = latencies[s_base]
        lat_noshrt = latencies[s_noshrt]
        lat_relax = latencies[s_relax]
        lat_fix = latencies[s_fix]

        print(f"{L_K:<6} {lat_base:<7.2f}(s={s_base}) {lat_noshrt:<7.2f}(s={s_noshrt}) "
              f"{lat_relax:<7.2f}(s={s_relax}) {lat_fix:<7.2f}(s={s_fix})")

        results.append({
            "L_K": L_K,
            "H_KV": H_KV,
            "B": B,
            "desc": cfg["desc"],
            "base": {"splits": s_base, "latency_us": round(lat_base, 2)},
            "no_shortcut": {"splits": s_noshrt, "latency_us": round(lat_noshrt, 2)},
            "relaxed": {"splits": s_relax, "latency_us": round(lat_relax, 2)},
            "fix": {"splits": s_fix, "latency_us": round(lat_fix, 2)},
        })

    os.makedirs("results", exist_ok=True)
    output = {
        "experiment": "guard_ablation",
        "device": device_name,
        "sm_count": sm_count,
        "results": results,
    }
    with open("results/guard_ablation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/guard_ablation.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Guard Ablation (Table 8)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer sampling iterations")
    args = parser.parse_args()
    run(quick=args.quick)
