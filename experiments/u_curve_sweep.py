#!/usr/bin/env python3
"""
U-Curve Sweep (Figure 2b)
═════════════════════════
Paper reference: Section 4.3, Figure 2b

Sweeps split values manually in *eager* mode (no CUDA graphs).
Eager mode includes python dispatch overhead (~30-55us), measuring the
true "service latency" (T_service = T_dispatch + T_kernel).
This reproduces the U-shaped curve that demonstrates why 4 (or 3) splits
is optimal, as excess splits incur linear dispatch penalties masking kernel gains.

Output: results/u_curve_sweep.json
"""

import sys
import os
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.bench_utils import make_decode_tensors, measure_eager_us


def run(quick: bool = False):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)

    # 1000 iter standard, 200 iter quick
    iters = 200 if quick else 1000

    # Test MQA boundary case: L_K=512
    B, H_KV, L_K = 1, 1, 512
    hq = max(H_KV, 64)

    print(f"Device: {device_name} ({sm_count} SMs)")
    print(f"Eager-mode sweep for B={B}, H_KV={H_KV}, L_K={L_K} (MQA boundary)")
    print("Measuring T_service (includes Python dispatch overhead)\n")

    splits_to_test = [1, 2, 3, 4, 8, 16]
    results = []

    q, k_cache, v_cache, cs, rcos, rsin, k, v = make_decode_tensors(
        b=B, lk=L_K, hq=hq, hkv=H_KV,
    )

    header = f"{'Splits':<8} {'Service Latency (us)'}"
    print(header)
    print("-" * 35)

    for s in splits_to_test:
        lat = measure_eager_us(
            q=q, k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
            k=k, v=v, num_splits=s,
            iterations=iters,
        )
        print(f"{s:<8} {lat:.2f}")

        results.append({
            "splits": s,
            "service_latency_us": round(lat, 2),
        })

    os.makedirs("results", exist_ok=True)
    output = {
        "experiment": "u_curve_sweep",
        "paper_reference": "Figure 2b",
        "device": device_name,
        "config": {"B": B, "H_KV": H_KV, "L_K": L_K},
        "results": results,
    }
    with open("results/u_curve_sweep.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/u_curve_sweep.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)
