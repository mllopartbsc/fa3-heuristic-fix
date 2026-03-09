#!/usr/bin/env python3
"""
MQA Crossover Boundary Sweep
══════════════════════════════════════

Sweeps L_K values focusing on MQA (H_KV=1) to confirm that 384 remains neutral
(Guard 1 trigger), 512 is a win (Guard 2 bypass), and 640/768 are neutral (baseline already splits).

Output: results/boundary_sweep.json
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
    measure_ab_interleaved,
    measure_kernel_us,
    DEFAULT_WARMUPS,
    DEFAULT_SAMPLE_ITERS,
)


def run(quick: bool = False):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)

    ab_rounds = 5 if quick else 20
    ab_iters_per_round = 100 if quick else 500

    L_values = [256, 384, 512, 640, 768]
    B = 1
    H_KV = 1
    hq = max(H_KV, 64)

    results = []
    print(f"Device: {device_name} ({sm_count} SMs)")
    print(f"Sweeping MQA (B={B}, H_KV={H_KV}) across crossover boundary\n")

    header = (f"{'L_K':<6} {'nBlk':<5} {'Base(s)':<8} {'Fix(s)':<7} "
              f"{'Base us':<10} {'Fix us':<10} {'Speedup':<9} {'Result'}")
    print(header)
    print("-" * 75)

    for L_K in L_values:
        nblk = (L_K + 127) // 128
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
            res = "WIN" if spd > 1.03 else ("REGRESSION" if spd < 0.97 else "NEUTRAL")
        else:
            lat = measure_kernel_us(
                q=q, k_cache=k_cache, v_cache=v_cache,
                cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
                k=k, v=v, num_splits=s_base, flash_kwargs=base_kwargs,
            )
            lat_b = lat
            lat_f = lat
            spd = 1.0
            res = "UNCHANGED"

        print(f"{L_K:<6} {nblk:<5} s={s_base:<5} s={s_fix:<4} "
              f"{lat_b:8.2f}us {lat_f:8.2f}us {spd:7.3f}x  {res}")

        results.append({
            "L_K": L_K, "H_KV": H_KV, "B": B, "nblk": nblk,
            "splits_base": s_base, "splits_fix": s_fix,
            "baseline_us": round(lat_b, 2), "fix_us": round(lat_f, 2),
            "speedup": round(spd, 4), "result": res,
        })

    os.makedirs("results", exist_ok=True)
    output = {
        "experiment": "boundary_sweep",
        "device": device_name,
        "results": results,
    }
    with open("results/boundary_sweep.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/boundary_sweep.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)
