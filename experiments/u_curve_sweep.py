#!/usr/bin/env python3
"""
U-Curve Sweep

Common mechanism experiment shared by both tracks.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.bench_utils import make_decode_tensors, measure_eager_us
from src.track_config import add_results_dir_argument, add_track_argument, write_track_json


def run(*, track: str, quick: bool = False, results_dir=None):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)

    # 1000 iter standard, 200 iter quick
    iters = 200 if quick else 1000

    # Test MQA boundary case: L_K=512
    B, H_KV, L_K = 1, 1, 512
    hq = max(H_KV, 64)

    print(f"Device: {device_name} ({sm_count} SMs)")
    print(f"Track: {track} (common experiment)")
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

    out_path = write_track_json(
        {
            "device": device_name,
            "config": {"B": B, "H_KV": H_KV, "L_K": L_K},
            "results": results,
        },
        experiment="u_curve_sweep",
        track=track,
        results_dir=results_dir,
        benchmark_mode="manual_split_eager_common",
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_track_argument(parser)
    add_results_dir_argument(parser)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(track=args.track, quick=args.quick, results_dir=args.results_dir)
