#!/usr/bin/env python3
"""
Guard Ablation Study

Compares baseline, no-shortcut, relaxed, and the active candidate policy.
This experiment intentionally uses policy injection so the policy decisions
can be compared side-by-side inside one benchmark run.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bench_utils import (
    DEFAULT_SAMPLE_ITERS,
    DEFAULT_TOTAL_ITERS,
    DEFAULT_WARMUPS,
    build_flash_kwargs,
    make_decode_tensors,
    measure_kernel_us,
)
from src.heuristics_reference import (
    baseline_num_splits,
    candidate_num_splits_for_track,
    no_shortcut_num_splits,
    relaxed_num_splits,
)
from src.track_config import add_results_dir_argument, add_track_argument, write_track_json


def run(*, track: str, quick: bool = False, results_dir=None):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)

    configs = [
        {"B": 1, "H_KV": 1, "L_K": 384, "desc": "Short boundary (Guard 1)"},
        {"B": 1, "H_KV": 1, "L_K": 512, "desc": "Low-tile boundary (nblk=4)"},
    ]
    total_iters = 2000 if quick else DEFAULT_TOTAL_ITERS

    results = []
    print(f"Device: {device_name} ({sm_count} SMs)")
    print(f"Track: {track}")
    header = f"{'L_K':<6} {'Base':<14} {'NoShortcut':<14} {'Relaxed':<14} {'Candidate':<14}"
    print(header)
    print("-" * 80)

    for cfg in configs:
        batch = cfg["B"]
        hkv = cfg["H_KV"]
        lk = cfg["L_K"]
        hq = max(hkv, 64)

        s_base = baseline_num_splits(b=batch, hkv=hkv, lq=1, lk=lk, d=128, num_sms=sm_count)
        s_noshrt = no_shortcut_num_splits(b=batch, hkv=hkv, lq=1, lk=lk, d=128, num_sms=sm_count)
        s_relax = relaxed_num_splits(b=batch, hkv=hkv, lq=1, lk=lk, d=128, num_sms=sm_count)
        s_candidate = candidate_num_splits_for_track(
            track, b=batch, hkv=hkv, lq=1, lk=lk, d=128, num_sms=sm_count
        )

        unique_splits = {s_base, s_noshrt, s_relax, s_candidate}
        latencies = {}

        q, k_cache, v_cache, cs, rcos, rsin, k, v = make_decode_tensors(
            b=batch, lk=lk, hq=hq, hkv=hkv
        )

        for splits in unique_splits:
            kwargs = build_flash_kwargs(
                batch=batch, lk=lk, hq=hq, hkv=hkv, d=128,
                cache_seqlens=cs, num_splits=splits, pack_gqa=None,
            )
            latencies[splits] = measure_kernel_us(
                q=q, k_cache=k_cache, v_cache=v_cache,
                cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
                k=k, v=v, num_splits=splits, flash_kwargs=kwargs,
                warmups=DEFAULT_WARMUPS, total_iters=total_iters, sample_iters=DEFAULT_SAMPLE_ITERS,
            )

        lat_base = latencies[s_base]
        lat_noshrt = latencies[s_noshrt]
        lat_relax = latencies[s_relax]
        lat_candidate = latencies[s_candidate]

        print(
            f"{lk:<6} "
            f"{lat_base:<9.2f}(s={s_base}) "
            f"{lat_noshrt:<9.2f}(s={s_noshrt}) "
            f"{lat_relax:<9.2f}(s={s_relax}) "
            f"{lat_candidate:<9.2f}(s={s_candidate})"
        )

        results.append({
            "L_K": lk,
            "H_KV": hkv,
            "B": batch,
            "desc": cfg["desc"],
            "base": {"splits": s_base, "latency_us": round(lat_base, 2), "speedup_vs_base": 1.0},
            "no_shortcut": {
                "splits": s_noshrt,
                "latency_us": round(lat_noshrt, 2),
                "speedup_vs_base": round(lat_base / lat_noshrt, 4),
            },
            "relaxed": {
                "splits": s_relax,
                "latency_us": round(lat_relax, 2),
                "speedup_vs_base": round(lat_base / lat_relax, 4),
            },
            "candidate": {
                "splits": s_candidate,
                "latency_us": round(lat_candidate, 2),
                "speedup_vs_base": round(lat_base / lat_candidate, 4),
            },
        })

    out_path = write_track_json(
        {
            "device": device_name,
            "sm_count": sm_count,
            "results": results,
        },
        experiment="guard_ablation",
        track=track,
        results_dir=results_dir,
        benchmark_mode="policy_injected_ablation",
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guard ablation")
    add_track_argument(parser)
    add_results_dir_argument(parser)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer sampling iterations")
    args = parser.parse_args()
    run(track=args.track, quick=args.quick, results_dir=args.results_dir)
