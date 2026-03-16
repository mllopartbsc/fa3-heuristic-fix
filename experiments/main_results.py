#!/usr/bin/env python3
"""
Main Results: Kernel-Level Latency

  - upstream_patch (Route 2): Heuristics.h patch only + precomputed metadata.
    What FA3 maintainers merge. Baseline vs patched with metadata on.
  - latest_stack_tuned (Route 1): Policy injection with precomputed metadata (same binary).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bench_utils import (
    DEFAULT_SAMPLE_ITERS,
    DEFAULT_WARMUPS,
    build_flash_kwargs,
    make_decode_tensors,
    measure_ab_interleaved,
    measure_compiled_profile_pair_with_metadata,
    measure_compiled_profile_detailed_with_metadata,
    measure_kernel_us_detailed,
)
from src.heuristics_reference import baseline_num_splits, candidate_num_splits_for_track
from src.track_config import (
    TRACK_LATEST_STACK_TUNED,
    add_results_dir_argument,
    add_track_argument,
    baseline_install_root_for_track,
    install_root_for_track,
    write_track_json,
)


def run(*, track: str, quick: bool = False, results_dir=None):
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name} ({sm_count} SMs)")

    ab_rounds = 5 if quick else 20
    ab_iters_per_round = 100 if quick else 500
    total_iters = 2000 if quick else 10000
    use_route2 = track != TRACK_LATEST_STACK_TUNED
    protocol = (
        "Route 2: heuristics.h patch + precomputed metadata (upstream merge)"
        if use_route2
        else "Route 1: policy injection, precomputed metadata"
    )
    print(f"Track: {track}")
    print(f"Protocol: {protocol}")
    print()

    # Evaluation table: L_K ∈ {128,256,384,512,2048,4096} × H_KV ∈ {1,2,8}
    configs = [
        {"B": 1, "H_KV": 1, "L_K": 128, "regime": "safe"},
        {"B": 1, "H_KV": 2, "L_K": 128, "regime": "safe"},
        {"B": 1, "H_KV": 8, "L_K": 128, "regime": "safe"},
        {"B": 1, "H_KV": 1, "L_K": 256, "regime": "safe"},
        {"B": 1, "H_KV": 2, "L_K": 256, "regime": "safe"},
        {"B": 1, "H_KV": 8, "L_K": 256, "regime": "safe"},
        {"B": 1, "H_KV": 1, "L_K": 384, "regime": "safe"},
        {"B": 1, "H_KV": 2, "L_K": 384, "regime": "safe"},
        {"B": 1, "H_KV": 8, "L_K": 384, "regime": "safe"},
        {"B": 1, "H_KV": 1, "L_K": 512, "regime": "win"},
        {"B": 1, "H_KV": 2, "L_K": 512, "regime": "win"},
        {"B": 1, "H_KV": 8, "L_K": 512, "regime": "safe"},
        {"B": 1, "H_KV": 1, "L_K": 2048, "regime": "long"},
        {"B": 1, "H_KV": 2, "L_K": 2048, "regime": "long"},
        {"B": 1, "H_KV": 8, "L_K": 2048, "regime": "long"},
        {"B": 1, "H_KV": 1, "L_K": 4096, "regime": "long"},
        {"B": 1, "H_KV": 2, "L_K": 4096, "regime": "long"},
        {"B": 1, "H_KV": 8, "L_K": 4096, "regime": "long"},
    ]

    results = []
    header = (f"{'L_K':<6} {'H_KV':<6} {'Tiles':<6} "
              f"{'Base (us)':<14} {'Cand (us)':<14} {'Speedup':<9} {'Regime'}")
    print(header)
    print("-" * 80)

    baseline_root = baseline_install_root_for_track(track)
    candidate_root = install_root_for_track(track)

    for cfg in configs:
        batch = cfg["B"]
        hkv = cfg["H_KV"]
        lk = cfg["L_K"]
        tiles = batch * hkv
        hq = max(hkv, 64)

        s_base = baseline_num_splits(b=batch, hkv=hkv, lq=1, lk=lk, d=128, num_sms=sm_count)
        s_candidate = candidate_num_splits_for_track(
            track, b=batch, hkv=hkv, lq=1, lk=lk, d=128, num_sms=sm_count
        )

        if use_route2:
            # Route 2: baseline vs patched binary, both with precomputed metadata
            if s_base != s_candidate:
                ab = measure_compiled_profile_pair_with_metadata(
                    baseline_profile_root=baseline_root,
                    candidate_profile_root=candidate_root,
                    batch=batch,
                    lk=lk,
                    hq=hq,
                    hkv=hkv,
                    d=128,
                    warmups=DEFAULT_WARMUPS,
                    total_iters=total_iters,
                    sample_iters=DEFAULT_SAMPLE_ITERS,
                )
                entry = {
                    **cfg,
                    "tiles": tiles,
                    "splits_base": s_base,
                    "splits_fix": s_candidate,
                    "baseline_median_us": round(ab["baseline"]["median"], 2),
                    "fix_median_us": round(ab["candidate"]["median"], 2),
                    "baseline_p5": round(ab["baseline"]["p5"], 2),
                    "baseline_p95": round(ab["baseline"]["p95"], 2),
                    "fix_p5": round(ab["candidate"]["p5"], 2),
                    "fix_p95": round(ab["candidate"]["p95"], 2),
                    "speedup": round(ab["speedup"], 4),
                    "significant": ab["significant"],
                }
            else:
                det = measure_compiled_profile_detailed_with_metadata(
                    profile_root=baseline_root,
                    policy="baseline",
                    batch=batch,
                    lk=lk,
                    hq=hq,
                    hkv=hkv,
                    d=128,
                    warmups=DEFAULT_WARMUPS,
                    total_iters=total_iters,
                    sample_iters=DEFAULT_SAMPLE_ITERS,
                )
                entry = {
                    **cfg,
                    "tiles": tiles,
                    "splits_base": s_base,
                    "splits_fix": s_candidate,
                    "baseline_median_us": round(det["median"], 2),
                    "fix_median_us": round(det["median"], 2),
                    "baseline_p5": round(det["p5"], 2),
                    "baseline_p95": round(det["p95"], 2),
                    "fix_p5": round(det["p5"], 2),
                    "fix_p95": round(det["p95"], 2),
                    "speedup": 1.0,
                    "significant": False,
                }
        else:
            # Route 1: policy injection with precomputed metadata (same binary)
            q, k_cache, v_cache, cs, rcos, rsin, k, v = make_decode_tensors(
                b=batch, lk=lk, hq=hq, hkv=hkv
            )
            base_kwargs = build_flash_kwargs(
                batch=batch, lk=lk, hq=hq, hkv=hkv, d=128,
                cache_seqlens=cs, num_splits=s_base, pack_gqa=None,
            )
            cand_kwargs = build_flash_kwargs(
                batch=batch, lk=lk, hq=hq, hkv=hkv, d=128,
                cache_seqlens=cs, num_splits=s_candidate, pack_gqa=None,
            )
            if s_base != s_candidate:
                ab = measure_ab_interleaved(
                    q=q, k_cache=k_cache, v_cache=v_cache,
                    cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
                    k=k, v=v, splits_a=s_base, splits_b=s_candidate,
                    flash_kwargs_a=base_kwargs, flash_kwargs_b=cand_kwargs,
                    warmups=DEFAULT_WARMUPS, rounds=ab_rounds, iters_per_round=ab_iters_per_round,
                    sample_iters=DEFAULT_SAMPLE_ITERS,
                )
                entry = {
                    **cfg,
                    "tiles": tiles,
                    "splits_base": s_base,
                    "splits_fix": s_candidate,
                    "baseline_median_us": round(ab["a_median"], 2),
                    "fix_median_us": round(ab["b_median"], 2),
                    "baseline_p5": round(ab["a_p5"], 2),
                    "baseline_p95": round(ab["a_p95"], 2),
                    "fix_p5": round(ab["b_p5"], 2),
                    "fix_p95": round(ab["b_p95"], 2),
                    "speedup": round(ab["speedup"], 4),
                    "significant": ab["b_p95"] < ab["a_p5"],
                }
            else:
                det = measure_kernel_us_detailed(
                    q=q, k_cache=k_cache, v_cache=v_cache,
                    cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
                    k=k, v=v, num_splits=s_base, flash_kwargs=base_kwargs,
                )
                entry = {
                    **cfg,
                    "tiles": tiles,
                    "splits_base": s_base,
                    "splits_fix": s_candidate,
                    "baseline_median_us": round(det["median"], 2),
                    "fix_median_us": round(det["median"], 2),
                    "baseline_p5": round(det["p5"], 2),
                    "baseline_p95": round(det["p95"], 2),
                    "fix_p5": round(det["p5"], 2),
                    "fix_p95": round(det["p95"], 2),
                    "speedup": 1.0,
                    "significant": False,
                }

        results.append(entry)
        print(
            f"{lk:<6} {hkv:<6} {tiles:<6} "
            f"{entry['baseline_median_us']:<14.2f} {entry['fix_median_us']:<14.2f} "
            f"{entry['speedup']:<9.3f}x {cfg['regime']}"
        )

    out_path = write_track_json(
        {
            "device": device_name,
            "sm_count": sm_count,
            "protocol": protocol,
            "results": results,
        },
        experiment="main_results",
        track=track,
        results_dir=results_dir,
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Results: kernel-level benchmark")
    add_track_argument(parser)
    add_results_dir_argument(parser)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer rounds / iterations")
    args = parser.parse_args()
    run(track=args.track, quick=args.quick, results_dir=args.results_dir)
