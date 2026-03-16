#!/usr/bin/env python3
"""
End-to-End Decode Step Simulation

Produces track-specific theoretical TPOT estimates from the single-layer kernel
delta. upstream_patch uses Route 2 (patched binary + metadata); latest_stack_tuned
uses Route 1 (policy injection). Numbers remain serving estimates, not measured E2E.
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

    ab_rounds = 5 if quick else 20
    ab_iters_per_round = 100 if quick else 500
    total_iters = 2000 if quick else 10000
    use_route2 = track != TRACK_LATEST_STACK_TUNED

    layers = 80
    tp_size = 8
    total_hq = 64
    total_hkv = 8
    hq_local = total_hq // tp_size
    hkv_local = total_hkv // tp_size
    batch = 1
    lk = 512

    print(f"Device: {device_name} ({sm_count} SMs)")
    print(f"Track: {track}")
    print(f"Simulating Llama-3 70B on TP={tp_size} (local H_Q={hq_local}, H_KV={hkv_local})")
    print(f"Batch=1, L_K=512, Layers={layers}")
    print("Measuring single-layer kernel delta...")

    s_base = baseline_num_splits(b=batch, hkv=hkv_local, lq=1, lk=lk, d=128, num_sms=sm_count)
    s_candidate = candidate_num_splits_for_track(
        track, b=batch, hkv=hkv_local, lq=1, lk=lk, d=128, num_sms=sm_count
    )

    if use_route2:
        ab = measure_compiled_profile_pair_with_metadata(
            baseline_profile_root=baseline_install_root_for_track(track),
            candidate_profile_root=install_root_for_track(track),
            batch=batch,
            lk=lk,
            hq=hq_local,
            hkv=hkv_local,
            d=128,
            warmups=DEFAULT_WARMUPS,
            total_iters=total_iters,
            sample_iters=DEFAULT_SAMPLE_ITERS,
        )
        lat_base = ab["baseline"]["median"]
        lat_candidate = ab["candidate"]["median"]
    else:
        q, k_cache, v_cache, cs, rcos, rsin, k, v = make_decode_tensors(
            b=batch, lk=lk, hq=hq_local, hkv=hkv_local
        )
        base_kwargs = build_flash_kwargs(
            batch=batch, lk=lk, hq=hq_local, hkv=hkv_local, d=128,
            cache_seqlens=cs, num_splits=s_base, pack_gqa=None,
        )
        cand_kwargs = build_flash_kwargs(
            batch=batch, lk=lk, hq=hq_local, hkv=hkv_local, d=128,
            cache_seqlens=cs, num_splits=s_candidate, pack_gqa=None,
        )
        ab = measure_ab_interleaved(
            q=q, k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
            k=k, v=v, splits_a=s_base, splits_b=s_candidate,
            flash_kwargs_a=base_kwargs, flash_kwargs_b=cand_kwargs,
            warmups=DEFAULT_WARMUPS, rounds=ab_rounds, iters_per_round=ab_iters_per_round,
            sample_iters=DEFAULT_SAMPLE_ITERS,
        )
        lat_base = ab["a_median"]
        lat_candidate = ab["b_median"]

    kernel_delta = max(lat_base - lat_candidate, 0.0)

    print(f"  Kernel baseline: {lat_base:.2f} us")
    print(f"  Kernel candidate: {lat_candidate:.2f} us")
    print(f"  Delta per layer: {kernel_delta:.2f} us")

    tpot_baseline = 28.04
    total_delta_ms = (kernel_delta * layers) / 1000.0
    tpot_candidate = tpot_baseline - total_delta_ms
    est_speedup = tpot_baseline / tpot_candidate if tpot_candidate > 0 else 1.0

    print("\nE2E Serving Estimates (theoretical, not measured):")
    print(f"  Baseline TPOT:    {tpot_baseline:.2f} ms")
    print(f"  Candidate TPOT:   {tpot_candidate:.2f} ms")
    print(f"  TPOT Reduction:   {total_delta_ms * 1000:.0f} us ({total_delta_ms / tpot_baseline * 100:.2f}%)")

    out_path = write_track_json(
        {
            "device": device_name,
            "single_layer_delta_us": round(kernel_delta, 2),
            "total_delta_ms": round(total_delta_ms, 3),
            "estimated_tpot_baseline_ms": tpot_baseline,
            "estimated_tpot_fix_ms": round(tpot_candidate, 3),
            "estimated_e2e_speedup": round(est_speedup, 4),
            "notes": [
                "This is a serving estimate derived from the measured single-layer kernel delta.",
                "It is not a measured end-to-end vLLM benchmark.",
            ],
        },
        experiment="e2e_decode_simulation",
        track=track,
        results_dir=results_dir,
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_track_argument(parser)
    add_results_dir_argument(parser)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(track=args.track, quick=args.quick, results_dir=args.results_dir)
