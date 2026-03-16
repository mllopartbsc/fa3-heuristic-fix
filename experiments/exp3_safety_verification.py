#!/usr/bin/env python3
"""
Experiment 3: Safety Contract Verification (160-Configuration Sweep)

  - upstream_patch (Route 2): baseline vs patched binary, precomputed metadata
  - latest_stack_tuned (Route 1): policy injection, precomputed metadata
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
    measure_compiled_profile_detailed_with_metadata,
    measure_compiled_profile_pair_with_metadata,
)
from src.heuristics_reference import _ceil_div, baseline_num_splits, candidate_num_splits_for_track
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
    print(f"Track: {track}")
    print()

    ab_rounds = 5 if quick else 20
    ab_iters_per_round = 100 if quick else 500
    total_iters = 2000 if quick else 10000

    b_values = [1, 2, 4, 8]
    h_values = [1, 2, 4, 8, 32]
    l_values = [128, 256, 384, 512, 1024, 2048, 4096, 8192]
    configs = [(b, h, l) for b in b_values for h in h_values for l in l_values]
    assert len(configs) == 160
    print(f"Testing {len(configs)} configurations (4 × 5 × 8 = 160)")
    print()

    wins = 0
    regressions = 0
    neutrals = 0
    unchanged = 0
    results = []
    use_route2 = track != TRACK_LATEST_STACK_TUNED
    baseline_root = baseline_install_root_for_track(track) if use_route2 else None
    candidate_root = install_root_for_track(track) if use_route2 else None

    header = (f"{'Config':<22} {'tiles':<6} {'nBlk':<5} {'Base(s)':<8} "
              f"{'Cand(s)':<8} {'Base us':<10} {'Cand us':<10} {'Speedup':<9} {'Result'}")
    print(header)
    print("-" * 110)

    for batch, hkv, lk in configs:
        tiles = batch * hkv
        nblk = _ceil_div(lk, 128)
        hq = max(hkv, 64)

        s_base = baseline_num_splits(b=batch, hkv=hkv, lq=1, lk=lk, d=128, num_sms=sm_count)
        s_candidate = candidate_num_splits_for_track(
            track, b=batch, hkv=hkv, lq=1, lk=lk, d=128, num_sms=sm_count
        )
        tag = f"B={batch},H={hkv},L={lk}"

        if s_base == s_candidate:
            unchanged += 1
            results.append({
                "B": batch,
                "H_KV": hkv,
                "L_K": lk,
                "tiles": tiles,
                "nblk": nblk,
                "splits_base": s_base,
                "splits_fix": s_candidate,
                "result": "UNCHANGED",
            })
            print(f"{tag:<22} {tiles:<6} {nblk:<5} s={s_base:<5} s={s_candidate:<5} {'—':>10} {'—':>10} {'—':>9} UNCHANGED")
            continue

        try:
            if use_route2:
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
                lat_b = ab["baseline"]["median"]
                lat_c = ab["candidate"]["median"]
                spd = lat_b / lat_c if lat_c > 0 else 1.0
            else:
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
                ab = measure_ab_interleaved(
                    q=q, k_cache=k_cache, v_cache=v_cache,
                    cache_seqlens=cs, rotary_cos=rcos, rotary_sin=rsin,
                    k=k, v=v, splits_a=s_base, splits_b=s_candidate,
                    flash_kwargs_a=base_kwargs, flash_kwargs_b=cand_kwargs,
                    warmups=DEFAULT_WARMUPS, rounds=ab_rounds, iters_per_round=ab_iters_per_round,
                    sample_iters=DEFAULT_SAMPLE_ITERS,
                )
                lat_b = ab["a_median"]
                lat_c = ab["b_median"]
                spd = lat_b / lat_c if lat_c > 0 else 1.0

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
                "B": batch,
                "H_KV": hkv,
                "L_K": lk,
                "tiles": tiles,
                "nblk": nblk,
                "splits_base": s_base,
                "splits_fix": s_candidate,
                "baseline_us": round(lat_b, 2),
                "fix_us": round(lat_c, 2),
                "speedup": round(spd, 4),
                "result": result,
            })
            print(f"{tag:<22} {tiles:<6} {nblk:<5} s={s_base:<5} s={s_candidate:<5} {lat_b:8.2f}us {lat_c:8.2f}us {spd:7.3f}x  {result}")
        except Exception as exc:
            results.append({
                "B": batch,
                "H_KV": hkv,
                "L_K": lk,
                "tiles": tiles,
                "nblk": nblk,
                "splits_base": s_base,
                "splits_fix": s_candidate,
                "result": "ERROR",
                "error": str(exc),
            })
            print(f"{tag:<22} {tiles:<6} {nblk:<5} ERROR: {exc}")

    print()
    print(f"Summary: {wins} WIN, {regressions} REGRESSION, {neutrals} NEUTRAL, {unchanged} UNCHANGED")
    print(f"Total: {len(configs)} configurations")

    out_path = write_track_json(
        {
            "device": device_name,
            "sm_count": sm_count,
            "total_configs": len(configs),
            "wins": wins,
            "regressions": regressions,
            "neutrals": neutrals,
            "unchanged": unchanged,
            "verdict": "PASS (zero regressions)" if regressions == 0 else "FAIL",
            "results": results,
        },
        experiment="exp3_safety_verification",
        track=track,
        results_dir=results_dir,
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: Safety Contract Verification")
    add_track_argument(parser)
    add_results_dir_argument(parser)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer rounds / iterations")
    args = parser.parse_args()
    run(track=args.track, quick=args.quick, results_dir=args.results_dir)
