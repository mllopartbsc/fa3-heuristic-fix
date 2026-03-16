#!/usr/bin/env python3
"""
Route 2: Heuristics.h patch + precomputed metadata.

The parent sets PYTHONPATH to the target FA3 build (baseline or patched). This script
uses the heuristic's output for num_splits (Python reference mirrors the C++ heuristic
in heuristics.h) and passes precomputed scheduler_metadata to the kernel (via
get_scheduler_metadata). This is what FA3 maintainers merge: heuristics.h change only,
with precomputed metadata enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bench_utils import (
    DEFAULT_SAMPLE_ITERS,
    DEFAULT_WARMUPS,
    build_flash_kwargs,
    make_decode_tensors,
    measure_kernel_us_detailed,
)
from src.heuristics_reference import baseline_num_splits, upstream_two_guard_num_splits


def _num_splits_for_policy(policy: str, batch: int, lk: int, hq: int, hkv: int, d: int) -> int:
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if policy == "baseline":
        return baseline_num_splits(b=batch, hkv=hkv, lq=1, lk=lk, d=d, num_sms=sm_count)
    if policy == "upstream_patch":
        return upstream_two_guard_num_splits(b=batch, hkv=hkv, lq=1, lk=lk, d=d, num_sms=sm_count)
    raise ValueError(f"Unknown policy: {policy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compiled-policy + precomputed metadata runner")
    parser.add_argument("--policy", choices=["baseline", "upstream_patch"], required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--lk", type=int, required=True)
    parser.add_argument("--hq", type=int, required=True)
    parser.add_argument("--hkv", type=int, required=True)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--total-iters", type=int, default=10000)
    parser.add_argument("--sample-iters", type=int, default=DEFAULT_SAMPLE_ITERS)
    parser.add_argument("--causal", action="store_true", default=True)
    args = parser.parse_args()

    num_splits = _num_splits_for_policy(
        args.policy, args.batch, args.lk, args.hq, args.hkv, args.d
    )
    q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v = make_decode_tensors(
        b=args.batch, lk=args.lk, hq=args.hq, hkv=args.hkv, d=args.d
    )
    flash_kwargs = build_flash_kwargs(
        batch=args.batch,
        lk=args.lk,
        hq=args.hq,
        hkv=args.hkv,
        d=args.d,
        cache_seqlens=cache_seqlens,
        num_splits=num_splits,
        pack_gqa=None,
        causal=args.causal,
    )
    stats = measure_kernel_us_detailed(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        k=k,
        v=v,
        num_splits=num_splits,
        flash_kwargs=flash_kwargs,
        causal=args.causal,
        warmups=args.warmups,
        total_iters=args.total_iters,
        sample_iters=args.sample_iters,
    )
    stats["policy"] = args.policy
    stats["num_splits"] = num_splits
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
