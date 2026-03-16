#!/usr/bin/env python3
"""
Run a single compiled-policy kernel timing measurement in a clean subprocess.

The parent process selects which installed FA3 build is visible by setting
PYTHONPATH to a single build root before invoking this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bench_utils import make_decode_tensors, measure_kernel_us_detailed_auto_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Compiled-policy FA3 timing runner")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--lk", type=int, required=True)
    parser.add_argument("--hq", type=int, required=True)
    parser.add_argument("--hkv", type=int, required=True)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--total-iters", type=int, required=True)
    parser.add_argument("--sample-iters", type=int, required=True)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v = make_decode_tensors(
        b=args.batch, lk=args.lk, hq=args.hq, hkv=args.hkv, d=args.d
    )
    stats = measure_kernel_us_detailed_auto_policy(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        k=k,
        v=v,
        causal=args.causal,
        warmups=args.warmups,
        total_iters=args.total_iters,
        sample_iters=args.sample_iters,
    )
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
