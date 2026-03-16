#!/usr/bin/env python3
"""
Experiment 1: Strict Correctness & Determinism

Verifies numerical equivalence for the candidate policy selected by each track.

Targeted cases (always run): H_KV ∈ {1, 2} at L_K=512 (the win regime),
plus (2, 512, 1) and (1, 384, 1). Random trials extend coverage to H_KV ∈ {1, 2, 4, 8, 16}.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.heuristics_reference import candidate_num_splits_for_track
from src.track_config import add_results_dir_argument, add_track_argument, write_track_json

# Import flash_attn_interface
try:
    import flash_attn_interface
except ImportError:
    print("[ERROR] flash_attn_interface not found. See setup_environment.sh")
    sys.exit(1)


def attention_ref_like_fa(q, k, v, *, upcast: bool, reorder_ops: bool):
    """Reference attention matching the FA3 test harness semantics."""
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(q.shape[-1])
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * scale)
    attn = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.einsum("bhts,bshd->bthd", attn, v)
    return out.to(dtype_og)


def _make_trial_cases(track: str, total_trials: int) -> list[tuple[int, int, int]]:
    targeted = [
        (1, 512, 1),
        (1, 512, 2),
        (2, 512, 1),
        (1, 384, 1),
    ]
    if total_trials <= len(targeted):
        return targeted[:total_trials]
    cases = list(targeted)
    random_trials = total_trials - len(targeted)
    for _ in range(random_trials):
        cases.append((
            random.choice([1, 2, 4]),
            random.randint(128, 1024),
            random.choice([1, 2, 4, 8, 16]),
        ))
    return cases


def run(args):
    device = "cuda"
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"Device: {torch.cuda.get_device_name(0)} ({sm_count} SMs)")
    print(f"Track: {args.track}")
    print(f"Trials: {args.trials}")
    print()

    results = []
    max_diff_global = 0.0
    failures = 0
    trial_cases = _make_trial_cases(args.track, args.trials)

    print(f"{'Trial':<6} {'B':<3} {'L':<5} {'H_KV':<5} {'Splits':<7} "
          f"{'MaxErr':<12} {'MeanErr':<12} {'Status'}")
    print("-" * 75)

    for i, (B, L, H_KV) in enumerate(trial_cases):
        H_Q = max(H_KV * 8, 64)
        D = 128
        dtype = torch.bfloat16

        torch.manual_seed(42 + i)

        q = torch.randn(B, 1, H_Q, D, dtype=dtype, device=device)
        k_cache = torch.randn(B, L, H_KV, D, dtype=dtype, device=device)
        v_cache = torch.randn(B, L, H_KV, D, dtype=dtype, device=device)

        s_fix = candidate_num_splits_for_track(
            args.track, b=B, hkv=H_KV, lq=1, lk=L, d=D, num_sms=sm_count
        )

        # Match FA3's decode benchmark path: query attends over an existing KV cache.
        cache_seqlens = torch.full((B,), L, dtype=torch.int32, device=device)
        out_test = flash_attn_interface.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            causal=True,
            num_splits=s_fix,
        )
        if isinstance(out_test, tuple):
            out_test = out_test[0]

        # Mirror the official FA3 kvcache test: compare FA3 against an upcast
        # reference, relative to PyTorch's own bf16 reorder error.
        k_rep = k_cache.repeat_interleave(H_Q // H_KV, dim=2)
        v_rep = v_cache.repeat_interleave(H_Q // H_KV, dim=2)
        out_ref = attention_ref_like_fa(q, k_rep, v_rep, upcast=True, reorder_ops=False)
        out_pt = attention_ref_like_fa(q, k_rep, v_rep, upcast=False, reorder_ops=True)

        diff = (out_test.float() - out_ref.float()).abs()
        pt_diff = (out_pt.float() - out_ref.float()).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        max_diff_global = max(max_diff_global, max_err)
        pt_max_err = pt_diff.max().item()
        pt_mean_err = pt_diff.mean().item()

        if torch.isnan(diff).any() or torch.isnan(pt_diff).any():
            max_rel = float("nan")
            ok = False
        else:
            denom = out_ref.float().abs().clamp(min=1e-6)
            max_rel = (diff / denom).max().item()
            ok = (
                max_err <= 3.0 * pt_max_err + 1e-5
                and mean_err <= 1.5 * pt_mean_err + 1e-6
            )
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1

        results.append({
            "trial": i, "B": B, "L": L, "H_KV": H_KV,
            "splits": s_fix,
            "case_type": "targeted" if i < 4 else "random",
            "max_abs": max_err, "mean_abs": mean_err,
            "pt_max_abs": pt_max_err, "pt_mean_abs": pt_mean_err,
            "max_rel": max_rel, "status": status,
        })

        if i % 100 == 0 or status == "FAIL":
            print(f"{i:<6} {B:<3} {L:<5} {H_KV:<5} s={s_fix:<4} "
                  f"{max_err:<12.2e} {mean_err:<12.2e} {status}")

    print("-" * 75)
    print(f"Total: {args.trials}, Pass: {args.trials - failures}, Fail: {failures}")
    print(f"Worst-case abs error: {max_diff_global:.2e}")

    # Determinism check
    print("\n=== Determinism Check (informational) ===")
    torch.manual_seed(123)
    q = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(1, 512, 1, 128, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(1, 512, 1, 128, dtype=torch.bfloat16, device=device)
    cache_seqlens = torch.tensor([512], dtype=torch.int32, device=device)
    s_det = candidate_num_splits_for_track(
        args.track, b=1, hkv=1, lq=1, lk=512, d=128, num_sms=sm_count
    )

    out1 = flash_attn_interface.flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True, num_splits=s_det
    )
    out2 = flash_attn_interface.flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True, num_splits=s_det
    )
    if isinstance(out1, tuple):
        out1 = out1[0]
    if isinstance(out2, tuple):
        out2 = out2[0]
    bitwise = (out1 - out2).abs().max().item() == 0.0
    print(f"Bitwise identical (splits={s_det}): {bitwise}")
    print("Note: Bitwise determinism under atomic reduction is best-effort.")

    out_path = write_track_json(
        {
            "device": torch.cuda.get_device_name(0),
            "sm_count": sm_count,
            "total_trials": args.trials,
            "failures": failures,
            "worst_case_abs_error": max_diff_global,
            "determinism_bitwise": bitwise,
            "verdict": "PASS" if failures == 0 else "FAIL",
            "trials": results,
        },
        experiment="exp1_correctness",
        track=args.track,
        results_dir=args.results_dir,
        benchmark_mode="explicit_num_splits_correctness",
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Correctness")
    add_track_argument(parser)
    add_results_dir_argument(parser)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: alias for --trials 100")
    args = parser.parse_args()
    if args.quick and args.trials == 1000:
        args.trials = 100
    run(args)
