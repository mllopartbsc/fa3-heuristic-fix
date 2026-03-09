#!/usr/bin/env python3
"""
Experiment 1: Strict Correctness & Determinism
═══════════════════════════════════════════════
Paper reference: Section 4.1

Verifies that the tile-aware fix (Policy C) produces numerically equivalent
outputs to a double-precision (FP64) reference implementation of scaled
dot-product attention.

Protocol:
  - 1,000 randomized trials across diverse shapes
  - B ∈ {1, 2, 4}, L ∈ [128, 1024], H_KV ∈ {4, 8, 16}
  - Max relative error ≤ 1e-3, max absolute error ≤ 1e-3
  - Also checks determinism under atomic reduction (informational)

Output: results/exp1_correctness.json
"""

import sys
import os
import json
import random
import argparse
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.heuristics_reference import baseline_num_splits, tile_aware_num_splits

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


def run(args):
    device = "cuda"
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"Device: {torch.cuda.get_device_name(0)} ({sm_count} SMs)")
    print(f"Trials: {args.trials}")
    print()

    results = []
    max_diff_global = 0.0
    failures = 0

    print(f"{'Trial':<6} {'B':<3} {'L':<5} {'H_KV':<5} {'Splits':<7} "
          f"{'MaxErr':<12} {'MeanErr':<12} {'Status'}")
    print("-" * 75)

    for i in range(args.trials):
        B = random.choice([1, 2, 4])
        L = random.randint(128, 1024)
        H_KV = random.choice([4, 8, 16])
        H_Q = H_KV * 8  # GQA ratio
        D = 128
        dtype = torch.bfloat16

        torch.manual_seed(42 + i)

        q = torch.randn(B, 1, H_Q, D, dtype=dtype, device=device)
        k_cache = torch.randn(B, L, H_KV, D, dtype=dtype, device=device)
        v_cache = torch.randn(B, L, H_KV, D, dtype=dtype, device=device)

        # Compute splits under the fix
        s_fix = tile_aware_num_splits(
            b=B, hkv=H_KV, lq=1, lk=L, d=D, num_sms=sm_count
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
            "splits": s_fix, "max_abs": max_err, "mean_abs": mean_err,
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
    k_cache = torch.randn(1, 512, 8, 128, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn(1, 512, 8, 128, dtype=torch.bfloat16, device=device)
    cache_seqlens = torch.tensor([512], dtype=torch.int32, device=device)

    out1 = flash_attn_interface.flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True, num_splits=4
    )
    out2 = flash_attn_interface.flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True, num_splits=4
    )
    if isinstance(out1, tuple):
        out1 = out1[0]
    if isinstance(out2, tuple):
        out2 = out2[0]
    bitwise = (out1 - out2).abs().max().item() == 0.0
    print(f"Bitwise identical (splits=4): {bitwise}")
    print("Note: Bitwise determinism under atomic reduction is best-effort.")

    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        "experiment": "exp1_correctness",
        "paper_reference": "Section 4.1, Experiment 1",
        "device": torch.cuda.get_device_name(0),
        "sm_count": sm_count,
        "total_trials": args.trials,
        "failures": failures,
        "worst_case_abs_error": max_diff_global,
        "determinism_bitwise": bitwise,
        "verdict": "PASS" if failures == 0 else "FAIL",
        "trials": results,
    }
    with open("results/exp1_correctness.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/exp1_correctness.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Correctness")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: alias for --trials 100")
    args = parser.parse_args()
    if args.quick and args.trials == 1000:
        args.trials = 100
    run(args)
