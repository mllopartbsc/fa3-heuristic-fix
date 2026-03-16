"""
Shared CUDA-Graph benchmarking utilities for all reproduction experiments.

Two benchmark routes:

  1. Route 1 (policy_injected): Python injects num_splits + precomputed
     scheduler_metadata. Same binary. Used for latest_stack_tuned track.

  2. Route 2 (patched_binary_with_metadata): Heuristics.h patch only + precomputed
     metadata. What FA3 maintainers merge. Baseline vs patched in subprocesses;
     each uses its heuristic's output for num_splits + precomputed metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch

# Lazy import — flash_attn_interface must be installed (see setup_environment.sh)
_flash_attn = None
REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILED_POLICY_RUNNER = REPO_ROOT / "src" / "compiled_policy_runner.py"
COMPILED_POLICY_RUNNER_WITH_METADATA = REPO_ROOT / "src" / "compiled_policy_runner_with_metadata.py"


def _get_flash_attn():
    """Lazy import of flash_attn_interface."""
    global _flash_attn
    if _flash_attn is None:
        try:
            import flash_attn_interface
            _flash_attn = flash_attn_interface
        except ImportError:
            raise ImportError(
                "flash_attn_interface not found. Please install FlashAttention-3 "
                "following the instructions in setup_environment.sh"
            )
    return _flash_attn


def build_flash_kwargs(
    *,
    batch: int,
    lk: int,
    hq: int,
    hkv: int,
    d: int,
    cache_seqlens,
    num_splits: int,
    pack_gqa=None,
    causal: bool = True,
):
    """Build kwargs for the policy-injected latest-stack benchmark path."""
    fa = _get_flash_attn()
    return {
        "pack_gqa": pack_gqa,
        "scheduler_metadata": fa.get_scheduler_metadata(
            batch_size=batch,
            max_seqlen_q=1,
            max_seqlen_k=lk,
            num_heads_q=hq,
            num_heads_kv=hkv,
            headdim=d,
            cache_seqlens=cache_seqlens,
            qkv_dtype=torch.bfloat16,
            headdim_v=d,
            causal=causal,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
        ),
    }


# ── Default timing parameters ───────────────────────────────────────────────
DEFAULT_WARMUPS = 200
DEFAULT_TOTAL_ITERS = 10_000
DEFAULT_SAMPLE_ITERS = 50


def _capture_graph(fn, *args, **kwargs):
    """Eager warmup + CUDA Graph capture. Returns (graph, output)."""
    for _ in range(10):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return g, out


def _timed_replay(g, warmups, total_iters, sample_iters):
    """Warm up graph and return list of per-call latencies in microseconds."""
    if sample_iters > total_iters:
        sample_iters = total_iters
    assert total_iters % sample_iters == 0, f"{total_iters} % {sample_iters} != 0"
    n_samples = total_iters // sample_iters

    for _ in range(warmups):
        g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us = []
    for _ in range(n_samples):
        start.record()
        for _ in range(sample_iters):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) / sample_iters * 1000.0)
    return samples_us


def measure_kernel_us(
    *,
    q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v,
    num_splits: int,
    flash_kwargs: dict | None = None,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> float:
    """
    Returns median per-call kernel latency in microseconds.
    Uses CUDA Graph capture to eliminate Python dispatch overhead.
    """
    fa = _get_flash_attn()
    flash_kwargs = flash_kwargs or {}

    def _kernel():
        return fa.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache_seqlens,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin,
            causal=causal, num_splits=num_splits,
            **flash_kwargs,
        )

    g, _ = _capture_graph(_kernel)
    samples = _timed_replay(g, warmups, total_iters, sample_iters)
    return float(statistics.median(samples))


def measure_kernel_us_auto_policy(
    *,
    q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> float:
    """
    Returns median per-call kernel latency with split selection left entirely to
    the compiled FA3 binary.
    """
    fa = _get_flash_attn()

    def _kernel():
        return fa.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache_seqlens,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin,
            causal=causal,
        )

    g, _ = _capture_graph(_kernel)
    samples = _timed_replay(g, warmups, total_iters, sample_iters)
    return float(statistics.median(samples))


def measure_kernel_us_detailed(
    *,
    q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v,
    num_splits: int,
    flash_kwargs: dict | None = None,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> dict:
    """
    Like measure_kernel_us but returns full statistics:
      median, mean, stdev, p5, p95, iqr, n_samples
    """
    fa = _get_flash_attn()
    flash_kwargs = flash_kwargs or {}

    def _kernel():
        return fa.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache_seqlens,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin,
            causal=causal, num_splits=num_splits,
            **flash_kwargs,
        )

    g, _ = _capture_graph(_kernel)
    samples = _timed_replay(g, warmups, total_iters, sample_iters)
    samples.sort()
    n = len(samples)
    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples) if n > 1 else 0.0,
        "p5": samples[max(0, int(n * 0.05))],
        "p95": samples[min(n - 1, int(n * 0.95))],
        "iqr": samples[int(n * 0.75)] - samples[int(n * 0.25)],
        "n_samples": n,
    }


def measure_kernel_us_detailed_auto_policy(
    *,
    q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> dict:
    """
    Like measure_kernel_us_detailed but leaves split selection to the compiled
    heuristic in the loaded FA3 binary.
    """
    fa = _get_flash_attn()

    def _kernel():
        return fa.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache_seqlens,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin,
            causal=causal,
        )

    g, _ = _capture_graph(_kernel)
    samples = _timed_replay(g, warmups, total_iters, sample_iters)
    samples.sort()
    n = len(samples)
    return {
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "stdev": statistics.stdev(samples) if n > 1 else 0.0,
        "p5": samples[max(0, int(n * 0.05))],
        "p95": samples[min(n - 1, int(n * 0.95))],
        "iqr": samples[int(n * 0.75)] - samples[int(n * 0.25)],
        "n_samples": n,
    }


def measure_ab_interleaved(
    *,
    q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v,
    splits_a: int,
    splits_b: int,
    flash_kwargs_a: dict | None = None,
    flash_kwargs_b: dict | None = None,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    rounds: int = 20,
    iters_per_round: int = 500,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> dict:
    """
    A/B interleaved measurement: alternates between splits_a and splits_b
    graphs each round, eliminating ordering/thermal bias.

    Returns dict: a_median, b_median, speedup, a_p5, a_p95, b_p5, b_p95
    """
    fa = _get_flash_attn()
    flash_kwargs_a = flash_kwargs_a or {}
    flash_kwargs_b = flash_kwargs_b or {}

    def _make_kernel(ns, extra_kwargs):
        def _fn():
            return fa.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=cache_seqlens,
                rotary_cos=rotary_cos, rotary_sin=rotary_sin,
                causal=causal, num_splits=ns,
                **extra_kwargs,
            )
        return _fn

    g_a, _ = _capture_graph(_make_kernel(splits_a, flash_kwargs_a))
    g_b, _ = _capture_graph(_make_kernel(splits_b, flash_kwargs_b))

    for _ in range(warmups):
        g_a.replay()
    torch.cuda.synchronize()
    for _ in range(warmups):
        g_b.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if sample_iters > iters_per_round:
        sample_iters = iters_per_round
    assert iters_per_round % sample_iters == 0
    n_samples_per_round = iters_per_round // sample_iters

    a_samples = []
    b_samples = []

    for r in range(rounds):
        order = [(g_a, a_samples), (g_b, b_samples)]
        if r % 2 == 1:
            order = order[::-1]

        for g, bucket in order:
            for _ in range(n_samples_per_round):
                start.record()
                for _ in range(sample_iters):
                    g.replay()
                end.record()
                torch.cuda.synchronize()
                bucket.append(start.elapsed_time(end) / sample_iters * 1000.0)

    a_sorted = sorted(a_samples)
    b_sorted = sorted(b_samples)
    na, nb = len(a_sorted), len(b_sorted)
    a_med = statistics.median(a_samples)
    b_med = statistics.median(b_samples)
    return {
        "a_median": a_med,
        "b_median": b_med,
        "speedup": a_med / b_med if b_med > 0 else float("nan"),
        "a_p5": a_sorted[max(0, int(na * 0.05))],
        "a_p95": a_sorted[min(na - 1, int(na * 0.95))],
        "b_p5": b_sorted[max(0, int(nb * 0.05))],
        "b_p95": b_sorted[min(nb - 1, int(nb * 0.95))],
        "a_stdev": statistics.stdev(a_samples) if na > 1 else 0,
        "b_stdev": statistics.stdev(b_samples) if nb > 1 else 0,
        "a_samples": na,
        "b_samples": nb,
    }


def measure_eager_us(
    *,
    q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v,
    num_splits: int,
    causal: bool = True,
    warmups: int = 100,
    iterations: int = 1000,
) -> float:
    """
    Eager-mode (no CUDA Graphs) timing — reports service latency T_service
    which includes Python dispatch overhead. Used for the U-curve figure.
    """
    fa = _get_flash_attn()

    def _call():
        return fa.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache_seqlens,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin,
            causal=causal, num_splits=num_splits,
        )

    # Warmup
    for _ in range(warmups):
        _call()
    torch.cuda.synchronize()

    # Timed iterations
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us = []
    for _ in range(iterations):
        start.record()
        _call()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

    return float(statistics.median(samples_us))


def find_optimal_splits(
    *,
    q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v,
    max_splits: int = 32,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    total_iters: int = 5000,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> dict:
    """
    Oracle sweep: tests num_splits from 1..max_splits and returns best.
    Used for Policy B (forced/oracle) comparisons.
    """
    lk = int(cache_seqlens[0].item())
    block_n = 128
    num_n_blocks = (lk + block_n - 1) // block_n
    effective_max = min(max_splits, max(num_n_blocks, 1))

    test_values = sorted(set(
        list(range(1, min(effective_max + 1, 17)))
        + [effective_max]
    ))

    results = []
    for ns in test_values:
        if ns < 1:
            continue
        try:
            lat = measure_kernel_us(
                q=q, k_cache=k_cache, v_cache=v_cache,
                cache_seqlens=cache_seqlens, rotary_cos=rotary_cos,
                rotary_sin=rotary_sin, k=k, v=v,
                num_splits=ns, causal=causal,
                warmups=warmups, total_iters=total_iters,
                sample_iters=sample_iters,
            )
            results.append((ns, lat))
        except Exception:
            continue

    if not results:
        return {"best_splits": 1, "best_latency": float("inf"), "all_results": []}

    best = min(results, key=lambda x: x[1])
    return {
        "best_splits": best[0],
        "best_latency": best[1],
        "all_results": results,
    }


def _runner_env(profile_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_entries = [str(profile_root), str(REPO_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        python_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_entries)
    return env


def measure_compiled_profile_detailed(
    *,
    profile_root: Path,
    batch: int,
    lk: int,
    hq: int,
    hkv: int,
    d: int = 128,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> dict:
    """
    Measure a compiled FA3 profile by spawning a clean Python subprocess whose
    PYTHONPATH points at a single installed build root.
    """
    cmd = [
        sys.executable,
        str(COMPILED_POLICY_RUNNER),
        "--batch", str(batch),
        "--lk", str(lk),
        "--hq", str(hq),
        "--hkv", str(hkv),
        "--d", str(d),
        "--warmups", str(warmups),
        "--total-iters", str(total_iters),
        "--sample-iters", str(sample_iters),
    ]
    if causal:
        cmd.append("--causal")
    out = subprocess.check_output(cmd, env=_runner_env(profile_root), text=True)
    return json.loads(out)


def measure_compiled_profile_pair(
    *,
    baseline_profile_root: Path,
    candidate_profile_root: Path,
    batch: int,
    lk: int,
    hq: int,
    hkv: int,
    d: int = 128,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> dict:
    base = measure_compiled_profile_detailed(
        profile_root=baseline_profile_root,
        batch=batch,
        lk=lk,
        hq=hq,
        hkv=hkv,
        d=d,
        causal=causal,
        warmups=warmups,
        total_iters=total_iters,
        sample_iters=sample_iters,
    )
    candidate = measure_compiled_profile_detailed(
        profile_root=candidate_profile_root,
        batch=batch,
        lk=lk,
        hq=hq,
        hkv=hkv,
        d=d,
        causal=causal,
        warmups=warmups,
        total_iters=total_iters,
        sample_iters=sample_iters,
    )
    base_med = float(base["median"])
    candidate_med = float(candidate["median"])
    return {
        "baseline": base,
        "candidate": candidate,
        "speedup": base_med / candidate_med if candidate_med > 0 else float("nan"),
        "significant": float(candidate["p95"]) < float(base["p5"]),
    }


def measure_compiled_profile_detailed_with_metadata(
    *,
    profile_root: Path,
    policy: str,
    batch: int,
    lk: int,
    hq: int,
    hkv: int,
    d: int = 128,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> dict:
    """
    Route 2: Measure a compiled FA3 profile with precomputed scheduler_metadata.
    The policy (baseline or upstream_patch) determines num_splits; the subprocess
    loads the profile's FA3 binary.
    """
    cmd = [
        sys.executable,
        str(COMPILED_POLICY_RUNNER_WITH_METADATA),
        "--policy", policy,
        "--batch", str(batch),
        "--lk", str(lk),
        "--hq", str(hq),
        "--hkv", str(hkv),
        "--d", str(d),
        "--warmups", str(warmups),
        "--total-iters", str(total_iters),
        "--sample-iters", str(sample_iters),
    ]
    if causal:
        cmd.append("--causal")
    out = subprocess.check_output(cmd, env=_runner_env(profile_root), text=True)
    return json.loads(out)


def measure_compiled_profile_pair_with_metadata(
    *,
    baseline_profile_root: Path,
    candidate_profile_root: Path,
    batch: int,
    lk: int,
    hq: int,
    hkv: int,
    d: int = 128,
    causal: bool = True,
    warmups: int = DEFAULT_WARMUPS,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    sample_iters: int = DEFAULT_SAMPLE_ITERS,
) -> dict:
    """
    Route 2: Baseline vs patched binary, both with precomputed scheduler_metadata.
    Shows the patch improves when used with precomputed metadata (upstream merge path).
    """
    base = measure_compiled_profile_detailed_with_metadata(
        profile_root=baseline_profile_root,
        policy="baseline",
        batch=batch,
        lk=lk,
        hq=hq,
        hkv=hkv,
        d=d,
        causal=causal,
        warmups=warmups,
        total_iters=total_iters,
        sample_iters=sample_iters,
    )
    candidate = measure_compiled_profile_detailed_with_metadata(
        profile_root=candidate_profile_root,
        policy="upstream_patch",
        batch=batch,
        lk=lk,
        hq=hq,
        hkv=hkv,
        d=d,
        causal=causal,
        warmups=warmups,
        total_iters=total_iters,
        sample_iters=sample_iters,
    )
    base_med = float(base["median"])
    candidate_med = float(candidate["median"])
    return {
        "baseline": base,
        "candidate": candidate,
        "speedup": base_med / candidate_med if candidate_med > 0 else float("nan"),
        "significant": float(candidate["p95"]) < float(base["p5"]),
    }


def make_decode_tensors(
    *,
    b: int,
    lk: int,
    hq: int = 64,
    hkv: int = 8,
    d: int = 128,
    new_kv: bool = False,
):
    """Create standard decode tensors (L_Q=1) for benchmarking.

    By default we benchmark the steady-state decode kernel: a single-token query
    attending over an already populated KV cache of total visible length ``lk``.
    Set ``new_kv=True`` only when you explicitly want to include cache-update
    work (append one new token plus rotary application) in the measurement.
    """
    dtype = torch.bfloat16
    device = "cuda"
    lq = 1
    max_k = lk + 128
    q = torch.randn(b, lq, hq, d, dtype=dtype, device=device)
    k_cache = torch.randn(b, max_k, hkv, d, dtype=dtype, device=device)
    v_cache = torch.randn(b, max_k, hkv, d, dtype=dtype, device=device)
    cache_len = max(lk - 1, 0) if new_kv else lk
    cache_seqlens = torch.full((b,), cache_len, dtype=torch.int32, device=device)
    if new_kv:
        rotary_dim = min(64, d // 2)
        rotary_cos = torch.randn(max_k, rotary_dim, dtype=dtype, device=device)
        rotary_sin = torch.randn(max_k, rotary_dim, dtype=dtype, device=device)
        k = torch.randn(b, lq, hkv, d, dtype=dtype, device=device)
        v = torch.randn(b, lq, hkv, d, dtype=dtype, device=device)
    else:
        rotary_cos = None
        rotary_sin = None
        k = None
        v = None
    return q, k_cache, v_cache, cache_seqlens, rotary_cos, rotary_sin, k, v
