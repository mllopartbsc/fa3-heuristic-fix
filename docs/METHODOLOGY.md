# Benchmark Methodology

This document describes the two benchmark routes and how they differ.

## Route 1: Policy Injection (latest_stack_tuned)

- **Same binary** (baseline FA3) for both baseline and candidate runs
- **Python injects** `num_splits` from the track's policy
- **Precomputed scheduler_metadata** via `get_scheduler_metadata()`
- A/B interleaved timing with CUDA Graph replay

Shows the full improvement (~1.18–1.25×) when the correct split count is chosen and precomputed metadata is used.

## Route 2: Heuristics.h Patch + Precomputed Metadata (upstream_patch)

**This is what FlashAttention-3 maintainers would merge.** The only change is `heuristics.h` — the `num_splits` heuristic. No other modifications.

- **Precomputed metadata** is enabled (caller uses `get_scheduler_metadata()` and passes it to the kernel)
- **Baseline vs patched**: We compare upstream FA3 (heuristic returns s=1) vs patched FA3 (heuristic returns s=3 for win regime)
- Each subprocess loads its FA3 via `PYTHONPATH`; the heuristic's output for `num_splits` is used with `get_scheduler_metadata()` (Python reference mirrors the C++ heuristic, since the API may not expose it directly)

The benchmark shows: when the heuristics.h patch is merged and consumers use precomputed metadata, you get ~1.20–1.24× in the win regime. This is the upstream-merge path.

## Build Setup

Two FA3 profiles (Hopper-only, SM90):

- **Baseline**: `.pydeps/baseline/` — upstream FA3 at pinned commit (no patch)
- **Patched**: `.pydeps/upstream_patch/` — same commit + **heuristics.h patch only**

## Summary

| Track | Route | Change | Precomputed metadata |
|-------|-------|--------|----------------------|
| upstream_patch | 2 | heuristics.h patch only (the upstream merge) | Yes |
| latest_stack_tuned | 1 | Python policy injection | Yes |
