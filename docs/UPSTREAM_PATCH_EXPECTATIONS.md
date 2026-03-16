# Expected Speedups

## Route 2 (upstream_patch): Heuristics.h Patch + Precomputed Metadata

**This is what FA3 maintainers merge: heuristics.h change only.** Precomputed metadata is enabled.

- **Baseline vs patched** in separate subprocesses
- Both use precomputed `scheduler_metadata` (get_scheduler_metadata)
- Patched heuristics.h returns s=3 for win regime (L_K=512, H_KV ∈ {1,2})
- **Expected speedup**: ~1.19–1.22× in win regime

This route shows the heuristics.h patch improves when precomputed metadata is used.

## Route 1 (latest_stack_tuned): Policy Injection

- Same binary, Python injects s=3 for win regime
- Precomputed `scheduler_metadata`
- **Expected speedup**: ~1.18–1.25× in win regime

## Why Precomputed Metadata Matters

FlashAttention-3 supports two invocation paths:

1. **Heuristic path** (`num_splits=0`): C++ heuristic runs at runtime; no `scheduler_metadata`.
2. **Scheduler metadata path**: Caller uses `get_scheduler_metadata()` and passes explicit `num_splits`.

Both routes use the scheduler metadata path so the full improvement is visible. Inference stacks that use `get_scheduler_metadata()` (e.g., vLLM) will see the same gains.

## Safety

- **Zero regressions**: 160-configuration sweep (exp3) must show 0 regressions.
- **Correctness**: exp1 verifies numerical equivalence.
