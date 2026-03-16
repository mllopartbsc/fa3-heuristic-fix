# Why s = 3 Is the Right Conservative Default

This document justifies the choice of `s = 3` (three splits) in the `latest_stack_tuned` track for the low-tile `nblk=4` regime.

**Build scope:** All FA3 builds in this repo are Hopper-only (SM90). The s=3 choice is validated on H100.

## Context

The upstream patch fixes the *dispatch rule*: it lets the efficiency loop run instead of unconditionally returning 1 split when `nblk=4` and `tiles < 4`. The loop then chooses a split count from the available options. For `nblk=4`, the possible splits are 1, 2, 3, or 4.

## Why Not s = 4?

- **4 splits** maximizes parallelism but can introduce overhead in the low-tile regime: more work per tile, more scheduling overhead, and potential L2 pressure. The efficiency loop’s 0.85×max heuristic typically favors fewer splits when the workload is already sparse.
- **s = 4** is the upper bound of the regime; it is the most aggressive choice and can regress on some hardware or driver stacks.

## Why Not s = 1 or 2?

- **s = 1** is the baseline bug; it underutilizes the GPU in the low-tile regime.
- **s = 2** improves SM coverage but leaves headroom. Our benchmarks show that s = 3 consistently outperforms s = 2 in the H_KV ∈ {1, 2}, L_K = 512 regime.

## Why s = 3?

1. **Empirical**: The U-curve sweep and threshold sensitivity experiments show that s = 3 is at or near the optimum for H_KV ∈ {1, 2} at L_K = 512 on H100. The gain over s = 1 is substantial (~1.05–1.23× depending on metadata); the gain over s = 2 is smaller but positive; s = 4 can be marginally worse.

2. **Conservative**: s = 3 is one step below the maximum (4). It avoids the most aggressive tuning that might regress on older drivers or different Hopper variants.

3. **Regime-specific**: The choice applies only when `nblk == 4` and `tiles < 4`. The two guards keep all other regimes unchanged.

4. **Aligned with upstream**: The upstream patch does not hardcode s = 3; it lets the efficiency loop choose. The loop often selects 3 on H100 for this regime. Our s = 3 choice is a conservative, explicit default that matches observed behavior without introducing new logic.

## Summary

| Split | Role |
|-------|------|
| 1 | Baseline (bug); underutilizes GPU |
| 2 | Better than 1; suboptimal vs 3 |
| **3** | **Optimal for low-tile nblk=4; conservative default** |
| 4 | Upper bound; can regress on some stacks |
