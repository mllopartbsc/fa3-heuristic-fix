# PR Evidence

See [`BENCHMARKING.md`](BENCHMARKING.md) for how to run the benchmarks and [`METHODOLOGY.md`](METHODOLOGY.md) for both routes.

**Route 2 (upstream_patch)** is what FA3 maintainers merge: heuristics.h change only. Precomputed metadata enabled. The patch is in [`patch/heuristics.patch`](../patch/heuristics.patch) — the only change is the `num_splits` heuristic (explicit `return 3` for low-tile boundary).
