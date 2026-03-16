# Latest Stack Findings

This note is the dedicated document for the `latest_stack_tuned` track.

It is intentionally separate from the reviewer-facing `upstream_patch` track.

**Build scope:** All FA3 builds in this repo are Hopper-only (SM90).
The upstream patch remains the real two-guard C++ diff; this document records
what the repo sees on newer H100 software stacks once the low-tile `nblk=4`
regime is tuned more aggressively.

## Main Conclusion

The performance gain on the latest stack is real, but it is not explained
simply by enabling `pack_gqa=True`.

Isolation benchmarks holding `pack_gqa` fixed and comparing baseline
`num_splits=1` against tuned `num_splits=3` reveal:

- Without precomputed scheduler metadata:
  - `H_KV=1`: about `1.05x`
  - `H_KV=2`: about `1.07x`
- With precomputed scheduler metadata on both sides:
  - `H_KV=1`: about `1.20x`
  - `H_KV=2`: about `1.23x`

The latest-stack improvement is best understood as a combination of:

1. using the efficient FA3 scheduling path (`get_scheduler_metadata()`), and
2. using `3` splits instead of `1` or `4` in the low-tile `nblk=4` regime.

## Current Tuned Policy

The tuned latest-stack policy implemented in this repository is:

- keep the same two safety guards as the upstream patch:
  - `nblk <= 3 -> 1 split`
  - `nblk <= 4 and tiles >= 4 -> 1 split`
- add one explicit low-tile boundary choice:
  - `nblk == 4 and tiles < 4 -> 3 splits`

This is a track-specific tuning decision, not the exact upstream patch.

## Why This Track Exists

If an inference stack already uses precomputed scheduler metadata, the practical
question becomes: does split tuning still matter once metadata is already on?

The answer from this track is yes. The tuned policy yields a large win over the
metadata-on baseline, and the gain appears to come primarily from split
selection rather than from `pack_gqa`.

## Relationship to the Upstream Patch

The two tracks should be read differently:

- `upstream_patch` exists to support a clean, reviewer-ready FlashAttention-3 PR.
- `latest_stack_tuned` exists to document stronger H100 results on the current
  stack, even when the policy is narrower and less upstream-minimal.

This tuned track therefore belongs in a reviewer appendix or follow-on tuning
discussion, not in the core upstream PR claim set.

## Broader Relevance

The underlying mechanism is still the same one highlighted by the upstream
patch track: low-head-count decode on Hopper can underfill the GPU badly when
the heuristic keeps `s=1` in the low-tile boundary regime.

Even though FlashAttention-4 is emerging, FlashAttention-3 remains the relevant
production kernel for Hopper MQA/GQA decode in many deployments. That makes the
latest-stack tuned findings operationally useful, even when they are not the
exact upstream patch artifact.
