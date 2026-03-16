#!/usr/bin/env python3
"""Apply heuristics.h patch by direct edit. Used when git apply fails (e.g. corrupt patch)."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))
HEURISTICS_DEFAULT = REPO_ROOT / "flash-attention" / "hopper" / "heuristics.h"

OLD = """    // If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if (num_n_blocks <= 4) { return 1; }
"""

NEW = """    // Guard 1: leave shorter contexts unchanged.
    if (num_n_blocks <= 3) { return 1; }
    // Guard 2: for nblk = 4, keep 1 split only when there are already enough tiles.
    // total_mblocks == batch_size * num_heads (already a function parameter).
    if (num_n_blocks <= 4 && total_mblocks >= 4) { return 1; }
    // Low-tile boundary case: explicit override for L_K=512 when tiles < 4.
    if (num_n_blocks == 4 && total_mblocks < 4) { return 3; }
    // For longer contexts, fall through to the existing efficiency loop.
"""


def main():
    heuristics_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HEURISTICS_DEFAULT
    if not heuristics_path.exists():
        print(f"ERROR: {heuristics_path} not found", file=sys.stderr)
        return 1
    content = heuristics_path.read_text()
    if NEW in content:
        print("Patch already applied.")
        return 0
    if OLD not in content:
        print("ERROR: Expected block not found in heuristics.h", file=sys.stderr)
        return 1
    heuristics_path.write_text(content.replace(OLD, NEW))
    print("Patch applied successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
