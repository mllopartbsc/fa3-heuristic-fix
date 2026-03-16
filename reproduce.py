#!/usr/bin/env python3
"""
Reproduce FA3 Heuristic Fix Results

Delegates to run_experiments.py for the full kernel-level reproduction workflow.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    cmd = [sys.executable, str(REPO_ROOT / "run_experiments.py")] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
