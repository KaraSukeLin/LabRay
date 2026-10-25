#!/usr/bin/env python3
"""Experiment B entry point; CPU commands need only Python 3.11+."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from b_orsharc.cli import main

if __name__ == "__main__":
    main()
