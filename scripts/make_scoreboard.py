#!/usr/bin/env python3
"""Deprecated shim — use make_scoreboard_and_learning_curves.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--scores-only", *sys.argv[1:]]
runpy.run_path(
    str(Path(__file__).with_name("make_scoreboard_and_learning_curves.py")),
    run_name="__main__",
)
