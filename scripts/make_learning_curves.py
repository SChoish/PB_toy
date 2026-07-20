#!/usr/bin/env python3
"""Deprecated shim — use make_scoreboard_and_learning_curves.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

remapped: list[str] = [sys.argv[0], "--curves-only"]
args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "--out-dir":
        remapped += ["--curves-dir", args[i + 1]]
        i += 2
        continue
    if a.startswith("--out-dir="):
        remapped.append("--curves-dir=" + a.split("=", 1)[1])
        i += 1
        continue
    if a == "--index":
        remapped += ["--curves-index", args[i + 1]]
        i += 2
        continue
    if a.startswith("--index="):
        remapped.append("--curves-index=" + a.split("=", 1)[1])
        i += 1
        continue
    remapped.append(a)
    i += 1
sys.argv = remapped
runpy.run_path(
    str(Path(__file__).with_name("make_scoreboard_and_learning_curves.py")),
    run_name="__main__",
)
