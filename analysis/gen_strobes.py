#!/usr/bin/env python3
"""Render the six strobe composites of paper Fig. 1 and Fig. 7 into data/02_deploy_20260520/analysis/report_strobes/.

Needs the runs' videos from the archive next to their run.csv. Trace drawn over the full circular fish
crops, no outline, 1920 x 1080 (`_traceover` suffix, the files the figures read).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_strobe_plots as b  # noqa: E402

b.OUT_DIR = str(b.ANALYSIS / "report_strobes")
os.makedirs(b.OUT_DIR, exist_ok=True)
b.TRACE_OUTLINE = False
b.FISH_ON_TOP = False
b.TRACE_FADE = False
b.OUT_SUFFIX = "_traceover"
b.main()
