#!/usr/bin/env python3
"""Generate the borderless report strobes (v2 trace-over + v3 fish-over) into report_strobes/."""
import os
import build_strobe_plots as b

RUNS = ["hope_didnt_work_2_highres", "insane_run_8_middle_bg_highres", "insane_run_2_highres",
        "finetune_run_7_highres", "finetune_run_8_highres", "figure_eight_6_clean_highres"]
FULL = dict(b.CONFIG)

b.OUT_DIR = os.path.join(b._HERE, "report_strobes")
os.makedirs(b.OUT_DIR, exist_ok=True)
b.TRACE_OUTLINE = False                          # no black border on the trace

# v2 — full circular fish crops, trace drawn on top
b.FISH_ON_TOP = False; b.TRACE_FADE = False; b.OUT_SUFFIX = "_traceover"
b.CONFIG = {k: FULL[k] for k in RUNS}
b.main()

# v3 — segmented full fish composited on top of the trace
b.FISH_ON_TOP = True; b.TRACE_FADE = False; b.OUT_SUFFIX = "_fishover"
b.CONFIG = {k: FULL[k] for k in RUNS}
b.main()
