"""Load / save / validate the curation manifest.

Manifest is a list of run entries; each run has multiple segments. Only
accept==True segments enter the dataset. seg_id = "<run_id>__seg{NN:02d}".
"""
from __future__ import annotations
import os
import yaml
from .schema import VALID_SOURCES as SOURCES


def load(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        doc = yaml.safe_load(f)
    return doc or []


def save(doc: list, path: str) -> None:
    validate(doc)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def seg_id(run_id: str, seg: int) -> str:
    return f"{run_id}__seg{seg:02d}"


def validate(doc: list) -> None:
    if not isinstance(doc, list):
        raise ValueError("manifest must be a list of run entries")
    for run in doc:
        rid = run.get("run_id")
        if not rid:
            raise ValueError("run entry missing run_id")
        if run.get("source") not in SOURCES:
            raise ValueError(f"{rid}: invalid source {run.get('source')!r}")
        windows = []
        seen_segs = []
        for seg in run.get("segments", []):
            if "seg" not in seg:
                raise ValueError(f"{rid}: segment missing 'seg' index")
            seg_index = seg["seg"]
            if "t_start" not in seg or "t_end" not in seg:
                raise ValueError(f"{rid} seg{seg_index}: missing t_start/t_end")
            ts, te = seg["t_start"], seg["t_end"]
            if te <= ts:
                raise ValueError(f"{rid} seg{seg_index}: t_end <= t_start")
            seen_segs.append(seg_index)
            windows.append((ts, te, seg_index))
        if len(seen_segs) != len(set(seen_segs)):
            raise ValueError(f"{rid}: duplicate seg index")
        windows.sort()
        for (a_s, a_e, a_i), (b_s, b_e, b_i) in zip(windows, windows[1:]):
            if b_s < a_e:
                raise ValueError(f"{rid}: seg{a_i} and seg{b_i} overlap ({a_e} > {b_s})")


def accepted_segments(doc: list) -> list:
    """Flatten to accepted segments with a stable seg_id, sorted by run then seg."""
    out = []
    for run in doc:
        for seg in run.get("segments", []):
            if not seg.get("accept"):
                continue
            out.append({
                "seg_id": seg_id(run["run_id"], seg["seg"]),
                "run_id": run["run_id"],
                "source": run["source"],
                "t_start": seg["t_start"],
                "t_end": seg["t_end"],
                "starts_at_rest": bool(seg.get("starts_at_rest", False)),
                "reached_target": bool(seg.get("reached_target", False)),
            })
    out.sort(key=lambda s: s["seg_id"])
    return out
