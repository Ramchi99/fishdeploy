"""Two-layer curation manifests for FluidID.

Layer 1 — rough triage (one entry per physical run):
    Fields: num, name, run_id, source, status, rough_start, rough_end,
            occluded (optional), boolean flags (reached_target, good_for_fluidid,
            highlight, has_error), plus any extra annotation keys.

    status is one of {"valid", "unsure", "invalid"}. Legacy entries carrying a
    boolean ``valid`` field are migrated forward (True->"valid", False->"invalid").

Layer 2 — fine parts (trajectory-validated windows, one or more per run):
    Fields: num, name, run_id, source, parts list where each part has
            part, t_start, t_end, starts_at_rest.
"""
from __future__ import annotations

import os
from typing import Any

import yaml

from .schema import VALID_SOURCES

VALID_STATUSES = ("valid", "unsure", "invalid")
# Independent boolean flags stored per layer-1 entry.
LAYER1_FLAGS = ("reached_target", "good_for_fluidid", "highlight", "has_error")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_slug(num: int, name: str) -> str:
    """Return a zero-padded run identifier string, e.g. '001_two_stroke_glide'."""
    return f"{num:03d}_{name}"


def part_id(num: int, name: str, part: int) -> str:
    """Return a unique part identifier string, e.g. '001_two_stroke_glide__p1'."""
    return f"{num:03d}_{name}__p{part}"


# ---------------------------------------------------------------------------
# Layer 1 — rough triage
# ---------------------------------------------------------------------------


def validate_layer1(doc: list) -> None:
    """Validate a layer-1 curation document in-place.

    Raises
    ------
    TypeError  if *doc* is not a list.
    ValueError for any logical constraint violation.
    """
    if not isinstance(doc, list):
        raise TypeError(f"layer-1 document must be a list, got {type(doc)}")

    seen_nums: set[int] = set()

    for i, entry in enumerate(doc):
        ctx = f"entry[{i}]"

        # --- required fields ---
        if not isinstance(entry.get("num"), int):
            raise ValueError(f"{ctx}: 'num' must be an int")
        num = entry["num"]

        if num in seen_nums:
            raise ValueError(
                f"duplicate num {num} in layer-1 doc — num values must be unique"
            )
        seen_nums.add(num)

        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{ctx}: 'name' must be a non-empty str")

        if "run_id" not in entry:
            raise ValueError(f"{ctx}: 'run_id' is required")

        source = entry.get("source")
        if source not in VALID_SOURCES:
            raise ValueError(
                f"{ctx}: 'source' must be one of {VALID_SOURCES}, got {source!r}"
            )

        # --- back-compat: migrate legacy boolean ``valid`` -> ``status`` ---
        if "status" not in entry and "valid" in entry:
            legacy_valid = entry.pop("valid")
            entry["status"] = "valid" if legacy_valid else "invalid"

        status = entry.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"{ctx}: 'status' must be one of {VALID_STATUSES}, got {status!r}"
            )

        # --- boolean flags (all optional; missing treated as False) ---
        for flag in LAYER1_FLAGS:
            if flag in entry and not isinstance(entry[flag], bool):
                raise ValueError(f"{ctx}: '{flag}' must be a bool")

        # --- window/occluded checks only for valid/unsure runs ---
        if status == "invalid":
            continue

        rough_start = entry.get("rough_start")
        rough_end = entry.get("rough_end")

        if rough_start is None or rough_end is None:
            raise ValueError(
                f"{ctx}: {status} run requires numeric 'rough_start' and 'rough_end'"
            )
        if not isinstance(rough_start, (int, float)) or not isinstance(
            rough_end, (int, float)
        ):
            raise ValueError(f"{ctx}: rough_start/rough_end must be numeric")
        if rough_end <= rough_start:
            raise ValueError(
                f"{ctx}: rough_end ({rough_end}) must be > rough_start ({rough_start})"
            )

        # --- occluded intervals ---
        occluded = entry.get("occluded", [])
        if not isinstance(occluded, list):
            raise ValueError(f"{ctx}: 'occluded' must be a list")
        for j, interval in enumerate(occluded):
            if (
                not isinstance(interval, (list, tuple))
                or len(interval) != 2
            ):
                raise ValueError(
                    f"{ctx}: occluded[{j}] must be a [t0, t1] pair"
                )
            t0, t1 = interval
            if t1 <= t0:
                raise ValueError(
                    f"{ctx}: occluded[{j}]: t1 ({t1}) must be > t0 ({t0})"
                )
            if t0 < rough_start or t1 > rough_end:
                raise ValueError(
                    f"{ctx}: occluded[{j}] [{t0}, {t1}] is outside the rough window "
                    f"[{rough_start}, {rough_end}]"
                )


def load_layer1(path: str) -> list:
    """Load a layer-1 YAML manifest; return [] if the file does not exist."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    return doc if doc is not None else []


def save_layer1(doc: list, path: str) -> None:
    """Validate *doc* and write it as YAML to *path* (creates parent dirs)."""
    validate_layer1(doc)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Layer 2 — fine parts
# ---------------------------------------------------------------------------


def validate_layer2(doc: list) -> None:
    """Validate a layer-2 curation document.

    Raises
    ------
    TypeError  if *doc* is not a list.
    ValueError for any logical constraint violation.
    """
    if not isinstance(doc, list):
        raise TypeError(f"layer-2 document must be a list, got {type(doc)}")

    seen_nums: set[int] = set()

    for i, entry in enumerate(doc):
        ctx = f"entry[{i}]"

        if not isinstance(entry.get("num"), int):
            raise ValueError(f"{ctx}: 'num' must be an int")
        num = entry["num"]

        if num in seen_nums:
            raise ValueError(
                f"duplicate num {num} in layer-2 doc — num values must be unique"
            )
        seen_nums.add(num)

        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{ctx}: 'name' must be a non-empty str")

        if "run_id" not in entry:
            raise ValueError(f"{ctx}: 'run_id' is required")

        source = entry.get("source")
        if source not in VALID_SOURCES:
            raise ValueError(
                f"{ctx}: 'source' must be one of {VALID_SOURCES}, got {source!r}"
            )

        parts = entry.get("parts")
        if not isinstance(parts, list):
            raise ValueError(f"{ctx}: 'parts' must be a list")

        seen_parts: set[int] = set()

        for k, p in enumerate(parts):
            pctx = f"{ctx}.parts[{k}]"

            part_num = p.get("part")
            if not isinstance(part_num, int):
                raise ValueError(f"{pctx}: 'part' must be an int")

            if part_num in seen_parts:
                raise ValueError(
                    f"{ctx}: duplicate part number {part_num} — part numbers must be "
                    "unique within a run"
                )
            seen_parts.add(part_num)

            t_start = p.get("t_start")
            t_end = p.get("t_end")

            if t_start is None or t_end is None:
                raise ValueError(f"{pctx}: 't_start' and 't_end' are required")
            if not isinstance(t_start, (int, float)) or not isinstance(
                t_end, (int, float)
            ):
                raise ValueError(f"{pctx}: t_start/t_end must be numeric")
            if t_end <= t_start:
                raise ValueError(
                    f"{pctx}: t_end ({t_end}) must be > t_start ({t_start})"
                )

            if not isinstance(p.get("starts_at_rest"), bool):
                raise ValueError(f"{pctx}: 'starts_at_rest' must be a bool")

        # Check for overlapping windows (sort by t_start)
        sorted_parts = sorted(parts, key=lambda p: p["t_start"])
        for k in range(len(sorted_parts) - 1):
            a, b = sorted_parts[k], sorted_parts[k + 1]
            if b["t_start"] < a["t_end"]:
                raise ValueError(
                    f"{ctx}: parts overlap — part {a['part']} ends at {a['t_end']} "
                    f"but part {b['part']} starts at {b['t_start']}"
                )


def load_layer2(path: str) -> list:
    """Load a layer-2 YAML manifest; return [] if the file does not exist."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh)
    return doc if doc is not None else []


def save_layer2(doc: list, path: str) -> None:
    """Validate *doc* and write it as YAML to *path* (creates parent dirs)."""
    validate_layer2(doc)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def accepted_parts(layer2_doc: list) -> list[dict[str, Any]]:
    """Flatten all parts from a layer-2 document into a sorted list of dicts.

    Each dict contains:
        part_id, num, name, run_id, source, part, t_start, t_end, starts_at_rest
    Sorted by part_id (lexicographic).
    """
    flat: list[dict[str, Any]] = []
    for entry in layer2_doc:
        num = entry["num"]
        name = entry["name"]
        run_id = entry["run_id"]
        source = entry["source"]
        for p in entry.get("parts", []):
            flat.append(
                {
                    "part_id": part_id(num, name, p["part"]),
                    "num": num,
                    "name": name,
                    "run_id": run_id,
                    "source": source,
                    "part": p["part"],
                    "t_start": p["t_start"],
                    "t_end": p["t_end"],
                    "starts_at_rest": p["starts_at_rest"],
                }
            )
    return sorted(flat, key=lambda d: d["part_id"])
