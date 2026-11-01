#!/usr/bin/env python3
"""Clean output_final Python files:
  1. Remove unused variable definitions.
  2. Renumber all var_* / array_* identifiers in sequential definition order.
  3. Strip comments except '# --- Body N ---' lines.
  4. Remove the export statement and '# === 2. Export ===' line.
  5. Save results into output_final_clean (same subfolder structure).
"""

import re
import sys
from pathlib import Path
from collections import OrderedDict


# ── helpers ──────────────────────────────────────────────────────────────────

_VAR_DEF_RE = re.compile(r"^(\s*)(var_\w+|array_\w+)\s*=\s*(.+)$")
_BODY_COMMENT_RE = re.compile(r"^#\s*---\s*Body\s+\d+\s*---\s*$")
_EXPORT_RE = re.compile(r"^cq\.exporters\.export\(")
_SECTION_COMMENT_RE = re.compile(r"^#\s*===")
_COMMENT_RE = re.compile(r"^\s*#")


def _find_used_names(lines: list[str], def_indices: dict[str, int]) -> set[str]:
    """Return names referenced outside their own definition line."""
    used: set[str] = set()
    for name, def_idx in def_indices.items():
        for i, line in enumerate(lines):
            if i == def_idx:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", line):
                used.add(name)
                break
    return used


def _build_rename_map(ordered_names: list[str]) -> dict[str, str]:
    """Build old->new rename map.

    Every var_* base gets a fresh sequential number:
      var_N_prop           -> var_M_prop   (M from a single counter)
      var_shared_param[_K] -> var_M        (standalone, no property suffix)
    array_* -> array_centers_M  (separate counter)
    """
    rename: dict[str, str] = {}
    var_counter = 0
    seen_bases: dict[str, int] = {}
    array_counter = 0

    for name in ordered_names:
        if name.startswith("array_"):
            new_name = f"array_centers_{array_counter}"
            array_counter += 1
            rename[name] = new_name
        else:
            m = re.match(r"^(var_\d+)(_\w+)$", name)
            if m:
                old_base = m.group(1)
                suffix = m.group(2)
            else:
                old_base = name
                suffix = ""

            if old_base not in seen_bases:
                var_counter += 1
                seen_bases[old_base] = var_counter

            new_num = seen_bases[old_base]
            rename[name] = f"var_{new_num}{suffix}"

    return rename


def _apply_renames(text: str, rename: dict[str, str]) -> str:
    if not rename:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(rename, key=len, reverse=True)) + r")\b"
    )
    return pattern.sub(lambda m: rename.get(m.group(1), m.group(1)), text)


def clean_file(src: Path) -> str:
    lines = src.read_text().splitlines()
    # ── pass 1: identify var defs and collect info ──
    def_indices: dict[str, int] = OrderedDict()
    for idx, line in enumerate(lines):
        m = _VAR_DEF_RE.match(line)
        if m:
            def_indices[m.group(2)] = idx
    # ── pass 2: find used names ──
    used = _find_used_names(lines, def_indices)
    # ── pass 3: remove unused defs, comments, export ──
    unused_indices = {def_indices[n] for n in def_indices if n not in used}
    kept: list[str] = []
    for idx, line in enumerate(lines):
        if idx in unused_indices:
            continue
        stripped = line.strip()
        if _EXPORT_RE.match(stripped):
            continue
        if _SECTION_COMMENT_RE.match(stripped):
            continue
        if _COMMENT_RE.match(stripped):
            if _BODY_COMMENT_RE.match(stripped):
                kept.append(line)
            continue
        if not stripped:
            continue
        kept.append(line)
    # ── pass 4: collect definition order of surviving names ──
    ordered_names: list[str] = []
    seen_names: set[str] = set()
    for line in kept:
        m = _VAR_DEF_RE.match(line)
        if m:
            name = m.group(2)
            if name not in seen_names:
                ordered_names.append(name)
                seen_names.add(name)
    # ── pass 5: build rename map and apply ──
    rename = _build_rename_map(ordered_names)
    trivial = all(k == v for k, v in rename.items())
    if trivial:
        return "\n".join(kept) + "\n"
    result = "\n".join(kept) + "\n"
    result = _apply_renames(result, rename)
    return result


