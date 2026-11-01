"""Intent-oriented description generation with GPT-5.4."""
from __future__ import annotations
import json
import re
from typing import Any, Optional

EMPTY_RELATION_SUMMARY: dict = {
    "explicit_shared_relations": [],
    "explicit_derived_relations": [],
    "numeric_equality_only_groups": [],
}

SYSTEM_PROMPT = """\
You are given a CadQuery script and a recovered relation summary, and must write one natural-language prompt for generating the CAD model.

Write a precise, low-ambiguity, intent-oriented engineering description that stays faithful to the code-defined geometry, preserves explicit parameter relations, and preserves the actual construction structure of loops, sketches, extrusions, and body operations.

Use the recovered relation summary as authoritative evidence for relation wording:
- Relations listed in explicit_shared_relations and explicit_derived_relations should be expressed as geometric, dimensional, or constraint-style relations when natural.
- Items in numeric_equality_only_groups must not be described as shared relations unless the code provides additional explicit support.

Rules:
1. Describe only geometry, dimensions, coordinates, planes, sketch regions, body operations, booleans, and relations that are directly supported by the script. Do not hallucinate unsupported features, topology, or relations.
2. Do not mention Python, CadQuery, variable names, API names, or source-code syntax. NEVER include identifiers like var_1, var_2_center, var_shared_param, loop_1, or any name containing underscores. NEVER include array-index syntax like [0], [1], or .x/.y.
3. Preserve explicit parameter dependencies already present in the code. Express them as geometric, dimensional, or constraint-style relations when they are explicitly encoded or explicitly recovered in the relation summary. Do not flatten such dependencies into standalone absolute values.
4. Only express shared, mirrored, aligned, offset, or other dependent relations when they are explicit in the code, explicitly listed in the relation summary, or directly structurally encoded. Do not promote mere numeric equality, repeated values, visually symmetric coordinates, or loop-generated patterns into relations unless they are explicitly supported.
5. When multiple sibling features explicitly share the same parameter, describe that parameter once at the group level. Do not reattach the same shared value to later sibling features.
6. Once a stated relation fully determines another feature's size or placement, give only the minimal anchor numerics and do not repeat derived values implied by that relation.
7. Preserve the original construction structure. Do not merge separately represented loops, sketches, extrusions, bodies, or booleans into a simpler inferred process.
8. If the code uses high-level primitives, describe them directly. If it uses low-level segments, arcs, or vertex lists, preserve the ordered boundary specification needed to reproduce the profile faithfully.
9. When a sketch plane is relevant, describe it only by its origin, x-axis, and normal exactly as given in the code.
10. Follow the logical construction order reflected by the script. Use neutral engineering wording such as model, part, body, or bodies.
11. When stating an explicit relation, name the relation ONLY; do not explain its mechanism, coordinate effect, or implementation.
12. The recovered relation summary uses code variable names (var_1, var_2_center, etc.) as internal labels to help you identify which code features are related. These labels are for your reference ONLY. You MUST resolve every such label to a concrete geometric description (e.g., "the first circle's center", "the third cutout's radius") or an ordinal reference to the feature it belongs to in the construction. Never let any code-level name leak into your output.
13. Express derived relations using geometric language: "mirror across the Y-axis", "reflection across … along the X-axis", "symmetric about the origin", "offset by …", etc. Do not reproduce code expressions, mathematical formulas with variable names, or coordinate-arithmetic involving internal labels.
14. Output ONLY one self-contained engineering-style paragraph. Do not include any reasoning steps, self-corrections, re-verifications, intermediate thoughts, or planning notes.

Example 1:
Code pattern: two rectangular loops explicitly share one width parameter, but their heights are separate parameters with the same numeric value.
Good output style:
“The second sketch contains two rectangular loops sharing a common width of 0.278351, centered at (0.326674, 0) and (0.798324, 0), with heights 0.386596 and 0.386596 respectively. Extrude this sketch by 0.487113 to form the second body.”
Bad output style:
“The first rectangle has width 0.278351 and height 0.386596, and the second rectangle shares the same width of 0.278351 and height 0.386596.”
Bad output style:
“The two rectangles share a common width and height” when only the width is explicitly shared.
Bad output style:
“The two rectangular loops share a common width of 0.278351, with the second loop having the same width (0.278351).”

Example 2:
Code pattern: one center is explicitly defined as the mirror of another across the Y axis.
Good output style:
“The sketch contains two circles sharing a common radius of 0.019067, with one center at (-0.538983, 0) and the other given by mirror symmetry across the Y axis.”
Bad output style:
“The sketch contains two circles sharing a common radius of 0.019067, with one center at (-0.538983, 0) and the other given by mirror symmetry across the Y axis (at (0.538983, 0)).”

Example 3:
Code pattern: a derived center is defined by reflecting one circle's center across another circle's center along the X-axis.
Good output style:
"The third circular cutout shares the same radius of 0.061813 and is centered at the reflection of the second cutout's center across the first cutout's center along the X-axis."
Bad output style (code expression leaked):
"The third circular cutout is centered at (var_1_center[0] + (var_1_center[0] - var_2_center[0]), var_2_center[1])."
Bad output style (variable name leaked):
"The third cutout is centered at the reflection of var_2_center across var_1_center."

Example 4:
Code pattern: a second polygon's vertices are the Y-axis mirror of the first polygon's vertices.
Good output style:
"The second polygonal loop is the Y-axis mirror of the first loop's vertices."
Bad output style:
"The second polygon's vertices are defined by [(p[0] * -1, p[1]) for p in var_1_points]."

Output exactly one concise but information-complete engineering-style paragraph and nothing else.
"""

def _join_items_natural(items: list[str]) -> str:
    """Join with commas and 'and' before the last item (Oxford comma for 3+)."""
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def _format_summary_value(v: Any) -> str:
    """Format JSON value (number, nested list, etc.) for one-line summary text."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v) if v != int(v) else str(int(v))
    if isinstance(v, list):
        inner = ", ".join(_format_summary_value(x) for x in v)
        return f"[{inner}]"
    return str(v)

def _naturalize_derived_relation(target: str, rhs: str, depends_on: list[str]) -> str:
    """Convert a code-level derived-relation RHS to a natural-language geometric description.

    Detects common mirror / reflection / offset patterns and expresses them
    in plain geometric language. Falls back to a cleaned-up expression with
    .x/.y instead of [0]/[1].
    """
    s = rhs.strip()

    # (-A[0], A[1]) → Y-axis mirror
    m = re.match(r'^\(\s*-(\w+)\[0\]\s*,\s*\1\[1\]\s*\)$', s)
    if m:
        return f"{target} is the mirror of {m.group(1)} across the Y-axis"

    # (A[0], -A[1]) → X-axis mirror
    m = re.match(r'^\(\s*(\w+)\[0\]\s*,\s*-\1\[1\]\s*\)$', s)
    if m:
        return f"{target} is the mirror of {m.group(1)} across the X-axis"

    # (-A[0], -A[1]) → origin point reflection
    m = re.match(r'^\(\s*-(\w+)\[0\]\s*,\s*-\1\[1\]\s*\)$', s)
    if m:
        return f"{target} is the point reflection of {m.group(1)} through the origin"

    # (A[0] + (A[0] - B[0]), B[1]) → X-reflection of B across A
    m = re.match(
        r'^\(\s*(\w+)\[0\]\s*\+\s*\(\s*\1\[0\]\s*-\s*(\w+)\[0\]\s*\)\s*,\s*\2\[1\]\s*\)$', s
    )
    if m:
        return (
            f"{target} is the reflection of {m.group(2)} across {m.group(1)} "
            f"along the X-axis"
        )

    # (B[0], A[1] + (A[1] - B[1])) → Y-reflection of B across A
    m = re.match(
        r'^\(\s*(\w+)\[0\]\s*,\s*(\w+)\[1\]\s*\+\s*\(\s*\2\[1\]\s*-\s*\1\[1\]\s*\)\s*\)$', s
    )
    if m:
        return (
            f"{target} is the reflection of {m.group(1)} across {m.group(2)} "
            f"along the Y-axis"
        )

    # [(-p[0], p[1]) for p in A] or [(p[0]*-1, p[1]) for p in A] → Y-axis mirror of points
    m = re.match(
        r'^\[\s*\(\s*(?:-p\[0\]|p\[0\]\s*\*\s*-1)\s*,\s*p\[1\]\s*\)\s*'
        r'for\s+p\s+in\s+(\w+)\s*\]$', s
    )
    if m:
        return f"{target} is the Y-axis mirror of all points in {m.group(1)}"

    # [(p[0], -p[1]) for p in A] → X-axis mirror of points
    m = re.match(
        r'^\[\s*\(\s*p\[0\]\s*,\s*(?:-p\[1\]|p\[1\]\s*\*\s*-1)\s*\)\s*'
        r'for\s+p\s+in\s+(\w+)\s*\]$', s
    )
    if m:
        return f"{target} is the X-axis mirror of all points in {m.group(1)}"

    # -A (scalar negation)
    m = re.match(r'^-(\w+)$', s)
    if m:
        return f"{target} is the negation of {m.group(1)}"

    # A + B (scalar sum)
    m = re.match(r'^(\w+)\s*\+\s*(\w+)$', s)
    if m:
        return f"{target} equals {m.group(1)} plus {m.group(2)}"

    # A - B (scalar difference)
    m = re.match(r'^(\w+)\s*-\s*(\w+)$', s)
    if m:
        return f"{target} equals {m.group(1)} minus {m.group(2)}"

    # A * B (scalar product)
    m = re.match(r'^(\w+)\s*\*\s*(\w+)$', s)
    if m:
        return f"{target} equals {m.group(1)} times {m.group(2)}"

    # A / B (scalar division)
    m = re.match(r'^(\w+)\s*/\s*(\w+)$', s)
    if m:
        return f"{target} equals {m.group(1)} divided by {m.group(2)}"

    # Fallback: replace array-index syntax for readability
    cleaned = re.sub(r'\[0\]', '.x', s)
    cleaned = re.sub(r'\[1\]', '.y', cleaned)
    cleaned = re.sub(r'\[2\]', '.z', cleaned)
    return f"{target} is defined by the expression: {cleaned}"

def format_relation_summary_as_text(relation_summary: Optional[dict] = None) -> str:
    rel = relation_summary if relation_summary is not None else dict(EMPTY_RELATION_SUMMARY)
    must_express = []
    must_not = []

    for item in rel.get("explicit_shared_relations", []):
        if not isinstance(item, dict):
            continue
        var = item.get("variable", "?")
        reused = item.get("reused_by", [])
        reused_s = _join_items_natural([str(x) for x in reused])
        must_express.append(f"{var} is reused by {reused_s}")

    for item in rel.get("explicit_derived_relations", []):
        if not isinstance(item, dict):
            continue
        target = item.get("target", "?")
        rhs = item.get("rhs", "")
        depends = item.get("depends_on", [])
        must_express.append(_naturalize_derived_relation(target, rhs, depends))

    for g in rel.get("numeric_equality_only_groups", []):
        if not isinstance(g, dict):
            continue
        val = _format_summary_value(g.get('value'))
        slots = _join_items_natural([str(x) for x in g.get("slots", [])])
        must_not.append(f"value {val} appears in {slots}")

    lines = ["Recovered relation summary (authoritative):"]
    if must_express:
        lines.append("MUST EXPRESS AS RELATIONS:")
        for x in must_express:
            lines.append(f"- {x}")
    else:
        lines.append("MUST EXPRESS AS RELATIONS:\n- none")

    if must_not:
        lines.append("MUST NOT UPGRADE TO SHARED RELATIONS:")
        for x in must_not:
            lines.append(f"- {x}")
    else:
        lines.append("MUST NOT UPGRADE TO SHARED RELATIONS:\n- none")

    return "\n".join(lines)

def build_user_message(code_text: str, relation_summary: Optional[dict] = None) -> str:
    rel_text = format_relation_summary_as_text(relation_summary)
    return (
        "Task:\n"
        "Write one natural-language prompt for generating the CAD model from the following script. "
        "Use the recovered relation summary as the authoritative guide for which shared and derived relations must be expressed. "
        "Important: never state both a recovered relation and its derived coordinate/value for the same attribute. "
        "Do not infer additional shared relations from repeated numeric values when they are listed under "
        "\"MUST NOT UPGRADE TO SHARED RELATIONS\".\n"
        "Reminder: variable names in the relation summary (var_1, var_2_center, etc.) are internal labels. "
        "Resolve them to ordinal feature references in your output. NEVER output any var_ name, loop_ name, "
        "array index, or code expression.\n\n"
        f"{rel_text}\n\n"
        "CadQuery script:\n"
        f"```python\n{code_text.strip()}\n```"
    )


def describe_program(code: str, summary: dict, llm) -> str:
    """Generate one shared description; reject code/API symbol leakage."""
    text = llm.generate(SYSTEM_PROMPT, build_user_message(code, summary))
    forbidden = r'\b(?:var_\w*|loop_\w*|array_\w*|CadQuery|Python|regularPolygon|slot2D|pushPoints|placeSketch|rarray|polarArray|rect\s*\(|circle\s*\(|fillet\s*\(|extrude\s*\(|union\s*\(|cut\s*\()'
    if not text.strip() or re.search(forbidden, text, re.IGNORECASE) or '```' in text:
        raise ValueError('Description contains code-level symbols or is empty.')
    return ' '.join(text.split()) + '\n'
