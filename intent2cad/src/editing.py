#!/usr/bin/env python3
"""Build downstream edit code/text/code triples (aligned across 3 representations).

Output: the new synthesis editing directory
  downstream_edit/
    semparam/test/src_code/<bucket>/<id>.py
    semparam/test/text/<bucket>/<id>.txt
    semparam/test/tgt_code/<bucket>/<id>.py
    semantic/test/src_code/<bucket>/<id>.py
    semantic/test/text/<bucket>/<id>.txt
    semantic/test/tgt_code/<bucket>/<id>.py
    primitive/test/src_code/<bucket>/<id>.py
    primitive/test/text/<bucket>/<id>.txt
    primitive/test/tgt_code/<bucket>/<id>.py
    kept_ops.tsv
    failed_cases.tsv
    summary.json

Notes for v1:
- Deterministic edits only; never call LLM.
- Skip array-instantiated features.
- Primitive lowering supports:
  - high-api loops for rect/circle/regularPolygon/polygon/rect+fillet/slot
  - explicit low-level chain regeneration for slot and rect+fillet
- For arc-based loops (slot/fillet), we regenerate the full loop chain so arc points and
  adjacent segment endpoints remain consistent.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOOP_RE = re.compile(r"^loop_\d+$")
VAR_RE = re.compile(r"^var_\w+$")

TARGET_CATEGORY_RATIOS = {
    "feature": 0.40,
    "consistency": 0.30,
    "dependency": 0.30,
}

BENCHMARK_TARGET_COUNTS = {
    "feature": 400,
    "consistency": 300,
    "dependency": 300,
}

RARE_FEATURE_FAMILIES = frozenset({"regularPolygon", "rect_fillet", "slot"})

SUPPORTED_FEATURE_KINDS = {
    "rect",
    "circle",
    "regularPolygon",
    "polygon",
    "rect_fillet",
    "slot",
}

REPLACEMENT_TARGETS = {
    "rect": ("circle", "rect_fillet", "slot"),
    "circle": ("regularPolygon", "rect", "slot"),
    "regularPolygon": ("circle",),
    "polygon": ("rect",),
    # Fillet edits are restricted to add/remove fillet on rect only.
    "rect_fillet": ("rect",),
    "slot": ("rect",),
}

PRIMITIVE_BANNED_APIS = frozenset({"rect", "regularPolygon", "polygon", "slot", "fillet"})
SIZE_SLOTS = frozenset({"radius", "width", "height", "length", "diameter", "fillet_radius", "sides"})
SLOT_LENGTH_EPS = 1e-6

SAFE_EVAL_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "len": len,
    "sum": sum,
    "range": range,
    "reversed": reversed,
    "list": list,
    "tuple": tuple,
    "set": set,
    "sorted": sorted,
    "zip": zip,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "bool": bool,
    "math": math,
}


def format_float(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    if s == "-0":
        return "0"
    return s or "0"


def value_to_source(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return format_float(v)
    if isinstance(v, tuple):
        return "(" + ", ".join(value_to_source(x) for x in v) + ")"
    if isinstance(v, list):
        return "[" + ", ".join(value_to_source(x) for x in v) + "]"
    raise TypeError(f"Unsupported value type: {type(v)}")


def almost_zero(x: float, eps: float = 1e-9) -> bool:
    return abs(x) <= eps


def as_xy(v: Any) -> tuple[float, float] | None:
    if isinstance(v, (tuple, list)) and len(v) == 2:
        a, b = v
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a), float(b)
    return None


def rotate_translate(pt: tuple[float, float], angle_deg: float, center: tuple[float, float]) -> tuple[float, float]:
    x, y = pt
    cx, cy = center
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    xr = x * c - y * s
    yr = x * s + y * c
    return (xr + cx, yr + cy)


def stable_int_from_text(text: str, seed: int = 0) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{seed}:".encode("utf-8"))
    h.update(text.encode("utf-8"))
    return int.from_bytes(h.digest(), "big", signed=False)


class SourceEditor:
    def __init__(self, source: str):
        self.source = source
        self._line_starts = self._build_line_starts(source)
        self._repls: list[tuple[int, int, str]] = []

    @staticmethod
    def _build_line_starts(s: str) -> list[int]:
        starts = [0]
        for i, ch in enumerate(s):
            if ch == "\n":
                starts.append(i + 1)
        return starts

    def _to_off(self, line: int, col: int) -> int:
        return self._line_starts[line - 1] + col

    def replace_node(self, node: ast.AST, text: str) -> None:
        if not all(hasattr(node, k) for k in ("lineno", "col_offset", "end_lineno", "end_col_offset")):
            raise ValueError("Node does not have location span")
        start = self._to_off(node.lineno, node.col_offset)
        end = self._to_off(node.end_lineno, node.end_col_offset)
        self._repls.append((start, end, text))

    def apply(self) -> str:
        if not self._repls:
            return self.source
        out = self.source
        for s, e, t in sorted(self._repls, key=lambda x: x[0], reverse=True):
            out = out[:s] + t + out[e:]
        return out


def find_kw(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def find_call_chain(expr: ast.AST) -> list[ast.Call]:
    chain: list[ast.Call] = []
    cur = expr
    while isinstance(cur, ast.Call):
        chain.append(cur)
        fn = cur.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Call):
            cur = fn.value
        else:
            break
    return chain


def call_attr_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


@dataclass
class FeatureInfo:
    loop_name: str
    kind: str
    assign: ast.Assign
    shape_call: ast.Call
    fillet_call: ast.Call | None
    push_call: ast.Call | None
    is_array: bool
    param_nodes: dict[str, ast.AST]


@dataclass
class Candidate:
    category: str
    op_type: str
    loop_name: str | None
    slot: str | None
    detail: dict[str, Any]


@dataclass
class ExecStat:
    ok: bool
    err: str
    volume: float | None
    bbox: tuple[float, float, float, float, float, float] | None
    shape: Any | None
    mesh_sig: tuple[Any, ...] | None


def extract_assignments(tree: ast.Module) -> tuple[dict[str, ast.Assign], dict[str, ast.Assign]]:
    loops: dict[str, ast.Assign] = {}
    vars_: dict[str, ast.Assign] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        name = tgt.id
        if LOOP_RE.match(name):
            loops[name] = node
        if VAR_RE.match(name):
            vars_[name] = node
    return loops, vars_


def extract_features(tree: ast.Module) -> dict[str, FeatureInfo]:
    loops, _ = extract_assignments(tree)
    out: dict[str, FeatureInfo] = {}

    for loop_name, asg in loops.items():
        chain = find_call_chain(asg.value)
        if not chain:
            continue

        shape_call: ast.Call | None = None
        fillet_call: ast.Call | None = None
        push_call: ast.Call | None = None

        attrs = [call_attr_name(c) for c in chain]
        for c, a in zip(chain, attrs):
            if a == "push":
                push_call = c
            if a == "fillet":
                fillet_call = c
            if a in {"rect", "circle", "regularPolygon", "polygon", "slot"} and shape_call is None:
                shape_call = c

        if shape_call is None:
            continue

        kind = call_attr_name(shape_call)
        assert kind is not None
        if kind == "rect" and fillet_call is not None:
            kind = "rect_fillet"

        if kind not in SUPPORTED_FEATURE_KINDS:
            continue

        is_array = False
        param_nodes: dict[str, ast.AST] = {}

        if push_call is not None and push_call.args:
            arg0 = push_call.args[0]
            if isinstance(arg0, ast.Name) and arg0.id.startswith("array_centers"):
                is_array = True
            if isinstance(arg0, ast.List):
                if len(arg0.elts) >= 2:
                    is_array = True
                elif len(arg0.elts) == 1:
                    param_nodes["center"] = arg0.elts[0]

        if kind == "circle":
            if shape_call.args:
                param_nodes["radius"] = shape_call.args[0]
        elif kind == "rect":
            if len(shape_call.args) >= 1:
                param_nodes["width"] = shape_call.args[0]
            if len(shape_call.args) >= 2:
                param_nodes["height"] = shape_call.args[1]
            kw = find_kw(shape_call, "angle")
            if kw is not None:
                param_nodes["angle"] = kw
        elif kind == "rect_fillet":
            if len(shape_call.args) >= 1:
                param_nodes["width"] = shape_call.args[0]
            if len(shape_call.args) >= 2:
                param_nodes["height"] = shape_call.args[1]
            kw = find_kw(shape_call, "angle")
            if kw is not None:
                param_nodes["angle"] = kw
            if fillet_call is not None and fillet_call.args:
                param_nodes["fillet_radius"] = fillet_call.args[0]
        elif kind == "slot":
            if len(shape_call.args) >= 1:
                param_nodes["length"] = shape_call.args[0]
            if len(shape_call.args) >= 2:
                param_nodes["diameter"] = shape_call.args[1]
            kw = find_kw(shape_call, "angle")
            if kw is not None:
                param_nodes["angle"] = kw
        elif kind == "regularPolygon":
            if len(shape_call.args) >= 1:
                param_nodes["radius"] = shape_call.args[0]
            if len(shape_call.args) >= 2:
                param_nodes["sides"] = shape_call.args[1]
            kw = find_kw(shape_call, "angle")
            if kw is not None:
                param_nodes["angle"] = kw
        elif kind == "polygon":
            if shape_call.args:
                param_nodes["points"] = shape_call.args[0]

        out[loop_name] = FeatureInfo(
            loop_name=loop_name,
            kind=kind,
            assign=asg,
            shape_call=shape_call,
            fillet_call=fillet_call,
            push_call=push_call,
            is_array=is_array,
            param_nodes=param_nodes,
        )

    return out


def eval_expr(node: ast.AST, env: dict[str, Any]) -> Any:
    code = compile(ast.Expression(node), "<eval_expr>", "eval")
    scope = dict(SAFE_EVAL_GLOBALS)
    scope.update(env)
    return eval(code, scope, {})


def eval_var_env(tree: ast.Module, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    ov = overrides or {}
    env: dict[str, Any] = {}
    pending: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if VAR_RE.match(name):
                pending.append((name, node.value))

    for name, _ in pending:
        if name in ov:
            env[name] = ov[name]

    unresolved = list(pending)
    changed = True
    while unresolved and changed:
        changed = False
        next_unresolved: list[tuple[str, ast.AST]] = []
        for name, rhs in unresolved:
            if name in ov:
                continue
            try:
                env[name] = eval_expr(rhs, env)
                changed = True
            except Exception:
                next_unresolved.append((name, rhs))
        unresolved = next_unresolved

    # keep unresolved vars absent from env; downstream logic uses membership checks.
    return env


def collect_var_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and VAR_RE.match(n.id):
            out.add(n.id)
    return out


def build_var_usage_map(feats: dict[str, FeatureInfo]) -> dict[str, list[tuple[str, str]]]:
    usage: dict[str, set[tuple[str, str]]] = {}
    for loop, feat in feats.items():
        for slot, node in feat.param_nodes.items():
            for var in collect_var_names(node):
                usage.setdefault(var, set()).add((loop, slot))
    out: dict[str, list[tuple[str, str]]] = {}
    for k, v in usage.items():
        out[k] = sorted(v)
    return out


def _subscript_index(node: ast.Subscript) -> int | None:
    s = node.slice
    if isinstance(s, ast.Constant) and isinstance(s.value, int):
        return int(s.value)
    if hasattr(ast, "Index") and isinstance(s, ast.Index) and isinstance(s.value, ast.Constant):
        if isinstance(s.value.value, int):
            return int(s.value.value)
    return None


def _is_base_axis(node: ast.AST, base: str, axis: int) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == base
        and _subscript_index(node) == axis
    )


def _as_num(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        c = _as_num(node.operand)
        if c is not None:
            return -c
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        c = _as_num(node.operand)
        if c is not None:
            return c
    return None


def _parse_linear(node: ast.AST, is_symbol: Any) -> tuple[float, float] | None:
    # Parse simple affine form: a * sym + b
    if is_symbol(node):
        return (1.0, 0.0)
    c = _as_num(node)
    if c is not None:
        return (0.0, c)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        z = _parse_linear(node.operand, is_symbol)
        if z is not None:
            return (-z[0], -z[1])
    if isinstance(node, ast.BinOp):
        l = _parse_linear(node.left, is_symbol)
        r = _parse_linear(node.right, is_symbol)
        if isinstance(node.op, ast.Add):
            if l is not None and r is not None:
                return (l[0] + r[0], l[1] + r[1])
        if isinstance(node.op, ast.Sub):
            if l is not None and r is not None:
                return (l[0] - r[0], l[1] - r[1])
        if isinstance(node.op, ast.Mult):
            if l is not None and r is not None:
                if abs(l[0]) <= 1e-12:
                    return (r[0] * l[1], r[1] * l[1])
                if abs(r[0]) <= 1e-12:
                    return (l[0] * r[1], l[1] * r[1])
        if isinstance(node.op, ast.Div):
            if l is not None and r is not None and abs(r[0]) <= 1e-12 and abs(r[1]) > 1e-12:
                return (l[0] / r[1], l[1] / r[1])
    return None


def _parse_affine_xy(node: ast.AST, base: str) -> tuple[float, float, float] | None:
    # Parse affine expression a*x + b*y + c where x=base[0], y=base[1]
    if _is_base_axis(node, base, 0):
        return (1.0, 0.0, 0.0)
    if _is_base_axis(node, base, 1):
        return (0.0, 1.0, 0.0)
    c = _as_num(node)
    if c is not None:
        return (0.0, 0.0, c)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        z = _parse_affine_xy(node.operand, base)
        if z is not None:
            return (-z[0], -z[1], -z[2])
    if isinstance(node, ast.BinOp):
        l = _parse_affine_xy(node.left, base)
        r = _parse_affine_xy(node.right, base)
        if isinstance(node.op, ast.Add):
            if l is not None and r is not None:
                return (l[0] + r[0], l[1] + r[1], l[2] + r[2])
        if isinstance(node.op, ast.Sub):
            if l is not None and r is not None:
                return (l[0] - r[0], l[1] - r[1], l[2] - r[2])
        if isinstance(node.op, ast.Mult):
            lc = _as_num(node.left)
            rc = _as_num(node.right)
            if lc is not None:
                rr = _parse_affine_xy(node.right, base)
                if rr is not None:
                    return (rr[0] * lc, rr[1] * lc, rr[2] * lc)
            if rc is not None:
                ll = _parse_affine_xy(node.left, base)
                if ll is not None:
                    return (ll[0] * rc, ll[1] * rc, ll[2] * rc)
        if isinstance(node.op, ast.Div):
            rc = _as_num(node.right)
            if rc is not None and abs(rc) > 1e-12:
                ll = _parse_affine_xy(node.left, base)
                if ll is not None:
                    return (ll[0] / rc, ll[1] / rc, ll[2] / rc)
    return None


def _heuristic_mirror_from_source(node: ast.AST, base: str) -> str | None:
    # Fallback for point-list / list-comp patterns (including for-loops).
    try:
        s = ast.unparse(node).replace(" ", "")
    except Exception:
        return None
    bx = f"{base}[0]"
    by = f"{base}[1]"
    has_bx = bx in s
    has_by = by in s
    neg_bx = f"-{bx}" in s
    neg_by = f"-{by}" in s
    if has_bx and has_by:
        if neg_bx and neg_by:
            return "origin-sign-flip pattern"
        if neg_bx:
            return "Y-axis sign-flip pattern"
        if neg_by:
            return "X-axis sign-flip pattern"
        if f"({by},{bx})" in s or f"({by},-{bx})" in s or f"(-{by},{bx})" in s or f"(-{by},-{bx})" in s:
            return "axis-swap pattern"
    return None


def _iter_uses_base(node: ast.AST, base: str) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id == base:
            return True
    return False


def _xy_target_names(target: ast.AST) -> tuple[str, str] | None:
    if isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2:
        a, b = target.elts
        if isinstance(a, ast.Name) and isinstance(b, ast.Name):
            return a.id, b.id
    return None


def _parse_affine_xy_names(node: ast.AST, x_name: str, y_name: str) -> tuple[float, float, float] | None:
    # Parse affine expression a*x + b*y + c
    if isinstance(node, ast.Name) and node.id == x_name:
        return (1.0, 0.0, 0.0)
    if isinstance(node, ast.Name) and node.id == y_name:
        return (0.0, 1.0, 0.0)
    c = _as_num(node)
    if c is not None:
        return (0.0, 0.0, c)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        z = _parse_affine_xy_names(node.operand, x_name, y_name)
        if z is not None:
            return (-z[0], -z[1], -z[2])
    if isinstance(node, ast.BinOp):
        l = _parse_affine_xy_names(node.left, x_name, y_name)
        r = _parse_affine_xy_names(node.right, x_name, y_name)
        if isinstance(node.op, ast.Add):
            if l is not None and r is not None:
                return (l[0] + r[0], l[1] + r[1], l[2] + r[2])
        if isinstance(node.op, ast.Sub):
            if l is not None and r is not None:
                return (l[0] - r[0], l[1] - r[1], l[2] - r[2])
        if isinstance(node.op, ast.Mult):
            lc = _as_num(node.left)
            rc = _as_num(node.right)
            if lc is not None:
                rr = _parse_affine_xy_names(node.right, x_name, y_name)
                if rr is not None:
                    return (rr[0] * lc, rr[1] * lc, rr[2] * lc)
            if rc is not None:
                ll = _parse_affine_xy_names(node.left, x_name, y_name)
                if ll is not None:
                    return (ll[0] * rc, ll[1] * rc, ll[2] * rc)
        if isinstance(node.op, ast.Div):
            rc = _as_num(node.right)
            if rc is not None and abs(rc) > 1e-12:
                ll = _parse_affine_xy_names(node.left, x_name, y_name)
                if ll is not None:
                    return (ll[0] / rc, ll[1] / rc, ll[2] / rc)
    return None


def _mirror_from_for_comp(node: ast.AST, base: str) -> str | None:
    # Detect mirror relations represented by list/generator comprehensions:
    # [(x, -y) for x, y in base], [(-x, y) for x, y in base], etc.
    if not isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return None
    if not node.generators:
        return None
    gen = node.generators[0]
    if not _iter_uses_base(gen.iter, base):
        return None
    xy = _xy_target_names(gen.target)
    if xy is None:
        return None
    x_name, y_name = xy
    elt = node.elt
    if not (isinstance(elt, ast.Tuple) and len(elt.elts) == 2):
        return None
    tx = _parse_affine_xy_names(elt.elts[0], x_name, y_name)
    ty = _parse_affine_xy_names(elt.elts[1], x_name, y_name)
    if tx is None or ty is None:
        return None
    a11, a12, b1 = tx
    a21, a22, b2 = ty
    near0 = lambda v: abs(v) <= 1e-9
    near = lambda v, t: abs(v - t) <= 1e-9
    if not (near0(b1) and near0(b2)):
        return None
    if near(a11, 1.0) and near(a12, 0.0) and near(a21, 0.0) and near(a22, -1.0):
        return "X-axis reflection (for-loop)"
    if near(a11, -1.0) and near(a12, 0.0) and near(a21, 0.0) and near(a22, 1.0):
        return "Y-axis reflection (for-loop)"
    if near(a11, -1.0) and near(a12, 0.0) and near(a21, 0.0) and near(a22, -1.0):
        return "origin reflection (for-loop)"
    if near(a11, 0.0) and near(a12, 1.0) and near(a21, 1.0) and near(a22, 0.0):
        return "axis-swap reflection (for-loop)"
    if near(a11, 0.0) and near(a12, -1.0) and near(a21, -1.0) and near(a22, 0.0):
        return "axis-swap sign reflection (for-loop)"
    return None


def _slots_from_refs(refs: list[tuple[str, str]] | None) -> set[str]:
    out: set[str] = set()
    if not isinstance(refs, list):
        return out
    for r in refs:
        if isinstance(r, (tuple, list)) and len(r) == 2 and isinstance(r[1], str):
            out.add(r[1])
    return out


def _is_spatial_relation(base_refs: list[tuple[str, str]] | None, target_refs: list[tuple[str, str]] | None) -> bool:
    # Spatial relation is mainly about coordinate-carrying parameters.
    spatial_slots = {"center", "points"}
    slots = _slots_from_refs(base_refs) | _slots_from_refs(target_refs)
    return any(s in spatial_slots for s in slots)


def refine_relation_type(
    cls: dict[str, Any],
    base_refs: list[tuple[str, str]] | None,
    target_refs: list[tuple[str, str]] | None,
) -> dict[str, Any]:
    out = dict(cls)
    t = out.get("relation_type", "UNCLASSIFIED")
    spatial = _is_spatial_relation(base_refs, target_refs)
    if t == "DISTANCE" and not spatial:
        out["relation_type"] = "UNCLASSIFIED"
        out["relation_note"] = "non-spatial scalar offset"
    if t == "CONCENTRIC" and not spatial:
        out["relation_type"] = "UNCLASSIFIED"
        out["relation_note"] = "non-spatial equality"
    return out


def classify_dependency_rhs(rhs: str, base: str) -> dict[str, Any]:
    out: dict[str, Any] = {"relation_type": "UNCLASSIFIED", "relation_note": ""}
    if not isinstance(rhs, str) or not rhs.strip() or not isinstance(base, str) or not base:
        return out
    try:
        e = ast.parse(rhs, mode="eval").body
    except Exception:
        return out

    if isinstance(e, ast.Tuple) and len(e.elts) == 2:
        tx = _parse_affine_xy(e.elts[0], base)
        ty = _parse_affine_xy(e.elts[1], base)
        if tx is not None and ty is not None:
            a11, a12, b1 = tx
            a21, a22, b2 = ty
            near0 = lambda v: abs(v) <= 1e-9
            near = lambda v, t: abs(v - t) <= 1e-9

            if near(a11, 1.0) and near(a12, 0.0) and near(a21, 0.0) and near(a22, 1.0):
                if near0(b1) and near0(b2):
                    return {"relation_type": "CONCENTRIC", "relation_note": "same center"}
                return {
                    "relation_type": "DISTANCE",
                    "relation_note": f"offset where dx is {format_float(b1)} and dy is {format_float(b2)}",
                }

            # MIRROR family: X-axis / Y-axis / origin / axis-swap reflections
            if near0(b1) and near0(b2):
                if near(a11, 1.0) and near(a12, 0.0) and near(a21, 0.0) and near(a22, -1.0):
                    return {"relation_type": "MIRROR", "relation_note": "X-axis reflection"}
                if near(a11, -1.0) and near(a12, 0.0) and near(a21, 0.0) and near(a22, 1.0):
                    return {"relation_type": "MIRROR", "relation_note": "Y-axis reflection"}
                if near(a11, -1.0) and near(a12, 0.0) and near(a21, 0.0) and near(a22, -1.0):
                    return {"relation_type": "MIRROR", "relation_note": "origin reflection"}
                if near(a11, 0.0) and near(a12, 1.0) and near(a21, 1.0) and near(a22, 0.0):
                    return {"relation_type": "MIRROR", "relation_note": "axis-swap reflection"}
                if near(a11, 0.0) and near(a12, -1.0) and near(a21, -1.0) and near(a22, 0.0):
                    return {"relation_type": "MIRROR", "relation_note": "axis-swap sign reflection"}

            # Pure scale relation
            if near0(b1) and near0(b2) and near(a12, 0.0) and near(a21, 0.0):
                if not (near(a11, 1.0) and near(a22, 1.0)):
                    return {
                        "relation_type": "RATIO",
                        "relation_note": f"scale where x is {format_float(a11)} and y is {format_float(a22)}",
                    }

        hm = _heuristic_mirror_from_source(e, base)
        if hm:
            return {"relation_type": "MIRROR", "relation_note": hm}

    hm_comp = _mirror_from_for_comp(e, base)
    if hm_comp:
        return {"relation_type": "MIRROR", "relation_note": hm_comp}

    if isinstance(e, (ast.List, ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
        hm = _heuristic_mirror_from_source(e, base)
        if hm:
            return {"relation_type": "MIRROR", "relation_note": hm}

    z = _parse_linear(e, lambda n: isinstance(n, ast.Name) and n.id == base)
    if z is not None:
        a, b = z
        if abs(a + 1.0) <= 1e-9 and abs(b) <= 1e-9:
            return {"relation_type": "RATIO", "relation_note": "scale is -1"}
        if abs(a - 1.0) <= 1e-9 and abs(b) > 1e-9:
            return {"relation_type": "DISTANCE", "relation_note": f"offset is {format_float(b)}"}
        if abs(b) <= 1e-9 and abs(a - 1.0) > 1e-9:
            return {"relation_type": "RATIO", "relation_note": f"scale is {format_float(a)}"}
        if abs(a - 1.0) <= 1e-9 and abs(b) <= 1e-9:
            return {"relation_type": "UNCLASSIFIED", "relation_note": "same-value scalar"}

    return out


def node_to_value_literal(node: ast.AST) -> Any | None:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def feature_params_from_literal(feat: FeatureInfo) -> dict[str, Any] | None:
    params: dict[str, Any] = {}
    for k, n in feat.param_nodes.items():
        v = node_to_value_literal(n)
        if v is None:
            return None
        params[k] = v

    if "angle" not in params:
        params["angle"] = 0.0
    if "center" not in params:
        params["center"] = (0.0, 0.0)

    return params


def mutate_scalar(slot: str, x: float) -> float:
    if slot == "angle":
        return x + 10.0
    if slot == "sides":
        v = int(round(x))
        if v < 12:
            return float(v + 1)
        return float(max(3, v - 1))
    if almost_zero(x):
        return x + 0.1
    return x * 1.1


def mutate_value(kind: str, slot: str, value: Any, all_params: dict[str, Any]) -> Any | None:
    if slot == "sides":
        if not isinstance(value, (int, float)):
            return None
        nv = int(round(mutate_scalar(slot, float(value))))
        return max(3, nv)

    if slot in {"radius", "width", "height", "length", "diameter", "fillet_radius", "angle"}:
        if not isinstance(value, (int, float)):
            return None
        nv = float(mutate_scalar(slot, float(value)))

        if slot in {"radius", "width", "height", "length", "diameter", "fillet_radius"}:
            nv = max(1e-4, nv)

        if kind == "slot":
            L = float(all_params.get("length", 1.0))
            D = float(all_params.get("diameter", 0.5))
            if slot == "length":
                L = nv
            elif slot == "diameter":
                D = nv
            if L <= D + SLOT_LENGTH_EPS:
                L = D + SLOT_LENGTH_EPS
            if slot == "length":
                nv = L
            if slot == "diameter":
                nv = D

        if kind == "rect_fillet":
            w = float(all_params.get("width", 1.0))
            h = float(all_params.get("height", 1.0))
            r = float(all_params.get("fillet_radius", 0.1))
            if slot == "width":
                w = nv
            elif slot == "height":
                h = nv
            elif slot == "fillet_radius":
                r = nv
            r = min(r, max(1e-4, min(w, h) * 0.49))
            if slot == "fillet_radius":
                nv = r

        return nv

    if slot == "center":
        xy = as_xy(value)
        if xy is None:
            return None
        x, y = xy
        return (x + 0.05, y)

    if slot == "points":
        if not isinstance(value, list) or not value:
            return None
        pts = []
        for p in value:
            xy = as_xy(p)
            if xy is None:
                return None
            pts.append(xy)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        scale = 1.1
        out = []
        for x, y in pts:
            out.append((cx + (x - cx) * scale, cy + (y - cy) * scale))
        return out

    return None


def replacement_params(
    src_kind: str,
    dst_kind: str,
    p: dict[str, Any],
    source_has_angle: bool = False,
) -> dict[str, Any] | None:
    center = as_xy(p.get("center", (0.0, 0.0))) or (0.0, 0.0)
    angle = float(p.get("angle", 0.0))
    target_accepts_angle = dst_kind in {"rect", "regularPolygon", "rect_fillet", "slot"}

    def with_preserved_angle(base: dict[str, Any]) -> dict[str, Any]:
        if not source_has_angle or not target_accepts_angle:
            return dict(base)
        out = dict(base)
        out["angle"] = angle
        return out

    if src_kind == "rect" and dst_kind == "circle":
        w = float(p["width"])
        h = float(p["height"])
        return {"center": center, "radius": max(1e-4, min(w, h) * 0.5)}

    if src_kind == "circle" and dst_kind == "regularPolygon":
        r = float(p["radius"])
        return {"center": center, "radius": max(1e-4, r), "sides": 6}

    if src_kind == "circle" and dst_kind == "rect":
        r = float(p["radius"])
        d = max(1e-4, 2.0 * r)
        return {"center": center, "width": d, "height": d}

    if src_kind == "regularPolygon" and dst_kind == "circle":
        r = float(p["radius"])
        return {"center": center, "radius": max(1e-4, r)}

    if src_kind == "polygon" and dst_kind == "rect":
        pts = p.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            return None
        xy = [as_xy(q) for q in pts]
        if any(v is None for v in xy):
            return None
        q2 = [v for v in xy if v is not None]
        xmin = min(v[0] for v in q2)
        xmax = max(v[0] for v in q2)
        ymin = min(v[1] for v in q2)
        ymax = max(v[1] for v in q2)
        return {
            "center": ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5),
            "width": max(1e-4, xmax - xmin),
            "height": max(1e-4, ymax - ymin),
        }

    if src_kind == "rect" and dst_kind == "rect_fillet":
        w = float(p["width"])
        h = float(p["height"])
        fr = max(1e-4, min(w, h) * 0.25)
        return with_preserved_angle(
            {"center": center, "width": max(1e-4, w), "height": max(1e-4, h), "fillet_radius": fr}
        )

    if src_kind == "rect" and dst_kind == "slot":
        w = float(p["width"])
        h = float(p["height"])
        L = max(w, h)
        D = min(w, h)
        return with_preserved_angle(
            {"center": center, "length": max(D + SLOT_LENGTH_EPS, L), "diameter": max(1e-4, D)}
        )

    if src_kind == "rect_fillet" and dst_kind == "rect":
        w = float(p["width"])
        h = float(p["height"])
        return with_preserved_angle(
            {"center": center, "width": max(1e-4, w), "height": max(1e-4, h)}
        )

    if src_kind == "circle" and dst_kind == "slot":
        r = float(p["radius"])
        D = max(1e-4, 2.0 * r)
        L = D + max(0.05, D * 0.1)
        return {"center": center, "length": L, "diameter": D}

    if src_kind == "slot" and dst_kind == "rect":
        L = float(p["length"])
        D = float(p["diameter"])
        w = max(D + SLOT_LENGTH_EPS, L)
        h = max(1e-4, D)
        return with_preserved_angle(
            {"center": center, "width": w, "height": h}
        )

    return None


def generate_high_api_expr(kind: str, p: dict[str, Any]) -> str:
    center = as_xy(p.get("center", (0.0, 0.0))) or (0.0, 0.0)
    angle = float(p.get("angle", 0.0))
    need_push = not (almost_zero(center[0]) and almost_zero(center[1]))
    push = f".push([{value_to_source(center)}])" if need_push else ""

    if kind == "circle":
        return f"cq.Sketch(){push}.circle({value_to_source(float(p['radius']))}).reset()"

    if kind == "rect":
        angle_part = f", angle={value_to_source(angle)}" if not almost_zero(angle) else ""
        return (
            f"cq.Sketch(){push}.rect({value_to_source(float(p['width']))}, "
            f"{value_to_source(float(p['height']))}{angle_part}).reset()"
        )

    if kind == "regularPolygon":
        angle_part = f", angle={value_to_source(angle)}" if not almost_zero(angle) else ""
        return (
            f"cq.Sketch(){push}.regularPolygon({value_to_source(float(p['radius']))}, "
            f"{int(p['sides'])}{angle_part}).reset()"
        )

    if kind == "polygon":
        pts = p["points"]
        return f"cq.Sketch().polygon({value_to_source(pts)}).reset()"

    if kind == "rect_fillet":
        angle_part = f", angle={value_to_source(angle)}" if not almost_zero(angle) else ""
        return (
            f"cq.Sketch(){push}.rect({value_to_source(float(p['width']))}, "
            f"{value_to_source(float(p['height']))}{angle_part}).reset()"
            f".vertices().fillet({value_to_source(float(p['fillet_radius']))}).reset()"
        )

    if kind == "slot":
        angle_part = f", angle={value_to_source(angle)}" if not almost_zero(angle) else ""
        return (
            f"cq.Sketch(){push}.slot({value_to_source(float(p['length']))}, "
            f"{value_to_source(float(p['diameter']))}{angle_part}).reset()"
        )

    raise ValueError(f"Unsupported kind: {kind}")


def generate_chain_rect_fillet_expr(p: dict[str, Any]) -> str:
    center = as_xy(p.get("center", (0.0, 0.0))) or (0.0, 0.0)
    angle = float(p.get("angle", 0.0))
    w = float(p["width"])
    h = float(p["height"])
    r = float(p["fillet_radius"])
    r = min(max(1e-4, r), max(1e-4, min(w, h) * 0.49))

    sx = w * 0.5
    sy = h * 0.5
    inv = math.sqrt(0.5) * r

    p1 = (sx - r, sy)
    p2 = (-sx + r, sy)
    c_tl = (-sx + r, sy - r)
    m_tl = (c_tl[0] - inv, c_tl[1] + inv)
    p3 = (-sx, sy - r)

    p4 = (-sx, -sy + r)
    c_bl = (-sx + r, -sy + r)
    m_bl = (c_bl[0] - inv, c_bl[1] - inv)
    p5 = (-sx + r, -sy)

    p6 = (sx - r, -sy)
    c_br = (sx - r, -sy + r)
    m_br = (c_br[0] + inv, c_br[1] - inv)
    p7 = (sx, -sy + r)

    p8 = (sx, sy - r)
    c_tr = (sx - r, sy - r)
    m_tr = (c_tr[0] + inv, c_tr[1] + inv)

    pts = [p1, p2, m_tl, p3, p4, m_bl, p5, p6, m_br, p7, p8, m_tr]
    pts = [rotate_translate(q, angle, center) for q in pts]
    p1, p2, m_tl, p3, p4, m_bl, p5, p6, m_br, p7, p8, m_tr = pts

    return (
        "cq.Sketch()"
        f".segment({value_to_source(p1)}, {value_to_source(p2)})"
        f".arc({value_to_source(p2)}, {value_to_source(m_tl)}, {value_to_source(p3)})"
        f".segment({value_to_source(p3)}, {value_to_source(p4)})"
        f".arc({value_to_source(p4)}, {value_to_source(m_bl)}, {value_to_source(p5)})"
        f".segment({value_to_source(p5)}, {value_to_source(p6)})"
        f".arc({value_to_source(p6)}, {value_to_source(m_br)}, {value_to_source(p7)})"
        f".segment({value_to_source(p7)}, {value_to_source(p8)})"
        f".arc({value_to_source(p8)}, {value_to_source(m_tr)}, {value_to_source(p1)})"
        ".assemble()"
    )


def generate_chain_slot_expr(p: dict[str, Any]) -> str:
    center = as_xy(p.get("center", (0.0, 0.0))) or (0.0, 0.0)
    angle = float(p.get("angle", 0.0))
    L = float(p["length"])
    D = float(p["diameter"])
    r = max(1e-4, D * 0.5)
    if L <= D + SLOT_LENGTH_EPS:
        L = D + SLOT_LENGTH_EPS
    a = L * 0.5 - r

    p1 = (-a, r)
    p2 = (a, r)
    mr = (a + r, 0.0)
    p3 = (a, -r)
    p4 = (-a, -r)
    ml = (-a - r, 0.0)

    pts = [p1, p2, mr, p3, p4, ml]
    pts = [rotate_translate(q, angle, center) for q in pts]
    p1, p2, mr, p3, p4, ml = pts

    return (
        "cq.Sketch()"
        f".segment({value_to_source(p1)}, {value_to_source(p2)})"
        f".arc({value_to_source(p2)}, {value_to_source(mr)}, {value_to_source(p3)})"
        f".segment({value_to_source(p3)}, {value_to_source(p4)})"
        f".arc({value_to_source(p4)}, {value_to_source(ml)}, {value_to_source(p1)})"
        ".assemble()"
    )


def generate_chain_rect_expr(p: dict[str, Any]) -> str:
    center = as_xy(p.get("center", (0.0, 0.0))) or (0.0, 0.0)
    angle = float(p.get("angle", 0.0))
    w = max(1e-4, float(p["width"]))
    h = max(1e-4, float(p["height"]))
    sx = w * 0.5
    sy = h * 0.5
    pts = [(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)]
    pts = [rotate_translate(q, angle, center) for q in pts]
    p1, p2, p3, p4 = pts
    return (
        "cq.Sketch()"
        f".segment({value_to_source(p1)}, {value_to_source(p2)})"
        f".segment({value_to_source(p2)}, {value_to_source(p3)})"
        f".segment({value_to_source(p3)}, {value_to_source(p4)})"
        f".segment({value_to_source(p4)}, {value_to_source(p1)})"
        ".assemble()"
    )


def generate_chain_polygon_expr(points: list[Any]) -> str:
    if len(points) < 3:
        raise ValueError("polygon needs at least 3 points")
    pts: list[tuple[float, float]] = []
    for p in points:
        xy = as_xy(p)
        if xy is None:
            raise ValueError("invalid polygon point")
        pts.append(xy)

    segs = []
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        segs.append(f".segment({value_to_source(a)}, {value_to_source(b)})")
    return "cq.Sketch()" + "".join(segs) + ".assemble()"


def generate_chain_regular_polygon_expr(p: dict[str, Any]) -> str:
    center = as_xy(p.get("center", (0.0, 0.0))) or (0.0, 0.0)
    angle = float(p.get("angle", 0.0))
    r = max(1e-4, float(p["radius"]))
    sides = max(3, int(p["sides"]))
    pts = []
    # Match CadQuery Sketch.regularPolygon(..., angle=...) orientation.
    # CadQuery builds base points via (sin, cos) then applies polygon(angle),
    # whose effective direction is opposite to our direct point-generation convention.
    # Equivalent direct form is x = -r*sin(angle + step*i), y = r*cos(angle + step*i).
    for i in range(sides):
        deg = angle + 360.0 * i / sides
        rad = math.radians(deg)
        pts.append((center[0] - r * math.sin(rad), center[1] + r * math.cos(rad)))
    return generate_chain_polygon_expr(pts)


def generate_primitive_circle_expr(p: dict[str, Any]) -> str:
    center = as_xy(p.get("center", (0.0, 0.0))) or (0.0, 0.0)
    r = max(1e-4, float(p["radius"]))
    need_push = not (almost_zero(center[0]) and almost_zero(center[1]))
    push = f".push([{value_to_source(center)}])" if need_push else ""
    return f"cq.Sketch(){push}.circle({value_to_source(r)}).reset()"


def _extract_loops_in_expr_order(expr: ast.AST) -> list[str]:
    out: list[str] = []

    def _walk(n: ast.AST) -> None:
        if isinstance(n, ast.Name) and LOOP_RE.match(n.id):
            out.append(n.id)
            return
        if isinstance(n, ast.BinOp):
            _walk(n.left)
            _walk(n.right)
            return
        for ch in ast.iter_child_nodes(n):
            _walk(ch)

    _walk(expr)
    return out


def _collect_placesketch_args(tree: ast.Module) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("sketch_")
        ):
            continue
        val = node.value
        if not (isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute) and val.func.attr == "placeSketch"):
            continue
        args: list[list[str]] = []
        for a in val.args:
            args.append(_extract_loops_in_expr_order(a))
        out[node.targets[0].id] = args
    return out


def build_semantic_to_primitive_loop_aliases(semantic_code: str, primitive_code: str) -> dict[str, set[str]]:
    sem_tree = ast.parse(semantic_code)
    pri_tree = ast.parse(primitive_code)
    sem_sketch = _collect_placesketch_args(sem_tree)
    pri_sketch = _collect_placesketch_args(pri_tree)

    aliases: dict[str, set[str]] = {}

    # Identity aliases first: if primitive has same loop name, keep it as alias.
    pri_loops, _ = extract_assignments(pri_tree)
    for name in pri_loops.keys():
        aliases.setdefault(name, set()).add(name)

    # Align by sketch name + argument index + ordered loop positions in each argument expression.
    for sk, sem_args in sem_sketch.items():
        pri_args = pri_sketch.get(sk)
        if pri_args is None:
            continue
        for i in range(min(len(sem_args), len(pri_args))):
            s_seq = sem_args[i]
            p_seq = pri_args[i]
            if not s_seq or not p_seq or len(s_seq) != len(p_seq):
                continue
            for s_lp, p_lp in zip(s_seq, p_seq):
                if not (LOOP_RE.match(s_lp) and LOOP_RE.match(p_lp)):
                    continue
                aliases.setdefault(s_lp, set()).add(p_lp)

    return aliases


_EXECUTION_ROOT = Path(__file__).resolve().parents[2] / 'data/synthesized/work/edit_execution'


def execute_cq(code: str, timeout: int = 30) -> ExecStat:
    # Keep the original volume/bbox/mesh criteria, with a killable 30s worker.
    from .validation import execute_code
    stat = execute_code(code, _EXECUTION_ROOT, include_mesh=True)
    if not stat['ok']:
        return ExecStat(False, stat.get('reason', 'execution_failed'), None, None, None, None)
    values = stat['invariants']
    bbox = (values[1], values[4], values[2], values[5], values[3], values[6])
    return ExecStat(True, '', values[0], bbox, None, stat.get('mesh_signature'))


def rel_diff(a: float, b: float, eps: float = 1e-9) -> float:
    den = max(abs(a), abs(b), eps)
    return abs(a - b) / den


def bbox_diff(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def parse_slot_ref(x: str) -> tuple[str, str] | None:
    # e.g. loop_2.width
    parts = x.split(".")
    if len(parts) != 2:
        return None
    lp, slot = parts
    if not LOOP_RE.match(lp):
        return None
    return lp, slot


def describe_value(v: Any) -> str:
    if isinstance(v, float):
        return format_float(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (tuple, list)):
        return value_to_source(v)
    return str(v)


def compact_params_text(params: dict[str, Any], max_items: int = 6) -> str:
    ordered_keys = ["center", "width", "height", "radius", "length", "diameter", "fillet_radius", "sides", "angle", "points"]
    items: list[str] = []
    seen = set()
    for k in ordered_keys:
        if k in params:
            items.append(f"{k}={describe_value(params[k])}")
            seen.add(k)
    for k in sorted(params.keys()):
        if k in seen:
            continue
        items.append(f"{k}={describe_value(params[k])}")
    if len(items) > max_items:
        items = items[:max_items] + ["..."]
    return ", ".join(items)


def params_natural_text(kind: str, params: dict[str, Any]) -> str:
    shown = params_for_instruction(kind, params)
    ordered_keys = [
        "center",
        "width",
        "height",
        "radius",
        "length",
        "diameter",
        "fillet_radius",
        "sides",
        "angle",
        "points",
    ]
    templates = {
        "center": "the center is {v}",
        "width": "the width is {v}",
        "height": "the height is {v}",
        "radius": "the radius is {v}",
        "length": "the length is {v}",
        "diameter": "the diameter is {v}",
        "fillet_radius": "the fillet radius is {v}",
        "sides": "the number of sides is {v}",
        "angle": "the angle is {v}",
        "points": "the points are {v}",
    }
    phrases: list[str] = []
    seen: set[str] = set()
    omitted: set[str] = set()
    for k in ordered_keys:
        if k in shown:
            if k == "center":
                cxy = as_xy(shown[k])
                if cxy is not None and almost_zero(cxy[0]) and almost_zero(cxy[1]):
                    # Omit default-origin center from instruction text.
                    omitted.add(k)
                    continue
            seen.add(k)
            vtxt = describe_value(shown[k])
            tmpl = templates.get(k, f"{k} is {{v}}")
            phrases.append(tmpl.format(v=vtxt))
    for k in sorted(shown.keys()):
        if k in seen or k in omitted:
            continue
        phrases.append(f"{k} is {describe_value(shown[k])}")
    return join_with_and(phrases)


def params_for_instruction(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    allowed: dict[str, tuple[str, ...]] = {
        "circle": ("center", "radius"),
        "rect": ("center", "width", "height", "angle"),
        "regularPolygon": ("center", "radius", "sides", "angle"),
        "polygon": ("points",),
        "rect_fillet": ("center", "width", "height", "fillet_radius", "angle"),
        "slot": ("center", "length", "diameter", "angle"),
    }
    keep = allowed.get(kind, tuple(params.keys()))
    out: dict[str, Any] = {}
    for k in keep:
        if k in params:
            out[k] = params[k]
    return out


def dependency_relation_phrase(changed: list[tuple[str, str, Any, Any]], max_terms: int | None = 8) -> str:
    names: list[str] = []
    for lp, s, _ov, _nv in changed:
        names.append(f"{lp}.{s}")
    uniq = sorted(set(names))
    if isinstance(max_terms, int) and max_terms > 0 and len(uniq) > max_terms:
        uniq = uniq[:max_terms] + ["..."]
    if not uniq:
        return "all dependent dimensions"
    return ", ".join(uniq)


def loop_slot_list_text(refs: list[tuple[str, str]] | list[str], max_terms: int | None = 6) -> str:
    items: list[str] = []
    for r in refs:
        if isinstance(r, str):
            items.append(r)
        elif isinstance(r, (tuple, list)) and len(r) == 2:
            items.append(f"{r[0]}.{r[1]}")
    items = sorted(set(items))
    if isinstance(max_terms, int) and max_terms > 0 and len(items) > max_terms:
        return ", ".join(items[:max_terms]) + ", ..."
    return ", ".join(items)


def join_with_and(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def affected_refs_text(slots: list[tuple[str, str]], max_terms: int | None = 12) -> str:
    refs = sorted({f"{lp}.{s}" for lp, s in slots})
    if isinstance(max_terms, int) and max_terms > 0 and len(refs) > max_terms:
        refs = refs[:max_terms] + ["..."]
    return join_with_and(refs)


def consistency_constraints_phrase(slots: list[tuple[str, str]]) -> str:
    if not slots:
        return "linked loop parameters remain equal"
    loop_names = sorted({lp for lp, _s in slots})
    slot_names = sorted({s for _lp, s in slots})
    refs_txt = affected_refs_text(slots, max_terms=None)

    if slot_names == ["center"] and len(loop_names) >= 2:
        loops_txt = join_with_and(loop_names)
        return f"{loops_txt} stay CONCENTRIC"

    if len(slot_names) == 1:
        return f"{refs_txt} EQUAL relation"

    return f"{refs_txt} EQUAL relation"


def consistency_change_target_text(slots: list[tuple[str, str]]) -> str:
    if not slots:
        return "loop_1.parameter"
    primary = sorted(set(slots))[0]
    return f"{primary[0]}.{primary[1]}"


def _valid_loop_slot_refs(refs: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(refs, list):
        return out
    for x in refs:
        if not (isinstance(x, (tuple, list)) and len(x) == 2):
            continue
        lp, s = x
        if isinstance(lp, str) and isinstance(s, str) and LOOP_RE.match(lp):
            out.append((lp, s))
    return out


def _dependency_change_target_text(c: Candidate) -> str:
    base_refs = _valid_loop_slot_refs(c.detail.get("base_refs", []))
    if base_refs:
        primary = sorted(set(base_refs))[0]
        return f"{primary[0]}.{primary[1]}"

    changed = c.detail.get("changed", [])
    from_changed: list[tuple[str, str]] = []
    if isinstance(changed, list):
        for x in changed:
            if not (isinstance(x, (tuple, list)) and len(x) == 4):
                continue
            lp, s = x[0], x[1]
            if isinstance(lp, str) and isinstance(s, str) and LOOP_RE.match(lp):
                from_changed.append((lp, s))
    if from_changed:
        primary = sorted(set(from_changed))[0]
        return f"{primary[0]}.{primary[1]}"

    return "loop_1.parameter"


def _normalize_relation_note(rt: str, note: str) -> str:
    n = " ".join(note.strip().split())
    if not n:
        return ""
    nl = n.lower()

    if rt == "DISTANCE":
        if nl.startswith("offset where "):
            return n
        if nl.startswith("offset is "):
            return f"offset where value is {n[10:].strip()}"
        if nl.startswith("offset "):
            tail = n[7:].strip()
            if tail.lower().startswith("where "):
                return f"offset {tail}"
            return f"offset where {tail}"
        if nl.startswith("offset"):
            tail = n[6:].strip()
            if tail:
                if tail.lower().startswith("where "):
                    return f"offset {tail}"
                return f"offset where {tail}"
            return "offset"
        return n

    if rt == "RATIO":
        if nl.startswith("scale where "):
            return n
        if nl.startswith("scale is "):
            return f"scale where value is {n[9:].strip()}"
        if nl.startswith("scale "):
            tail = n[6:].strip()
            if tail.lower().startswith("where "):
                return f"scale {tail}"
            return f"scale where {tail}"
        if nl.startswith("scale"):
            tail = n[5:].strip()
            if tail:
                if tail.lower().startswith("where "):
                    return f"scale {tail}"
                return f"scale where {tail}"
            return "scale"
        return n

    return n


def _dependency_relation_rule_text(target_txt: str, base_txt: str, rt_raw: str, note_raw: str) -> str:
    rt = rt_raw.strip().upper()
    if rt == "CONCENTRIC":
        return f"{target_txt} stays CONCENTRIC with {base_txt}"

    note = _normalize_relation_note(rt, note_raw)
    if note:
        return f"{target_txt} keeps a {rt} relation with {note} of {base_txt}"
    return f"{target_txt} keeps a {rt} relation of {base_txt}"


def dependency_constraints_phrase(c: Candidate, max_rules: int | None = None) -> str:
    detail = c.detail
    base_refs = detail.get("base_refs", [])
    dep_specs = detail.get("dep_specs", [])

    base_txt = loop_slot_list_text(base_refs, max_terms=None) if isinstance(base_refs, list) else ""
    if not base_txt:
        # Prefer loop-level wording in instructions; avoid exposing var_* names.
        base_txt = _dependency_change_target_text(c)

    rules: list[str] = []

    if isinstance(dep_specs, list):
        for spec in dep_specs:
            if not isinstance(spec, dict):
                continue
            target_refs = spec.get("target_refs", [])
            target_txt = loop_slot_list_text(target_refs, max_terms=None) if isinstance(target_refs, list) else ""
            if not target_txt:
                tv = spec.get("target_var")
                target_txt = tv if isinstance(tv, str) else "dependent terms"

            rt = spec.get("relation_type")
            rn = spec.get("relation_note")
            if isinstance(rt, str) and rt and rt != "UNCLASSIFIED":
                note = rn if isinstance(rn, str) else ""
                rules.append(_dependency_relation_rule_text(target_txt, base_txt, rt, note))
            else:
                rules.append(f"{target_txt} remain expression-dependent on {base_txt}")

            if isinstance(max_rules, int) and max_rules > 0 and len(rules) >= max_rules:
                break

    if rules:
        return "; ".join(rules)

    changed = detail.get("changed", [])
    if isinstance(changed, list):
        dep_phrase = dependency_relation_phrase(changed, max_terms=None)
        return f"{dep_phrase} remain dependency-linked"
    return "linked loop parameters preserve their dependency relation"


def instruction_for_candidate(c: Candidate) -> str:
    if c.category == "feature" and c.op_type.startswith("param:"):
        kind = c.detail.get("kind", "feature")
        loop = c.loop_name or "the target loop"
        slot = c.slot or "parameter"
        oldv = c.detail.get("old")
        newv = c.detail.get("new")
        if oldv is not None and newv is not None:
            return (
                f"Change the {slot} of the {kind} in {loop} from {describe_value(oldv)} "
                f"to {describe_value(newv)}, and keep the other design intent unchanged."
            )
        return ""

    if c.category == "consistency":
        slots = c.detail.get("slots", [])
        oldv = c.detail.get("old")
        newv = c.detail.get("new")
        refs_txt = consistency_change_target_text(slots) if isinstance(slots, list) else "loop_1.parameter"
        rule_txt = consistency_constraints_phrase(slots) if isinstance(slots, list) else "linked loop parameters remain equal"
        if oldv is not None and newv is not None:
            return (
                f"We have the following design intent: {rule_txt}. Change {refs_txt} from "
                f"{describe_value(oldv)} to {describe_value(newv)}, and keep the design intent unchanged."
            )
        return ""

    if c.category == "dependency":
        base_txt = _dependency_change_target_text(c)
        oldv = c.detail.get("old")
        newv = c.detail.get("new")
        dep_rule = dependency_constraints_phrase(c)
        if oldv is not None and newv is not None:
            return (
                f"We have the following design intent: {dep_rule}. Change {base_txt} "
                f"from {describe_value(oldv)} to {describe_value(newv)}, and keep the design intent unchanged."
            )
        return ""

    if c.op_type.startswith("replace:"):
        _, src, dst = c.op_type.split(":")
        loop = c.loop_name or "the target loop"
        new_params = c.detail.get("new_params", {})
        if isinstance(new_params, dict):
            params_txt = params_natural_text(dst, new_params)
        else:
            params_txt = ""
        if params_txt:
            if src == "rect" and dst == "rect_fillet":
                fr = new_params.get("fillet_radius")
                if fr is not None:
                    return (
                        f"Add a fillet feature with a radius that is {describe_value(fr)} to the rect in {loop}, "
                        f"and keep the design intent unchanged."
                    )
                return (
                    f"Add a fillet feature to the rect in {loop}, "
                    f"and keep the design intent unchanged."
                )
            if src == "rect_fillet" and dst == "rect":
                return (
                    f"Remove the fillet feature from the rect in {loop}, "
                    f"and keep the design intent unchanged."
                )
            return (
                f"Replace the geometry in {loop} with a {dst} feature, where {params_txt}, "
                f"and keep the design intent unchanged."
            )
        return ""
    return ""


def collect_relation_bound_slots(summary: dict[str, Any], semparam_feats: dict[str, FeatureInfo]) -> set[str]:
    bound: set[str] = set()

    for rel in summary.get("explicit_shared_relations", []):
        refs = rel.get("reused_by") or []
        if not isinstance(refs, list):
            continue
        for r in refs:
            if isinstance(r, str) and parse_slot_ref(r) is not None:
                bound.add(r)

    var_usage = build_var_usage_map(semparam_feats)
    for drel in summary.get("explicit_derived_relations", []):
        if not isinstance(drel, dict):
            continue
        tv = drel.get("target")
        if isinstance(tv, str):
            for lp, slot in var_usage.get(tv, []):
                bound.add(f"{lp}.{slot}")
        deps = drel.get("depends_on") or []
        if isinstance(deps, list):
            for dep in deps:
                if isinstance(dep, str):
                    for lp, slot in var_usage.get(dep, []):
                        bound.add(f"{lp}.{slot}")
    return bound


def replacement_required_params(src: str, dst: str) -> set[str]:
    key = f"{src}->{dst}"
    required: dict[str, set[str]] = {
        "rect->circle": {"center", "radius"},
        "rect->rect_fillet": {"center", "width", "height", "fillet_radius"},
        "rect->slot": {"center", "length", "diameter"},
        "circle->regularPolygon": {"center", "radius", "sides"},
        "circle->rect": {"center", "width", "height"},
        "circle->slot": {"center", "length", "diameter"},
        "regularPolygon->circle": {"center", "radius"},
        "polygon->rect": {"center", "width", "height"},
        "rect_fillet->rect": {"center", "width", "height"},
        "slot->rect": {"center", "width", "height"},
    }
    return required.get(key, set())


def candidate_is_well_specified(c: Candidate) -> bool:
    if c.category == "feature" and c.op_type.startswith("param:"):
        if c.loop_name is None or c.slot is None:
            return False
        return ("old" in c.detail) and ("new" in c.detail) and c.detail.get("kind") is not None

    if c.category == "consistency":
        slots = c.detail.get("slots")
        if not isinstance(slots, list) or not slots:
            return False
        if c.detail.get("old") is None or c.detail.get("new") is None:
            return False
        return all(isinstance(x, (tuple, list)) and len(x) == 2 for x in slots)

    if c.category == "dependency":
        if c.detail.get("old") is None or c.detail.get("new") is None:
            return False
        dep_specs = c.detail.get("dep_specs")
        if not isinstance(dep_specs, list) or not dep_specs:
            return False
        has_valid_rel = False
        for s in dep_specs:
            if not isinstance(s, dict):
                continue
            rt = s.get("relation_type")
            if isinstance(rt, str) and rt and rt != "UNCLASSIFIED":
                has_valid_rel = True
        return has_valid_rel

    if c.op_type.startswith("replace:"):
        parts = c.op_type.split(":")
        if len(parts) != 3:
            return False
        _k, src, dst = parts
        np = c.detail.get("new_params")
        if not isinstance(np, dict):
            return False
        shown = params_for_instruction(dst, np)
        req = replacement_required_params(src, dst)
        return req.issubset(set(shown.keys()))

    return False


def build_feature_candidates(
    semantic_feats: dict[str, FeatureInfo],
    blocked_param_refs: set[str] | None = None,
) -> list[Candidate]:
    out: list[Candidate] = []
    blocked = blocked_param_refs or set()

    for loop, feat in semantic_feats.items():
        if feat.is_array:
            continue
        params = feature_params_from_literal(feat)
        if params is None:
            continue

        # parameter edits
        for slot in sorted(feat.param_nodes.keys()):
            if f"{loop}.{slot}" in blocked:
                continue
            oldv = params.get(slot)
            if oldv is None:
                continue
            newv = mutate_value(feat.kind, slot, oldv, params)
            if newv is None:
                continue
            if newv == oldv:
                continue
            out.append(
                Candidate(
                    category="feature",
                    op_type=f"param:{feat.kind}:{slot}",
                    loop_name=loop,
                    slot=slot,
                    detail={"kind": feat.kind, "old": oldv, "new": newv},
                )
            )

        # replacement edits (no array)
        for dst in REPLACEMENT_TARGETS.get(feat.kind, ()):
            rp = replacement_params(feat.kind, dst, params, source_has_angle=("angle" in feat.param_nodes))
            if rp is None:
                continue
            out.append(
                Candidate(
                    category="feature",
                    op_type=f"replace:{feat.kind}:{dst}",
                    loop_name=loop,
                    slot=None,
                    detail={"kind": feat.kind, "target_kind": dst, "new_params": rp},
                )
            )

    return out


def build_consistency_candidates(
    summary: dict[str, Any],
    semparam_env: dict[str, Any],
    semantic_feats: dict[str, FeatureInfo],
) -> list[Candidate]:
    out: list[Candidate] = []
    for rel in summary.get("explicit_shared_relations", []):
        var = rel.get("variable")
        if not isinstance(var, str) or var not in semparam_env:
            continue
        old = semparam_env[var]
        if isinstance(old, (int, float)):
            new = mutate_value("shared", "radius", float(old), {"radius": float(old)})
        elif as_xy(old) is not None:
            x, y = as_xy(old) or (0.0, 0.0)
            new = (x + 0.05, y)
        else:
            continue

        refs = rel.get("reused_by") or []
        slots: list[tuple[str, str]] = []
        ok = True
        for r in refs:
            if not isinstance(r, str):
                ok = False
                break
            sr = parse_slot_ref(r)
            if sr is None:
                ok = False
                break
            lp, slot = sr
            if lp not in semantic_feats:
                ok = False
                break
            if slot not in semantic_feats[lp].param_nodes:
                ok = False
                break
            if semantic_feats[lp].is_array:
                ok = False
                break
            slots.append((lp, slot))
        if not ok or not slots:
            continue

        out.append(
            Candidate(
                category="consistency",
                op_type="consistency:shared",
                loop_name=None,
                slot=None,
                detail={"variable": var, "old": old, "new": new, "slots": slots},
            )
        )

    return out


def build_dependency_candidates(
    summary: dict[str, Any],
    semparam_tree: ast.Module,
    semparam_feats: dict[str, FeatureInfo],
    semantic_feats: dict[str, FeatureInfo],
) -> list[Candidate]:
    out: list[Candidate] = []
    old_env = eval_var_env(semparam_tree)
    var_usage = build_var_usage_map(semparam_feats)
    derived_all = summary.get("explicit_derived_relations", [])
    base_vars: list[str] = []
    seen_base: set[str] = set()
    for rel in derived_all:
        deps = rel.get("depends_on") or []
        if not isinstance(deps, list):
            continue
        for dep in deps:
            if isinstance(dep, str) and dep in old_env and dep not in seen_base:
                seen_base.add(dep)
                base_vars.append(dep)

    for base in base_vars:

        old_base = old_env[base]
        if isinstance(old_base, (int, float)):
            new_base = mutate_value("dep", "radius", float(old_base), {"radius": float(old_base)})
        elif as_xy(old_base) is not None:
            x, y = as_xy(old_base) or (0.0, 0.0)
            new_base = (x + 0.05, y)
        else:
            continue

        try:
            new_env = eval_var_env(semparam_tree, overrides={base: new_base})
        except Exception:
            continue

        def _value_changed(ov: Any, nv: Any) -> bool:
            if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
                return abs(float(ov) - float(nv)) > 1e-8
            if isinstance(ov, (tuple, list)) and isinstance(nv, (tuple, list)):
                return json.dumps(ov, sort_keys=True) != json.dumps(nv, sort_keys=True)
            return ov != nv

        changed_map: dict[tuple[str, str], tuple[Any, Any]] = {}
        for lp, feat in semparam_feats.items():
            if feat.is_array or lp not in semantic_feats:
                continue
            for slot, node in feat.param_nodes.items():
                try:
                    ov = eval_expr(node, old_env)
                    nv = eval_expr(node, new_env)
                except Exception:
                    continue
                if slot in semantic_feats[lp].param_nodes and _value_changed(ov, nv):
                    changed_map[(lp, slot)] = (ov, nv)

        dep_specs: list[dict[str, Any]] = []
        for drel in derived_all:
            if not isinstance(drel, dict):
                continue
            d_deps = drel.get("depends_on") or []
            if not isinstance(d_deps, list) or base not in d_deps:
                continue
            tvar = drel.get("target")
            if not isinstance(tvar, str):
                continue
            trefs = var_usage.get(tvar, [])
            rhs = drel.get("rhs", "")
            cls0 = classify_dependency_rhs(rhs, base)
            cls = refine_relation_type(cls0, var_usage.get(base, []), trefs)
            dep_specs.append(
                {
                    "target_var": tvar,
                    "rhs": rhs,
                    "target_refs": trefs,
                    "relation_type": cls.get("relation_type", "UNCLASSIFIED"),
                    "relation_note": cls.get("relation_note", ""),
                }
            )
        if not dep_specs:
            continue
        # Filter out unclassified dependency relations.
        if any(s.get("relation_type") == "UNCLASSIFIED" for s in dep_specs if isinstance(s, dict)):
            continue

        # If derived relation metadata exists but the source code has flattened constants
        # (no symbolic propagation), force deterministic dependent updates from rhs.
        dep_var_updates: dict[str, Any] = {}
        for spec in dep_specs:
            if not isinstance(spec, dict):
                continue
            tvar = spec.get("target_var")
            rhs = spec.get("rhs", "")
            if not isinstance(tvar, str) or tvar == base or tvar not in old_env:
                continue
            if not isinstance(rhs, str) or not rhs.strip():
                continue
            try:
                rhs_node = ast.parse(rhs, mode="eval").body
                tv_new = eval_expr(rhs_node, new_env)
            except Exception:
                continue
            tv_old = old_env.get(tvar)
            if not _value_changed(tv_old, tv_new):
                continue
            dep_var_updates[tvar] = tv_new

            for lp, slot in var_usage.get(tvar, []):
                if lp not in semparam_feats or lp not in semantic_feats:
                    continue
                if semparam_feats[lp].is_array:
                    continue
                node = semparam_feats[lp].param_nodes.get(slot)
                if node is None:
                    continue
                try:
                    ov = eval_expr(node, old_env)
                    env_for_target = dict(new_env)
                    env_for_target[tvar] = tv_new
                    nv = eval_expr(node, env_for_target)
                except Exception:
                    ov, nv = tv_old, tv_new
                if _value_changed(ov, nv):
                    changed_map[(lp, slot)] = (ov, nv)

        changed: list[tuple[str, str, Any, Any]] = [
            (lp, slot, ov, nv) for (lp, slot), (ov, nv) in sorted(changed_map.items())
        ]
        if not changed:
            continue

        size_changed = 0
        total_delta = 0.0
        for _lp, slot, ov, nv in changed:
            if slot in SIZE_SLOTS:
                size_changed += 1
            if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
                total_delta += abs(float(nv) - float(ov))

        # Prefer dependency edits that create measurable geometric change (size-like slots),
        # but still keep complex relation edits available.
        rel_bonus = 0.0
        for s in dep_specs:
            t = s.get("relation_type")
            if t == "CONCENTRIC":
                rel_bonus += 1.5
            elif t == "MIRROR":
                rel_bonus += 1.0
            elif t == "DISTANCE":
                rel_bonus += 0.8
            elif t == "RATIO":
                rel_bonus += 0.6
        score = float(len(changed)) + float(size_changed) * 2.0 + min(total_delta, 10.0) + rel_bonus

        out.append(
            Candidate(
                category="dependency",
                op_type="dependency:base",
                loop_name=None,
                slot=None,
                detail={
                    "base_var": base,
                    "old": old_base,
                    "new": new_base,
                    "changed": changed,
                    "base_refs": var_usage.get(base, []),
                    "dep_specs": dep_specs,
                    "dep_var_updates": dep_var_updates,
                    "score": score,
                    "size_changed": size_changed,
                },
            )
        )

    return out


def choose_candidate(
    rel: str,
    candidates: dict[str, list[Candidate]],
    cat_counts: dict[str, int],
    total_kept: int,
    feature_op_counts: dict[str, int],
    dep_relation_counts: dict[str, int],
    max_feature_op_ratio: float,
    seed: int,
) -> Candidate | None:
    available = [c for c in ("feature", "consistency", "dependency") if candidates.get(c)]
    if not available:
        return None

    deficits: dict[str, float] = {}
    if total_kept == 0:
        for c in available:
            deficits[c] = TARGET_CATEGORY_RATIOS[c]
    else:
        for c in available:
            cur = cat_counts.get(c, 0) / total_kept
            deficits[c] = TARGET_CATEGORY_RATIOS[c] - cur

    rnd = random.Random(stable_int_from_text(rel, seed=seed))
    selected_cat = max(available, key=lambda c: (deficits[c], rnd.random()))
    cand_list = list(candidates[selected_cat])
    if selected_cat == "dependency":
        def dep_types(c: Candidate) -> set[str]:
            out: set[str] = set()
            for s in c.detail.get("dep_specs", []):
                if isinstance(s, dict):
                    t = s.get("relation_type")
                    if isinstance(t, str) and t:
                        out.add(t)
            return out

        need_concentric = dep_relation_counts.get("CONCENTRIC", 0) <= 0
        cand_list.sort(
            key=lambda c: (
                1 if (need_concentric and "CONCENTRIC" in dep_types(c)) else 0,
                1 if "CONCENTRIC" in dep_types(c) else 0,
                1 if "MIRROR" in dep_types(c) else 0,
                1 if "DISTANCE" in dep_types(c) else 0,
                1 if "RATIO" in dep_types(c) else 0,
                0 if "UNCLASSIFIED" in dep_types(c) else 1,
                float(c.detail.get("score", 0.0)),
            ),
            reverse=True,
        )
    else:
        rnd.shuffle(cand_list)

    if selected_cat == "feature":
        filtered: list[Candidate] = []
        total_f = max(1, sum(feature_op_counts.values()))
        for c in cand_list:
            op = c.op_type
            cur_ratio = feature_op_counts.get(op, 0) / total_f
            if cur_ratio < max_feature_op_ratio:
                filtered.append(c)
        if filtered:
            return filtered[0]

    return cand_list[0]


def apply_semantic_edit(code: str, cand: Candidate) -> tuple[str, dict[str, Any], set[str]]:
    tree = ast.parse(code)
    feats = extract_features(tree)
    editor = SourceEditor(code)
    touched_loops: set[str] = set()

    if cand.category == "feature" and cand.op_type.startswith("param:"):
        loop = cand.loop_name
        slot = cand.slot
        if loop is None or slot is None or loop not in feats:
            raise ValueError("feature param target missing")
        node = feats[loop].param_nodes.get(slot)
        if node is None:
            raise ValueError("slot node missing in semantic")
        editor.replace_node(node, value_to_source(cand.detail["new"]))
        touched_loops.add(loop)

    elif cand.category == "feature" and cand.op_type.startswith("replace:"):
        loop = cand.loop_name
        if loop is None or loop not in feats:
            raise ValueError("feature replacement target missing")
        dst_kind = cand.detail["target_kind"]
        new_params = cand.detail["new_params"]
        expr = generate_high_api_expr(dst_kind, new_params)
        editor.replace_node(feats[loop].assign.value, expr)
        touched_loops.add(loop)

    elif cand.category == "consistency":
        for lp, slot in cand.detail["slots"]:
            if lp not in feats:
                raise ValueError(f"semantic loop missing: {lp}")
            node = feats[lp].param_nodes.get(slot)
            if node is None:
                raise ValueError(f"semantic slot missing: {lp}.{slot}")
            editor.replace_node(node, value_to_source(cand.detail["new"]))
            touched_loops.add(lp)

    elif cand.category == "dependency":
        for lp, slot, _ov, nv in cand.detail["changed"]:
            if lp not in feats:
                continue
            node = feats[lp].param_nodes.get(slot)
            if node is None:
                continue
            editor.replace_node(node, value_to_source(nv))
            touched_loops.add(lp)

    new_code = editor.apply()
    parsed = ast.parse(new_code)
    new_feats = extract_features(parsed)

    loop_params: dict[str, Any] = {}
    for lp in touched_loops:
        f = new_feats.get(lp)
        if f is None:
            continue
        p = feature_params_from_literal(f)
        if p is None:
            continue
        loop_params[lp] = {"kind": f.kind, "params": p}

    return new_code, loop_params, touched_loops


def _loop_index(loop_name: str) -> str:
    if not LOOP_RE.match(loop_name):
        return "x"
    return loop_name.split("_", 1)[1]


def _literal_key(v: Any) -> str:
    try:
        return value_to_source(v)
    except Exception:
        return repr(v)


def _new_var_name(existing: set[str], loop_name: str, slot: str) -> str:
    base = f"var_{_loop_index(loop_name)}_{slot}"
    if base not in existing:
        return base
    i = 2
    while True:
        cand = f"{base}_{i}"
        if cand not in existing:
            return cand
        i += 1


def _insert_var_assignments(code: str, new_vars: list[tuple[str, Any]]) -> str:
    if not new_vars:
        return code
    tree = ast.parse(code)
    _, vars_ = extract_assignments(tree)
    if vars_:
        after_line = max(asg.end_lineno for asg in vars_.values())
    else:
        after_line = 0
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                after_line = max(after_line, node.end_lineno)

    ins = "".join(f"{name} = {value_to_source(v)}\n" for name, v in new_vars)
    lines = code.splitlines(keepends=True)
    idx = max(0, min(after_line, len(lines)))
    return "".join(lines[:idx]) + ins + "".join(lines[idx:])


def _remove_unused_top_level_vars(code: str) -> str:
    tree = ast.parse(code)
    _, vars_ = extract_assignments(tree)
    if not vars_:
        return code
    name_counts: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            name_counts[n.id] = name_counts.get(n.id, 0) + 1
    editor = SourceEditor(code)
    for name, asg in vars_.items():
        if name_counts.get(name, 0) <= 1:
            editor.replace_node(asg, "")
    return editor.apply()


def _lift_touched_loop_literals_to_vars(code: str, touched_loops: set[str]) -> str:
    if not touched_loops:
        return code

    tree = ast.parse(code)
    feats = extract_features(tree)
    _, vars_ = extract_assignments(tree)

    # Existing var literals for potential reuse.
    value_to_vars: dict[str, list[str]] = {}
    for name, asg in vars_.items():
        lv = node_to_value_literal(asg.value)
        if lv is None:
            continue
        value_to_vars.setdefault(_literal_key(lv), []).append(name)

    name_counts: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            name_counts[n.id] = name_counts.get(n.id, 0) + 1

    existing_names: set[str] = set(vars_.keys())
    pending_new_vars: list[tuple[str, Any]] = []
    editor = SourceEditor(code)
    key_to_chosen_var: dict[str, str] = {}

    for lp in sorted(touched_loops):
        feat = feats.get(lp)
        if feat is None:
            continue
        for slot, node in feat.param_nodes.items():
            if slot == "points":
                continue
            if isinstance(node, ast.Name) and VAR_RE.match(node.id):
                continue
            lit = node_to_value_literal(node)
            if lit is None:
                continue
            k = _literal_key(lit)

            chosen: str | None = key_to_chosen_var.get(k)
            if chosen is None:
                reuse_list = value_to_vars.get(k, [])
                if reuse_list:
                    chosen = reuse_list[0]

                if chosen is None:
                    chosen = _new_var_name(existing_names, lp, slot)
                    existing_names.add(chosen)
                    pending_new_vars.append((chosen, lit))

                key_to_chosen_var[k] = chosen

            editor.replace_node(node, chosen)

    out = editor.apply()
    out = _insert_var_assignments(out, pending_new_vars)
    return out


def apply_semparam_edit(code: str, cand: Candidate) -> str:
    tree = ast.parse(code)
    feats = extract_features(tree)
    _, vars_ = extract_assignments(tree)
    editor = SourceEditor(code)
    touched_loops: set[str] = set()

    if cand.category == "feature" and cand.op_type.startswith("param:"):
        loop = cand.loop_name
        slot = cand.slot
        if loop is None or slot is None or loop not in feats:
            raise ValueError("semparam feature target missing")
        node = feats[loop].param_nodes.get(slot)
        if node is None:
            raise ValueError("semparam slot missing")
        touched_loops.add(loop)
        if isinstance(node, ast.Name) and VAR_RE.match(node.id) and node.id in vars_:
            editor.replace_node(vars_[node.id].value, value_to_source(cand.detail["new"]))
        else:
            editor.replace_node(node, value_to_source(cand.detail["new"]))

    elif cand.category == "feature" and cand.op_type.startswith("replace:"):
        loop = cand.loop_name
        if loop is None or loop not in feats:
            raise ValueError("semparam replace target missing")
        touched_loops.add(loop)
        dst_kind = cand.detail["target_kind"]
        new_params = cand.detail["new_params"]
        expr = generate_high_api_expr(dst_kind, new_params)
        editor.replace_node(feats[loop].assign.value, expr)

    elif cand.category == "consistency":
        var = cand.detail["variable"]
        asg = vars_.get(var)
        if asg is None:
            raise ValueError(f"shared var missing in semparam: {var}")
        editor.replace_node(asg.value, value_to_source(cand.detail["new"]))
        for lp, _slot in cand.detail.get("slots", []):
            if isinstance(lp, str):
                touched_loops.add(lp)

    elif cand.category == "dependency":
        var = cand.detail["base_var"]
        asg = vars_.get(var)
        if asg is None:
            raise ValueError(f"base var missing in semparam: {var}")
        editor.replace_node(asg.value, value_to_source(cand.detail["new"]))
        for lp, _slot, _ov, _nv in cand.detail.get("changed", []):
            if isinstance(lp, str):
                touched_loops.add(lp)

        dep_updates = cand.detail.get("dep_var_updates", {})
        if isinstance(dep_updates, dict):
            for tvar, tnew in dep_updates.items():
                if not isinstance(tvar, str) or tvar == var:
                    continue
                asg2 = vars_.get(tvar)
                if asg2 is None:
                    continue
                # Preserve symbolic dependency when target assignment already references base var.
                try:
                    if var in collect_var_names(asg2.value):
                        continue
                except Exception:
                    pass
                editor.replace_node(asg2.value, value_to_source(tnew))

    out = editor.apply()
    out = _lift_touched_loop_literals_to_vars(out, touched_loops)
    out = _remove_unused_top_level_vars(out)
    return out


def apply_primitive_edit(
    code: str,
    loop_params: dict[str, Any],
    touched_loops: set[str],
    semantic_code: str,
) -> str:
    tree = ast.parse(code)
    loops, _ = extract_assignments(tree)
    editor = SourceEditor(code)
    alias_map = build_semantic_to_primitive_loop_aliases(semantic_code, code)

    for lp in sorted(touched_loops):
        if lp not in loop_params:
            raise ValueError(f"primitive target loop params missing: {lp}")

        target_loops = sorted(alias_map.get(lp, {lp}))
        valid_targets = [x for x in target_loops if x in loops]
        if not valid_targets:
            raise ValueError(f"primitive loop missing: {lp}")

        kind = loop_params[lp]["kind"]
        params = loop_params[lp]["params"]

        if kind == "slot":
            expr = generate_chain_slot_expr(params)
        elif kind == "rect_fillet":
            expr = generate_chain_rect_fillet_expr(params)
        elif kind == "rect":
            expr = generate_chain_rect_expr(params)
        elif kind == "polygon":
            expr = generate_chain_polygon_expr(params["points"])
        elif kind == "regularPolygon":
            expr = generate_chain_regular_polygon_expr(params)
        elif kind == "circle":
            expr = generate_primitive_circle_expr(params)
        else:
            raise ValueError(f"unsupported primitive kind: {kind}")

        # Guardrail: primitive target must not introduce semantic feature APIs.
        tmp_tree = ast.parse(expr, mode="eval")
        for n in ast.walk(tmp_tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr in PRIMITIVE_BANNED_APIS:
                    raise ValueError(f"primitive semantic API leakage detected: {n.func.attr}")

        for plp in valid_targets:
            editor.replace_node(loops[plp].value, expr)

    return editor.apply()


def validate_triplet(
    src_codes: dict[str, str],
    tgt_codes: dict[str, str],
    cross_rel_tol: float,
    min_change_tol: float,
) -> tuple[bool, dict[str, Any]]:
    detail: dict[str, Any] = {"per_repr": {}}

    for r in ("semparam", "semantic", "primitive"):
        try:
            ast.parse(tgt_codes[r])
            syn_ok = True
            syn_err = ""
        except SyntaxError as e:
            syn_ok = False
            syn_err = str(e)

        s0 = execute_cq(src_codes[r]) if syn_ok else ExecStat(False, "syntax failed", None, None, None, None)
        s1 = execute_cq(tgt_codes[r]) if syn_ok else ExecStat(False, "syntax failed", None, None, None, None)

        changed = False
        mesh_changed = False
        if s0.ok and s1.ok and s0.volume is not None and s1.volume is not None and s0.bbox and s1.bbox:
            vol_ch = rel_diff(s0.volume, s1.volume)
            bb_ch = bbox_diff(s0.bbox, s1.bbox)
            if s0.mesh_sig is not None and s1.mesh_sig is not None:
                mesh_changed = s0.mesh_sig != s1.mesh_sig
            changed = (vol_ch > min_change_tol) or (bb_ch > min_change_tol) or mesh_changed
        detail["per_repr"][r] = {
            "syntax_ok": syn_ok,
            "syntax_err": syn_err,
            "src_exec": s0.ok,
            "src_err": s0.err,
            "tgt_exec": s1.ok,
            "tgt_err": s1.err,
            "src_volume": s0.volume,
            "tgt_volume": s1.volume,
            "src_bbox": s0.bbox,
            "tgt_bbox": s1.bbox,
            "mesh_sig_src": s0.mesh_sig,
            "mesh_sig_tgt": s1.mesh_sig,
            "mesh_changed": mesh_changed,
            "changed": changed,
        }

    all_ok = True
    for r in ("semparam", "semantic", "primitive"):
        d = detail["per_repr"][r]
        if not (d["syntax_ok"] and d["src_exec"] and d["tgt_exec"] and d["changed"]):
            all_ok = False

    cross_ok = False
    vols = []
    for r in ("semparam", "semantic", "primitive"):
        v = detail["per_repr"][r]["tgt_volume"]
        if isinstance(v, (int, float)):
            vols.append(float(v))
    if len(vols) == 3:
        diffs = [
            rel_diff(vols[0], vols[1]),
            rel_diff(vols[0], vols[2]),
            rel_diff(vols[1], vols[2]),
        ]
        cross_ok = max(diffs) <= cross_rel_tol
    detail["cross_repr_equiv"] = cross_ok

    return all_ok and cross_ok, detail


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def tsv_escape(v: Any) -> str:
    s = "" if v is None else str(v)
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def relation_tags_for_candidate(cand: Candidate) -> list[str]:
    tags: set[str] = set()
    if cand.category == "dependency":
        for s in cand.detail.get("dep_specs", []):
            if not isinstance(s, dict):
                continue
            t = s.get("relation_type", "UNCLASSIFIED")
            if isinstance(t, str) and t:
                tags.add(t)
        # Match the paper's exclusive DE categories; retain all relation rules
        # in dep_relation_detail and the edit instruction.
        if tags == {"MIRROR", "RATIO"}:
            tags = {"MIRROR"}
    elif cand.category == "consistency":
        slots = cand.detail.get("slots", [])
        if isinstance(slots, list) and slots:
            slot_names = {str(s) for _lp, s in slots if isinstance(_lp, str)}
            if slot_names == {"center"}:
                tags.add("CONCENTRIC")
            else:
                tags.add("SHARED")
    return sorted(tags)


def relation_detail_for_candidate(cand: Candidate, max_items: int = 8) -> str:
    if cand.category != "dependency":
        return ""
    items: list[str] = []
    for s in cand.detail.get("dep_specs", []):
        if not isinstance(s, dict):
            continue
        tv = s.get("target_var", "target")
        rt = s.get("relation_type", "UNCLASSIFIED")
        rn = s.get("relation_note", "")
        if isinstance(rn, str) and rn:
            items.append(f"{tv}:{rt}({rn})")
        else:
            items.append(f"{tv}:{rt}")
    items = sorted(set(items))
    if len(items) > max_items:
        items = items[:max_items] + ["..."]
    return "; ".join(items)


def write_triplet(output_root: Path, rel: Path, instruction: str, src_codes: dict[str, str], tgt_codes: dict[str, str]) -> None:
    for rep in ("semparam", "semantic", "primitive"):
        src_path = output_root / rep / "test" / "src_code" / rel
        txt_path = output_root / rep / "test" / "text" / rel.with_suffix(".txt")
        tgt_path = output_root / rep / "test" / "tgt_code" / rel

        ensure_parent(src_path)
        ensure_parent(txt_path)
        ensure_parent(tgt_path)

        src_path.write_text(src_codes[rep], encoding="utf-8")
        txt_path.write_text(instruction + "\n", encoding="utf-8")
        tgt_path.write_text(tgt_codes[rep], encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"explicit_shared_relations": [], "explicit_derived_relations": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"explicit_shared_relations": [], "explicit_derived_relations": []}


def feature_family_from_op_type(op_type: str) -> str:
    parts = op_type.split(":")
    if len(parts) >= 2 and parts[0] == "param":
        return parts[1]
    if len(parts) >= 2 and parts[0] == "replace":
        return parts[1]
    return ""


def consistency_subtype_from_slots(slots: list[tuple[str, str]]) -> str:
    if not slots:
        return "same-shared"
    slot_names = sorted({s for _lp, s in slots})
    if len(slot_names) == 1:
        return f"same-{slot_names[0]}"
    return "same-" + "+".join(slot_names)


def stringify_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float, bool, tuple, list)):
        try:
            return value_to_source(v)
        except Exception:
            return str(v)
    return str(v)


def candidate_old_new(cand: Candidate) -> tuple[str, str]:
    d = cand.detail
    if cand.category == "dependency":
        return stringify_value(d.get("old")), stringify_value(d.get("new"))
    if cand.category == "consistency":
        return stringify_value(d.get("old")), stringify_value(d.get("new"))
    if cand.category == "feature":
        return stringify_value(d.get("old")), stringify_value(d.get("new"))
    return "", ""


def candidate_affected_slots(cand: Candidate) -> str:
    if cand.category == "consistency":
        out = [f"{lp}.{s}" for lp, s in cand.detail.get("slots", []) if isinstance(lp, str) and isinstance(s, str)]
        return ",".join(sorted(set(out)))
    if cand.category == "dependency":
        out = []
        for lp, s, _ov, _nv in cand.detail.get("changed", []):
            if isinstance(lp, str) and isinstance(s, str):
                out.append(f"{lp}.{s}")
        return ",".join(sorted(set(out)))
    if cand.loop_name and cand.slot:
        return f"{cand.loop_name}.{cand.slot}"
    return ""


def relation_subtype_from_candidate(cand: Candidate, rel_tags: list[str]) -> str:
    if cand.category == "consistency":
        return consistency_subtype_from_slots(cand.detail.get("slots", []))
    if cand.category == "dependency":
        if not rel_tags:
            return ""
        if len(rel_tags) == 1:
            return rel_tags[0]
        return "+".join(sorted(rel_tags))
    return ""


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    cat_counts = {"feature": 0, "consistency": 0, "dependency": 0}
    for r in rows:
        c = r.get("category", "")
        if c in cat_counts:
            cat_counts[c] += 1

    feature_rows = [r for r in rows if r.get("category") == "feature"]
    consistency_rows = [r for r in rows if r.get("category") == "consistency"]
    dependency_rows = [r for r in rows if r.get("category") == "dependency"]

    feature_family_counts: dict[str, int] = {}
    feature_op_counts: dict[str, int] = {}
    rect_related = 0
    for r in feature_rows:
        fam = str(r.get("feature_family", ""))
        op = str(r.get("op_type", ""))
        if fam:
            feature_family_counts[fam] = feature_family_counts.get(fam, 0) + 1
        if op:
            feature_op_counts[op] = feature_op_counts.get(op, 0) + 1
        if fam == "rect":
            rect_related += 1

    consistency_subtype_counts: dict[str, int] = {}
    for r in consistency_rows:
        st = str(r.get("relation_subtype", "")) or "unknown"
        consistency_subtype_counts[st] = consistency_subtype_counts.get(st, 0) + 1

    dependency_subtype_counts: dict[str, int] = {}
    for r in dependency_rows:
        tags = [t for t in str(r.get("dep_relation_tags", "")).split(",") if t]
        for t in sorted(set(tags)):
            dependency_subtype_counts[t] = dependency_subtype_counts.get(t, 0) + 1

    return {
        "total": total,
        "category_counts": cat_counts,
        "category_ratios": {k: (cat_counts[k] / total if total else 0.0) for k in cat_counts},
        "feature_family_counts": feature_family_counts,
        "feature_op_counts": feature_op_counts,
        "rect_related_feature_count": rect_related,
        "rect_related_feature_ratio": (rect_related / len(feature_rows) if feature_rows else 0.0),
        "consistency_subtype_counts": consistency_subtype_counts,
        "dependency_relation_counts": dependency_subtype_counts,
    }


def select_feature_rows(
    rows: list[dict[str, Any]],
    target_n: int,
    rect_max_ratio: float,
    op_max_ratio: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not rows or target_n <= 0:
        return []
    rows0 = list(rows)
    rng.shuffle(rows0)
    target_n = min(target_n, len(rows0))
    rect_cap = max(1, int(math.floor(target_n * rect_max_ratio)))
    op_cap = max(1, int(math.floor(target_n * op_max_ratio)))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    op_counts: dict[str, int] = {}
    rect_count = 0
    kind_counts = {"replacement": 0, "param": 0}

    def row_kind(r: dict[str, Any]) -> str:
        op = str(r.get("op_type", ""))
        return "replacement" if op.startswith("replace:") else "param"

    total_repl = sum(1 for r in rows0 if row_kind(r) == "replacement")
    total_param = sum(1 for r in rows0 if row_kind(r) == "param")

    # Prefer 1:1 mix between replacement and parameter edits inside feature category.
    kind_quota = {
        "replacement": target_n // 2,
        "param": target_n - (target_n // 2),
    }
    if total_repl < kind_quota["replacement"]:
        deficit = kind_quota["replacement"] - total_repl
        kind_quota["replacement"] = total_repl
        kind_quota["param"] = min(total_param, kind_quota["param"] + deficit)
    if total_param < kind_quota["param"]:
        deficit = kind_quota["param"] - total_param
        kind_quota["param"] = total_param
        kind_quota["replacement"] = min(total_repl, kind_quota["replacement"] + deficit)

    def can_add(r: dict[str, Any], use_rect_cap: bool, use_op_cap: bool, use_kind_quota: bool) -> bool:
        nonlocal rect_count
        rid = str(r.get("id", ""))
        if rid in selected_ids:
            return False
        kind = row_kind(r)
        if use_kind_quota and kind_counts[kind] >= kind_quota[kind]:
            return False
        fam = str(r.get("feature_family", ""))
        op = str(r.get("op_type", ""))
        if use_rect_cap and fam == "rect" and rect_count >= rect_cap:
            return False
        if use_op_cap and op_counts.get(op, 0) >= op_cap:
            return False
        return True

    def do_add(r: dict[str, Any]) -> None:
        nonlocal rect_count
        rid = str(r.get("id", ""))
        fam = str(r.get("feature_family", ""))
        op = str(r.get("op_type", ""))
        kind = row_kind(r)
        selected.append(r)
        selected_ids.add(rid)
        op_counts[op] = op_counts.get(op, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if fam == "rect":
            rect_count += 1

    def fill_rows(use_rect_cap: bool, use_op_cap: bool, use_kind_quota: bool, rare_only: bool = False) -> None:
        for r in rows0:
            if len(selected) >= target_n:
                break
            if rare_only and str(r.get("feature_family", "")) not in RARE_FEATURE_FAMILIES:
                continue
            if can_add(r, use_rect_cap, use_op_cap, use_kind_quota):
                do_add(r)

    # Stage 1: prioritize rare feature families first under 1:1 quota.
    fill_rows(use_rect_cap=True, use_op_cap=True, use_kind_quota=True, rare_only=True)

    # Stage 2-4: fill to kind quota (replacement:param ~= 1:1) with progressive cap relaxation.
    fill_rows(use_rect_cap=True, use_op_cap=True, use_kind_quota=True)
    fill_rows(use_rect_cap=True, use_op_cap=False, use_kind_quota=True)
    fill_rows(use_rect_cap=False, use_op_cap=False, use_kind_quota=True)

    # Stage 5-7: if quota cannot be met due to availability, fill remaining slots.
    fill_rows(use_rect_cap=True, use_op_cap=True, use_kind_quota=False)
    fill_rows(use_rect_cap=True, use_op_cap=False, use_kind_quota=False)
    fill_rows(use_rect_cap=False, use_op_cap=False, use_kind_quota=False)

    return selected


def select_rows_by_subtype(
    rows: list[dict[str, Any]],
    target_n: int,
    subtype_key: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not rows or target_n <= 0:
        return []
    target_n = min(target_n, len(rows))
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        st = str(r.get(subtype_key, "")) or "unknown"
        groups.setdefault(st, []).append(r)
    for g in groups.values():
        rng.shuffle(g)

    keys = sorted(groups.keys())
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(out) < target_n:
        progressed = False
        for k in keys:
            g = groups[k]
            while g and str(g[-1].get("id", "")) in seen:
                g.pop()
            if not g:
                continue
            r = g.pop()
            rid = str(r.get("id", ""))
            if rid in seen:
                continue
            out.append(r)
            seen.add(rid)
            progressed = True
            if len(out) >= target_n:
                break
        if not progressed:
            break
    return out


def copy_selected_triplets(pool_root: Path, bench_root: Path, selected_rows: list[dict[str, Any]]) -> None:
    for row in selected_rows:
        rel = Path(str(row["rel"]))
        for rep in ("semparam", "semantic", "primitive"):
            for part, suffix in (("src_code", ".py"), ("text", ".txt"), ("tgt_code", ".py")):
                src = pool_root / rep / "test" / part / (rel if suffix == ".py" else rel.with_suffix(".txt"))
                dst = bench_root / rep / "test" / part / (rel if suffix == ".py" else rel.with_suffix(".txt"))
                ensure_parent(dst)
                shutil.copy2(src, dst)


def eligible_for_edit(code_path: Path, summary_path: Path) -> bool:
    summary = load_json(summary_path)
    if not (summary.get('explicit_shared_relations') or summary.get('explicit_derived_relations')):
        return False
    tree = ast.parse(code_path.read_text())
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {'rect', 'polygon', 'regularPolygon', 'fillet', 'slot', 'slot2D', 'rarray', 'polarArray'} for n in ast.walk(tree))


def build_edit_dataset(generation_root: Path, summary_root: Path, output_root: Path) -> int:
    """Build deterministic aligned edits; retain the existing 2% volume criterion."""
    from types import SimpleNamespace
    global _EXECUTION_ROOT
    _EXECUTION_ROOT = Path(output_root) / 'work'
    args = SimpleNamespace(
        semparam_root=generation_root/'semparam/test/code',
        semantic_root=generation_root/'semantic/test/code',
        primitive_root=generation_root/'primitive/test/code',
        param_summary_root=summary_root, output_root=output_root, seed=42,
        max_feature_op_ratio=0.35,
        benchmark_feature_target=400, benchmark_consistency_target=300,
        benchmark_dependency_target=300, benchmark_size=1000,
        feature_rect_max_ratio=0.60, benchmark_feature_op_max_ratio=0.30,
        cross_rel_tol=0.02, min_change_tol=1e-5,
    )
    random.seed(args.seed)

    semparam_root = args.semparam_root.resolve()
    semantic_root = args.semantic_root.resolve()
    primitive_root = args.primitive_root.resolve()
    summary_root = args.param_summary_root.resolve()
    out_root = args.output_root.resolve()
    pool_root = out_root / "validated_pool"
    bench_root = out_root / "benchmark_1k"

    rels = sorted(p.relative_to(semparam_root) for p in semparam_root.rglob("*.py"))
    rels = [rel for rel in rels if eligible_for_edit(semparam_root / rel, summary_root / rel.with_suffix('.json'))]

    out_root.mkdir(parents=True, exist_ok=True)
    pool_root.mkdir(parents=True, exist_ok=True)
    bench_root.mkdir(parents=True, exist_ok=True)

    kept_tsv_path = pool_root / "kept_ops.tsv"
    failed_tsv_path = pool_root / "failed_cases.tsv"
    unclassified_review_path = pool_root / "dependency_unclassified_review.tsv"
    pool_summary_path = pool_root / "summary.json"
    bench_selected_tsv_path = bench_root / "selected_ops.tsv"
    bench_summary_path = bench_root / "summary.json"
    summary_path = out_root / "summary.json"

    if kept_tsv_path.exists() or failed_tsv_path.exists() or pool_summary_path.exists():
        raise SystemExit(
            f"Output metadata already exists in {out_root}; choose a fresh output directory."
        )

    cat_counts = {"feature": 0, "consistency": 0, "dependency": 0}
    feature_op_counts: dict[str, int] = {}
    dep_relation_counts: dict[str, int] = {}
    consistency_relation_counts: dict[str, int] = {}
    relation_tag_counts: dict[str, int] = {}
    relation_tag_counts_by_category: dict[str, dict[str, int]] = {
        "feature": {},
        "consistency": {},
        "dependency": {},
    }

    kept = 0
    skipped = 0
    reasons: dict[str, int] = {}
    kept_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for i, rel in enumerate(rels, 1):
        sid = rel.with_suffix("").as_posix().replace("/", "_")

        sp = semparam_root / rel
        se = semantic_root / rel
        pr = primitive_root / rel

        if not (sp.exists() and se.exists() and pr.exists()):
            skipped += 1
            reasons["missing_representation"] = reasons.get("missing_representation", 0) + 1
            failed_rows.append(
                {
                    "id": sid,
                    "rel": rel.as_posix(),
                    "reason": "missing_representation",
                    "category": "",
                    "op_type": "",
                    "loop": "",
                    "slot": "",
                    "error": "",
                }
            )
            continue

        src_codes = {
            "semparam": sp.read_text(encoding="utf-8"),
            "semantic": se.read_text(encoding="utf-8"),
            "primitive": pr.read_text(encoding="utf-8"),
        }

        try:
            semparam_tree = ast.parse(src_codes["semparam"])
            semantic_tree = ast.parse(src_codes["semantic"])
            primitive_tree = ast.parse(src_codes["primitive"])
        except SyntaxError as e:
            skipped += 1
            reasons["parse_error"] = reasons.get("parse_error", 0) + 1
            failed_rows.append(
                {
                    "id": sid,
                    "rel": rel.as_posix(),
                    "reason": "parse_error",
                    "category": "",
                    "op_type": "",
                    "loop": "",
                    "slot": "",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue

        semparam_feats = extract_features(semparam_tree)
        semantic_feats = extract_features(semantic_tree)
        _primitive_feats = extract_features(primitive_tree)

        summary = load_json(summary_root / rel.with_suffix(".json"))
        relation_bound_slots = collect_relation_bound_slots(summary, semparam_feats)

        try:
            semparam_env = eval_var_env(semparam_tree)
        except Exception:
            semparam_env = {}

        cands_feature = build_feature_candidates(semantic_feats, relation_bound_slots)
        cands_consistency = build_consistency_candidates(summary, semparam_env, semantic_feats)
        cands_dependency = build_dependency_candidates(summary, semparam_tree, semparam_feats, semantic_feats)
        cands_feature = [c for c in cands_feature if candidate_is_well_specified(c)]
        cands_consistency = [c for c in cands_consistency if candidate_is_well_specified(c)]
        cands_dependency = [c for c in cands_dependency if candidate_is_well_specified(c)]

        cands = {
            "feature": cands_feature,
            "consistency": cands_consistency,
            "dependency": cands_dependency,
        }

        cand = choose_candidate(
            rel.as_posix(),
            cands,
            cat_counts,
            max(1, kept),
            feature_op_counts,
            dep_relation_counts,
            args.max_feature_op_ratio,
            args.seed,
        )
        if cand is None:
            skipped += 1
            reasons["no_candidate"] = reasons.get("no_candidate", 0) + 1
            failed_rows.append(
                {
                    "id": sid,
                    "rel": rel.as_posix(),
                    "reason": "no_candidate",
                    "category": "",
                    "op_type": "",
                    "loop": "",
                    "slot": "",
                    "error": "",
                }
            )
            continue

        instruction = instruction_for_candidate(cand)
        if not instruction.strip():
            skipped += 1
            reasons["underspecified_instruction"] = reasons.get("underspecified_instruction", 0) + 1
            failed_rows.append(
                {
                    "id": sid,
                    "rel": rel.as_posix(),
                    "reason": "underspecified_instruction",
                    "category": cand.category,
                    "op_type": cand.op_type,
                    "loop": cand.loop_name or "",
                    "slot": cand.slot or "",
                    "error": "",
                }
            )
            continue

        try:
            tgt_semantic, loop_params, touched_loops = apply_semantic_edit(src_codes["semantic"], cand)
            tgt_semparam = apply_semparam_edit(src_codes["semparam"], cand)
            tgt_primitive = apply_primitive_edit(
                src_codes["primitive"],
                loop_params,
                touched_loops,
                src_codes["semantic"],
            )

            tgt_codes = {
                "semparam": tgt_semparam,
                "semantic": tgt_semantic,
                "primitive": tgt_primitive,
            }

            ok, vdetail = validate_triplet(
                src_codes=src_codes,
                tgt_codes=tgt_codes,
                cross_rel_tol=args.cross_rel_tol,
                min_change_tol=args.min_change_tol,
            )
            if not ok:
                skipped += 1
                reasons["validation_failed"] = reasons.get("validation_failed", 0) + 1
                failed_rows.append(
                    {
                        "id": sid,
                        "rel": rel.as_posix(),
                        "reason": "validation_failed",
                        "category": cand.category,
                        "op_type": cand.op_type,
                        "loop": cand.loop_name or "",
                        "slot": cand.slot or "",
                        "error": json.dumps(vdetail, ensure_ascii=False),
                    }
                )
                continue

        except Exception as e:
            skipped += 1
            reasons["rewrite_failed"] = reasons.get("rewrite_failed", 0) + 1
            failed_rows.append(
                {
                    "id": sid,
                    "rel": rel.as_posix(),
                    "reason": "rewrite_failed",
                    "category": cand.category,
                    "op_type": cand.op_type,
                    "loop": cand.loop_name or "",
                    "slot": cand.slot or "",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue

        write_triplet(pool_root, rel, instruction, src_codes, tgt_codes)

        kept += 1
        cat_counts[cand.category] += 1
        if cand.category == "feature":
            feature_op_counts[cand.op_type] = feature_op_counts.get(cand.op_type, 0) + 1
        if cand.category == "dependency":
            for s in cand.detail.get("dep_specs", []):
                if not isinstance(s, dict):
                    continue
                rt = s.get("relation_type", "UNCLASSIFIED")
                if not isinstance(rt, str) or not rt:
                    rt = "UNCLASSIFIED"
                dep_relation_counts[rt] = dep_relation_counts.get(rt, 0) + 1
        rel_tags = relation_tags_for_candidate(cand)
        for t in rel_tags:
            relation_tag_counts[t] = relation_tag_counts.get(t, 0) + 1
            cat_map = relation_tag_counts_by_category.setdefault(cand.category, {})
            cat_map[t] = cat_map.get(t, 0) + 1
            if cand.category == "consistency":
                consistency_relation_counts[t] = consistency_relation_counts.get(t, 0) + 1

        kept_rows.append(
            {
                "id": sid,
                "rel": rel.as_posix(),
                "category": cand.category,
                "op_type": cand.op_type,
                "feature_family": feature_family_from_op_type(cand.op_type),
                "relation_subtype": relation_subtype_from_candidate(cand, rel_tags),
                "loop": cand.loop_name or "",
                "slot": cand.slot or "",
                "old_value": candidate_old_new(cand)[0],
                "new_value": candidate_old_new(cand)[1],
                "affected_slots": candidate_affected_slots(cand),
                "instruction": instruction,
                "touched_loops": ",".join(sorted(list(touched_loops))),
                "touched_code_locations": ",".join(sorted(list(touched_loops))),
                "dep_relation_tags": ",".join(rel_tags),
                "dep_relation_detail": relation_detail_for_candidate(cand),
                "validation_status": "passed",
            }
        )

        if i % 50 == 0:
            print(f"processed {i}/{len(rels)} | kept={kept} skipped={skipped}")

    with kept_tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "id\trel\tcategory\top_type\tfeature_family\trelation_subtype\tloop\tslot\told_value\tnew_value\taffected_slots\tinstruction\ttouched_loops\ttouched_code_locations\tdep_relation_tags\tdep_relation_detail\tvalidation_status\n"
        )
        for row in kept_rows:
            f.write(
                "\t".join(
                    tsv_escape(row[k])
                    for k in (
                        "id",
                        "rel",
                        "category",
                        "op_type",
                        "feature_family",
                        "relation_subtype",
                        "loop",
                        "slot",
                        "old_value",
                        "new_value",
                        "affected_slots",
                        "instruction",
                        "touched_loops",
                        "touched_code_locations",
                        "dep_relation_tags",
                        "dep_relation_detail",
                        "validation_status",
                    )
                )
                + "\n"
            )

    with failed_tsv_path.open("w", encoding="utf-8") as f:
        f.write("id\trel\treason\tcategory\top_type\tloop\tslot\terror\n")
        for row in failed_rows:
            f.write(
                "\t".join(
                    tsv_escape(row[k])
                    for k in ("id", "rel", "reason", "category", "op_type", "loop", "slot", "error")
                )
                + "\n"
            )

    with unclassified_review_path.open("w", encoding="utf-8") as f:
        f.write("id\trel\top_type\tdep_relation_tags\tdep_relation_detail\tinstruction\n")
        for row in kept_rows:
            if row.get("category") != "dependency":
                continue
            tags = row.get("dep_relation_tags", "")
            if "UNCLASSIFIED" not in tags.split(","):
                continue
            f.write(
                "\t".join(
                    tsv_escape(row[k])
                    for k in ("id", "rel", "op_type", "dep_relation_tags", "dep_relation_detail", "instruction")
                )
                + "\n"
            )

    total = kept + skipped
    pool_stats = summarize_rows(kept_rows)
    pool_summary = {
        "semparam_root": str(semparam_root),
        "semantic_root": str(semantic_root),
        "primitive_root": str(primitive_root),
        "summary_root": str(summary_root),
        "output_root": str(pool_root),
        "total_processed": total,
        "kept": kept,
        "skipped": skipped,
        "kept_tsv": str(kept_tsv_path),
        "failed_tsv": str(failed_tsv_path),
        "dependency_unclassified_review_tsv": str(unclassified_review_path),
        "category_counts": cat_counts,
        "feature_op_counts": feature_op_counts,
        "dependency_relation_counts": dep_relation_counts,
        "consistency_relation_counts": consistency_relation_counts,
        "relation_tag_counts": relation_tag_counts,
        "relation_tag_counts_by_category": relation_tag_counts_by_category,
        "skip_reasons": reasons,
        "target_ratios": TARGET_CATEGORY_RATIOS,
        "realized_ratios": {k: (cat_counts[k] / kept if kept else 0.0) for k in cat_counts},
        "distribution": pool_stats,
        "notes": "validated edit pool (all validated triples generated by this script).",
    }
    pool_summary_path.write_text(json.dumps(pool_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Stage 2: stratified benchmark sampling from validated pool.
    target_counts = {
        "feature": max(0, int(args.benchmark_feature_target)),
        "consistency": max(0, int(args.benchmark_consistency_target)),
        "dependency": max(0, int(args.benchmark_dependency_target)),
    }
    benchmark_size = max(0, int(args.benchmark_size))
    desired_total = min(benchmark_size, len(kept_rows))

    rows_feature = [r for r in kept_rows if r.get("category") == "feature"]
    rows_consistency = [r for r in kept_rows if r.get("category") == "consistency"]
    rows_dependency = [r for r in kept_rows if r.get("category") == "dependency"]

    sample_rng = random.Random(args.seed + 1009)
    selected_feature = select_feature_rows(
        rows_feature,
        min(target_counts["feature"], desired_total),
        rect_max_ratio=float(args.feature_rect_max_ratio),
        op_max_ratio=float(args.benchmark_feature_op_max_ratio),
        rng=sample_rng,
    )
    selected_consistency = select_rows_by_subtype(
        rows_consistency,
        min(target_counts["consistency"], desired_total),
        subtype_key="relation_subtype",
        rng=sample_rng,
    )
    selected_dependency = select_rows_by_subtype(
        rows_dependency,
        min(target_counts["dependency"], desired_total),
        subtype_key="relation_subtype",
        rng=sample_rng,
    )

    selected_map: dict[str, dict[str, Any]] = {}
    for r in selected_feature + selected_consistency + selected_dependency:
        rid = str(r["id"])
        selected_map[rid] = r

    selected_rows = list(selected_map.values())
    if len(selected_rows) > desired_total:
        sample_rng.shuffle(selected_rows)
        selected_rows = selected_rows[:desired_total]
    if len(selected_rows) < desired_total:
        selected_ids = {str(r["id"]) for r in selected_rows}
        need = desired_total - len(selected_rows)
        leftovers_by_cat: dict[str, list[dict[str, Any]]] = {
            "feature": [r for r in rows_feature if str(r["id"]) not in selected_ids],
            "consistency": [r for r in rows_consistency if str(r["id"]) not in selected_ids],
            "dependency": [r for r in rows_dependency if str(r["id"]) not in selected_ids],
        }
        for k in leftovers_by_cat:
            sample_rng.shuffle(leftovers_by_cat[k])

        selected_counts = {
            "feature": sum(1 for r in selected_rows if r.get("category") == "feature"),
            "consistency": sum(1 for r in selected_rows if r.get("category") == "consistency"),
            "dependency": sum(1 for r in selected_rows if r.get("category") == "dependency"),
        }

        # Backfill by dynamically matching target proportions among currently-fillable categories.
        while need > 0:
            fillable = [c for c in ("feature", "consistency", "dependency") if leftovers_by_cat[c]]
            if not fillable:
                break
            wsum = sum(max(1, target_counts[c]) for c in fillable)
            total_fillable_now = sum(selected_counts[c] for c in fillable)
            total_fillable_final = total_fillable_now + need
            ideal = {
                c: total_fillable_final * (max(1, target_counts[c]) / wsum)
                for c in fillable
            }
            chosen_cat = max(
                fillable,
                key=lambda c: (
                    ideal[c] - selected_counts[c],
                    target_counts[c],
                    -("feature", "consistency", "dependency").index(c),
                ),
            )
            r = leftovers_by_cat[chosen_cat].pop()
            rid = str(r["id"])
            if rid in selected_ids:
                continue
            selected_rows.append(r)
            selected_ids.add(rid)
            selected_counts[chosen_cat] += 1
            need -= 1

    # Stable ordering by id for reproducibility.
    selected_rows = sorted(selected_rows, key=lambda r: str(r.get("id", "")))

    copy_selected_triplets(pool_root, bench_root, selected_rows)

    with bench_selected_tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "id\trel\tcategory\top_type\tfeature_family\trelation_subtype\tloop\tslot\told_value\tnew_value\taffected_slots\tinstruction\ttouched_loops\ttouched_code_locations\tdep_relation_tags\tdep_relation_detail\tvalidation_status\n"
        )
        for row in selected_rows:
            f.write(
                "\t".join(
                    tsv_escape(row[k])
                    for k in (
                        "id",
                        "rel",
                        "category",
                        "op_type",
                        "feature_family",
                        "relation_subtype",
                        "loop",
                        "slot",
                        "old_value",
                        "new_value",
                        "affected_slots",
                        "instruction",
                        "touched_loops",
                        "touched_code_locations",
                        "dep_relation_tags",
                        "dep_relation_detail",
                        "validation_status",
                    )
                )
                + "\n"
            )

    benchmark_stats = summarize_rows(selected_rows)
    benchmark_summary = {
        "source_pool": str(pool_root),
        "output_root": str(bench_root),
        "benchmark_size_target": benchmark_size,
        "benchmark_size_realized": len(selected_rows),
        "category_target_counts": target_counts,
        "category_available_counts": {
            "feature": len(rows_feature),
            "consistency": len(rows_consistency),
            "dependency": len(rows_dependency),
        },
        "category_realized_counts": benchmark_stats["category_counts"],
        "category_realized_ratios": benchmark_stats["category_ratios"],
        "feature_rect_max_ratio": float(args.feature_rect_max_ratio),
        "benchmark_feature_op_max_ratio": float(args.benchmark_feature_op_max_ratio),
        "distribution": benchmark_stats,
        "selected_tsv": str(bench_selected_tsv_path),
    }
    bench_summary_path.write_text(json.dumps(benchmark_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "output_root": str(out_root),
        "validated_pool_summary": str(pool_summary_path),
        "benchmark_summary": str(bench_summary_path),
        "validated_pool": {
            "kept": kept,
            "skipped": skipped,
            "category_counts": cat_counts,
        },
        "benchmark_1k": {
            "target_counts": target_counts,
            "realized_counts": benchmark_stats["category_counts"],
            "realized_total": len(selected_rows),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0
