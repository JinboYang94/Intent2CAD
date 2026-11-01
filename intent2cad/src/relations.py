"""Recover explicit shared and derived relations from accepted programs."""
from __future__ import annotations
import ast
import re
from collections import defaultdict
from typing import Any

LOOP_NAME_RE = re.compile(r"^loop_\d+$")
RECT_SLOTS = ("width", "height")
ELLIPSE_SLOTS = ("x_radius", "y_radius")
_SLOT_PRIORITY = {"width": 0, "height": 1, "x_radius": 0, "y_radius": 1,
                  "radius": 2, "center": 3, "points": 4, "fillet_radius": 5}

VAR_NAME_RE = re.compile(r"^var_[a-zA-Z0-9_]+$")

FLOAT_TOL = 1e-9

def _normalize_number(x: int | float) -> float:
    if isinstance(x, bool):
        raise TypeError
    v = float(x)
    # Stable bucket key for grouping equal literals
    if abs(v) < FLOAT_TOL:
        v = 0.0
    return round(v, 12)

def _literal_key(node: ast.AST) -> tuple[Any, ...] | None:
    """
    If node is a fully static literal (numbers, tuples/lists of literals, unary +/- on number),
    return a hashable key for grouping. Otherwise None.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ("num", _normalize_number(node.value))
        if isinstance(node.value, str):
            return ("str", node.value)
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal_key(node.operand)
        if inner is None or inner[0] != "num":
            return None
        v = inner[1]
        if isinstance(node.op, ast.USub):
            return ("num", _normalize_number(-v))
        return ("num", v)
    if isinstance(node, (ast.Tuple, ast.List)):
        elts = [_literal_key(e) for e in node.elts]
        if any(e is None for e in elts):
            return None
        kind = "tuple" if isinstance(node, ast.Tuple) else "list"
        return (kind, tuple(elts))
    return None

def duplicate_literal_groups_from_ast(tree: ast.Module) -> list[dict[str, Any]]:
    """
    Same duplicate-literal grouping as analyze_file, but from an already-parsed AST.
    """
    assignments: dict[str, tuple[Any, ...]] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if not VAR_NAME_RE.match(target.id):
                continue
            key = _literal_key(node.value)
            if key is None:
                continue
            assignments[target.id] = key

    key_to_names: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for name, key in assignments.items():
        key_to_names[key].append(name)

    groups: list[dict[str, Any]] = []
    for key, names in sorted(key_to_names.items(), key=lambda kv: (str(kv[0]), kv[1])):
        if len(names) <= 1:
            continue
        sorted_names = sorted(names, key=lambda s: (len(s), s))
        groups.append(
            {
                "literal_key": _key_to_display(key),
                "value": literal_key_to_json_value(key),
                "names": sorted_names,
                "count": len(sorted_names),
            }
        )
    return groups

def literal_key_to_json_value(key: tuple[Any, ...]) -> Any:
    """Convert internal literal key to JSON-serializable Python values (numbers, lists, str)."""
    if not key:
        return None
    k0 = key[0]
    if k0 == "num":
        return key[1]
    if k0 == "str":
        return key[1]
    if k0 == "tuple":
        return [literal_key_to_json_value(x) for x in key[1]]
    if k0 == "list":
        return [literal_key_to_json_value(x) for x in key[1]]
    return _key_to_display(key)

def _key_to_display(key: tuple[Any, ...]) -> str:
    if not key:
        return "()"
    if key[0] == "num":
        return repr(key[1])
    if key[0] == "str":
        return repr(key[1])
    if key[0] in ("tuple", "list"):
        return _format_nested(key)
    return repr(key)


def _format_nested(key: tuple[Any, ...]) -> str:
    def format_item(item):
        if item[0] == "num":
            return str(item[1])
        if item[0] == "str":
            return repr(item[1])
        if item[0] in {"tuple", "list"}:
            inner = ", ".join(format_item(child) for child in item[1])
            return f"({inner})" if item[0] == "tuple" else f"[{inner}]"
        return repr(item)
    return format_item(key)

def _collect_sketch_calls(value: ast.AST) -> list[tuple[str, ast.Call]]:
    """Collect (method_name, Call) for all Call nodes under value (any nesting)."""
    out: list[tuple[str, ast.Call]] = []
    for node in ast.walk(value):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        meth = node.func.attr
        out.append((meth, node))
    return out

def _var_names_in_expr(node: ast.AST) -> list[str]:
    names: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and VAR_NAME_RE.match(n.id):
            names.append(n.id)
    return names

def _first_var_name(node: ast.AST) -> str | None:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and VAR_NAME_RE.match(n.id):
            return n.id
    return None

def extract_loop_var_slots(tree: ast.Module) -> list[tuple[str, str, str]]:
    """
    Return list of (var_name, loop_id, slot) for var_* used in loop_* sketch RHS.
    """
    rows: list[tuple[str, str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or not LOOP_NAME_RE.match(t.id):
            continue
        loop_id = t.id
        for meth, call in _collect_sketch_calls(node.value):
            if meth == "rect" and len(call.args) >= 2:
                for slot, arg in zip(RECT_SLOTS, (call.args[0], call.args[1])):
                    for vn in _var_names_in_expr(arg):
                        rows.append((vn, loop_id, slot))
            elif meth == "ellipse" and len(call.args) >= 2:
                for slot, arg in zip(ELLIPSE_SLOTS, (call.args[0], call.args[1])):
                    for vn in _var_names_in_expr(arg):
                        rows.append((vn, loop_id, slot))
            elif meth == "circle" and call.args:
                for vn in _var_names_in_expr(call.args[0]):
                    rows.append((vn, loop_id, "radius"))
            elif meth == "fillet" and call.args:
                for vn in _var_names_in_expr(call.args[0]):
                    rows.append((vn, loop_id, "fillet_radius"))
            elif meth == "push" and call.args:
                a0 = call.args[0]
                if isinstance(a0, (ast.List, ast.Tuple)):
                    for elt in a0.elts:
                        vn = _first_var_name(elt)
                        if vn:
                            rows.append((vn, loop_id, "center"))
                else:
                    for vn in _var_names_in_expr(a0):
                        rows.append((vn, loop_id, "center"))
            elif meth == "polygon" and call.args:
                for vn in _var_names_in_expr(call.args[0]):
                    rows.append((vn, loop_id, "points"))
    return rows

def top_level_literal_map(tree: ast.Module) -> dict[str, tuple[Any, ...]]:
    m: dict[str, tuple[Any, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if not VAR_NAME_RE.match(target.id):
                continue
            k = _literal_key(node.value)
            if k is not None:
                m[target.id] = k
    return m

def extract_derived_assignments(tree: ast.Module) -> list[tuple[str, str, list[str]]]:
    """
    Top-level var_* = expr where expr is not a pure literal and references other var_*.
    Returns (lhs, rhs_unparsed, sorted dependency var names).
    """
    out: list[tuple[str, str, list[str]]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if not VAR_NAME_RE.match(target.id):
                continue
            lhs = target.id
            rhs = node.value
            if _literal_key(rhs) is not None:
                continue
            deps = sorted({n for n in _var_names_in_expr(rhs) if n != lhs})
            if not deps:
                continue
            try:
                rhs_s = ast.unparse(rhs)
            except AttributeError:
                rhs_s = "<unparse unavailable>"
            out.append((lhs, rhs_s, deps))
    return out

def _slot_sort_key(slot: str) -> tuple[int, int, str]:
    m = re.match(r"^loop_(\d+)\.(.+)$", slot)
    if m:
        sn = m.group(2)
        pri = _SLOT_PRIORITY.get(sn, 50)
        return (int(m.group(1)), pri, sn)
    return (10**9, 99, slot)

def build_summary_dict(tree: ast.Module, duplicate_groups: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build JSON-serializable summary. Numeric-equality groups drop vars (and their slots)
    already listed as explicit shared parameters (same var name in 2+ loop slots).
    Empty sections are [].
    """
    lit_map = top_level_literal_map(tree)
    loop_rows = extract_loop_var_slots(tree)
    derived = extract_derived_assignments(tree)

    var_to_slots: defaultdict[str, list[str]] = defaultdict(list)
    for vn, loop_id, slot in loop_rows:
        var_to_slots[vn].append(f"{loop_id}.{slot}")

    explicit_shared_vars: set[str] = {
        vn for vn, slots in var_to_slots.items() if len(set(slots)) >= 2
    }

    explicit_shared_relations: list[dict[str, Any]] = []
    for vn in sorted(var_to_slots.keys()):
        places = sorted(set(var_to_slots[vn]), key=_slot_sort_key)
        if len(places) < 2:
            continue
        lit = lit_map.get(vn)
        entry: dict[str, Any] = {"variable": vn}
        if lit is not None:
            entry["value"] = literal_key_to_json_value(lit)
        entry["reused_by"] = places
        entry["relation_scope"] = "explicit_symbolic_reuse"
        explicit_shared_relations.append(entry)

    explicit_derived_relations: list[dict[str, Any]] = []
    for lhs, rhs_s, deps in derived:
        explicit_derived_relations.append(
            {
                "target": lhs,
                "rhs": rhs_s,
                "depends_on": deps,
                "relation_scope": "explicit_derived",
            }
        )

    numeric_equality_only_groups: list[dict[str, Any]] = []
    for g in duplicate_groups:
        names: list[str] = g["names"]
        non_shared = [n for n in names if n not in explicit_shared_vars]
        if not non_shared:
            continue
        slot_list: list[str] = []
        for n in non_shared:
            slot_list.extend(var_to_slots.get(n, []))
        uniq_slots = sorted(set(slot_list), key=_slot_sort_key)
        numeric_equality_only_groups.append(
            {
                "value": g["value"],
                "variables": sorted(non_shared, key=lambda s: (len(s), s)),
                "slots": uniq_slots,
                "shared_relation_allowed": False,
            }
        )

    return {
        "explicit_shared_relations": explicit_shared_relations,
        "explicit_derived_relations": explicit_derived_relations,
        "numeric_equality_only_groups": numeric_equality_only_groups,
    }


def summarize(code: str) -> dict:
    tree = ast.parse(code)
    return build_summary_dict(tree, duplicate_literal_groups_from_ast(tree))
