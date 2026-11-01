"""Constraint-backed, local parameter rewriting with GPT-5.4."""
from __future__ import annotations
import ast
import json
import re
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)
EQUAL_TOLERANCE = 1e-5
SHARED_STRICT_TOL = 1e-5
_SUSPICIOUS_MUL_DIV_MAX_DENOM = 48
_SUSPICIOUS_MUL_DIV_RAT_TOL = 1e-5
_SUSPICIOUS_MUL_DIV_EPS_ZERO = 1e-12
_VAR_NAME_FOR_SUSPICIOUS_MUL_DIV = re.compile(r"^var_[a-zA-Z0-9_]+$")

def _append_result_log(log_path, fmt, *args, level="INFO"):
    getattr(log, level.lower())(fmt, *args)

SYSTEM_PROMPT = """\
You are a CAD parametrization expert. Analyze an already high-level CadQuery sketch script together with sketch-level geometric constraints, and rewrite only variable definitions into shared variables or simple expressions.

## Task
The code already uses high-level APIs such as `rect`, `circle`, `slot`, `regularPolygon`, and `fillet`. Do NOT change code structure, API calls, loops, or body logic. Only improve parameter definitions (`var_*`). Your goal is to recover human-intuitive parametric design intent, not to numerically fit arbitrary relations between variable values.

## Input Notes
- Code: Comments map constraint tags to loops, e.g. # 'JGV', 'JGp' -> loop_2. Use these comments to connect constraint entities to the corresponding high-level geometry.
- Constraints: A constraint entity may correspond to multiple sketch tags because one geometric entity can be represented by multiple sketch segments. Treat such tags as one grouped entity.

## Rules
- Only modify `var_*` definitions. Do not change geometry, APIs, loops, or body logic.
- Infer parameter meaning from the existing high-level API, variable names, and provided sketch constraints.
- Preserve geometry exactly (tolerance <= 1e-5) after substitutions.
- Prefer shared parameters or simple human-like expressions over repeated constants. Do NOT replace entirely with bare numbers.
- If evidence is weak or ambiguous, make no change.

## Constraint Use
- `EQUAL` is strong evidence for shared parameters.
- `CONCENTRIC`, `DISTANCE`, and `MIRROR` may justify shared centers or simple symmetric expressions.
- `LENGTH`, `RADIUS`, and `DIAMETER` are dimension evidence only; use them only for equality or clearly intended simple relations.
- Other constraints are supporting context only, not sole evidence.

## Compatibility
Relations may be inferred between geometrically compatible quantities, not only identical property names, e.g.:
- width <-> height
- radius <-> radius
- radius <-> diameter
- line length <-> width/height
- distance <-> width/height/diameter

Do NOT relate incompatible quantities such as size <-> angle or size <-> side count.

## Output
Return a JSON object with one field:
**changes** (required): Array of substitution objects. If no changes are justified, use [].

Schema:
{
  "changes": [
    {
      "variable": "<variable_name>",
      "new_value": "<expression_or_variable>",
      "relation_type": "shared" | "expression",
      "primary_constraints": [
        {
          "type": "EQUAL" | "CONCENTRIC" | "DISTANCE" | "RADIUS" | "LENGTH" | "DIAMETER" | "MIRROR",
          "entities": ["<tag1>", "<tag2>"]
        }
      ],
      "reason": "<brief explanation, must be a direct and simple sentence under 20 words. Do NOT write long explanations.>"
    }
  ]
}
"""

PARAMETRIZATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "description": "List of variable substitutions.",
            "items": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string", "description": "Variable name to replace"},
                    "new_value": {"type": "string", "description": "Expression or variable reference"},
                    "relation_type": {
                        "type": "string",
                        "enum": ["shared", "expression"],
                        "description": "Whether the substitution is a direct shared parameter or a derived expression",
                    },
                    "primary_constraints": {
                        "type": "array",
                        "description": "The most important constraints supporting this substitution",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "entities": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["type", "entities"]
                        }
                    },
                    "reason": {"type": "string", "description": "Brief explanation"},
                },
                "required": ["variable", "new_value"],
            },
        },
    },
    "required": ["changes"],
}

def parse_code(code_text: str) -> tuple[OrderedDict, dict]:
    variables: OrderedDict = OrderedDict()
    tag_loop: dict[str, str] = {}

    for line in code_text.splitlines():
        stripped = line.strip()

        m = re.match(r'^(var_\w+)\s*=\s*(.+)$', stripped)
        if m:
            try:
                variables[m.group(1)] = ast.literal_eval(m.group(2).strip())
            except Exception:
                pass

        m = re.match(r"#\s*((?:'[^']+',?\s*)+)->\s*(loop_\d+)", stripped)
        if m:
            loop_name = m.group(2)
            for tag in re.findall(r"'([^']+)'", m.group(1)):
                tag_loop[tag.split('.')[0]] = loop_name

    return variables, tag_loop

def parse_constraints(data: dict) -> list[dict]:
    return data.get("constraints", [])

USABLE_CONSTRAINT_TYPES = {"EQUAL", "CONCENTRIC", "DISTANCE", "LENGTH", "RADIUS", "DIAMETER", "MIRROR"}

def build_usable_constraints_summary(constraints: list[dict], tag_loop: dict[str, str]) -> list[dict]:
    results = []
    for c in constraints:
        ctype = c.get("type")
        if ctype not in USABLE_CONSTRAINT_TYPES:
            continue
        entities = c.get("entities", [])

        def loops_of(entity: dict) -> set[str]:
            return {
                tag_loop[tag.split('.')[0]]
                for tag in entity.get("json_ids", [])
                if tag.split('.')[0] in tag_loop
            }

        entry = {
            "type": ctype,
            "entities": [e.get("json_ids", []) for e in entities],
            "loops": [sorted(loops_of(e)) for e in entities],
        }
        if "value_normalized" in c:
            entry["value"] = c["value_normalized"]
        elif "value" in c and ctype in {"DISTANCE", "LENGTH", "RADIUS", "DIAMETER"}:
            entry["value"] = c.get("value")
        results.append(entry)
    return results

def build_constraints_summary(constraints_json: dict) -> list[dict]:
    summary = []
    for c in constraints_json.get("constraints", []):
        entry: dict = {"type": c["type"]}
        entry["entities"] = [e.get("json_ids", []) for e in c.get("entities", [])]
        if "value_normalized" in c:
            entry["value"] = c["value_normalized"]
        summary.append(entry)
    return summary

def build_user_message(code_text: str, constraints_json: dict, usable_constraints: list[dict]) -> str:
    cs = build_constraints_summary(constraints_json)
    return (
        f"## CadQuery Code\n\n```python\n{code_text.strip()}\n```\n\n"
        f"## Constraints (normalized values)\n\n```json\n{json.dumps(cs, indent=2)}\n```\n\n"
        f"## Primary Usable Constraints\n\n"
        f"```json\n{json.dumps(usable_constraints, indent=2)}\n```\n\n"
        f"Identify all valid variable substitutions based on EQUAL, CONCENTRIC, DISTANCE, LENGTH, RADIUS, DIAMETER, or MIRROR. Do NOT infer directly from PARALLEL, PERPENDICULAR, TANGENT, or other non-substitution constraints. Output only the JSON object."
    )

_EVAL_BUILTINS_FOR_SUBST: dict[str, Any] = {
    "abs": abs,
    "enumerate": enumerate,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _pattern_bind_names(target: ast.AST) -> set[str]:
    """Names bound by a comprehension / for-loop target pattern."""
    found: set[str] = set()

    def rec(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            found.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                rec(e)
        elif isinstance(t, ast.Starred):
            rec(t.value)

    rec(target)
    return found


def _scoped_comp(
    generators: list[ast.comprehension],
    elt: ast.AST,
    outer_bound: frozenset[str],
) -> set[str]:
    """Free names in a list/set/generator comprehension (respects nested gens)."""
    if not generators:
        return _expr_free_var_names(elt, outer_bound)
    gen = generators[0]
    rest = generators[1:]
    local = _pattern_bind_names(gen.target)
    inner_bound = frozenset(outer_bound | local)
    out = _expr_free_var_names(gen.iter, outer_bound)
    for if_ in gen.ifs:
        out |= _expr_free_var_names(if_, inner_bound)
    if rest:
        out |= _scoped_comp(rest, elt, inner_bound)
    else:
        out |= _expr_free_var_names(elt, inner_bound)
    return out


def _scoped_comp_dict(
    generators: list[ast.comprehension],
    key: ast.AST,
    value: ast.AST,
    outer_bound: frozenset[str],
) -> set[str]:
    if not generators:
        return _expr_free_var_names(key, outer_bound) | _expr_free_var_names(value, outer_bound)
    gen = generators[0]
    rest = generators[1:]
    local = _pattern_bind_names(gen.target)
    inner_bound = frozenset(outer_bound | local)
    out = _expr_free_var_names(gen.iter, outer_bound)
    for if_ in gen.ifs:
        out |= _expr_free_var_names(if_, inner_bound)
    if rest:
        out |= _scoped_comp_dict(rest, key, value, inner_bound)
    else:
        out |= _expr_free_var_names(key, inner_bound) | _expr_free_var_names(value, inner_bound)
    return out


def _expr_free_var_names(node: ast.AST | None, bound: frozenset[str]) -> set[str]:
    """Names loaded from outer scope (must exist in variables dict)."""
    if node is None:
        return set()
    if isinstance(node, ast.Constant):
        return set()
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id not in bound:
            return {node.id}
        return set()
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return _scoped_comp(node.generators, node.elt, bound)
    if isinstance(node, ast.DictComp):
        return _scoped_comp_dict(node.generators, node.key, node.value, bound)
    if isinstance(node, ast.Lambda):
        lam_bound: set[str] = set(bound)
        for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            lam_bound.add(a.arg)
        out: set[str] = set()
        for d in node.args.defaults:
            out |= _expr_free_var_names(d, bound)
        for d in node.args.kw_defaults:
            if d is not None:
                out |= _expr_free_var_names(d, bound)
        out |= _expr_free_var_names(node.body, frozenset(lam_bound))
        return out
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return set().union(*(_expr_free_var_names(e, bound) for e in node.elts))
    if isinstance(node, ast.Dict):
        s = set()
        for k, v in zip(node.keys, node.values):
            if k is not None:
                s |= _expr_free_var_names(k, bound)
            s |= _expr_free_var_names(v, bound)
        return s
    if isinstance(node, ast.UnaryOp):
        return _expr_free_var_names(node.operand, bound)
    if isinstance(node, ast.BinOp):
        return _expr_free_var_names(node.left, bound) | _expr_free_var_names(node.right, bound)
    if isinstance(node, ast.BoolOp):
        return set().union(*(_expr_free_var_names(v, bound) for v in node.values))
    if isinstance(node, ast.IfExp):
        return (
            _expr_free_var_names(node.test, bound)
            | _expr_free_var_names(node.body, bound)
            | _expr_free_var_names(node.orelse, bound)
        )
    if isinstance(node, ast.Compare):
        s = _expr_free_var_names(node.left, bound)
        for comp in node.comparators:
            s |= _expr_free_var_names(comp, bound)
        return s
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _EVAL_BUILTINS_FOR_SUBST:
            s = set()
        else:
            s = _expr_free_var_names(node.func, bound)
        for a in node.args:
            s |= _expr_free_var_names(a, bound)
        for kw in node.keywords:
            s |= _expr_free_var_names(kw.value, bound)
        return s
    if isinstance(node, ast.Subscript):
        s = _expr_free_var_names(node.value, bound)
        sl = node.slice
        if isinstance(sl, ast.Slice):
            s |= _expr_free_var_names(sl.lower, bound)
            s |= _expr_free_var_names(sl.upper, bound)
            s |= _expr_free_var_names(sl.step, bound)
        else:
            s |= _expr_free_var_names(sl, bound)
        return s
    if isinstance(node, ast.Slice):
        return (
            _expr_free_var_names(node.lower, bound)
            | _expr_free_var_names(node.upper, bound)
            | _expr_free_var_names(node.step, bound)
        )
    if isinstance(node, ast.Attribute):
        return _expr_free_var_names(node.value, bound)
    if isinstance(node, ast.Starred):
        return _expr_free_var_names(node.value, bound)
    if isinstance(node, ast.FormattedValue):
        return _expr_free_var_names(node.value, bound)
    if isinstance(node, ast.JoinedStr):
        return set().union(*(_expr_free_var_names(v, bound) for v in node.values))
    return set().union(
        *(
            _expr_free_var_names(child, bound)
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, (ast.Load, ast.Store, ast.Del))
        )
    )


def _expr_var_names(expr_str: str) -> set[str]:
    """Script variable names (var_*) that an expression reads from outer scope."""
    try:
        tree = ast.parse(expr_str.strip(), mode="eval")
        return _expr_free_var_names(tree.body, frozenset())
    except SyntaxError:
        return set()


def _eval_expr(expr_str: str, variables: dict):
    """Safely evaluate expression using only variables. Returns value or None if invalid."""
    names = _expr_var_names(expr_str)
    if not names or not names.issubset(variables.keys()):
        return None
    ctx = {k: v for k, v in variables.items() if k in names}
    try:
        return eval(expr_str.strip(), {"__builtins__": _EVAL_BUILTINS_FOR_SUBST}, ctx)
    except Exception:
        return None


def _muldiv_numeric_value(node: ast.AST) -> float | None:
    """Single int/float literal node, including unary +/-."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    ):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _muldiv_numeric_value(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _muldiv_is_simple_rational(x: float) -> bool:
    """True if x is ~integer, ~zero, or ~p/q with 1 <= q <= _SUSPICIOUS_MUL_DIV_MAX_DENOM."""
    x = float(x)
    ax = abs(x)
    if ax < _SUSPICIOUS_MUL_DIV_EPS_ZERO:
        return True
    if abs(x - round(x)) < _SUSPICIOUS_MUL_DIV_RAT_TOL:
        return True
    for q in range(1, _SUSPICIOUS_MUL_DIV_MAX_DENOM + 1):
        p = round(x * q)
        if abs(x - p / q) < _SUSPICIOUS_MUL_DIV_RAT_TOL:
            return True
    return False


def _muldiv_subtree_references_var(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and _VAR_NAME_FOR_SUSPICIOUS_MUL_DIV.match(n.id):
            return True
    return False


def _expr_tree_has_suspicious_mul_div_literal(node: ast.AST) -> bool:
    """
    True if any Div/Mul has a numeric literal on one side and a var_*-referencing subtree on the other,
    and that literal is not a simple rational (same detection as check_suspicious_var_mul_div.py).
    """
    for n in ast.walk(node):
        if not isinstance(n, ast.BinOp) or not isinstance(n.op, (ast.Div, ast.Mult)):
            continue
        left_num = _muldiv_numeric_value(n.left)
        right_num = _muldiv_numeric_value(n.right)
        for side_val, other in ((left_num, n.right), (right_num, n.left)):
            if side_val is None:
                continue
            fv = float(side_val)
            if not _muldiv_subtree_references_var(other):
                continue
            if not _muldiv_is_simple_rational(fv):
                return True
    return False


def new_value_has_suspicious_mul_div_literal(new_val: str) -> bool:
    """Reject substitutions like var_2 = var_1 / 2.5846 with non-simple decimal factors."""
    try:
        tree = ast.parse(new_val.strip(), mode="eval")
    except SyntaxError:
        return False
    return _expr_tree_has_suspicious_mul_div_literal(tree.body)


def values_approx_equal(a, b, tol: float = EQUAL_TOLERANCE) -> bool:
    """Compare values with tolerance. Handles numbers, tuples, and nested structures (e.g. center coords)."""
    if isinstance(a, list):
        a = tuple(a)
    if isinstance(b, list):
        b = tuple(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return False
        return all(
            values_approx_equal(x, y, tol) for x, y in zip(a, b)
        )
    return a == b


def _is_sequence_of_coordinate_tuples(v) -> bool:
    """True if v is a non-empty sequence of fixed-length numeric tuples (2D+ points)."""
    if not isinstance(v, (list, tuple)) or not v:
        return False
    n_dim: int | None = None
    for item in v:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return False
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in item):
            return False
        if n_dim is None:
            n_dim = len(item)
        elif len(item) != n_dim:
            return False
    return True


def _norm_coord_tuple(el) -> tuple:
    return tuple(el) if isinstance(el, list) else el


def _multiset_approx_equal_sequence(a, b, tol: float) -> bool:
    """Order-independent equality: each element of a matches a distinct element of b within tol."""
    if len(a) != len(b):
        return False
    b_list = list(b)
    used = [False] * len(b_list)
    for x in a:
        xn = _norm_coord_tuple(x)
        matched = False
        for j, y in enumerate(b_list):
            if used[j]:
                continue
            if values_approx_equal(xn, _norm_coord_tuple(y), tol):
                used[j] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _values_match_for_substitution(
    current, evaluated, var_name: str, tol: float = EQUAL_TOLERANCE
) -> bool:
    """Like values_approx_equal, but *_points lists match as multisets (ignore vertex order)."""
    if var_name.endswith("_points"):
        ca, cb = current, evaluated
        if isinstance(ca, list):
            ca = tuple(ca)
        if isinstance(cb, list):
            cb = tuple(cb)
        if (
            isinstance(ca, tuple)
            and isinstance(cb, tuple)
            and _is_sequence_of_coordinate_tuples(ca)
            and _is_sequence_of_coordinate_tuples(cb)
        ):
            return _multiset_approx_equal_sequence(ca, cb, tol)
    return values_approx_equal(current, evaluated, tol)


def _ast_expr_has_var_reference(node: ast.AST) -> bool:
    """True if AST subtree references a parametrization name var_*."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id.startswith("var_"):
            return True
    return False


def _ast_expr_numeric_literal_only(node: ast.AST) -> bool:
    """True if expression is built only from int/float constants and + - * / (no var_*)."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _ast_expr_numeric_literal_only(node.operand)
    if isinstance(node, ast.BinOp):
        return _ast_expr_numeric_literal_only(node.left) and _ast_expr_numeric_literal_only(
            node.right
        )
    return False


def _center_tuple_mixes_literal_and_var(new_val: str) -> bool:
    """
    Reject e.g. (-0.124999, -var_1_center[1]): one axis is a bare number, the other uses var_*.
    Keeps centers either all-literal tuples, all-var expressions, or a single var_* reference.
    """
    try:
        tree = ast.parse(new_val.strip(), mode="eval")
    except SyntaxError:
        return False
    expr = tree.body
    if not isinstance(expr, ast.Tuple):
        return False
    has_literal_component = False
    has_var_component = False
    for elt in expr.elts:
        if _ast_expr_has_var_reference(elt):
            has_var_component = True
        elif _ast_expr_numeric_literal_only(elt):
            has_literal_component = True
        else:
            # Non-numeric, no var_* (unlikely for sketch centers) — do not treat as "safe literal"
            pass
    return has_literal_component and has_var_component


def _ast_subtree_has_center_var(node: ast.AST) -> bool:
    """True if subtree contains a Name ending with _center (covers var_1_center[0] via walk)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id.endswith("_center"):
            return True
    return False


def _reject_center_new_value(new_val: str, variables: dict) -> tuple[bool, str]:
    """
    Reject rhs for *_center targets when:
    - new_value is a tuple: each element's AST subtree must reference at least one *_center
      (so (0.0, var_2_radius) and (var_1_center[0], var_2_radius) are rejected);
    - otherwise: if there is at least one var_* ref and all are non-*_center, reject.
    """
    try:
        tree = ast.parse(new_val.strip(), mode="eval")
    except SyntaxError:
        return False, ""
    expr = tree.body
    if isinstance(expr, ast.Tuple):
        if not expr.elts:
            return True, "empty coordinate tuple"
        for i, elt in enumerate(expr.elts):
            if not _ast_subtree_has_center_var(elt):
                return True, f"axis {i} has no *_center variable"
        return False, ""
    if new_val in variables:
        refs = {new_val}
    else:
        refs = _expr_var_names(new_val)
    if refs and all(not r.endswith("_center") for r in refs):
        return True, "only non-center var_* refs"
    return False, ""


def _parse_var_name(name: str) -> tuple[Optional[str], Optional[str]]:
    """Split `var_2_width` -> (`var_2`, `width`)."""
    m = re.match(r"^(var_\d+)_(.+)$", name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _infer_shared_param_base(component_vars: list[str]) -> Optional[str]:
    """
    Return var_shared_param for size-like shared params (width, height, radius, etc.).
    Return None for non-size params (e.g. center) so we collapse to the first variable.
    """
    if not component_vars:
        return None
    size_props = {"width", "height", "radius", "diameter", "length"}
    for name in component_vars:
        _, prop = _parse_var_name(name)
        if prop and prop in size_props:
            return "var_shared_param"
    return None  # center, etc. — collapse to first var


_NO_INLINE_EXPR_PROPS = frozenset({
    "center", "points",
    "width", "height", "radius", "diameter", "length",
    "fillet_radius",
})


def _should_inline_expr(var_name: str) -> bool:
    """Whether an expression substitution should be inlined at use sites (definition removed).

    Returns False for center, points, size properties, and array/coordinate-list
    variables — these are kept as named definitions for human readability.
    """
    if var_name.startswith("array_"):
        return False
    _, prop = _parse_var_name(var_name)
    if prop is None:
        return False
    return prop not in _NO_INLINE_EXPR_PROPS


def _make_unique_name(base_name: str, existing_names: set[str]) -> str:
    """Create a non-conflicting variable name."""
    if base_name not in existing_names:
        return base_name
    idx = 2
    while f"{base_name}_{idx}" in existing_names:
        idx += 1
    return f"{base_name}_{idx}"


def _connected_components(edges: set[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    seen: set[str] = set()
    components: list[list[str]] = []
    for start in graph:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in graph.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def _wrap_expr_if_compound(repl: str) -> str:
    """Wrap replacement in parentheses if it's a compound expression (avoids operator precedence bugs)."""
    if re.match(r"^\w+$", repl.strip()):
        return repl
    return f"({repl})"


def _replace_var_refs(text: str, replacements: dict[str, str], wrap_compound: bool = False) -> str:
    """Replace whole variable-name references in arbitrary code text."""
    if not replacements:
        return text
    pattern = r"\b(" + "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)) + r")\b"

    def replacer(m: re.Match) -> str:
        r = replacements[m.group(1)]
        return _wrap_expr_if_compound(r) if wrap_compound else r

    return re.sub(pattern, replacer, text)


def _format_value_for_inline(val) -> str:
    """Format a Python value for inline use (max 6 decimals)."""
    if isinstance(val, (int, float)):
        r = round(float(val), 6)
        if r == int(r):
            return str(int(r))
        return f"{r:.6f}".rstrip("0").rstrip(".")
    if isinstance(val, tuple):
        return "(" + ", ".join(_format_value_for_inline(x) for x in val) + ")"
    return str(val)


def _format_numeric_literals_6(code: str) -> str:
    """Format float literals to max 6 decimal places, strip trailing zeros."""
    def _repl(m: re.Match) -> str:
        s = m.group(0)
        try:
            val = float(s)
            r = round(val, 6)
            if r == int(r):
                return str(int(r))
            formatted = f"{r:.6f}".rstrip("0").rstrip(".")
            return formatted
        except ValueError:
            return s
    return re.sub(r"-?\d+\.\d+", _repl, code)


def _remove_unused_vars(code: str) -> str:
    """Remove variable definition lines for vars that are never used elsewhere."""
    lines = code.splitlines(keepends=True)
    var_defs: dict[str, tuple[int, str]] = {}
    for idx, line in enumerate(lines):
        m = re.match(r"^(\s*)(var_\w+)(\s*=\s*)(.+)$", line)
        if m:
            var_defs[m.group(2)] = (idx, line)

    used: set[str] = set()
    for var_name, (def_idx, def_line) in var_defs.items():
        for i, line in enumerate(lines):
            if i == def_idx:
                continue
            if re.search(r"\b" + re.escape(var_name) + r"\b", line):
                used.add(var_name)
                break

    indices_to_remove = {var_defs[u][0] for u in var_defs if u not in used}
    new_lines = [line for i, line in enumerate(lines) if i not in indices_to_remove]
    return "".join(new_lines)


def _reorder_for_expr_forward_refs(code_text: str, changes: list[dict], variables: OrderedDict) -> tuple[str, OrderedDict]:
    """
    Reorder var definition lines so that variables referenced in expressions appear before
    the vars that reference them. Fixes expr_forward_ref by moving late-defined refs earlier.
    Returns (reordered_code, reordered_variables) or (code_text, variables) if no reorder needed.
    """
    lines = code_text.splitlines(keepends=True)
    var_defs: dict[str, dict] = {}
    for idx, line in enumerate(lines):
        m = re.match(r"^(\s*)(var_\w+)(\s*=\s*)(.+)$", line)
        if m:
            var_defs[m.group(2)] = {"line_idx": idx, "line": line}

    var_order = {name: i for i, name in enumerate(variables)}
    refs_to_move: set[str] = set()
    earliest_line = len(lines)

    for change in changes:
        var = change.get("variable", "")
        new_val = change.get("new_value", "")
        if var not in variables or not isinstance(new_val, str) or not new_val.strip():
            continue
        if new_val in variables:
            continue  # direct ref, not expression
        refs = _expr_var_names(new_val)
        if not refs or not refs.issubset(variables.keys()):
            continue
        late_refs = [r for r in refs if var_order.get(r, 9999) >= var_order.get(var, 9999)]
        if not late_refs:
            continue
        refs_to_move.update(late_refs)
        if var in var_defs:
            earliest_line = min(earliest_line, var_defs[var]["line_idx"])

    if not refs_to_move or earliest_line >= len(lines):
        return code_text, variables

    # Sort refs by original line order; others by original line order
    refs_sorted = sorted(refs_to_move, key=lambda v: var_defs.get(v, {}).get("line_idx", 9999))
    others = [v for v in variables if v not in refs_to_move]
    others_sorted = sorted(others, key=lambda v: var_defs.get(v, {}).get("line_idx", 9999))
    new_order = refs_sorted + others_sorted

    var_line_indices = sorted({d["line_idx"] for d in var_defs.values()})
    first_var_line = min(var_line_indices)
    last_var_line = max(var_line_indices)

    before = lines[:first_var_line]
    after = lines[last_var_line + 1:]
    block_lines = [var_defs[v]["line"] for v in new_order if v in var_defs]
    new_code = "".join(before + block_lines + after)

    new_variables, _ = parse_code(new_code)
    return new_code, new_variables


def apply_changes(
    code_text: str,
    changes: list[dict],
    variables: OrderedDict,
    log_path: Optional[Path] = None,
    file_name: Optional[str] = None,
) -> tuple[str, int, dict]:
    """
    Apply variable substitutions to code. Returns (modified_code, n_applied, skip_reasons).
    When log_path and file_name are provided, skip reasons are written via _append_result_log
    (log.warning is ineffective in Ray/vLLM workers).
    """

    def _append_skip(fmt: str, *args) -> None:
        if log_path and file_name:
            _append_result_log(log_path, "%-25s  WARNING: " + fmt, file_name, *args, level="WARNING")

    skip_reasons = {
        "unknown_var": 0,
        "invalid_new_value": 0,
        "direct_forward_ref": 0,
        "direct_value_mismatch": 0,
        "expr_unknown_vars": 0,
        "expr_forward_ref": 0,
        "expr_eval_failed": 0,
        "expr_value_mismatch": 0,
        "expr_inline_precision_revert": 0,
        "center_mixed_literal_var": 0,
        "center_only_non_center_refs": 0,
        "suspicious_mul_div_literal": 0,
    }
    if not changes:
        return _remove_unused_vars(code_text), 0, skip_reasons

    # Reorder var definitions to fix expr_forward_ref before applying
    code_text, variables = _reorder_for_expr_forward_refs(code_text, changes, variables)

    lines = code_text.splitlines(keepends=True)
    var_defs: dict[str, dict] = {}
    for idx, line in enumerate(lines):
        m = re.match(r"^(\s*)(var_\w+)(\s*=\s*)(.+)$", line)
        if not m:
            continue
        var_defs[m.group(2)] = {
            "line_idx": idx,
            "indent": m.group(1),
            "rhs": m.group(4).rstrip("\n"),
        }

    var_order = {name: i for i, name in enumerate(variables)}
    substitutions: dict[str, str] = {}
    shared_edges: set[tuple[str, str]] = set()
    accepted_changes = 0

    for change in sorted(changes, key=lambda c: var_order.get(c.get("variable", ""), 9999)):
        var     = change.get("variable", "")
        new_val = change.get("new_value", "")

        if var not in variables:
            skip_reasons["unknown_var"] += 1
            _append_skip("Unknown variable '%s' — skipping.", var)
            continue
        if not isinstance(new_val, str) or not new_val.strip():
            skip_reasons["invalid_new_value"] += 1
            _append_skip("new_value '%s' for '%s' is invalid — skipping.", new_val, var)
            continue
        if var == new_val.strip():
            continue  # self-reference (e.g. var_1_length = var_1_length), no-op, skip

        if var.endswith("_center"):
            reject, reject_why = _reject_center_new_value(new_val, variables)
            if reject:
                skip_reasons["center_only_non_center_refs"] += 1
                _append_skip(
                    "Center '%s': rhs invalid (%s) for '%s' — skipping.",
                    var,
                    reject_why,
                    new_val,
                )
                continue

        # Direct variable reference (e.g. var_1_height)
        if new_val in variables:
            if var_order.get(new_val, 9999) >= var_order.get(var, 9999):
                # Forward ref: new_val defined after var. If values equal, apply reverse (var_1_width=var_1_height instead of var_1_height=var_1_width)
                if values_approx_equal(variables[var], variables[new_val], tol=SHARED_STRICT_TOL):
                    substitutions[new_val] = var
                    shared_edges.add(tuple(sorted((var, new_val))))
                    accepted_changes += 1
                    continue
                skip_reasons["direct_forward_ref"] += 1
                _append_skip("'%s' defined after '%s'; forward reference — skipping.", new_val, var)
                continue
            if not values_approx_equal(variables[var], variables[new_val], tol=SHARED_STRICT_TOL):
                skip_reasons["direct_value_mismatch"] += 1
                _append_skip(
                    "Values of '%s'=%s and '%s'=%s differ beyond tolerance — skipping.",
                    var, variables[var], new_val, variables[new_val],
                )
                continue
            shared_edges.add(tuple(sorted((var, new_val))))
        else:
            # Expression (e.g. var_1_height / 12, 2 * var_1_radius)
            if new_value_has_suspicious_mul_div_literal(new_val):
                skip_reasons["suspicious_mul_div_literal"] += 1
                _append_skip(
                    "new_value '%s' for '%s' uses * or / with a non-simple rational literal — skipping.",
                    new_val,
                    var,
                )
                continue
            refs = _expr_var_names(new_val)
            if not refs or not refs.issubset(variables.keys()):
                skip_reasons["expr_unknown_vars"] += 1
                _append_skip("new_value '%s' for '%s' references unknown variables — skipping.", new_val, var)
                continue
            if any(var_order.get(r, 9999) >= var_order.get(var, 9999) for r in refs):
                skip_reasons["expr_forward_ref"] += 1
                _append_skip("Expression '%s' has forward reference — skipping.", new_val)
                continue
            eval_val = _eval_expr(new_val, variables)
            if eval_val is None:
                skip_reasons["expr_eval_failed"] += 1
                log.warning("Cannot evaluate expression '%s' for '%s' — skipping.", new_val, var)
                continue
            if not _values_match_for_substitution(variables[var], eval_val, var):
                skip_reasons["expr_value_mismatch"] += 1
                log.warning(
                    "Expression '%s' evaluates to %s, but '%s'=%s — skipping.",
                    new_val, eval_val, var, variables[var],
                )
                continue

        if var.endswith("_center") and _center_tuple_mixes_literal_and_var(new_val):
            skip_reasons["center_mixed_literal_var"] += 1
            _append_skip(
                "Center '%s': tuple must not mix numeric literals with var_* in '%s' — skipping.",
                var,
                new_val,
            )
            continue

        substitutions[var] = new_val
        accepted_changes += 1

    if not substitutions:
        return _remove_unused_vars(code_text), 0, skip_reasons

    # Canonicalize shared groups. For strongly human-readable cases (such as
    # repeated square sizes), introduce a new shared parameter name. Otherwise,
    # collapse the group to an existing canonical variable and remove the
    # redundant aliases from both definitions and downstream usage sites.
    insertions: dict[int, list[str]] = {}
    existing_names = set(variables.keys()) | set(substitutions.keys())
    eliminated_defs: set[str] = set()
    global_ref_replacements: dict[str, str] = {}
    for component in _connected_components(shared_edges):
        defined_component = [name for name in component if name in var_defs]
        if len(defined_component) < 2:
            continue

        defined_component.sort(key=lambda name: var_defs[name]["line_idx"])
        first_name = defined_component[0]
        first_def = var_defs[first_name]
        base_name = _infer_shared_param_base(component)

        if base_name is not None:
            canonical_name = _make_unique_name(base_name, existing_names)
            existing_names.add(canonical_name)
            insertions.setdefault(first_def["line_idx"], []).append(
                f"{first_def['indent']}{canonical_name} = {first_def['rhs']}\n"
            )
            names_to_eliminate = component
        else:
            canonical_name = first_name
            names_to_eliminate = [name for name in component if name != canonical_name]

        for name in names_to_eliminate:
            substitutions[name] = canonical_name
            global_ref_replacements[name] = canonical_name
            eliminated_defs.add(name)

    # Expression substitutions: decide which expression vars to inline (definition
    # removed, expression expanded at every use site) vs keep as named definitions
    # (RHS rewritten but name preserved).  Named properties — center, points, width,
    # height, radius, diameter, length, fillet_radius, array_* — are NEVER inlined
    # so the output stays human-readable per prompt requirements.
    expr_vars = [
        v for v in substitutions
        if v in var_defs and v not in eliminated_defs
        and substitutions[v] not in variables  # expression, not direct var ref
        and _should_inline_expr(v)
    ]
    expr_orig: dict[str, str] = {v: substitutions[v] for v in expr_vars}

    # Inline only the anonymous/exotic expression vars selected above.
    # If inline expr would disagree with stored value beyond tolerance,
    # keep the original variable definition and usages — do not inline.
    for v in expr_vars:
        expr_str = substitutions[v]
        eval_val = _eval_expr(expr_orig.get(v, expr_str), variables)
        if eval_val is not None and not _values_match_for_substitution(
            variables[v], eval_val, v, tol=EQUAL_TOLERANCE
        ):
            del substitutions[v]
            accepted_changes -= 1
            skip_reasons["expr_inline_precision_revert"] += 1
            continue
        eliminated_defs.add(v)
        global_ref_replacements[v] = expr_str

    # Expand global_ref_replacements so no value contains refs to other eliminated vars
    for _ in range(len(global_ref_replacements) + 1):
        changed = False
        for k in list(global_ref_replacements.keys()):
            v = global_ref_replacements[k]
            new_v = _replace_var_refs(v, global_ref_replacements, wrap_compound=True)
            if new_v != v:
                global_ref_replacements[k] = new_v
                changed = True
        if not changed:
            break

    result_lines = []
    for idx, line in enumerate(lines):
        for inserted in insertions.get(idx, []):
            result_lines.append(inserted)
        m = re.match(r'^(\s*)(var_\w+)(\s*=\s*)(.+)$', line)
        if m and m.group(2) in eliminated_defs:
            continue
        if m and m.group(2) in substitutions:
            new_rhs = _replace_var_refs(substitutions[m.group(2)], global_ref_replacements, wrap_compound=True)
            result_lines.append(f"{m.group(1)}{m.group(2)}{m.group(3)}{new_rhs}\n")
        else:
            result_lines.append(_replace_var_refs(line, global_ref_replacements, wrap_compound=True))

    result = "".join(result_lines)
    result = _format_numeric_literals_6(result)
    result = _remove_unused_vars(result)
    return result, accepted_changes, skip_reasons




def lift_relations(code: str, constraints: dict, llm, working_root, baseline=None, source_code=None):
    """Propose local rewrites and accept only after all deterministic checks."""
    from .validation import accept_rewrite
    from .grounding import grounded_changes
    variables, tags = parse_code(code)
    usable = build_usable_constraints_summary(parse_constraints(constraints), tags)
    if not variables or not usable:
        return code, {'accepted': False, 'reason': 'no_recoverable_relations', 'changes': []}
    try:
        response = llm.generate(SYSTEM_PROMPT, build_user_message(code, constraints, usable), PARAMETRIZATION_JSON_SCHEMA)
        proposal = json.loads(response)
        changes = proposal['changes']
        if not isinstance(changes, list):
            raise ValueError('changes must be a list.')
        valid, grounding_rejections = grounded_changes(code, changes, usable, variables, _expr_var_names, source_code)
        candidate, applied, skipped = apply_changes(code, valid, variables)
        if not applied:
            return code, {'accepted': False, 'reason': 'no_valid_substitution', 'proposal': proposal,
                          'skip_reasons': skipped, 'grounding_rejections': grounding_rejections}
        check = accept_rewrite(code, candidate, working_root, baseline)
        check.update({'proposal': proposal, 'applied': applied, 'skip_reasons': skipped,
                      'grounding_rejections': grounding_rejections})
        return (candidate if check['accepted'] else code), check
    except Exception as error:
        log.warning('Relation proposal rejected: %s', type(error).__name__)
        return code, {'accepted': False, 'reason': 'proposal_error:' + type(error).__name__}
