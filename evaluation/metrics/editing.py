"""Original loop-based edit success and numerical comparison definitions."""
from __future__ import annotations
import ast
import math
import re
from typing import Any

def normalize_code_for_exact_match(code: str) -> str:
    return code.strip()


def almost_equal(a: Any, b: Any, tol: float = 1e-5) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(almost_equal(x, y, tol=tol) for x, y in zip(a, b))
    return a == b


def canonical_value(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, tuple):
        return tuple(canonical_value(x) for x in v)
    if isinstance(v, list):
        return tuple(canonical_value(x) for x in v)
    return v


def eval_simple_expr(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"Unknown name: {node.id}")
        return env[node.id]
    if isinstance(node, ast.Subscript):
        value = eval_simple_expr(node.value, env)
        idx_node = node.slice
        if isinstance(idx_node, ast.Constant):
            idx = idx_node.value
        elif hasattr(ast, "Index") and isinstance(idx_node, ast.Index):
            idx = eval_simple_expr(idx_node.value, env)
        else:
            idx = eval_simple_expr(idx_node, env)
        return value[idx]
    if isinstance(node, ast.Tuple):
        return tuple(eval_simple_expr(e, env) for e in node.elts)
    if isinstance(node, ast.List):
        return [eval_simple_expr(e, env) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = eval_simple_expr(node.operand, env)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = eval_simple_expr(node.left, env)
        right = eval_simple_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"abs", "round"}:
        args = [eval_simple_expr(arg, env) for arg in node.args]
        if node.func.id == "abs":
            return abs(*args)
        return round(*args)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def eval_var_env(tree: ast.Module) -> dict[str, Any]:
    pending: list[tuple[str, ast.AST]] = []
    env: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and tgt.id.startswith("var_"):
            pending.append((tgt.id, node.value))

    changed = True
    while pending and changed:
        changed = False
        rest: list[tuple[str, ast.AST]] = []
        for name, value_node in pending:
            try:
                env[name] = eval_simple_expr(value_node, env)
                changed = True
            except Exception:
                rest.append((name, value_node))
        pending = rest
    return env


def call_attr_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
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


def find_kw(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def loop_assignments(tree: ast.Module) -> dict[str, ast.Assign]:
    out: dict[str, ast.Assign] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and re.fullmatch(r"loop_\d+", tgt.id):
            out[tgt.id] = node
    return out


def parse_high_level_loop(assign: ast.Assign, env: dict[str, Any]) -> dict[str, Any] | None:
    chain = find_call_chain(assign.value)
    shape_call: ast.Call | None = None
    fillet_call: ast.Call | None = None
    push_call: ast.Call | None = None
    for call in chain:
        attr = call_attr_name(call)
        if attr == "push":
            push_call = call
        elif attr == "fillet":
            fillet_call = call
        elif attr in {"circle", "rect", "regularPolygon", "polygon", "slot"} and shape_call is None:
            shape_call = call

    if shape_call is None:
        return None

    kind = call_attr_name(shape_call)
    if kind == "rect" and fillet_call is not None:
        kind = "rect_fillet"
    if kind not in {"circle", "rect", "regularPolygon", "polygon", "slot", "rect_fillet"}:
        return None

    params: dict[str, Any] = {"center": (0.0, 0.0), "angle": 0.0}
    if push_call is not None and push_call.args:
        pushed = eval_simple_expr(push_call.args[0], env)
        if isinstance(pushed, list) and len(pushed) == 1:
            params["center"] = tuple(pushed[0])

    if kind == "circle":
        params["radius"] = eval_simple_expr(shape_call.args[0], env)
    elif kind == "rect":
        params["width"] = eval_simple_expr(shape_call.args[0], env)
        params["height"] = eval_simple_expr(shape_call.args[1], env)
    elif kind == "rect_fillet":
        params["width"] = eval_simple_expr(shape_call.args[0], env)
        params["height"] = eval_simple_expr(shape_call.args[1], env)
        if fillet_call is None or not fillet_call.args:
            return None
        params["fillet_radius"] = eval_simple_expr(fillet_call.args[0], env)
    elif kind == "regularPolygon":
        params["radius"] = eval_simple_expr(shape_call.args[0], env)
        params["sides"] = int(eval_simple_expr(shape_call.args[1], env))
    elif kind == "polygon":
        params["points"] = eval_simple_expr(shape_call.args[0], env)
    elif kind == "slot":
        params["length"] = eval_simple_expr(shape_call.args[0], env)
        params["diameter"] = eval_simple_expr(shape_call.args[1], env)

    angle_node = find_kw(shape_call, "angle")
    if angle_node is not None:
        params["angle"] = eval_simple_expr(angle_node, env)

    return {"kind": kind, "params": {k: canonical_value(v) for k, v in sorted(params.items())}}


def parse_primitive_loop(assign: ast.Assign, env: dict[str, Any]) -> tuple[tuple[Any, ...], ...] | None:
    edges: list[tuple[Any, ...]] = []
    for call in reversed(find_call_chain(assign.value)):
        attr = call_attr_name(call)
        if attr == "segment" and len(call.args) >= 2:
            edges.append((
                "segment",
                canonical_value(eval_simple_expr(call.args[0], env)),
                canonical_value(eval_simple_expr(call.args[1], env)),
            ))
        elif attr == "arc" and len(call.args) >= 3:
            edges.append((
                "arc",
                canonical_value(eval_simple_expr(call.args[0], env)),
                canonical_value(eval_simple_expr(call.args[1], env)),
                canonical_value(eval_simple_expr(call.args[2], env)),
            ))
    return tuple(edges) if edges else None


def reverse_edge(edge: tuple[Any, ...]) -> tuple[Any, ...]:
    if edge[0] == "segment":
        return ("segment", edge[2], edge[1])
    if edge[0] == "arc":
        return ("arc", edge[3], edge[2], edge[1])
    return edge


def primitive_loops_equal(a: tuple[tuple[Any, ...], ...], b: tuple[tuple[Any, ...], ...]) -> bool:
    if len(a) != len(b):
        return False

    def seq_equal(x: tuple[tuple[Any, ...], ...], y: tuple[tuple[Any, ...], ...]) -> bool:
        return all(
            xe[0] == ye[0]
            and len(xe) == len(ye)
            and all(almost_equal(xv, yv) for xv, yv in zip(xe[1:], ye[1:]))
            for xe, ye in zip(x, y)
        )

    variants = [b, tuple(reverse_edge(e) for e in reversed(b))]
    for candidate in variants:
        for shift in range(len(candidate)):
            rotated = candidate[shift:] + candidate[:shift]
            if seq_equal(a, rotated):
                return True
    return False


def loop_repr(code: str, loop_name: str) -> dict[str, Any] | None:
    tree = ast.parse(code)
    env = eval_var_env(tree)
    assigns = loop_assignments(tree)
    assign = assigns.get(loop_name)
    if assign is None:
        return None
    high = parse_high_level_loop(assign, env)
    if high is not None:
        return {"type": "high", "value": high}
    primitive = parse_primitive_loop(assign, env)
    if primitive is not None:
        return {"type": "primitive", "value": primitive}
    return None


def loop_reprs_equal(pred: dict[str, Any] | None, gt: dict[str, Any] | None) -> bool:
    if pred is None or gt is None or pred["type"] != gt["type"]:
        return False
    if pred["type"] == "primitive":
        return primitive_loops_equal(pred["value"], gt["value"])
    p_val = pred["value"]
    g_val = gt["value"]
    if p_val["kind"] != g_val["kind"] or set(p_val["params"]) != set(g_val["params"]):
        return False
    return all(almost_equal(p_val["params"][k], g_val["params"][k]) for k in p_val["params"])


def touched_loops_from_metadata(meta: dict[str, str]) -> list[str]:
    loops = [x.strip() for x in meta.get("touched_loops", "").split(",") if x.strip()]
    if not loops and meta.get("loop"):
        loops = [meta["loop"]]
    return loops


def compute_edit_success(pred_code: str, gt_code: str, metadata: dict[str, str], pred_ok: bool) -> dict[str, Any]:
    if not pred_ok:
        return {"success": False, "error": "prediction did not execute"}
    loops = touched_loops_from_metadata(metadata)
    if not loops:
        return {"success": False, "error": "missing touched loops"}
    try:
        for loop in loops:
            if not loop_reprs_equal(loop_repr(pred_code, loop), loop_repr(gt_code, loop)):
                return {"success": False, "error": f"loop mismatch: {loop}", "loops": loops}
        return {"success": True, "error": "", "loops": loops}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "loops": loops}
