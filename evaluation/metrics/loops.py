"""Resolved loop comparison used by preservation and VLM localization analysis."""
from __future__ import annotations
import ast
from typing import Any
from . import editing as base

def eval_analysis_expr(node: ast.AST, env: dict[str, Any]) -> Any:
    """Evaluate dataset expressions, including simple point-list comprehensions."""
    if not isinstance(node, ast.ListComp):
        return base.eval_simple_expr(node, env)
    if len(node.generators) != 1 or node.generators[0].ifs:
        raise ValueError("Unsupported list comprehension")
    generator = node.generators[0]
    iterable = base.eval_simple_expr(generator.iter, env)
    values = []
    for item in iterable:
        local_env = dict(env)
        if isinstance(generator.target, ast.Name):
            local_env[generator.target.id] = item
        elif isinstance(generator.target, ast.Tuple):
            if len(generator.target.elts) != len(item):
                raise ValueError("List-comprehension unpack mismatch")
            for target, value in zip(generator.target.elts, item):
                if not isinstance(target, ast.Name):
                    raise ValueError("Unsupported list-comprehension target")
                local_env[target.id] = value
        else:
            raise ValueError("Unsupported list-comprehension target")
        values.append(base.eval_simple_expr(node.elt, local_env))
    return values


def eval_named_env(tree: ast.Module) -> dict[str, Any]:
    """Resolve simple parameter and array-center assignments."""
    pending: list[tuple[str, ast.AST]] = []
    env: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id.startswith("var_") or target.id.startswith("array_centers_"):
            pending.append((target.id, node.value))

    changed = True
    while pending and changed:
        changed = False
        rest: list[tuple[str, ast.AST]] = []
        for name, value_node in pending:
            try:
                env[name] = eval_analysis_expr(value_node, env)
                changed = True
            except Exception:
                rest.append((name, value_node))
        pending = rest
    return env


def parse_high_level_loop(assign: ast.Assign, env: dict[str, Any]) -> dict[str, Any] | None:
    chain = base.find_call_chain(assign.value)
    shape_call: ast.Call | None = None
    fillet_call: ast.Call | None = None
    push_call: ast.Call | None = None
    for call in chain:
        attr = base.call_attr_name(call)
        if attr == "push":
            push_call = call
        elif attr == "fillet":
            fillet_call = call
        elif attr in {"circle", "rect", "regularPolygon", "polygon", "slot"} and shape_call is None:
            shape_call = call

    if shape_call is None:
        return None
    kind = base.call_attr_name(shape_call)
    if kind == "rect" and fillet_call is not None:
        kind = "rect_fillet"
    if kind not in {"circle", "rect", "regularPolygon", "polygon", "slot", "rect_fillet"}:
        return None

    params: dict[str, Any] = {"centers": ((0.0, 0.0),), "angle": 0.0}
    if push_call is not None and push_call.args:
        pushed = base.eval_simple_expr(push_call.args[0], env)
        if isinstance(pushed, list):
            params["centers"] = tuple(tuple(p) for p in pushed)

    if kind == "circle":
        params["radius"] = base.eval_simple_expr(shape_call.args[0], env)
    elif kind == "rect":
        params["width"] = base.eval_simple_expr(shape_call.args[0], env)
        params["height"] = base.eval_simple_expr(shape_call.args[1], env)
    elif kind == "rect_fillet":
        params["width"] = base.eval_simple_expr(shape_call.args[0], env)
        params["height"] = base.eval_simple_expr(shape_call.args[1], env)
        if fillet_call is None or not fillet_call.args:
            return None
        params["fillet_radius"] = base.eval_simple_expr(fillet_call.args[0], env)
    elif kind == "regularPolygon":
        params["radius"] = base.eval_simple_expr(shape_call.args[0], env)
        params["sides"] = int(base.eval_simple_expr(shape_call.args[1], env))
    elif kind == "polygon":
        params["points"] = base.eval_simple_expr(shape_call.args[0], env)
    elif kind == "slot":
        params["length"] = base.eval_simple_expr(shape_call.args[0], env)
        params["diameter"] = base.eval_simple_expr(shape_call.args[1], env)

    angle_node = base.find_kw(shape_call, "angle")
    if angle_node is not None:
        params["angle"] = base.eval_simple_expr(angle_node, env)
    mode_node = base.find_kw(shape_call, "mode")
    if mode_node is not None:
        params["mode"] = base.eval_simple_expr(mode_node, env)

    return {
        "type": "high",
        "value": {
            "kind": kind,
            "params": {k: base.canonical_value(v) for k, v in sorted(params.items())},
        },
    }


def parse_loop_reprs(code: str) -> tuple[dict[str, dict[str, Any] | None], str]:
    try:
        tree = ast.parse(code)
        env = eval_named_env(tree)
        out: dict[str, dict[str, Any] | None] = {}
        for name, assign in base.loop_assignments(tree).items():
            high = parse_high_level_loop(assign, env)
            if high is not None:
                out[name] = high
                continue
            primitive = base.parse_primitive_loop(assign, env)
            out[name] = {"type": "primitive", "value": primitive} if primitive is not None else None
        return out, ""
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def reprs_equal(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None or a.get("type") != b.get("type"):
        return False
    if a["type"] == "primitive":
        return base.primitive_loops_equal(a["value"], b["value"])
    av = a["value"]
    bv = b["value"]
    if av["kind"] != bv["kind"] or set(av["params"]) != set(bv["params"]):
        return False
    return all(base.almost_equal(av["params"][k], bv["params"][k]) for k in av["params"])
