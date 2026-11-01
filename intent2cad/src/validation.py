"""Deterministic locality and solid validation; every execution has a 30s limit."""
from __future__ import annotations

import ast
import copy
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile

TOLERANCE = 1e-5
EXECUTION_TIMEOUT = 30
PARAM_PREFIXES = ("var_", "array_")
import builtins
SAFE_FUNCTIONS = {name: getattr(builtins, name) for name in (
    "abs", "min", "max", "round", "len", "sum", "range", "reversed", "list",
    "tuple", "zip", "enumerate", "sorted", "float", "int", "bool",
)}


def parameter_assignment(node):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith(PARAM_PREFIXES))


def _expression_allowed(node):
    for item in ast.walk(node):
        if isinstance(item, (ast.Attribute, ast.Lambda, ast.NamedExpr, ast.Await)):
            return False
        if isinstance(item, ast.Call):
            if not isinstance(item.func, ast.Name) or item.func.id not in SAFE_FUNCTIONS:
                return False
    return True


def parameter_environment(tree: ast.Module) -> dict:
    environment = {}
    for node in tree.body:
        if not parameter_assignment(node):
            continue
        if not _expression_allowed(node.value):
            raise ValueError("Parameter expression contains a disallowed operation.")
        environment[node.targets[0].id] = eval(
            compile(ast.Expression(node.value), "<parameter>", "eval"),
            {"__builtins__": SAFE_FUNCTIONS, **environment}, {},
        )
    return environment


def _close_values(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(left - right) <= TOLERANCE
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_close_values(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def locality_check(before: str, after: str) -> tuple[bool, str]:
    """Resolve only parameter values; require the rest of the AST to stay intact."""
    try:
        trees = [ast.parse(before), ast.parse(after)]
        environments = [parameter_environment(t) for t in trees]
        for name in environments[0].keys() & environments[1].keys():
            if not _close_values(environments[0][name], environments[1][name]):
                return False, f"parameter_value_changed:{name}"

        class Resolve(ast.NodeTransformer):
            def __init__(self, environment):
                self.environment = environment

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load) and node.id in self.environment:
                    return ast.copy_location(ast.parse(repr(self.environment[node.id]), mode="eval").body, node)
                return node

        resolved = []
        for tree, environment in zip(trees, environments):
            tree = copy.deepcopy(tree)
            tree.body = [node for node in tree.body if not parameter_assignment(node)]
            resolved.append(Resolve(environment).visit(tree))

        def same(left, right):
            if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
                return _close_values(left.value, right.value)
            if isinstance(left, ast.AST) and isinstance(right, ast.AST):
                return type(left) is type(right) and all(same(getattr(left, name), getattr(right, name)) for name in left._fields)
            if isinstance(left, list) and isinstance(right, list):
                return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
            return type(left) is type(right) and left == right

        return (True, "ok") if same(*resolved) else (False, "program_skeleton_or_arguments_changed")
    except Exception as error:
        return False, f"locality_error:{type(error).__name__}"


def strip_exports(code: str) -> str:
    tree = ast.parse(code)
    excluded = [node for node in tree.body if (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and ast.unparse(node.value.func).startswith("cq.exporters.")
    )]
    removed = {line for node in excluded for line in range(node.lineno, node.end_lineno + 1)}
    return "\n".join(line for i, line in enumerate(code.splitlines(), 1) if i not in removed) + "\n"


def _worker(code: str, working_root: str, connection, include_mesh: bool):
    try:
        import cadquery as cq

        with tempfile.TemporaryDirectory(prefix="cq-", dir=working_root) as directory:
            os.chdir(directory)
            namespace = {"cq": cq, "__builtins__": builtins.__dict__}
            exec(compile(strip_exports(code), "<cad>", "exec"), namespace)
            result = namespace.get("result")
            if result is None:
                result = namespace.get("r")
            if result is None:
                raise ValueError("Missing result/r.")
            shape = result.val() if hasattr(result, "val") else result
            volume = float(shape.Volume())
            if not shape.isValid() or not shape.Solids() or not math.isfinite(volume) or volume <= 0:
                raise ValueError("Not a valid non-empty solid.")
            bbox = shape.BoundingBox()
            center = shape.Center()
            values = [volume, bbox.xmin, bbox.ymin, bbox.zmin,
                      bbox.xmax, bbox.ymax, bbox.zmax, center.x, center.y, center.z]
            if not all(math.isfinite(float(x)) for x in values):
                raise ValueError("Non-finite geometric invariant.")
            mesh_signature = None
            if include_mesh:
                try:
                    vertices, faces = shape.tessellate(0.02)
                    xs, ys, zs = ([float(getattr(v, axis)) for v in vertices] for axis in ("x", "y", "z"))
                    if xs:
                        mesh_signature = (len(xs), len(faces), round(sum(xs), 6),
                                          round(sum(ys), 6), round(sum(zs), 6),
                                          round(sum(x*x + y*y + z*z for x, y, z in zip(xs, ys, zs)), 6))
                except Exception:
                    pass
            connection.send({"ok": True, "invariants": [float(x) for x in values],
                             "mesh_signature": mesh_signature})
    except Exception as error:
        connection.send({"ok": False, "reason": f"{type(error).__name__}:{error}"})
    finally:
        connection.close()


def execute_code(code: str, working_root: Path, include_mesh: bool = False) -> dict:
    working_root = Path(working_root).resolve()
    working_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(code, str(working_root), send, include_mesh))
    process.start()
    send.close()
    reported = False
    try:
        if not receive.poll(EXECUTION_TIMEOUT):
            return {"ok": False, "reason": "execution_timeout_30s"}
        try:
            result = receive.recv()
            reported = True
            return result
        except EOFError:
            return {"ok": False, "reason": "cad_worker_exited"}
    finally:
        if reported:
            process.join(timeout=1)
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()
        receive.close()


def geometry_consistent(before: dict, after: dict) -> bool:
    if not before.get("ok") or not after.get("ok"):
        return False
    left, right = before.get("invariants", []), after.get("invariants", [])
    return len(left) == len(right) == 10 and all(
        math.isfinite(a) and math.isfinite(b) and abs(a - b) <= TOLERANCE
        for a, b in zip(left, right)
    )


def accept_rewrite(before: str, after: str, working_root: Path, baseline: dict | None = None) -> dict:
    local, reason = locality_check(before, after)
    if not local:
        return {"accepted": False, "reason": reason}
    baseline = baseline if baseline is not None else execute_code(before, working_root)
    candidate = execute_code(after, working_root)
    accepted = geometry_consistent(baseline, candidate)
    return {"accepted": accepted, "reason": "ok" if accepted else "execution_or_geometry_failed",
            "baseline": baseline, "candidate": candidate}
