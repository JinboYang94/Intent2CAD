"""Original generation metric definitions, without model-loading dependencies."""
from __future__ import annotations
import ast
import re
import signal
from collections import Counter, defaultdict
from typing import Any, Optional
import numpy as np
import trimesh

CD_NUM_SAMPLES = 8192
EXEC_TIMEOUT = 30
VAR_RE = re.compile(r"^var_\w+$")
LOOP_RE = re.compile(r"^loop_\d+$")
CRS_EPS = 1e-5
CANONICAL_FEATURES = frozenset({
    "array",
    "rect",
    "polygon",
    "regularPolygon",
    "fillet",
    "slot",
})
FEATURE_API_ALIASES = {
    "array": "array",
    "rect": "rect",
    "polygon": "polygon",
    "regularPolygon": "regularPolygon",
    "fillet": "fillet",
    "slot": "slot",      # Sketch.slot(...)
    "slot2D": "slot",    # Workplane.slot2D(...)
}
ARRAY_PLACEMENT_APIS = frozenset({
    "pushPoints",
    "rarray",
    "polarArray",
    "parray",
})
_PRED_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

def strip_pred_file_artifacts(raw: str) -> str:
    """Normalize saved predictions (e.g. CAD-Coder) for exec/AST metrics.

    Strips optional ``<think>`` blocks and prefers the first fenced
    `` ```python `` segment when present, without changing already-clean CadQuery.
    """
    text = raw.strip()
    if "<think>" in text:
        end = text.rfind("</think>")
        if end != -1:
            text = text[end + len("</think>") :].strip()
    m = _PRED_CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


class _ExecutionTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ExecutionTimeout()


def execute_cadquery(code_str: str, timeout: int = EXEC_TIMEOUT):
    """Execute CadQuery code and return (success, result_workplane, error_msg).

    Accepts either ``result`` or ``r`` as the final solid (CAD-Coder uses ``r``).
    Uses SIGALRM for timeout (Unix only).  Set timeout=0 to disable.
    """
    import cadquery as cq

    use_alarm = timeout > 0 and hasattr(signal, "SIGALRM")
    if use_alarm:
        prev = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)
    try:
        ns: dict[str, Any] = {"cq": cq, "__builtins__": __builtins__}
        exec(code_str, ns)
        result = ns.get("result")
        if result is None:
            result = ns.get("r")
        if result is None:
            return False, None, "No 'result' or 'r' variable produced"
        _ = result.val().Volume()
        return True, result, ""
    except _ExecutionTimeout:
        return False, None, "Execution timed out"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)


def tessellate(result, tolerance: float = 0.001):
    """Return (vertices_ndarray, face_index_lists) from a CadQuery Workplane result."""
    verts, faces = result.val().tessellate(tolerance)
    vertices = np.array([[v.x, v.y, v.z] for v in verts])
    face_lists = [list(f) for f in faces]
    return vertices, face_lists


def sample_surface_points_trimesh(vertices: np.ndarray, faces: list[list[int]], n: int) -> np.ndarray:
    """使用工业级库 trimesh 进行严谨的表面采样"""
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3))
    
    # 构建 trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # 调用 trimesh 内置的均匀采样 (Text2CAD 默认行为)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts


def normalize_pc(points: np.ndarray) -> np.ndarray:
    """Normalize point cloud by its maximum bounding box extent (from Text2CAD)."""
    if len(points) == 0:
        return points
    scale = np.abs(np.max(points) - np.min(points))
    # 防止完全相同的点(如除以0)导致 NaN
    if scale < 1e-8:
        scale = 1e-8
    return points / scale


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer Distance (mean of squared nearest-neighbour distances)."""
    from scipy.spatial import KDTree

    if len(a) == 0 or len(b) == 0:
        return float("inf")

    d_a2b, _ = KDTree(b).query(a)
    d_b2a, _ = KDTree(a).query(b)
    return float(np.mean(d_a2b ** 2) + np.mean(d_b2a ** 2))


def compute_cd(gt_result, pred_result, n_points: int = 8192) -> dict:
    try:
        # 使用 0.001 容差对齐 Text2CAD 的 linear_deflection
        gt_v, gt_f = tessellate(gt_result, tolerance=0.001) 
        pred_v, pred_f = tessellate(pred_result, tolerance=0.001)
    except Exception as e:
        return {"cd": None, "error": f"Tessellation: {e}"}

    if len(gt_v) < 3 or len(pred_v) < 3:
        return {"cd": None, "error": "Too few vertices"}

    # 使用 trimesh 采样 8192 个点
    gt_pts = sample_surface_points_trimesh(gt_v, gt_f, n_points)
    pred_pts = sample_surface_points_trimesh(pred_v, pred_f, n_points)
    
    # 归一化点云
    gt_pts = normalize_pc(gt_pts)
    pred_pts = normalize_pc(pred_pts)
    
    # 计算双向 Chamfer Distance 并放大 1000 倍
    cd = chamfer_distance(gt_pts, pred_pts) * 1000.0
    
    return {"cd": cd, "error": ""}


def check_syntax(code_str: str) -> tuple[bool, str]:
    try:
        ast.parse(code_str)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def compute_sv(pred_code: str) -> dict:
    """SV: syntax validity of predicted code (AST-parseable)."""
    ok, err = check_syntax(pred_code)
    return {"sv": ok, "error": err}


def _extract_canonical_feature_multiset(code_str: str) -> Counter[str]:
    """Extract multiset of canonical high-level feature calls from code AST.

    Only semantic feature labels in CANONICAL_FEATURES are counted.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return Counter()

    feats: Counter[str] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            api_name = func.attr
        elif isinstance(func, ast.Name):
            api_name = func.id
        else:
            continue

        if api_name in ARRAY_PLACEMENT_APIS:
            feats["array"] += 1
            continue

        canonical = FEATURE_API_ALIASES.get(api_name)
        if canonical in CANONICAL_FEATURES:
            feats[canonical] += 1
    return feats


def _multiset_intersection_size(a: Counter[str], b: Counter[str]) -> int:
    """Return |a ∩ b| for multisets represented by Counters."""
    return int(sum(min(a[k], b[k]) for k in (a.keys() & b.keys())))


def compute_api_recovery_f1(gt_code: str, pred_code: str) -> dict:
    """Code-level API Recovery F1 based on canonical feature-call multisets."""
    gt_ms = _extract_canonical_feature_multiset(gt_code)
    pred_ms = _extract_canonical_feature_multiset(pred_code)

    gt_total = int(sum(gt_ms.values()))
    pred_total = int(sum(pred_ms.values()))
    if gt_total == 0:
        return {
            "api_recovery_f1": None,
            "gt_count": 0,
            "pred_count": pred_total,
            "matched": 0,
            "detail": "no_gt_canonical_features",
        }

    matched = _multiset_intersection_size(gt_ms, pred_ms)
    precision = (matched / pred_total) if pred_total > 0 else 0.0
    recall = matched / gt_total
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "api_recovery_f1": f1,
        "precision": precision,
        "recall": recall,
        "gt_count": gt_total,
        "pred_count": pred_total,
        "matched": matched,
    }


def _var_names_in_ast(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and VAR_RE.match(n.id)}


def _safe_eval_expr(node: ast.AST, env: dict[str, Any]) -> Any:
    return eval(compile(ast.Expression(node), "<>", "eval"), {"__builtins__": {}, "abs": abs}, env)


def resolve_symbol_values(code_str: str) -> dict[str, Any]:
    """Execute var_* and array_* assignments in order, return {name: resolved_value}."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return {}
    env: dict[str, Any] = {"abs": abs}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        name = tgt.id
        if not (VAR_RE.match(name) or name.startswith("array_")):
            continue
        try:
            env[name] = _safe_eval_expr(node.value, env)
        except Exception:
            pass
    return env


def resolve_var_values(code_str: str) -> dict[str, Any]:
    """Execute all var_* assignments in order, return {name: resolved_value}."""
    return {k: v for k, v in resolve_symbol_values(code_str).items() if VAR_RE.match(k)}


def _flatten_numeric(val) -> tuple[float, ...]:
    """Flatten a value (scalar, tuple, list of tuples) into a flat tuple of floats."""
    if isinstance(val, (int, float)):
        return (float(val),)
    if isinstance(val, (tuple, list)):
        result: list[float] = []
        for item in val:
            result.extend(_flatten_numeric(item))
        return tuple(result)
    return ()


def _values_close(a, b, eps: float = CRS_EPS) -> bool:
    fa, fb = _flatten_numeric(a), _flatten_numeric(b)
    if len(fa) != len(fb):
        return False
    return all(abs(x - y) < eps for x, y in zip(fa, fb))


def _greedy_value_match(gt_vals: list, pred_vals: list, eps: float = CRS_EPS) -> int:
    """Greedy 1-to-1 matching of two value lists by numeric closeness."""
    used = [False] * len(pred_vals)
    matched = 0
    for gv in gt_vals:
        for j, pv in enumerate(pred_vals):
            if not used[j] and _values_close(gv, pv, eps):
                used[j] = True
                matched += 1
                break
    return matched


def _find_tracked_feature_call(node: ast.AST) -> Optional[ast.Call]:
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in FEATURE_API_ALIASES
        ):
            return n
    return None


def _find_call_in_chain(node: ast.AST, attr_name: str) -> Optional[ast.Call]:
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr_name:
            return n
    return None


def _expr_skeleton(node: ast.AST, refs: Optional[dict[str, str]] = None) -> str:
    if refs is None:
        refs = {}
    if isinstance(node, ast.Constant):
        return "CONST"
    if isinstance(node, ast.Name):
        if VAR_RE.match(node.id):
            if node.id not in refs:
                refs[node.id] = f"REF{len(refs)}"
            return refs[node.id]
        return f"NAME({node.id})"
    if isinstance(node, ast.Tuple):
        return "TUP(" + ",".join(_expr_skeleton(e, refs) for e in node.elts) + ")"
    if isinstance(node, ast.List):
        return "LIST(" + ",".join(_expr_skeleton(e, refs) for e in node.elts) + ")"
    if isinstance(node, ast.BinOp):
        return f"BIN[{type(node.op).__name__}]({_expr_skeleton(node.left, refs)},{_expr_skeleton(node.right, refs)})"
    if isinstance(node, ast.UnaryOp):
        return f"UNARY[{type(node.op).__name__}]({_expr_skeleton(node.operand, refs)})"
    if isinstance(node, ast.Call):
        func_name = ast.unparse(node.func)
        args = ",".join(_expr_skeleton(a, refs) for a in node.args)
        kws = ",".join(f"{k.arg}={_expr_skeleton(k.value, refs)}" for k in node.keywords)
        return f"CALL({func_name}|{args}|{kws})"
    if isinstance(node, ast.Subscript):
        return f"SUB({_expr_skeleton(node.value, refs)}[{_expr_skeleton(node.slice, refs)}])"
    return type(node).__name__


def _extract_loop_api_uses(loop_expr: ast.AST) -> dict[str, set[tuple[str, str]]]:
    uses: dict[str, set[tuple[str, str]]] = defaultdict(set)
    api_call = _find_tracked_feature_call(loop_expr)
    if api_call is None or not isinstance(api_call.func, ast.Attribute):
        return uses
    api = api_call.func.attr
    for idx, arg in enumerate(api_call.args):
        for vn in _var_names_in_ast(arg):
            uses[vn].add((api, f"arg{idx}"))
    for kw in api_call.keywords:
        label = kw.arg if kw.arg is not None else "kw"
        for vn in _var_names_in_ast(kw.value):
            uses[vn].add((api, f"kw:{label}"))
    push_call = _find_call_in_chain(api_call.func.value, "push")
    if push_call is not None:
        for arg in push_call.args:
            for vn in _var_names_in_ast(arg):
                uses[vn].add((api, "push"))
    return uses


def _collect_var_def_skeletons(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name) and VAR_RE.match(tgt.id):
            out[tgt.id] = _expr_skeleton(node.value)
    return out


def _get_shared_var_structs(code_str: str) -> list[tuple[str, tuple[tuple[str, str], ...], int]]:
    """Return structural signatures for shared vars used in >=2 distinct loops."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return []
    def_sigs = _collect_var_def_skeletons(tree)
    var_loops: dict[str, set[str]] = defaultdict(set)
    var_uses: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name) or not LOOP_RE.match(tgt.id):
            continue
        loop_uses = _extract_loop_api_uses(node.value)
        for vn in _var_names_in_ast(node.value):
            var_loops[vn].add(tgt.id)
        for vn, toks in loop_uses.items():
            var_uses[vn].update(toks)
    feats: list[tuple[str, tuple[tuple[str, str], ...], int]] = []
    for vn, loops in var_loops.items():
        if len(loops) >= 2:
            feats.append((def_sigs.get(vn, "UNDEF"), tuple(sorted(var_uses.get(vn, set()))), len(loops)))
    return feats


def _get_derived_var_structs(code_str: str) -> list[tuple[str, int]]:
    """Return structural signatures for derived var_* (RHS references another var_*)."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return []
    feats: list[tuple[str, int]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name) or not VAR_RE.match(tgt.id):
            continue
        refs = _var_names_in_ast(node.value) - {tgt.id}
        if refs:
            feats.append((_expr_skeleton(node.value), len(refs)))
    return feats


def _greedy_struct_match(gt_feats: list[Any], pred_feats: list[Any]) -> int:
    used = [False] * len(pred_feats)
    matched = 0
    for gf in gt_feats:
        for j, pf in enumerate(pred_feats):
            if not used[j] and gf == pf:
                used[j] = True
                matched += 1
                break
    return matched


def compute_spf(gt_code: str, pred_code: str) -> dict:
    """SPF = |struct-matched shared vars| / |GT shared vars|.

    A shared var is one referenced in >=2 loop_* assignments.
    Matching is by structural signature (definition skeleton + API usage roles).
    """
    gt_feats = _get_shared_var_structs(gt_code)
    pred_feats = _get_shared_var_structs(pred_code)
    n_gt = len(gt_feats)
    n_pred = len(pred_feats)
    if n_gt == 0:
        return {"spf": None, "gt_shared": 0, "pred_shared": n_pred,
                "detail": "no_shared_params"}
    matched = _greedy_struct_match(gt_feats, pred_feats)
    return {"spf": matched / n_gt, "gt_shared": n_gt, "pred_shared": n_pred,
            "matched": matched}


def compute_dpf(gt_code: str, pred_code: str) -> dict:
    """DPF = |struct-matched derived vars| / |GT derived vars|.

    A derived var is a var_* whose RHS references another var_*.
    Matching is by expression skeleton (independent of concrete var names/values).
    """
    gt_feats = _get_derived_var_structs(gt_code)
    pred_feats = _get_derived_var_structs(pred_code)
    n_gt = len(gt_feats)
    n_pred = len(pred_feats)
    if n_gt == 0:
        return {"dpf": None, "gt_derived": 0, "pred_derived": n_pred,
                "detail": "no_derived_params"}
    matched = _greedy_struct_match(gt_feats, pred_feats)
    return {"dpf": matched / n_gt, "gt_derived": n_gt, "pred_derived": n_pred,
            "matched": matched}
