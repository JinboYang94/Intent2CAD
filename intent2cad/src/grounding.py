"""Bind proposed relations to actual parameter uses and source entities.

Loop comments alone do not identify an edge of a rectangle or a slot. Such
evidence is deliberately rejected when its parameter cannot be identified.
Numeric equality is checked later and is never used to establish provenance.
"""
from __future__ import annotations

import ast
import math
import re
from collections import defaultdict


def parameter_uses(code):
    tree = ast.parse(code)
    definitions = {}
    loops = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            definitions[name] = node.value
            if re.fullmatch(r"loop_\d+", name):
                loops[name] = node.value

    def dependencies(node, seen=frozenset()):
        result = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and item.id not in seen:
                if item.id.startswith('var_'):
                    result.add(item.id)
                if item.id in definitions and item.id not in loops:
                    result |= dependencies(definitions[item.id], seen | {item.id})
        return result

    uses = defaultdict(set)
    kinds = {}
    roles = {'circle': ['radius'], 'rect': ['width', 'height', 'angle'],
             'slot': ['length', 'diameter', 'angle'],
             'regularPolygon': ['radius', 'sides', 'angle'], 'polygon': ['points'],
             'push': ['center'], 'fillet': ['fillet_radius']}
    for loop, expression in loops.items():
        calls = [n for n in ast.walk(expression) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        shapes = [n.func.attr for n in calls if n.func.attr in {'circle', 'rect', 'slot', 'regularPolygon', 'polygon'}]
        kinds[loop] = shapes[0] if len(shapes) == 1 else 'unknown'
        pushes = [n for n in calls if n.func.attr == 'push']
        if any(not n.args or not isinstance(n.args[0], (ast.List, ast.Tuple)) or len(n.args[0].elts) != 1 for n in pushes):
            kinds[loop] = 'unknown'  # A loop alias may contain several array members.
        for call in calls:
            names = roles.get(call.func.attr, [])
            arguments = [(names[i] if i < len(names) else 'unknown', a) for i, a in enumerate(call.args)]
            arguments += [(kw.arg if kw.arg in names else 'unknown', kw.value) for kw in call.keywords]
            for role, argument in arguments:
                direct = isinstance(argument, ast.Name)
                if role == 'center':
                    direct = (isinstance(argument, (ast.List, ast.Tuple)) and len(argument.elts) == 1
                              and isinstance(argument.elts[0], ast.Name))
                if not direct:
                    role = 'unknown'  # An arithmetic term is not the entire feature dimension.
                for name in dependencies(argument):
                    uses[name].add((loop, role))
    # A parameter reused in an extrusion, another assignment or an unsupported
    # operation must not be silently bound using evidence for only one sketch.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)]
            if all(n.startswith(('var_', 'array_centers_')) or n in loops for n in names):
                continue
        for name in dependencies(node):
            uses[name].add(('outside_sketch', 'unknown'))
    return uses, kinds


def tag_bindings(code):
    bindings = defaultdict(set)
    for line in code.splitlines():
        match = re.match(r"\s*#\s*((?:'[^']+',?\s*)+)->\s*(loop_\d+)", line)
        if match:
            for tag in re.findall(r"'([^']+)'", match[1]):
                bindings[tag.split('.')[0]].add(match[2])
    return bindings


def entity_role(group, loop, kind, constraint_type, source_loops):
    if kind == 'circle':
        if constraint_type in {'EQUAL', 'RADIUS', 'DIAMETER'} and all('.' not in t for t in group):
            return 'radius'
        if constraint_type in {'CONCENTRIC', 'MIRROR'}:
            return 'center'
        if constraint_type == 'DISTANCE' and all(t.endswith('.center') for t in group):
            return 'center'
        return None
    source = source_loops.get(loop)
    if not source or any('.' in t for t in group):
        return None
    from .gate import Evidence
    from .semantic import FeatureRecognizer
    evidence = Evidence({}, {}, FeatureRecognizer)
    matches = [unit for unit in evidence.units(source) if set(group) == unit['tags']]
    if len(matches) != 1:
        return None
    unit = matches[0]
    if unit['kind'] == 'arc' and constraint_type in {'EQUAL', 'RADIUS', 'DIAMETER'}:
        if kind == 'slot':
            return 'diameter'
        if kind == 'rect' and source['type'] == 'fillet':
            return 'fillet_radius'
    if kind == 'rect' and source['type'] == 'rect' and unit['kind'] == 'line' and constraint_type in {'EQUAL', 'LENGTH'}:
        # Orientations, rather than coincident numerical lengths, distinguish
        # width from height even for a square or a rotated rectangle.
        angle = math.radians(source.get('angle', 0))
        dx, dy = evidence.direction(unit)
        along = abs(dx * math.cos(angle) + dy * math.sin(angle))
        across = abs(-dx * math.sin(angle) + dy * math.cos(angle))
        if along > 1 - 1e-8 and across < 1e-8:
            return 'width'
        if across > 1 - 1e-8 and along < 1e-8:
            return 'height'
    return None


def grounded_changes(code, changes, usable, variables, expression_names, source_code=None):
    uses, kinds = parameter_uses(code)
    tags = tag_bindings(code)
    source_loops = {}
    if source_code is not None:
        from .semantic import ScriptParser
        try:
            source_loops = ScriptParser(source_code).parse()
        except (ValueError, KeyError, TypeError):
            pass  # Missing geometry is not permission to guess an edge binding.
    evidence = defaultdict(list)
    for item in usable:
        key = (item['type'], tuple(sorted(t for group in item['entities'] for t in group)))
        evidence[key].append(item)
    accepted, rejected = [], []
    for change in changes:
        reason = None
        if not isinstance(change, dict):
            rejected.append({'reason': 'malformed_change'})
            continue
        variable = change.get('variable')
        refs = expression_names(str(change.get('new_value', '')))
        names = refs | {variable}
        primary = change.get('primary_constraints', [])
        if not refs or variable in refs or not names <= variables.keys() or not primary:
            reason = 'missing_parameter_or_evidence'
        support = defaultdict(set)
        links = set()
        dimension_names = set()
        for cited in primary if isinstance(primary, list) else []:
            if reason:
                break
            if not isinstance(cited, dict) or not isinstance(cited.get('entities'), list):
                reason = 'malformed_evidence'
                break
            key = (cited.get('type'), tuple(sorted(cited['entities'])))
            matched = evidence.get(key, [])
            if len(matched) != 1:
                reason = 'missing_or_ambiguous_evidence'
                break
            item = matched[0]
            entity_loops = []
            for group in item['entities']:
                mapped = [tags.get(t.split('.')[0], set()) for t in group]
                if not mapped or any(len(m) != 1 for m in mapped) or len(set.union(*mapped)) != 1:
                    reason = 'ambiguous_entity_mapping'
                    break
                entity_loops.append(next(iter(mapped[0])))
            if reason:
                break
            ctype = item['type']
            bindings = {(loop, entity_role(group, loop, kinds.get(loop), ctype, source_loops))
                        for group, loop in zip(item['entities'], entity_loops)}
            if any(role is None for _, role in bindings):
                reason = 'ambiguous_entity_parameter'
                break
            if ctype not in {'RADIUS', 'DIAMETER', 'LENGTH'} and len(entity_loops) < 2:
                reason = 'unsupported_parameter_evidence'
                break
            covered = set()
            for name in names:
                supported = bindings & uses.get(name, set())
                if supported:
                    support[name] |= supported
                    covered.add(name)
            if ctype in {'RADIUS', 'DIAMETER', 'LENGTH'}:
                dimension_names |= covered
            else:
                links |= {(a, b) for a in covered for b in covered if a != b}
        if not reason:
            if any(not uses.get(n) or support[n] != uses[n] for n in names):
                reason = 'parameter_use_not_covered'
            else:
                # Independent dimension evidence may support a simple ratio;
                # relational evidence must connect every referenced parameter.
                links |= {(a, b) for a in dimension_names for b in dimension_names if a != b}
                connected = {variable}
                while True:
                    expanded = connected | {b for a, b in links if a in connected}
                    if expanded == connected:
                        break
                    connected = expanded
                if not names <= connected:
                    reason = 'disconnected_parameter_evidence'
        if reason:
            rejected.append({'variable': variable, 'reason': reason})
        else:
            accepted.append(change)
    return accepted, rejected
