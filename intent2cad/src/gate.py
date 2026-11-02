"""Feature-specific source-constraint evidence, applied after geometry recognition."""
from __future__ import annotations

import math

TOL = 1e-5


def tag_base(tag):
    return tag.split(".", 1)[0]


def tags_of(loop):
    return set(loop.get("tags", []) + loop.get("segment_tags", []) + loop.get("arc_tags", []))


def connected(nodes, edges):
    if not nodes:
        return False
    seen = {next(iter(nodes))}
    while True:
        expanded = seen | {b for a, b in edges if a in seen} | {a for a, b in edges if b in seen}
        if expanded == seen:
            return nodes <= seen
        seen = expanded


def circle_from_arc(arc):
    (ax, ay), (bx, by), (cx, cy) = arc["start"], arc["mid"], arc["end"]
    determinant = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(determinant) < 1e-15:
        raise ValueError("Degenerate arc.")
    aa, bb, cc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    center = ((aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / determinant,
              (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / determinant)
    return center, math.dist(center, arc["start"])


class Evidence:
    def __init__(self, loops, constraints, recognizer):
        self.loops = loops
        self.constraints = constraints.get("constraints", [])
        self.recognizer = recognizer
        self.tag_loops = {}
        for name, loop in loops.items():
            for tag in tags_of(loop):
                self.tag_loops.setdefault(tag, set()).add(name)

    def units(self, loop):
        segments = self.recognizer._merge_collinear_segments(loop.get("segments", []))
        result = []
        for segment in segments:
            result.append({"kind": "line", "geometry": segment,
                           "tags": set(segment.get("tags", []) or [segment.get("tag")]) - {None}})
        for arc in loop.get("arcs", []):
            result.append({"kind": "arc", "geometry": arc, "tags": {arc.get("tag")} - {None}})
        return result

    @staticmethod
    def _references(constraint, units):
        entities = []
        for entity in constraint.get("entities", []):
            tags = {tag_base(tag) for tag in entity.get("json_ids", [])}
            matches = [i for i, unit in enumerate(units) if tags and tags <= unit["tags"]]
            if len(matches) != 1:
                return None
            entities.append(matches[0])
        return entities or None

    @staticmethod
    def direction(unit):
        if unit["kind"] != "line":
            return None
        g = unit["geometry"]
        dx, dy = g["end"][0] - g["start"][0], g["end"][1] - g["start"][1]
        length = math.hypot(dx, dy)
        return (dx / length, dy / length) if length > TOL else None

    @staticmethod
    def size(unit):
        if unit["kind"] == "arc":
            return circle_from_arc(unit["geometry"])[1]
        return math.dist(unit["geometry"]["start"], unit["geometry"]["end"])

    @staticmethod
    def endpoint(entity, loop):
        points = []
        edges = loop.get("segments", []) + loop.get("arcs", [])
        for ref in entity.get("json_ids", []):
            tag, dot, suffix = ref.partition(".")
            matches = [e for e in edges if e.get("tag") == tag]
            if not dot or suffix not in {"start", "end"} or not matches:
                return None
            points.extend(e[suffix] for e in matches)
        if not points or any(math.dist(points[0], p) > TOL for p in points[1:]):
            return None
        return points[0]

    @staticmethod
    def entity_size(entity, loop, constraint_type):
        ids = entity.get("json_ids", [])
        if not ids or any("." in ref for ref in ids):
            return None
        edges = loop.get("segments", []) + loop.get("arcs", [])
        selected = [[e for e in edges if e.get("tag") == tag] for tag in ids]
        if any(len(group) != 1 for group in selected):
            return None
        selected = [group[0] for group in selected]
        arcs = [e for e in selected if "mid" in e]
        if arcs:
            if len(arcs) != len(selected) or constraint_type == "LENGTH":
                return None
            radii = [circle_from_arc(e)[1] for e in arcs]
            if max(radii) - min(radii) > TOL:
                return None
            return radii[0] * (2 if constraint_type == "DIAMETER" else 1)
        if constraint_type not in {"LENGTH", "EQUAL"}:
            return None
        return sum(math.dist(e["start"], e["end"]) for e in selected)

    def check(self, loop):
        kind = loop.get("type")
        if kind in {"circle", "unknown"}:
            return True, "primitive_preserved"
        units = self.units(loop)
        if not units or any(not unit["tags"] for unit in units):
            return False, "missing_primitive_tags"
        relations, unary = {}, {}
        endpoints = set()
        dimensions = {}
        for constraint in self.constraints:
            refs = self._references(constraint, units)
            if not refs:
                continue
            ctype = constraint.get("type")
            if len(refs) == 1:
                i = refs[0]
                geometry = units[i]["geometry"]
                direction = self.direction(units[i])
                # A relation on .center or .start is not evidence for the whole edge.
                ids = constraint.get("entities", [{}])[0].get("json_ids", [])
                if ctype in {"HORIZONTAL", "VERTICAL"} and direction and all("." not in x for x in ids):
                    component = direction[1] if ctype == "HORIZONTAL" else direction[0]
                    if abs(component) > TOL:
                        return False, "source_constraint_conflicts_with_geometry"
                    unary.setdefault(ctype, set()).add(i)
                if ctype in {"RADIUS", "DIAMETER", "LENGTH"} and "value_normalized" in constraint:
                    value = float(constraint["value_normalized"])
                    if not math.isfinite(value):
                        return False, "nonfinite_source_dimension"
                    expected = self.entity_size(constraint["entities"][0], loop, ctype)
                    if expected is None:
                        continue
                    if abs(value - expected) > TOL:
                        return False, "source_dimension_conflicts_with_geometry"
                    if units[i]["kind"] == "arc" and ctype in {"RADIUS", "DIAMETER"}:
                        dimensions[i] = value / 2 if ctype == "DIAMETER" else value
                continue
            if len(refs) != 2 or refs[0] == refs[1]:
                continue
            i, j = refs
            pair = tuple(sorted(refs))
            if ctype in {"PARALLEL", "PERPENDICULAR", "TANGENT", "ANGLE", "EQUAL"}:
                if any("." in tag for entity in constraint["entities"] for tag in entity.get("json_ids", [])):
                    continue
            if ctype == "ANGLE" and "value_degrees" not in constraint:
                continue
            if ctype == "COINCIDENT":
                points = [self.endpoint(entity, loop) for entity in constraint.get("entities", [])]
                if len(points) == 2 and all(p is not None for p in points):
                    if math.dist(*points) > TOL:
                        return False, "coincident_endpoints_conflict"
                    endpoints.add(pair)
            first, second = self.direction(units[i]), self.direction(units[j])
            if ctype in {"PARALLEL", "PERPENDICULAR"} and first and second:
                residual = (first[0] * second[1] - first[1] * second[0]) if ctype == "PARALLEL" else sum(a * b for a, b in zip(first, second))
                if abs(residual) > TOL:
                    return False, "source_constraint_conflicts_with_geometry"
            if ctype == "EQUAL":
                if units[i]["kind"] != units[j]["kind"]:
                    continue
                sizes = [self.entity_size(e, loop, ctype) for e in constraint["entities"]]
                if any(v is None for v in sizes):
                    continue
                if abs(sizes[0] - sizes[1]) > TOL:
                    return False, "equal_constraint_conflicts_with_geometry"
                if any({tag_base(t) for t in e["json_ids"]} != units[k]["tags"]
                       for e, k in zip(constraint["entities"], refs)):
                    continue  # Equal subedges do not imply equal merged edges.
            if ctype == "ANGLE" and "value_degrees" in constraint and first and second:
                angle = float(constraint["value_degrees"]) % 180
                if not math.isfinite(angle):
                    return False, "nonfinite_source_angle"
                actual = math.degrees(math.acos(max(-1, min(1, sum(a * b for a, b in zip(first, second))))))
                if min(abs(actual - angle), abs(180 - actual - angle)) > TOL:
                    return False, "angle_constraint_conflicts_with_geometry"
                if min(abs(angle - 90), abs(angle + 90)) <= TOL:
                    ctype = "PERPENDICULAR"
                elif min(abs(angle), abs(angle - 180)) <= TOL:
                    ctype = "PARALLEL"
            relations.setdefault(ctype, set()).add(pair)

        lines = {i for i, unit in enumerate(units) if unit["kind"] == "line"}
        arcs = set(range(len(units))) - lines
        parity = [(a, b, 0) for a, b in relations.get("PARALLEL", set())]
        parity += [(a, b, 1) for a, b in relations.get("PERPENDICULAR", set())]
        for axis in ("HORIZONTAL", "VERTICAL"):
            members = sorted(unary.get(axis, set()))
            parity += [(a, b, 0) for a, b in zip(members, members[1:])]
        parity += [(a, b, 1) for a in unary.get("HORIZONTAL", set()) for b in unary.get("VERTICAL", set())]

        def oriented_rectangle():
            if len(lines) != 4:
                return False
            colors = {min(lines): 0}
            for _ in range(len(lines)):
                for a, b, opposite in parity:
                    if a not in lines or b not in lines:
                        continue
                    if a in colors:
                        if b in colors and colors[b] != colors[a] ^ opposite:
                            return False
                        colors[b] = colors[a] ^ opposite
                    elif b in colors:
                        colors[a] = colors[b] ^ opposite
            if set(colors) != lines or list(colors.values()).count(0) != 2:
                return False
            # Source parity must agree with every candidate edge direction.
            base = self.direction(units[min(lines)])
            for i in lines:
                v = self.direction(units[i])
                if v is None:
                    return False
                residual = sum(a * b for a, b in zip(base, v)) if colors[i] else base[0] * v[1] - base[1] * v[0]
                if abs(residual) > TOL:
                    return False
            return True

        equal = set(relations.get("EQUAL", set()))
        equal |= {(a, b) for a in dimensions for b in dimensions if a < b and abs(dimensions[a] - dimensions[b]) <= TOL}

        if kind == "rect":
            ok = oriented_rectangle()
        elif kind in {"slot", "fillet"}:
            tangencies = set()
            all_joins = set()
            for a in lines:
                for b in arcs:
                    shared = [(p, q) for p in (units[a]["geometry"]["start"], units[a]["geometry"]["end"])
                              for q in (units[b]["geometry"]["start"], units[b]["geometry"]["end"]) if math.dist(p, q) <= TOL]
                    if shared:
                        pair = tuple(sorted((a, b)))
                        all_joins.add(pair)
                        center, _ = circle_from_arc(units[b]["geometry"])
                        radial = (shared[0][0][0] - center[0], shared[0][0][1] - center[1])
                        direction = self.direction(units[a])
                        if direction and abs(sum(x * y for x, y in zip(radial, direction))) <= TOL:
                            if pair in relations.get("TANGENT", set()):
                                tangencies.add(pair)
            supported_radius = connected(arcs, equal)
            if kind == "slot":
                line_list = sorted(lines)
                parallel = len(line_list) == 2 and connected(lines, [(a, b) for a, b, v in parity if v == 0])
                ok = parallel and len(all_joins) == 4 and tangencies == all_joins and supported_radius
            else:
                ok = oriented_rectangle() and len(all_joins) == 8 and tangencies == all_joins and supported_radius
        elif kind == "regularPolygon":
            angles = relations.get("ANGLE", set()) | relations.get("PERPENDICULAR", set())
            # Equal edge lengths and source angle constraints must cover the boundary.
            adjacent = set()
            for a in lines:
                for b in lines:
                    if a < b and any(math.dist(p, q) <= TOL for p in (units[a]["geometry"]["start"], units[a]["geometry"]["end"])
                                     for q in (units[b]["geometry"]["start"], units[b]["geometry"]["end"])):
                        adjacent.add((a, b))
            ok = connected(lines, equal) and len(adjacent) == len(lines) and adjacent <= angles
        elif kind == "polygon":
            # All boundary vertices need source endpoint coincidence, not a stray edge relation.
            degrees = {i: 0 for i in lines}
            for a, b in endpoints:
                if a in lines and b in lines:
                    degrees[a] += 1
                    degrees[b] += 1
            ok = bool(lines) and all(v == 2 for v in degrees.values()) and connected(lines, endpoints)
        else:
            ok = False
        return bool(ok), "constraint_supported" if ok else "insufficient_template_evidence"

    def array_supported(self, array):
        members = array["loops"]
        if (not array.get("is_regular_pattern") or len(members) < 3
                or any(self.loops[name].get("type") != "circle" for name in members)):
            # A constraint on one rectangle edge does not establish equality
            # of the whole repeated feature. Only whole circles are bound here.
            return False
        size_edges, position_edges = set(), set()
        for constraint in self.constraints:
            groups = constraint.get("entities", [])
            names = []
            for entity in groups:
                ids = entity.get("json_ids", [])
                mapped = set().union(*(self.tag_loops.get(tag_base(t), set()) for t in ids))
                if len(mapped) != 1:
                    names = []
                    break
                names.append(next(iter(mapped)))
            if len(names) != 2 or names[0] == names[1] or not set(names) <= set(members):
                continue
            first, second = (self.loops[n] for n in names)
            pair = tuple(names)
            kind = constraint.get("type")
            if kind == "EQUAL":
                if (all(all("." not in t for t in e["json_ids"]) for e in groups)
                        and abs(first["radius"] - second["radius"]) <= TOL):
                    size_edges.add(pair)
                continue
            if not all(all(t.endswith(".center") for t in e["json_ids"]) for e in groups):
                continue
            a, b = first["center"], second["center"]
            if kind == "HORIZONTAL" and abs(a[1] - b[1]) <= TOL:
                position_edges.add(pair)
            elif kind == "VERTICAL" and abs(a[0] - b[0]) <= TOL:
                position_edges.add(pair)
            elif kind == "CONCENTRIC" and math.dist(a, b) <= TOL:
                position_edges.add(pair)
            elif kind == "DISTANCE" and "value_normalized" in constraint:
                if abs(math.dist(a, b) - float(constraint["value_normalized"])) <= TOL:
                    position_edges.add(pair)
        return connected(set(members), size_edges) and connected(set(members), position_edges)
