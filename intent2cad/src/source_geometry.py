"""Source-curve geometry in sketch-local coordinates, before normalization."""
import math

TAU = 2 * math.pi


def source_point(entity, suffix):
    message = entity['entity_msg']
    geometry = entity['geometry']
    g = geometry.get('message', {})
    if suffix == 'center' and geometry.get('typeName') == 'BTCurveGeometryCircle':
        return (float(g['xCenter']), float(g['yCenter']))
    if suffix not in {'start', 'end'} or suffix + 'Param' not in message:
        return None
    t = float(message[suffix + 'Param'])
    if geometry.get('typeName') == 'BTCurveGeometryLine':
        return (g['pntX'] + t * g['dirX'], g['pntY'] + t * g['dirY'])
    if geometry.get('typeName') == 'BTCurveGeometryCircle':
        direction = -1 if g.get('clockwise', False) else 1
        x, y = g['xDir'], g['yDir']
        return (g['xCenter'] + g['radius'] * (x * math.cos(t) - direction * y * math.sin(t)),
                g['yCenter'] + g['radius'] * (y * math.cos(t) + direction * x * math.sin(t)))
    return None


def arc_is_contained(curve, geometry, message, tolerance):
    """Equal circles are insufficient: require the entire JSON arc in the source trim."""
    if message is None:
        return False
    if 'startParam' not in message or 'endParam' not in message:
        # An untrimmed Onshape circle has no endpoints.
        return not message.get('startPointId') and not message.get('endPointId')
    g = geometry['message']
    span = float(message['endParam']) - float(message['startParam'])
    radius = float(g['radius'])
    if radius <= tolerance or span < 0 or span > TAU + 1e-8:
        return False
    angle_tol = tolerance / radius
    if span >= TAU - angle_tol:
        return True
    if curve['type'] != 'Arc3D':
        return False
    try:
        start, end = float(curve['start_angle']), float(curve['end_angle'])
        extent = end - start
        if extent < 0 or extent > span + angle_tol:
            return False
        reference = curve['reference_vector']
        normal = curve['normal']
        if abs(abs(normal['z']) - 1) > 1e-8:
            return False
        sign = 1 if normal['z'] > 0 else -1
        ref_angle = math.atan2(reference['y'], reference['x'])
        center = curve['center_point']
        points = [(center['x'] + curve['radius'] * math.cos(ref_angle + sign * t),
                   center['y'] + curve['radius'] * math.sin(ref_angle + sign * t))
                  for t in (start, (start + end) / 2, end)]
        for point, field in ((points[0], 'start_point'), (points[-1], 'end_point')):
            if math.dist(point, (curve[field]['x'], curve[field]['y'])) > tolerance:
                return False
        source_ref = math.atan2(g['yDir'], g['xDir'])
        source_sign = -1 if g.get('clockwise', False) else 1
        for x, y in points:
            t = source_sign * (math.atan2(y - g['yCenter'], x - g['xCenter']) - source_ref)
            relative = (t - message['startParam']) % TAU
            if relative > span + angle_tol and TAU - relative > angle_tol:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def primitive_endpoints(path):
    """Read actual emitted endpoints, including reversals made during loop ordering."""
    from .semantic import ScriptParser
    result = {}
    for loop in ScriptParser(path.read_text()).parse().values():
        for edge in loop.get('segments', []) + loop.get('arcs', []):
            tag = edge.get('tag')
            if tag:
                result.setdefault(tag, []).append(edge)
    return result


def map_endpoint(entity, suffix, tags, endpoints, scale, tolerance=1e-5):
    """Map to actual primitive endpoints; never copy a suffix across a split or reversal."""
    try:
        point = source_point(entity, suffix)
        if point is None or not all(math.isfinite(v) for v in point):
            return None
        point = tuple(v * scale for v in point)
        mapped = []
        for tag in tags:
            edges = endpoints.get(tag, [])
            # Repeated tags can refer to different loops or opposite orientations.
            for end in ('start', 'end'):
                matches = [math.dist(point, edge[end]) <= tolerance for edge in edges]
                if matches and all(matches):
                    mapped.append(f'{tag}.{end}')
                elif any(matches):
                    return None
        return mapped or None
    except (KeyError, TypeError, ValueError):
        return None


def covered_intervals(intervals, length, tolerance):
    """Require complete, non-overlapping coverage; a surviving fragment is not a whole edge."""
    cursor = 0.0
    for start, end in sorted(intervals):
        if abs(start - cursor) > tolerance or end <= start or end > length + tolerance:
            return False
        cursor = end
    return bool(intervals) and abs(cursor - length) <= tolerance


def source_is_covered(entity, curves, tolerance):
    try:
        g = entity['geometry']['message']
        if entity['geometry']['typeName'] == 'BTCurveGeometryLine':
            start, end = source_point(entity, 'start'), source_point(entity, 'end')
            length = math.dist(start, end)
            if length <= tolerance:
                return False
            direction = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
            intervals = []
            for curve in curves:
                points = [curve[key] for key in ('start_point', 'end_point')]
                projected = [sum((p[axis] - start[i]) * direction[i] for i, axis in enumerate(('x', 'y'))) for p in points]
                intervals.append(sorted(projected))
            return covered_intervals(intervals, length, tolerance)
        if entity['geometry']['typeName'] != 'BTCurveGeometryCircle':
            return False
        message = entity['entity_msg']
        start = float(message.get('startParam', 0))
        span = float(message['endParam']) - start if 'endParam' in message else TAU
        tol = tolerance / g['radius']
        if any(c['type'] == 'Circle3D' for c in curves):
            return len(curves) == 1 and span >= TAU - tol
        intervals = []
        sign = -1 if g.get('clockwise', False) else 1
        ref = math.atan2(g['yDir'], g['xDir'])
        for curve in curves:
            point = curve['start_point'] if curve['normal']['z'] * sign > 0 else curve['end_point']
            position = (sign * (math.atan2(point['y'] - g['yCenter'], point['x'] - g['xCenter']) - ref) - start) % TAU
            if TAU - position <= tol:
                position = 0.0
            extent = curve['end_angle'] - curve['start_angle']
            if position + extent > TAU and span >= TAU - tol:
                intervals.extend([(position, TAU), (0, position + extent - TAU)])
            else:
                intervals.append((position, position + extent))
        return covered_intervals(intervals, span, tol)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
