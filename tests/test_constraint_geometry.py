"""Regressions for source identity, endpoint orientation and merged-edge evidence."""
import copy
import math
import unittest

from intent2cad.src.constraints import ConstraintExtractor
from intent2cad.src.source_geometry import map_endpoint, source_is_covered, arc_is_contained
from intent2cad.src.gate import Evidence
from intent2cad.src.semantic import FeatureRecognizer
from intent2cad.src.grounding import entity_role


def line_entity(start=0, end=2):
    return {'geometry': {'typeName': 'BTCurveGeometryLine',
                         'message': {'pntX': 0, 'pntY': 0, 'dirX': 1, 'dirY': 0}},
            'entity_msg': {'startParam': start, 'endParam': end}}


def line(a, b):
    return {'type': 'Line3D', 'start_point': {'x': a, 'y': 0}, 'end_point': {'x': b, 'y': 0}}


def arc(a, b):
    return {'type': 'Arc3D', 'radius': 1, 'center_point': {'x': 0, 'y': 0},
            'reference_vector': {'x': 1, 'y': 0}, 'normal': {'z': 1},
            'start_angle': a, 'end_angle': b,
            'start_point': {'x': math.cos(a), 'y': math.sin(a)},
            'end_point': {'x': math.cos(b), 'y': math.sin(b)}}


def circle_entity(start=0, end=math.pi):
    return {'geometry': {'typeName': 'BTCurveGeometryCircle',
                         'message': {'xCenter': 0, 'yCenter': 0, 'radius': 1,
                                     'xDir': 1, 'yDir': 0, 'clockwise': False}},
            'entity_msg': {'startParam': start, 'endParam': end}}


class SourceGeometryTests(unittest.TestCase):
    def test_reversed_emitted_edge(self):
        edges = {'A': [{'start': (2, 0), 'end': (0, 0)}]}
        self.assertEqual(map_endpoint(line_entity(), 'start', ['A'], edges, 1), ['A.end'])
        self.assertEqual(map_endpoint(line_entity(), 'end', ['A'], edges, 1), ['A.start'])

    def test_split_edge_endpoint_is_not_broadcast(self):
        edges = {'A': [{'start': (0, 0), 'end': (1, 0)}],
                 'B': [{'start': (2, 0), 'end': (1, 0)}]}
        self.assertEqual(map_endpoint(line_entity(), 'start', ['A', 'B'], edges, 1), ['A.start'])
        self.assertEqual(map_endpoint(line_entity(), 'end', ['A', 'B'], edges, 1), ['B.start'])

    def test_ambiguous_reused_tag_is_rejected(self):
        edges = {'A': [{'start': (0, 0), 'end': (2, 0)}, {'start': (2, 0), 'end': (0, 0)}]}
        self.assertIsNone(map_endpoint(line_entity(), 'start', ['A'], edges, 1))

    def test_unmapped_and_unknown_endpoints_are_rejected(self):
        edges = {'A': [{'start': (.2, 0), 'end': (1.8, 0)}]}
        self.assertIsNone(map_endpoint(line_entity(), 'start', ['A'], edges, 1))
        self.assertIsNone(map_endpoint(line_entity(), 'startSnap', ['A'], edges, 1))

    def test_endpoint_tolerance_is_not_relaxed(self):
        self.assertEqual(map_endpoint(line_entity(), 'start', ['A'],
                         {'A': [{'start': (9e-6, 0), 'end': (2, 0)}]}, 1), ['A.start'])
        self.assertIsNone(map_endpoint(line_entity(), 'start', ['A'],
                          {'A': [{'start': (11e-6, 0), 'end': (2, 0)}]}, 1))

    def test_split_coverage(self):
        self.assertTrue(source_is_covered(line_entity(), [line(0, 1), line(2, 1)], 1e-5))
        for pieces in ([line(0, 1)], [line(0, .8), line(1, 2)],
                       [line(0, 1.2), line(1, 2)], [line(0, 2), line(0, 2)]):
            self.assertFalse(source_is_covered(line_entity(), pieces, 1e-5))

    def test_same_circle_different_arc_is_not_a_match(self):
        entity = circle_entity()
        self.assertTrue(arc_is_contained(arc(0, math.pi), entity['geometry'], entity['entity_msg'], 1e-5))
        self.assertFalse(arc_is_contained(arc(math.pi, 2*math.pi), entity['geometry'], entity['entity_msg'], 1e-5))
        self.assertFalse(arc_is_contained({'type': 'Circle3D'}, entity['geometry'], entity['entity_msg'], 1e-5))

    def test_split_arc_requires_complete_coverage(self):
        entity = circle_entity()
        pieces = [arc(0, math.pi/2), arc(math.pi/2, math.pi)]
        self.assertTrue(source_is_covered(entity, pieces, 1e-5))
        self.assertFalse(source_is_covered(entity, pieces[:1], 1e-5))
        self.assertFalse(source_is_covered(entity, [pieces[0], pieces[0]], 1e-5))

    def test_clockwise_source_endpoint(self):
        entity = circle_entity(0, math.pi/2)
        entity['geometry']['message']['clockwise'] = True
        edges = {'A': [{'start': (0, -1), 'end': (1, 0)}]}
        self.assertEqual(map_endpoint(entity, 'end', ['A'], edges, 1), ['A.start'])

    def test_sketch_local_geometry_does_not_use_world_origin(self):
        extractor = ConstraintExtractor('', '')
        extractor.scale_factor = 1
        entity = line_entity()
        self.assertTrue(extractor.match_geometry(line(0, 2), entity['geometry'], entity['entity_msg'], {'x': 10, 'y': -5}))


class MergedEdgeTests(unittest.TestCase):
    def setUp(self):
        self.loop = {'type': 'rect', 'angle': 0, 'segments': [
            {'tag': 'A', 'start': (0, 0), 'end': (1, 0)},
            {'tag': 'B', 'start': (1, 0), 'end': (3, 0)},
            {'tag': 'C', 'start': (3, 0), 'end': (3, 2)},
            {'tag': 'D', 'start': (3, 2), 'end': (0, 2)},
            {'tag': 'E', 'start': (0, 2), 'end': (0, 0)}], 'arcs': []}
        self.constraints = [{'type': kind, 'entities': [{'json_ids': [tag]}]}
                            for kind, tag in [('HORIZONTAL', 'A'), ('VERTICAL', 'C'),
                                              ('HORIZONTAL', 'D'), ('VERTICAL', 'E')]]

    def check(self, extra):
        return Evidence({}, {'constraints': self.constraints + extra}, FeatureRecognizer).check(self.loop)

    def test_internal_endpoint_is_not_the_merged_edge_endpoint(self):
        self.assertEqual(Evidence.endpoint({'json_ids': ['B.start']}, self.loop), (1, 0))
        good = {'type': 'COINCIDENT', 'entities': [{'json_ids': ['B.end']}, {'json_ids': ['C.start']}]}
        self.assertTrue(self.check([good])[0])
        bad = copy.deepcopy(good);bad['entities'][0]['json_ids'] = ['B.start']
        self.assertEqual(self.check([bad]), (False, 'coincident_endpoints_conflict'))

    def test_short_edge_dimension_is_not_merged_length(self):
        good = {'type': 'LENGTH', 'entities': [{'json_ids': ['A']}], 'value_normalized': 1}
        self.assertTrue(self.check([good])[0])
        bad = dict(good, value_normalized=3)
        self.assertEqual(self.check([bad]), (False, 'source_dimension_conflicts_with_geometry'))

    def test_partial_edge_does_not_ground_whole_feature_width(self):
        source = {'loop_1': self.loop}
        self.assertIsNone(entity_role({'A'}, 'loop_1', 'rect', 'LENGTH', source))
        self.assertEqual(entity_role({'A', 'B'}, 'loop_1', 'rect', 'LENGTH', source), 'width')

    def test_point_references_cannot_establish_edge_orientation(self):
        constraints = [{'type': kind, 'entities': [{'json_ids': [a]}, {'json_ids': [b]}]}
                       for kind, a, b in [('PARALLEL', 'A', 'D'), ('PARALLEL', 'C', 'E'),
                                          ('PERPENDICULAR', 'A', 'C')]]
        self.assertTrue(Evidence({}, {'constraints': constraints}, FeatureRecognizer).check(self.loop)[0])
        for constraint in constraints:
            for entity in constraint['entities']:
                entity['json_ids'] = [entity['json_ids'][0] + '.start']
        self.assertFalse(Evidence({}, {'constraints': constraints}, FeatureRecognizer).check(self.loop)[0])


class ScopeAndArrayTests(unittest.TestCase):
    def test_duplicate_curve_ids_across_sketches_are_not_overwritten(self):
        extractor = ConstraintExtractor('', '')
        curve = dict(line(0, 2), curve='A')
        sketch = {'type': 'Sketch', 'profiles': {'p': {'loops': [{'profile_curves': [curve]}]}}}
        extractor.json_data = {'entities': {'first': sketch, 'second': copy.deepcopy(sketch)}}
        self.assertEqual(extractor.extract_json_entities(), {})

    def test_duplicate_source_ids_across_features_are_not_overwritten(self):
        extractor = ConstraintExtractor('', '')
        entity = line_entity()
        item = {'typeName': 'BTMSketchCurveSegment', 'message': dict(entity['entity_msg'],
                entityId='same', geometry=entity['geometry'])}
        extractor.yml_data = {'features': [{'message': {'featureId': name, 'entities': [item]}}
                                           for name in ('first', 'second')]}
        self.assertEqual(extractor.extract_yml_entities(), {})

    def test_circle_array_requires_center_position_evidence(self):
        loops = {f'loop_{i}': {'type': 'circle', 'tags': [f'T{i}'], 'radius': 1,
                              'center': (i * 3, 0)} for i in range(3)}
        constraints = []
        for i in range(2):
            constraints.append({'type': 'EQUAL', 'entities': [{'json_ids': [f'T{i}']}, {'json_ids': [f'T{i+1}']}]})
            constraints.append({'type': 'HORIZONTAL', 'entities': [{'json_ids': [f'T{i}.center']}, {'json_ids': [f'T{i+1}.center']}]})
        array = {'loops': list(loops), 'is_regular_pattern': True}
        self.assertTrue(Evidence(loops, {'constraints': constraints}, FeatureRecognizer).array_supported(array))
        bad = copy.deepcopy(constraints)
        for c in bad:
            if c['type'] == 'HORIZONTAL':
                for entity in c['entities']:entity['json_ids'] = [entity['json_ids'][0].replace('.center', '.start')]
        self.assertFalse(Evidence(loops, {'constraints': bad}, FeatureRecognizer).array_supported(array))
        wrong_geometry = copy.deepcopy(loops);wrong_geometry['loop_1']['center'] = (3, 1)
        self.assertFalse(Evidence(wrong_geometry, {'constraints': constraints}, FeatureRecognizer).array_supported(array))


if __name__ == '__main__':
    unittest.main()
