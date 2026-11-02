import json
import yaml
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import math

from .source_geometry import arc_is_contained, map_endpoint, primitive_endpoints, source_is_covered


def normalize_feature_list(payload: dict) -> dict:
    """Read local feature exports in either message or btType serialization."""
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        fields = {key: convert(item) for key, item in value.items() if key != 'btType'}
        if 'btType' not in value:
            return fields
        match = re.fullmatch(r'(.+)-(\d+)', value['btType'])
        if not match:
            raise ValueError('Unrecognized btType in the local feature export.')
        return {'typeName': match[1], 'type': int(match[2]), 'message': fields}

    if not isinstance(payload, dict) or not isinstance(payload.get('features'), list):
        raise ValueError('Local feature export must contain a features list with sketch geometry.')
    if payload.get('isComplete') is False:
        raise ValueError('Local feature export is incomplete.')
    result = dict(payload)
    result['features'] = convert(payload['features'])
    return result


class ConstraintExtractor:
    def __init__(self, json_path: str, yml_path: str, dedup_path: Optional[str] = None, norm_factor: float = 0.75):
        """
        初始化约束提取器
        
        Args:
            json_path: CAD JSON文件路径
            yml_path: 已准备好的本地FeatureScript YML或JSON文件路径
            dedup_path: output_deduplicates中对应的.py文件路径（可选）
            norm_factor: 归一化因子，默认0.75（与json2cadquery.py保持一致）
        """
        self.json_path = json_path
        self.yml_path = yml_path
        self.dedup_path = dedup_path
        self.norm_factor = norm_factor
        self.json_data = None
        self.yml_data = None
        
        # 存储ID映射：yml_id -> json_id
        self.id_mapping = {}
        self.primitive_endpoints = {}
        
        # 存储dedup文件中实际使用的TAG
        self.valid_tags: Optional[Set[str]] = None
        
        # 存储归一化参数
        self.scale_factor: Optional[float] = None
        self.bbox_center: Optional[np.ndarray] = None
        
        # 支持的约束类型
        self.supported_constraints = {
            'PARALLEL', 'PERPENDICULAR', 'TANGENT', 
            'COINCIDENT', 'CONCENTRIC', 'EQUAL',
            'HORIZONTAL', 'VERTICAL',  # 水平和垂直约束
            'LENGTH', 'ANGLE', 'RADIUS', 'DISTANCE', 'DIAMETER',  # 尺寸约束
            'MIRROR'  # 镜像约束
        }
        
        # 尺寸约束类型
        self.dimensional_constraints = {
            'LENGTH', 'RADIUS', 'DISTANCE', 'DIAMETER'
        }
        
        # 角度约束类型
        self.angular_constraints = {'ANGLE'}
        
    def load_files(self):
        """加载原始CAD JSON和本地特征导出文件。"""
        with open(self.json_path, 'r') as f:
            self.json_data = json.load(f)
        
        with open(self.yml_path, 'r') as f:
            self.yml_data = normalize_feature_list(yaml.safe_load(f))
        for feature in self.yml_data.get('features', []):
            for item in feature.get('message', {}).get('constraints', []):
                message = item.get('message', {})
                message['constraintType'] = {'LEN': 'LENGTH'}.get(message.get('constraintType'), message.get('constraintType'))
        
        # 计算归一化参数（与json2cadquery.py保持一致）
        self._compute_normalization_params()
        
        # 如果提供了dedup文件路径，加载有效的TAG
        if self.dedup_path:
            self.valid_tags = self._extract_tags_from_dedup()
            self.primitive_endpoints = primitive_endpoints(Path(self.dedup_path))
    
    def _compute_normalization_params(self):
        """
        计算归一化参数（与json2cadquery.py保持一致的逻辑）
        
        从JSON的bounding_box中提取归一化参数，用于将约束的尺寸值也归一化
        """
        if 'properties' not in self.json_data or 'bounding_box' not in self.json_data['properties']:
            print("Warning: No bounding_box found in JSON, skipping normalization")
            self.scale_factor = 1.0
            self.bbox_center = np.array([0.0, 0.0, 0.0])
            return
        
        bbox = self.json_data['properties']['bounding_box']
        min_pt = bbox['min_point']
        max_pt = bbox['max_point']
        
        bbox_min = np.array([min_pt['x'], min_pt['y'], min_pt['z']])
        bbox_max = np.array([max_pt['x'], max_pt['y'], max_pt['z']])
        
        # 计算bbox中心
        self.bbox_center = (bbox_min + bbox_max) / 2.0
        
        # 计算最大半轴长度
        half_extent = (bbox_max - bbox_min) / 2.0
        max_abs_distance = np.max(half_extent)
        
        # 处理边界情况
        if max_abs_distance < 1e-10:
            self.scale_factor = 1.0
        else:
            self.scale_factor = self.norm_factor / max_abs_distance
        
        print(f"归一化参数:")
        print(f"  Bounding box: min={bbox_min}, max={bbox_max}")
        print(f"  Center: {self.bbox_center}")
        print(f"  Scale factor: {self.scale_factor:.10f}")
    
    def _extract_tags_from_dedup(self) -> Set[str]:
        """
        从dedup文件中提取所有使用的TAG
        
        Returns:
            TAG集合
        """
        tags = set()
        
        if not self.dedup_path or not Path(self.dedup_path).exists():
            return tags
        
        try:
            with open(self.dedup_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则表达式提取所有 tag='XXX' 中的XXX
            tag_pattern = r"tag=['\"]([^'\"]+)['\"]"
            matches = re.findall(tag_pattern, content)
            tags.update(matches)
            
        except Exception as e:
            print(f"Warning: Failed to extract tags from {self.dedup_path}: {e}")
        
        return tags
    
    def extract_json_entities(self) -> Dict[str, Dict]:
        """
        从JSON文件中提取实线图元及其几何参数
        
        Returns:
            字典：curve_id -> {type, data, transform, sketch_id}
        """
        entities = {}
        ambiguous = set()
        
        if 'entities' not in self.json_data:
            return entities
        
        for entity_id, entity in self.json_data['entities'].items():
            if entity.get('type') == 'Sketch':
                # 获取Sketch的坐标变换
                transform = entity.get('transform', {})
                origin = transform.get('origin', {'x': 0, 'y': 0, 'z': 0})
                
                profiles = entity.get('profiles', {})
                for profile_id, profile in profiles.items():
                    loops = profile.get('loops', [])
                    for loop in loops:
                        profile_curves = loop.get('profile_curves', [])
                        for curve in profile_curves:
                            curve_id = curve.get('curve')
                            curve_type = curve.get('type')
                            
                            if curve_id and curve_type:
                                if curve_id in entities and entities[curve_id]['sketch_id'] != entity_id:
                                    ambiguous.add(curve_id)
                                entities[curve_id] = {
                                    'type': curve_type,
                                    'data': curve,
                                    'transform': origin,  # 保存Sketch的origin
                                    'sketch_id': entity_id  # 记录所属Sketch
                                }
        
        for identifier in ambiguous:
            entities.pop(identifier, None)
        return entities
    
    def extract_yml_entities(self) -> Dict[str, Dict]:
        """
        从YML文件中提取实线图元（isConstruction=False）及其几何参数
        
        Returns:
            字典：entity_id -> {geometry, type, isConstruction}
        """
        entities = {}
        ambiguous = set()
        
        if 'features' not in self.yml_data:
            return entities
        
        for feature in self.yml_data['features']:
            if 'message' in feature:
                message = feature['message']
                feature_id = message.get('featureId')  # 获取Feature ID
                feature_entities = message.get('entities', [])
                
                for entity in feature_entities:
                    if 'message' not in entity:
                        continue
                    
                    entity_msg = entity['message']
                    entity_id = entity_msg.get('entityId')
                    is_construction = entity_msg.get('isConstruction', False)
                    geometry = entity_msg.get('geometry')
                    type_name = entity.get('typeName')
                    
                    # 只保留非构造线和非点的实体
                    if (entity_id and not is_construction and 
                        type_name not in ['BTMSketchPoint'] and
                        geometry is not None):
                        if entity_id in entities and entities[entity_id]['feature_id'] != feature_id:
                            ambiguous.add(entity_id)
                        entities[entity_id] = {
                            'geometry': geometry,
                            'type': type_name,
                            'isConstruction': is_construction,
                            'entity_msg': entity_msg,
                            'feature_id': feature_id  # 记录所属Feature
                        }
        
        for identifier in ambiguous:
            entities.pop(identifier, None)
        return entities
    
    def match_geometry(self, json_geom: Dict, yml_geom: Dict, yml_entity_msg: Dict = None, 
                      json_transform: Dict = None) -> bool:
        """
        比较两个几何参数是否匹配
        
        Args:
            json_geom: JSON中的几何参数
            yml_geom: YML中的几何参数
            yml_entity_msg: YML实体的完整消息（包含startParam/endParam）
            json_transform: JSON Sketch的坐标变换（origin）
            
        Returns:
            是否匹配
        """
        if not json_geom or not yml_geom:
            return False
        
        json_type = json_geom.get('type')
        yml_type_name = yml_geom.get('typeName')
        
        # 类型映射
        type_mapping = {
            'Line3D': 'BTCurveGeometryLine',
            'Arc3D': 'BTCurveGeometryCircle',
            'Circle3D': 'BTCurveGeometryCircle'
        }
        
        if json_type not in type_mapping:
            return False
        
        if type_mapping[json_type] != yml_type_name:
            return False
        
        # 根据类型比较几何参数
        if json_type == 'Line3D':
            return self._match_line(json_geom, yml_geom, yml_entity_msg, json_transform)
        elif json_type in ['Arc3D', 'Circle3D']:
            return self._match_arc_or_circle(json_geom, yml_geom, json_transform, yml_entity_msg=yml_entity_msg)
        
        return False
    
    def _match_line(self, json_geom: Dict, yml_geom: Dict, yml_entity_msg: Dict = None, 
                    json_transform: Dict = None, tol: float = None) -> bool:
        """
        匹配直线
        直线段由参数方程定义：P(t) = (pntX, pntY) + t·(dirX, dirY)
        起点 = pnt + startParam * dir
        终点 = pnt + endParam * dir
        
        支持两种匹配模式：
        1. 精确匹配：YML线段 = JSON线段（端点重合）
        2. 部分匹配：JSON线段是YML线段的子集（JSON被打断的情况）
        """
        tol = 1e-5 / self.scale_factor if tol is None else tol
        if not yml_entity_msg or not all(k in yml_entity_msg for k in ('startParam', 'endParam')):
            return False
        json_start = json_geom.get('start_point', {})
        json_end = json_geom.get('end_point', {})
        
        yml_msg = yml_geom.get('message', {})
        yml_dir = np.array([yml_msg.get('dirX', 0), yml_msg.get('dirY', 0)])
        yml_pnt = np.array([yml_msg.get('pntX', 0), yml_msg.get('pntY', 0)])
        
        # 使用startParam和endParam计算YML线段的实际起点和终点
        start_param = yml_entity_msg.get('startParam', 0)
        end_param = yml_entity_msg.get('endParam', 0)
        yml_start = yml_pnt + start_param * yml_dir
        yml_end = yml_pnt + end_param * yml_dir
        
        # 计算JSON线段的端点（局部坐标）
        json_start_arr = np.array([json_start.get('x', 0), json_start.get('y', 0)])
        json_end_arr = np.array([json_end.get('x', 0), json_end.get('y', 0)])
        
        # Both exports describe the same sketch-local plane; world origin is irrelevant.

        # 模式1: 精确匹配（端点重合）
        
        if ((np.linalg.norm(json_start_arr - yml_start) <= tol and np.linalg.norm(json_end_arr - yml_end) <= tol)
                or (np.linalg.norm(json_start_arr - yml_end) <= tol and np.linalg.norm(json_end_arr - yml_start) <= tol)):
            return True
        
        # 模式2: 部分匹配（JSON是YML的子集）
        # 检查JSON线段是否与YML线段共线，且JSON的端点在YML线段上
        yml_vec = yml_end - yml_start
        yml_length = np.linalg.norm(yml_vec)
        
        if yml_length < tol:
            return False
        
        yml_dir_normalized = yml_vec / yml_length
        
        # 检查JSON的方向是否与YML平行
        json_vec = json_end_arr - json_start_arr
        json_length = np.linalg.norm(json_vec)
        
        if json_length < tol:
            return False
        
        json_dir_normalized = json_vec / json_length
        
        # 方向平行检查（允许反向）
        if abs(abs(np.dot(json_dir_normalized, yml_dir_normalized)) - 1.0) > 0.01:
            return False
        
        # 检查JSON的两个端点是否都在YML线段上（或其延长线上，但在容差范围内）
        def point_on_segment(point, seg_start, seg_end, tolerance):
            """检查点是否在线段上（含端点）"""
            seg_vec = seg_end - seg_start
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < tolerance:
                return np.linalg.norm(point - seg_start) < tolerance
            
            seg_dir = seg_vec / seg_len
            to_point = point - seg_start
            projection = np.dot(to_point, seg_dir)
            
            # 投影在线段范围内（允许小误差）
            if projection < -tolerance or projection > seg_len + tolerance:
                return False
            
            # 点到线段的垂直距离
            closest_point = seg_start + projection * seg_dir
            distance = np.linalg.norm(point - closest_point)
            
            return distance < tolerance
        
        json_start_on_yml = point_on_segment(json_start_arr, yml_start, yml_end, tol)
        json_end_on_yml = point_on_segment(json_end_arr, yml_start, yml_end, tol)
        
        return json_start_on_yml and json_end_on_yml
    
    def _match_arc_or_circle(self, json_geom: Dict, yml_geom: Dict, json_transform: Dict = None, tol: float = None, yml_entity_msg: Dict = None) -> bool:
        """
        匹配圆弧或圆
        
        支持两种匹配模式：
        1. 精确匹配：圆心和半径相同
        2. 部分匹配：JSON圆弧是YML圆弧的子集（共圆心共半径）
        """
        tol = 1e-5 / self.scale_factor if tol is None else tol
        # JSON中的圆心和半径
        json_center = json_geom.get('center_point', {})
        json_radius = json_geom.get('radius', 0)
        
        # YML中的圆心和半径
        yml_msg = yml_geom.get('message', {})
        yml_center = np.array([yml_msg.get('xCenter', 0), yml_msg.get('yCenter', 0)])
        yml_radius = yml_msg.get('radius', 0)
        
        json_center_arr = np.array([json_center.get('x', 0), json_center.get('y', 0)])
        
        # 比较圆心和半径
        center_dist = np.linalg.norm(json_center_arr - yml_center)
        radius_diff = abs(json_radius - yml_radius)
        
        # Circle identity and trim containment must both match.
        return (center_dist < tol and radius_diff < tol
                and arc_is_contained(json_geom, yml_geom, yml_entity_msg, tol))
    
    def _filter_composite_ids(self, json_ids: list) -> list:
        """
        过滤组合ID中的重复局部ID
        如果列表中有组合ID（包含'+'），则移除该组合ID中包含的所有局部ID
        
        例如：['JG9+JG5+JG1', 'JG9', 'JG5'] -> ['JG9+JG5+JG1']
        """
        # 找出所有组合ID
        composite_ids = [jid for jid in json_ids if '+' in jid]
        
        if not composite_ids:
            return json_ids
        
        # 收集组合ID中包含的所有局部ID
        parts_in_composites = set()
        for comp_id in composite_ids:
            parts = comp_id.split('+')
            parts_in_composites.update(parts)
        
        # 过滤：保留组合ID和不在组合中的独立ID
        filtered = []
        for jid in json_ids:
            if '+' in jid:
                # 保留组合ID
                filtered.append(jid)
            elif jid not in parts_in_composites:
                # 保留不在任何组合中的独立ID
                filtered.append(jid)
            # 跳过在组合中的局部ID
        
        return filtered
    
    def build_id_mapping(self):
        """
        建立JSON和YML之间的ID映射
        支持：
        1. 一对多映射：一个YML图元可能对应多个JSON图元（被打断的情况）
        2. 只在同一个Sketch/Feature内匹配（避免跨Sketch的错误匹配）
        """
        json_entities = self.extract_json_entities()
        yml_entities = self.extract_yml_entities()
        
        print(f"JSON实线数量: {len(json_entities)}")
        print(f"YML实线数量: {len(yml_entities)}")
        
        # 按Sketch/Feature分组
        json_by_sketch = {}
        for json_id, json_data in json_entities.items():
            sketch_id = json_data.get('sketch_id')
            if sketch_id not in json_by_sketch:
                json_by_sketch[sketch_id] = {}
            json_by_sketch[sketch_id][json_id] = json_data
        
        yml_by_feature = {}
        for yml_id, yml_data in yml_entities.items():
            feature_id = yml_data.get('feature_id')
            if feature_id not in yml_by_feature:
                yml_by_feature[feature_id] = {}
            yml_by_feature[feature_id][yml_id] = yml_data
        
        # 在同一个Sketch/Feature内进行匹配
        matched_count = 0
        total_json_used = set()
        
        for feature_id, feature_yml_entities in yml_by_feature.items():
            # Feature ID 应该对应 Sketch ID
            sketch_json_entities = json_by_sketch.get(feature_id, {})
            
            if not sketch_json_entities:
                print(f"警告: YML Feature {feature_id} 没有对应的JSON Sketch")
                continue
            
            # 在当前Sketch/Feature内匹配
            for yml_id, yml_data in feature_yml_entities.items():
                yml_geom = yml_data['geometry']
                yml_entity_msg = yml_data['entity_msg']
                
                # 收集所有匹配的JSON图元（仅限当前Sketch）
                matching_json_ids = []
                
                for json_id, json_data in sketch_json_entities.items():
                    json_geom_data = json_data['data']
                    json_transform = json_data.get('transform')  # 获取坐标变换
                    
                    if self.match_geometry(json_geom_data, yml_geom, yml_entity_msg, json_transform):
                        matching_json_ids.append(json_id)
                
                # 如果找到匹配，优化并建立映射
                if matching_json_ids:
                    # 去除重复：如果有组合ID（包含'+'），则移除其中的局部ID
                    filtered_ids = self._filter_composite_ids(matching_json_ids)
                    
                    if not source_is_covered(yml_data,
                                             [sketch_json_entities[jid]['data'] for jid in filtered_ids],
                                             1e-5 / self.scale_factor):
                        continue
                    self.id_mapping[yml_id] = filtered_ids
                    
                    # 统计消耗：所有匹配的JSON实体都算作被消耗（包括组合ID中的局部ID）
                    total_json_used.update(matching_json_ids)
                    matched_count += 1
                    
                    if len(filtered_ids) == 1:
                        print(f"匹配: {yml_id} -> {filtered_ids[0]}")
                    else:
                        print(f"匹配: {yml_id} -> [{', '.join(filtered_ids)}] (组合)")
        
        # 统计匹配完成度
        total_json_in_features = sum(len(entities) for entities in json_by_sketch.values())
        json_unused = total_json_in_features - len(total_json_used)
        yml_unmatched = len(yml_entities) - matched_count
        
        print(f"\n成功匹配: {matched_count}/{len(yml_entities)}")
        print(f"JSON实体使用: {len(total_json_used)}/{total_json_in_features}, 剩余: {json_unused}")
        print(f"YML实体匹配: {matched_count}/{len(yml_entities)}, 未匹配: {yml_unmatched}")

        # 输出未匹配的YML实体
        unmatched_yml = [yml_id for yml_id in yml_entities.keys() if yml_id not in self.id_mapping]
        if unmatched_yml:
            print(f"\n未匹配的YML实体:")
            for yml_id in unmatched_yml:
                print(f"  - {yml_id}")
        
        # 返回统计信息（用于batch处理）
        return {
            'json_total': total_json_in_features,
            'json_used': len(total_json_used),
            'json_remaining': json_unused,
            'yml_total': len(yml_entities),
            'yml_matched': matched_count,
            'yml_remaining': yml_unmatched
        }
    
    def extract_constraints(self) -> List[Dict]:
        """
        提取约束信息
        
        Returns:
            约束列表
        """
        constraints = []
        
        if 'features' not in self.yml_data:
            return constraints
        
        # 获取所有实线ID（YML中的）
        yml_entities = self.extract_yml_entities()
        solid_line_ids = set(yml_entities.keys())
        
        total_constraints = 0
        type_filtered = 0
        entity_filtered = 0
        mapping_filtered = 0
        coincident_filtered = 0  # COINCIDENT非圆心重合的过滤数
        local_filtered = 0  # 局部约束过滤数
        
        for feature in self.yml_data['features']:
            if 'message' not in feature:
                continue
            
            message = feature['message']
            constraint_list = message.get('constraints', [])
            
            for constraint in constraint_list:
                if 'message' not in constraint:
                    continue
                
                total_constraints += 1
                constraint_msg = constraint['message']
                constraint_type = constraint_msg.get('constraintType')
                
                # 检查约束类型是否支持
                if constraint_type not in self.supported_constraints:
                    type_filtered += 1
                    continue
                
                # 提取约束相关的实体ID
                entity_ids = self._extract_constraint_entities(constraint_msg)
                
                # 检查所有实体是否都是实线
                if not entity_ids:
                    entity_filtered += 1
                    continue
                
                # 过滤掉包含构造线或点的约束
                valid_entities = []
                for entity_id in entity_ids:
                    # 去掉可能的后缀（如.start, .end, .center等）
                    base_entity_id = self._get_base_entity_id(entity_id, solid_line_ids)
                    
                    if (base_entity_id in solid_line_ids
                            and yml_entities[base_entity_id]['feature_id'] == message.get('featureId')):
                        valid_entities.append(entity_id)
                
                # 只有当所有涉及的实体都是实线时才保留
                if len(valid_entities) != len(entity_ids) or len(valid_entities) == 0:
                    entity_filtered += 1
                    continue  # 跳过包含构造线/点的约束
                
                # Center coincidence becomes CONCENTRIC; endpoint coincidence retains its location.
                if constraint_type == 'COINCIDENT':
                    # 检查是否所有实体都是圆心（.center后缀）
                    is_concentric = all('.center' in eid for eid in valid_entities)
                    
                    if is_concentric and len(valid_entities) >= 2:
                        # 圆心重合且至少2个实体 -> 转换为CONCENTRIC
                        constraint_type = 'CONCENTRIC'
                    else:
                        pass  # Endpoint identity is resolved against the emitted primitive geometry below.
                
                # 过滤局部约束（涉及.start, .end等局部点）
                has_local_constraint = any(
                    any(suffix in eid for suffix in ['.start', '.end', '.startSnap', '.endSnap'])
                    for eid in valid_entities
                )
                
                if has_local_constraint and constraint_type not in {'COINCIDENT', 'TANGENT'}:
                    local_filtered += 1
                    continue
                
                # VERTICAL和HORIZONTAL只针对线段整体或圆心，不应包含端点后缀
                # 允许.center后缀（圆心对齐），但不允许其他后缀
                if constraint_type in ['VERTICAL', 'HORIZONTAL']:
                    # 检查1: 是否有不允许的后缀（端点后缀）
                    has_invalid_suffix = False
                    for eid in valid_entities:
                        base_id = self._get_base_entity_id(eid, solid_line_ids)
                        if eid != base_id:
                            suffix = eid[len(base_id):]
                            # 只允许.center后缀，不允许其他后缀
                            if not suffix.startswith('.center'):
                                has_invalid_suffix = True
                                break
                    
                    if has_invalid_suffix:
                        local_filtered += 1  # 计入局部约束过滤
                        continue
                    
                    # 检查2: 如果所有实体都是点（都带.center），则至少需要2个点
                    # 单个点的竖直/水平约束没有意义
                    all_are_points = all(
                        eid != self._get_base_entity_id(eid, solid_line_ids) and '.center' in eid
                        for eid in valid_entities
                    )
                    if all_are_points and len(valid_entities) < 2:
                        local_filtered += 1
                        continue
                
                if True:  # 进入映射处理
                    # 将YML ID映射到JSON ID
                    mapped_entities = []
                    for entity_id in valid_entities:
                        base_entity_id = self._get_base_entity_id(entity_id, solid_line_ids)
                        if base_entity_id in self.id_mapping:
                            # 获取映射的JSON ID（可能是列表，表示被打断的情况）
                            json_ids = self.id_mapping[base_entity_id]
                            # 如果不是列表，转成列表处理
                            if not isinstance(json_ids, list):
                                json_ids = [json_ids]
                            
                            # 保留后缀
                            suffix = entity_id[len(base_entity_id):]
                            if suffix in {'.start', '.end'}:
                                mapped_ids = map_endpoint(yml_entities[base_entity_id], suffix[1:],
                                                          json_ids, self.primitive_endpoints, self.scale_factor)
                            elif suffix in {'', '.center'}:
                                mapped_ids = [jid + suffix for jid in json_ids]
                            else:
                                mapped_ids = None  # Unknown local-point semantics are not evidence.
                            mapped_entities.append({
                                'yml_id': entity_id,
                                'json_ids': mapped_ids,
                                'base_json_ids': sorted({jid.split('.', 1)[0] for jid in mapped_ids}) if mapped_ids else None,
                                'is_combined': len(json_ids) > 1  # 是否为组合（被打断）
                            })
                        else:
                            # 如果没有映射，保留原ID
                            mapped_entities.append({
                                'yml_id': entity_id,
                                'json_ids': None,
                                'base_json_ids': None,
                                'is_combined': False
                            })
                    
                    # 只有当所有实体都成功映射时才添加约束
                    if all(e['json_ids'] is not None for e in mapped_entities):
                        # 如果指定了valid_tags，检查约束涉及的JSON ID是否都在valid_tags中
                        if self.valid_tags is not None:
                            # 收集约束中所有的base_json_ids
                            all_json_ids = []
                            for entity in mapped_entities:
                                if entity['base_json_ids']:
                                    all_json_ids.extend(entity['base_json_ids'])
                            
                            # 检查是否所有JSON ID都在valid_tags中
                            if not all(jid in self.valid_tags for jid in all_json_ids):
                                mapping_filtered += 1
                                continue
                            
                            # if any(entity['is_combined'] for entity in mapped_entities):
                            #     mapping_filtered += 1
                            #     continue

                        # 构建约束数据
                        constraint_data = {
                            'type': constraint_type,
                            'entities': mapped_entities
                        }
                        
                        # 如果是尺寸约束，提取数值
                        if constraint_type in self.dimensional_constraints or constraint_type in self.angular_constraints:
                            dimension_value = self._extract_dimension_value(constraint_msg, constraint_type, entity_ids)
                            if dimension_value is not None:
                                if constraint_type in self.angular_constraints:
                                    # 角度约束：保存原始度数值
                                    constraint_data['value_degrees'] = round(dimension_value, 6)
                                else:
                                    # 长度约束：保存原始mm值和归一化值
                                    constraint_data['value_mm'] = round(dimension_value, 6)
                                    
                                    # 计算归一化后的值（与test_model.py中的几何保持同一尺度）
                                    # value_mm -> 米 -> 归一化
                                    value_m = dimension_value / 1000.0
                                    value_normalized = value_m * self.scale_factor
                                    constraint_data['value_normalized'] = round(value_normalized, 6)
                        
                        # 如果是MIRROR约束，提取镜像轴信息
                        if constraint_type == 'MIRROR':
                            mirror_axis_info = self._extract_mirror_axis(constraint_msg)
                            if mirror_axis_info:
                                constraint_data['mirror_axis'] = mirror_axis_info
                        
                        constraints.append(constraint_data)
                    else:
                        mapping_filtered += 1
        
        # 输出约束过滤统计
        print(f"\n约束提取统计:")
        print(f"  总约束数: {total_constraints}")
        print(f"  类型过滤: {type_filtered} (不支持的类型)")
        print(f"  实体过滤: {entity_filtered} (包含构造线/点)")
        print(f"  COINCIDENT过滤: {coincident_filtered} (非圆心重合)")
        print(f"  局部约束过滤: {local_filtered} (.start/.end等)")
        filter_reason = "映射失败"
        if self.valid_tags is not None:
            filter_reason += "/TAG不在dedup中"
        print(f"  映射过滤: {mapping_filtered} ({filter_reason})")
        print(f"  成功提取: {len(constraints)}")
        if self.valid_tags is not None:
            print(f"  有效TAG数量: {len(self.valid_tags)}")
        
        return constraints
    
    def _get_base_entity_id(self, entity_id: str, solid_line_ids: set) -> str:
        """
        智能识别基础实体ID
        
        某些实体ID本身包含数字后缀（如 entityId.0, entityId.1），
        而约束可能引用这些实体的点（如 entityId.0.start）。
        
        Args:
            entity_id: 完整的实体ID（可能包含后缀）
            solid_line_ids: 实线ID集合
            
        Returns:
            基础实体ID
        """
        # 先检查完整ID是否就是实线
        if entity_id in solid_line_ids:
            return entity_id
        
        # 尝试逐步去掉后缀，找到匹配的实线ID
        parts = entity_id.split('.')
        
        # 从最长到最短尝试组合
        for i in range(len(parts), 0, -1):
            candidate = '.'.join(parts[:i])
            if candidate in solid_line_ids:
                return candidate
        
        # 如果都没找到，返回第一个部分（原来的逻辑）
        return parts[0]
    
    def _extract_constraint_entities(self, constraint_msg: Dict) -> List[str]:
        """从约束消息中提取实体ID"""
        entity_ids = []
        parameters = constraint_msg.get('parameters', [])
        
        # 首先检查是否是MIRROR约束，如果是，需要排除镜像轴参数
        constraint_type = constraint_msg.get('constraintType')
        mirror_axis_param_id = None
        
        if constraint_type == 'MIRROR':
            # 查找镜像轴参数ID
            for param in parameters:
                if 'message' not in param:
                    continue
                param_msg = param['message']
                param_id = param_msg.get('parameterId', '')
                
                # 情况1: 明确标注为 localMirror
                if 'mirror' in param_id.lower() and 'type' not in param_id.lower():
                    mirror_axis_param_id = param_id
                    break
            
            # 情况2: 如果没有找到 localMirror，检查是否有 local2（这是另一种命名约定）
            if not mirror_axis_param_id:
                for param in parameters:
                    if 'message' not in param:
                        continue
                    param_msg = param['message']
                    param_id = param_msg.get('parameterId', '')
                    if param_id == 'local2':
                        mirror_axis_param_id = 'local2'
                        break
        
        # 提取实体ID，排除镜像轴参数
        for param in parameters:
            if 'message' not in param:
                continue
            
            param_msg = param['message']
            param_id = param_msg.get('parameterId', '')
            
            # 跳过镜像轴参数
            if mirror_axis_param_id and param_id == mirror_axis_param_id:
                continue
            
            # 查找包含实体ID的参数
            # 支持: localFirst, localSecond, local0, local1, localEntity1, localEntity2等
            if any(keyword in param_id.lower() for keyword in ['first', 'second', 'master', 'offset', 'entity', 'local']):
                value = param_msg.get('value')
                if value and isinstance(value, str):
                    entity_ids.append(value)
        
        return entity_ids
    
    def _extract_mirror_axis(self, constraint_msg: Dict) -> Optional[Dict]:
        """
        提取MIRROR约束的镜像轴信息
        
        Args:
            constraint_msg: 约束消息字典
            
        Returns:
            镜像轴信息字典，包含yml_id, point_normalized, direction_normalized, equation
        """
        # 从参数中查找镜像轴参数
        mirror_axis_id = None
        parameters = constraint_msg.get('parameters', [])
        
        # 情况1: 查找 localMirror 参数（明确标注的镜像轴）
        for param in parameters:
            if 'message' not in param:
                continue
            
            param_msg = param['message']
            param_id = param_msg.get('parameterId', '')
            
            # 查找镜像轴参数（parameterId 包含 'mirror'，但排除 sketchToolType）
            if 'mirror' in param_id.lower() and 'type' not in param_id.lower():
                value = param_msg.get('value')
                if value and isinstance(value, str):
                    mirror_axis_id = value
                    break
        
        # 情况2: 如果没有找到 localMirror，则查找 local2（另一种命名约定）
        if not mirror_axis_id:
            for param in parameters:
                if 'message' not in param:
                    continue
                
                param_msg = param['message']
                param_id = param_msg.get('parameterId', '')
                
                if param_id == 'local2':
                    value = param_msg.get('value')
                    if value and isinstance(value, str):
                        mirror_axis_id = value
                        break
        
        if not mirror_axis_id:
            return None
        
        # 直接从YML数据中查找镜像轴（包括构造线）
        if 'features' not in self.yml_data:
            return None
        
        mirror_axis_entity = None
        for feature in self.yml_data['features']:
            if 'message' not in feature:
                continue
            
            message = feature['message']
            feature_entities = message.get('entities', [])
            
            for entity in feature_entities:
                if 'message' not in entity:
                    continue
                
                entity_msg = entity['message']
                entity_id = entity_msg.get('entityId')
                
                if entity_id == mirror_axis_id or (entity_id and entity_id.startswith(mirror_axis_id)):
                    mirror_axis_entity = entity_msg
                    break
            
            if mirror_axis_entity:
                break
        
        if not mirror_axis_entity:
            return None
        
        # 获取几何信息
        geometry = mirror_axis_entity.get('geometry')
        if not geometry or 'message' not in geometry:
            return None
        
        geom_msg = geometry['message']
        type_name = geometry.get('typeName')
        
        # 只处理线段（BTCurveGeometryLine）
        if type_name != 'BTCurveGeometryLine':
            return None
        
        # 提取线段的参数方程：P(t) = (pntX, pntY) + t * (dirX, dirY)
        pnt_x = geom_msg.get('pntX', 0)
        pnt_y = geom_msg.get('pntY', 0)
        dir_x = geom_msg.get('dirX', 0)
        dir_y = geom_msg.get('dirY', 0)
        
        # 获取线段的起点和终点参数
        start_param = mirror_axis_entity.get('startParam', 0)
        end_param = mirror_axis_entity.get('endParam', 1)
        
        # 计算线段的中点（作为代表点）
        mid_param = (start_param + end_param) / 2
        point_x = pnt_x + mid_param * dir_x
        point_y = pnt_y + mid_param * dir_y
        
        # 归一化点坐标（应用scale_factor）
        point_normalized = [
            float(round(point_x * self.scale_factor, 6)),
            float(round(point_y * self.scale_factor, 6))
        ]
        
        # 方向向量归一化（单位向量）
        direction_length = np.sqrt(dir_x**2 + dir_y**2)
        if direction_length < 1e-10:
            return None
        
        direction_normalized = [
            float(round(dir_x / direction_length, 6)),
            float(round(dir_y / direction_length, 6))
        ]
        
        # 计算直线方程 ax + by + c = 0
        # 从点向式推导：方向向量 (dx, dy)，点 (x0, y0)
        # 法向量 (-dy, dx)，直线方程：-dy*(x-x0) + dx*(y-y0) = 0
        # 即：-dy*x + dx*y + (dy*x0 - dx*y0) = 0
        # 归一化后的坐标
        x0, y0 = point_normalized
        dx, dy = direction_normalized
        
        a = -dy
        b = dx
        c = dy * x0 - dx * y0
        
        # 归一化方程系数（使得 a^2 + b^2 = 1）
        norm = np.sqrt(a**2 + b**2)
        if norm > 1e-10:
            a = float(round(a / norm, 6))
            b = float(round(b / norm, 6))
            c = float(round(c / norm, 6))
        
        equation = f"{a}*x + {b}*y + {c}"
        
        return {
            'yml_id': mirror_axis_id,
            'point_normalized': point_normalized,
            'direction_normalized': direction_normalized,
            'equation': equation
        }
    
    def _parse_expression_to_mm(self, expression: str) -> Optional[float]:
        """
        解析expression字符串并转换为mm
        
        支持的格式：
        - 单一单位：'10 mm', '2 inch', 'in'（当作1 in）
        - 复数单位：'48 inches', '24 feet'
        - 多单位相加：'24*foot+4*inch', '7.5 cm + .5 in'
        - 数学常量：'2*pi cm', '(pi/2) cm'
        - 数学函数：'sqrt(5) in'（需要支持math库）
        
        Args:
            expression: 表达式字符串
            
        Returns:
            转换后的mm值，如果解析失败返回None
        """
        if not expression:
            return None
        
        try:
            expr = expression.strip()
            
            # 处理只有单位的情况（如 'in' -> '1 in'）
            unit_only_pattern = r'^(millimeter|centimeter|meter|foot|feet|inch|inches|mm|cm|ft|in|m)s?$'
            if re.match(unit_only_pattern, expr, flags=re.IGNORECASE):
                expr = '1 ' + expr
            
            # 支持数学常量
            expr = expr.replace('pi', str(math.pi))
            
            # 检查是否包含加减运算（多单位相加）
            # 需要在单位外部进行加减（如 '24*foot+4*inch'）
            if self._has_multi_unit_operation(expr):
                return self._parse_multi_unit_expression(expr)
            
            # 单一单位的情况，使用原来的逻辑
            return self._parse_single_unit_expression(expr)
                
        except Exception as e:
            print(f"Warning: Failed to parse expression '{expression}': {e}")
            return None
    
    def _has_multi_unit_operation(self, expr: str) -> bool:
        """检查表达式是否包含多单位运算（如 '24*foot+4*inch'）"""
        # 单位列表
        units = ['millimeter', 'centimeter', 'meter', 'foot', 'feet', 'inch', 'inches', 'mm', 'cm', 'ft', 'in', 'm']
        
        # 检查是否有多个单位
        unit_count = 0
        for unit in units:
            # 使用单词边界确保完整匹配
            pattern = r'(?<![a-zA-Z])' + re.escape(unit) + r's?(?![a-zA-Z])'
            if re.search(pattern, expr, flags=re.IGNORECASE):
                unit_count += 1
                if unit_count > 1:
                    return True
        
        return False
    
    def _parse_multi_unit_expression(self, expr: str) -> Optional[float]:
        """
        解析包含多个单位的表达式（如 '24*foot+4*inch', '7.5 cm + .5 in'）
        
        策略：分割加减运算，分别解析每部分，然后相加/相减
        注意：需要考虑括号内的运算，不能简单分割
        """
        # 分割表达式，保留运算符，但要考虑括号
        # 使用正则表达式分割，只在括号外的+/-处分割
        parts = []
        current_part = ''
        paren_depth = 0
        
        for i, char in enumerate(expr):
            if char == '(':
                paren_depth += 1
                current_part += char
            elif char == ')':
                paren_depth -= 1
                current_part += char
            elif char in ['+', '-'] and paren_depth == 0:
                # 在括号外的+/-处分割
                if current_part.strip():
                    parts.append(current_part.strip())
                parts.append(char)
                current_part = ''
            else:
                current_part += char
        
        # 添加最后一部分
        if current_part.strip():
            parts.append(current_part.strip())
        
        total_mm = 0.0
        current_sign = 1.0  # 1表示正，-1表示负
        
        for part in parts:
            if part == '+':
                current_sign = 1.0
            elif part == '-':
                current_sign = -1.0
            else:
                # 解析这一部分（单一单位或括号内的表达式）
                value_mm = self._parse_single_unit_expression(part)
                if value_mm is None:
                    return None
                total_mm += current_sign * value_mm
        
        return total_mm
    
    def _parse_single_unit_expression(self, expr: str) -> Optional[float]:
        """
        解析单一单位的表达式（如 '10 mm', '0.5*inch', '2*pi cm'）
        支持表达式内部的角度单位和三角函数（如 '(400-81.5*cos(45 deg)) mm'）
        
        注意：如果是括号内的多单位表达式（如 '(31ft+7in-20ft-11in)'），
        需要递归调用_parse_multi_unit_expression
        """
        try:
            # 移除多余空格
            expr = expr.strip()
            
            # 如果是括号包裹的多单位表达式，去掉括号后递归处理
            if expr.startswith('(') and expr.endswith(')'):
                inner_expr = expr[1:-1].strip()
                # 检查是否包含多个单位
                if self._has_multi_unit_operation(inner_expr):
                    return self._parse_multi_unit_expression(inner_expr)
            
            # 预处理：先处理表达式内部的角度单位（deg/degree），将其转换为弧度
            # 匹配：数字 + 可选空格 + deg/degree（不在表达式末尾）
            # 例如：cos(45 deg) -> cos(math.radians(45))
            def replace_angle_to_radians(match):
                angle_value = match.group(1)
                return f'math.radians({angle_value})'
            
            # 查找三角函数内的角度：func(数字 deg/degree)
            expr = re.sub(r'(\d+(?:\.\d+)?)\s*(?:deg|degree)(?!\w)', replace_angle_to_radians, expr, flags=re.IGNORECASE)
            
            # 检测单位并转换
            # 注意：按长度从长到短排序，避免短单位被误识别
            # 包含复数形式：inches, feet, yards
            unit_conversion = {
                'millimeters': 1.0,
                'millimeter': 1.0,
                'centimeters': 10.0,
                'centimeter': 10.0,
                'meters': 1000.0,
                'meter': 1000.0,
                'yards': 914.4,
                'yard': 914.4,
                'inches': 25.4,
                'inch': 25.4,
                'feet': 304.8,
                'foot': 304.8,
                'yd': 914.4,
                'mm': 1.0,
                'cm': 10.0,
                'ft': 304.8,
                'in': 25.4,
                'm': 1000.0,
            }
            
            # 查找单位（按长度从长到短匹配，避免短单位覆盖长单位）
            unit = None
            conversion_factor = 1.0
            
            # 按单位名称长度从长到短排序
            sorted_units = sorted(unit_conversion.items(), key=lambda x: len(x[0]), reverse=True)
            
            for unit_name, factor in sorted_units:
                # 使用负向前瞻/后顾确保单位前后不是字母（支持'10mm'这样没有空格的格式）
                pattern = r'(?<![a-zA-Z])' + re.escape(unit_name) + r'(?![a-zA-Z])'
                if re.search(pattern, expr, flags=re.IGNORECASE):
                    unit = unit_name
                    conversion_factor = factor
                    break
            
            # 替换单位为空，保留数学表达式
            if unit:
                # 使用正则表达式移除单位（不区分大小写，支持没有空格的情况如'10mm'）
                pattern = r'\s*' + re.escape(unit) + r'(?![a-zA-Z])'
                expr_no_unit = re.sub(pattern, '', expr, flags=re.IGNORECASE)
                
                # 清理表达式中多余的空格和运算符
                # 1. 移除末尾的空格和运算符（如 "0.02*in" -> "0.02*" -> "0.02"）
                expr_no_unit = re.sub(r'[\+\-\*\/\s]+$', '', expr_no_unit)
                
                # 2. 清理括号前的运算符（如 ".5*(2*" -> ".5*(2"，处理 (2*inch) 这种情况）
                # 匹配：左括号 + 数字/小数 + 运算符 + 右括号
                # 例如：(2*) -> (2), (.065*) -> (.065)
                expr_no_unit = re.sub(r'\(([0-9.]+)\s*[\+\-\*\/]+\s*\)', r'(\1)', expr_no_unit)
                
                # 3. 移除括号内的尾随空格（如 "(8 )" -> "(8)"）
                expr_no_unit = re.sub(r'\(\s+', '(', expr_no_unit)
                expr_no_unit = re.sub(r'\s+\)', ')', expr_no_unit)
                
                # 4. 清理多余的空格
                expr_no_unit = ' '.join(expr_no_unit.split())
            else:
                expr_no_unit = expr
            
            # 支持数学函数
            # 1. sqrt -> math.sqrt
            if 'sqrt' in expr_no_unit.lower():
                expr_no_unit = re.sub(r'\bsqrt\b', 'math.sqrt', expr_no_unit, flags=re.IGNORECASE)
            
            # 2. sin/cos/tan -> math.sin/cos/tan
            for func in ['sin', 'cos', 'tan']:
                if func in expr_no_unit.lower():
                    expr_no_unit = re.sub(r'\b' + func + r'\b', f'math.{func}', expr_no_unit, flags=re.IGNORECASE)
            
            # 安全评估数学表达式
            # 允许数字、运算符、括号、科学计数法、字母（用于math.sqrt）和点（用于math.）
            # 简化检查：只要不包含危险字符即可
            if re.match(r'^[a-zA-Z0-9\s\+\-\*\/\(\)\.\^eE]+$', expr_no_unit):
                # 替换^为**（Python的幂运算符）
                expr_no_unit = expr_no_unit.replace('^', '**')
                
                # 移除前导零（如 "00300" -> "300", "020" -> "20"）
                # 但保留小数点前的零（如 "0.5" 不变）
                # 匹配：可选的运算符/括号 + 前导零 + 数字
                expr_no_unit = re.sub(r'(^|[\+\-\*\/\(])\s*0+(\d+)', r'\1\2', expr_no_unit)
                
                # 评估表达式（允许使用math模块）
                value = eval(expr_no_unit, {"__builtins__": {}}, {"math": math})
                # 转换为mm
                value_mm = value * conversion_factor
                return value_mm
            else:
                return None
                
        except Exception as e:
            print(f"Warning: Failed to parse single unit expression '{expr}': {e}")
            return None
    
    def _parse_angle_to_degrees(self, expression: str) -> Optional[float]:
        """
        解析角度表达式并转换为度数
        
        支持的单位：
        - rad (弧度): 转换为度
        - deg/degree (度): 保持
        
        Args:
            expression: 表达式字符串，如 "1.5707963267948966*rad" 或 "90 deg"
            
        Returns:
            转换后的度数值，如果解析失败返回None
        """
        if not expression:
            return None
        
        try:
            expr = expression.strip()
            
            # 检测是否为弧度
            if 'rad' in expr.lower():
                # 移除'rad'并评估表达式
                expr_no_unit = re.sub(r'\s*\*?\s*rad\b', '', expr, flags=re.IGNORECASE)
                if re.match(r'^[\d\s\+\-\*\/\(\)\.\^]+$', expr_no_unit):
                    expr_no_unit = expr_no_unit.replace('^', '**')
                    radians = eval(expr_no_unit)
                    degrees = math.degrees(radians)
                    return degrees
            
            # 检测是否为度数
            elif any(unit in expr.lower() for unit in ['deg', 'degree']):
                expr_no_unit = re.sub(r'\s*(deg|degree)\b', '', expr, flags=re.IGNORECASE)
                if re.match(r'^[\d\s\+\-\*\/\(\)\.\^]+$', expr_no_unit):
                    expr_no_unit = expr_no_unit.replace('^', '**')
                    degrees = eval(expr_no_unit)
                    return degrees
            
            # 默认假设为度数
            else:
                if re.match(r'^[\d\s\+\-\*\/\(\)\.\^]+$', expr):
                    expr = expr.replace('^', '**')
                    degrees = eval(expr)
                    return degrees
                    
        except Exception as e:
            print(f"Warning: Failed to parse angle expression '{expression}': {e}")
            return None
        
        return None
    
    def _extract_dimension_value(self, constraint_msg: Dict, constraint_type: str, entity_ids: List[str] = None) -> Optional[float]:
        """
        从约束消息中提取尺寸数值
        
        Args:
            constraint_msg: 约束消息字典
            constraint_type: 约束类型
            entity_ids: 约束涉及的实体ID列表
            
        Returns:
            转换后的数值（长度单位为mm，角度单位为度）
            
        Raises:
            ValueError: 如果无法从expression中提取数值
        """
        parameters = constraint_msg.get('parameters', [])
        
        # 查找length参数
        for param in parameters:
            if 'message' not in param:
                continue
            
            param_msg = param['message']
            param_id = param_msg.get('parameterId', '')
            
            # 查找长度参数
            if param_id in {'length', 'angle'}:
                expression = param_msg.get('expression', '')
                
                # 必须有expression
                if not expression:
                    entity_ids_str = ', '.join(entity_ids) if entity_ids else 'N/A'
                    raise ValueError(
                        f"约束类型 {constraint_type} 的length参数没有expression!\n"
                        f"  entityId: {constraint_msg.get('entityId')}\n"
                        f"  涉及实体: {entity_ids_str}\n"
                        f"  parameterId: {param_id}\n"
                        f"  value: {param_msg.get('value')}"
                    )
                
                # 解析expression
                if constraint_type in self.angular_constraints:
                    result = self._parse_angle_to_degrees(expression)
                else:
                    result = self._parse_expression_to_mm(expression)
                
                # 如果解析失败，报错
                if result is None:
                    entity_ids_str = ', '.join(entity_ids) if entity_ids else 'N/A'
                    raise ValueError(
                        f"无法解析约束类型 {constraint_type} 的expression: '{expression}'!\n"
                        f"  entityId: {constraint_msg.get('entityId')}\n"
                        f"  涉及实体: {entity_ids_str}\n"
                        f"  parameterId: {param_id}"
                    )
                
                return result
        
        return None
    
    def save_constraints(self, output_path: str):
        """保存约束到JSON文件"""
        constraints = self.extract_constraints()
        
        # 清理约束数据：保留yml_id和完整的json_ids（包含后缀），以及尺寸数值
        cleaned_constraints = []
        for constraint in constraints:
            cleaned_entities = []
            for entity in constraint['entities']:
                cleaned_entities.append({
                    'yml_id': entity['yml_id'],
                    'json_ids': entity['json_ids']  # 使用完整的json_ids（包含.center等后缀）
                })
            
            cleaned_constraint = {
                'type': constraint['type'],
                'entities': cleaned_entities
            }
            
            # 保留尺寸数值字段
            if 'value_mm' in constraint:
                cleaned_constraint['value_mm'] = constraint['value_mm']
            if 'value_normalized' in constraint:
                cleaned_constraint['value_normalized'] = constraint['value_normalized']
            if 'value_degrees' in constraint:
                cleaned_constraint['value_degrees'] = constraint['value_degrees']
            
            # 保留镜像轴信息
            if 'mirror_axis' in constraint:
                cleaned_constraint['mirror_axis'] = constraint['mirror_axis']
            
            cleaned_constraints.append(cleaned_constraint)
        
        # 构建输出数据
        output_data = {
            'metadata': {
                'json_file': self.json_path,
                'yml_file': self.yml_path,
                'total_constraints': len(constraints),
                'id_mapping_count': len(self.id_mapping),
                'scale_factor': self.scale_factor,
                'bbox_center': self.bbox_center.tolist() if self.bbox_center is not None else None
            },
            'id_mapping': self.id_mapping,
            'constraints': cleaned_constraints
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 紧凑格式：使用indent=1减少空行
            json.dump(output_data, f, indent=1, ensure_ascii=False)
        
        print(f"\n约束已保存到: {output_path}")
        print(f"总计提取约束: {len(constraints)}")
        print(f"约束类型统计:")
        
        type_count = {}
        for c in constraints:
            ctype = c['type']
            type_count[ctype] = type_count.get(ctype, 0) + 1
        
        for ctype, count in sorted(type_count.items()):
            print(f"  {ctype}: {count}")
    
    def process(self, output_path: str):
        """完整处理流程"""
        print("="*60)
        print("开始处理CAD约束提取")
        print("="*60)
        
        print("\n1. 加载文件...")
        self.load_files()
        
        print("\n2. 建立ID映射...")
        mapping_stats = self.build_id_mapping()
        
        print("\n3. 提取并保存约束...")
        self.save_constraints(output_path)
        
        print("\n" + "="*60)
        print("处理完成!")
        print("="*60)




def extract_for_sample(json_path: Path, features_path: Path, primitive_path: Path) -> dict:
    """Map local sketch entities and constraints to reconstructed primitive tags."""
    extractor = ConstraintExtractor(str(json_path), str(features_path), str(primitive_path))
    extractor.load_files()
    extractor.build_id_mapping()
    constraints = extractor.extract_constraints()
    return {'constraints': constraints, 'id_mapping': extractor.id_mapping}
