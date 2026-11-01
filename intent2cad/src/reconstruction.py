#!/usr/bin/env python3
"""
json2cadquery.py

Convert parsed JSON CAD files to CADQuery scripts.

The JSON format contains:
- entities: CAD entities with their geometric data
- sequence: ordered list of operations
- properties: metadata

Features:
- Parse sketches with circles, lines, arcs
- Handle extrude operations
- Generate CADQuery code for 3D modeling
- Support for multiple profiles and loops

Usage:
    python json2cadquery.py input.json --out-prefix model_out
"""

import json
import math
import numpy as np
import copy
import re
import os
from pathlib import Path
from collections import defaultdict

# ===== 全局精度设置 =====
# 控制所有坐标的小数位数
COORDINATE_PRECISION = 6  # 输出到生成代码的精度（4位小数）

# 根据精度动态计算容差
# 示例：COORDINATE_PRECISION = 4 时
#   - COORDINATE_TOLERANCE = 0.001 (10^-3) - 用于比较坐标、识别矩形等
#   - ZERO_TOLERANCE = 0.0001 (10^-4) - 用于判断坐标是否接近零
COORDINATE_TOLERANCE = 10 ** (-(COORDINATE_PRECISION - 1))  # 10^-(PRECISION-1)
ZERO_TOLERANCE = 10 ** (-COORDINATE_PRECISION)  # 10^-PRECISION

def format_coord(value):
    """格式化坐标值，截断到指定精度（用于内部计算），避免四舍五入放大 EQUAL 约束的微小差异"""
    v = float(value)
    return math.trunc(v * 10**COORDINATE_PRECISION) / 10**COORDINATE_PRECISION

# def format_coord_str(value):
#     """格式化坐标值为字符串，补齐位数（用于代码输出）"""
#     return f"{float(value):.{COORDINATE_PRECISION}f}"

def format_coord_str(val):
    """
    格式化浮点数：截断到 6 位小数（不四舍五入），避免 EQUAL 约束的相近值被舍入到不同结果。
    去除尾零，避免科学计数法。
    """
    if val == 0:
        return "0"
    truncated = math.trunc(val * 1e6) / 1e6
    formatted = f"{truncated:.6f}".rstrip('0').rstrip('.')
    if formatted == '-0':
        return "0"
    return formatted

def normalize_cad_points(points, norm_factor=0.75):
    """
    Normalize 2D/3D CAD sketch data following the DeepCAD paper's preprocessing strategy.
    
    This normalization involves two steps:
    1. Centering: Translate all points so that the bounding box center is at (0, 0, 0)
    2. Scaling: Uniformly scale all points to fit within [-1, 1] with a safety margin
    
    Args:
        points: Numpy array of shape (N, 3) representing vertices or control points
        norm_factor: Safety margin factor (default: 0.75), prevents points from touching 
                    the exact boundary of [-1, 1]
    
    Returns:
        normalized_points: Normalized points of shape (N, 3)
        translation: Translation vector that was applied (for inverse transformation)
        scale_factor: Uniform scale factor that was applied (for inverse transformation)
    
    Example:
        >>> points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> normalized, translation, scale = normalize_cad_points(points)
        >>> # To reverse: original = normalized / scale - translation
    """
    
    # Input validation
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N, 3), got {points.shape}")
    
    if points.shape[0] == 0:
        raise ValueError("Points array is empty")
    
    # Step 1: Centering
    # Calculate bounding box
    bbox_min = np.min(points, axis=0)  # Shape: (3,)
    bbox_max = np.max(points, axis=0)  # Shape: (3,)
    bbox_center = (bbox_min + bbox_max) / 2.0
    
    # Translate points to center the bounding box at origin
    centered_points = points - bbox_center
    
    # Step 2: Uniform Scaling with safety margin
    # Calculate the maximum absolute distance from origin after centering
    max_abs_distance = np.max(np.abs(centered_points))
    
    # Handle edge case: all points are at the same location
    if max_abs_distance < 1e-10:
        # Points are essentially at the same location, no scaling needed
        scale_factor = 1.0
        normalized_points = centered_points
    else:
        # Compute scale factor: we want max_abs_distance to map to norm_factor
        # This ensures points fit within [-norm_factor, norm_factor] ⊂ [-1, 1]
        scale_factor = norm_factor / max_abs_distance
        
        # Apply uniform scaling
        normalized_points = centered_points * scale_factor
    
    return normalized_points, bbox_center, scale_factor

def denormalize_cad_points(normalized_points, translation, scale_factor):
    """
    Reverse the normalization process to recover original CAD points.
    
    Args:
        normalized_points: Normalized points of shape (N, 3)
        translation: Translation vector that was applied during normalization
        scale_factor: Scale factor that was applied during normalization
    
    Returns:
        original_points: Recovered original points of shape (N, 3)
    
    Example:
        >>> points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> normalized, translation, scale = normalize_cad_points(points)
        >>> recovered = denormalize_cad_points(normalized, translation, scale)
        >>> np.allclose(points, recovered)  # Should be True
    """    
    # Reverse scaling
    centered_points = normalized_points / scale_factor
    
    # Reverse translation
    original_points = centered_points + translation
    
    return original_points

def load_json(path):
    """Load JSON file"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {path}: {e}")
        return None

def get_workplane_from_transform(transform, norm_params=None):
    """
    Determines the correct CadQuery workplane based on the transform block.
    Always returns a custom plane definition to correctly handle origin.
    
    Args:
        transform: Transform data containing origin and axis information
        norm_params: Normalization parameters (if None, assumes mm unit and no normalization)
    """
    if not transform:
        return "XY"  # Default to XY if no transform is provided

    origin = transform.get('origin', {})
    x_axis = transform.get('x_axis', {})
    y_axis = transform.get('y_axis', {})
    z_axis = transform.get('z_axis', {})
    
    # 处理 origin 坐标：如果有归一化参数，应用归一化；否则米转毫米，最终输出为毫米单位
    if norm_params:
        # 应用归一化：(point - translation) * scale
        translation = norm_params['translation']
        scale = norm_params['scale']
        origin_pt = (
            format_coord((origin.get('x', 0.0) - translation[0]) * scale),
            format_coord((origin.get('y', 0.0) - translation[1]) * scale),
            format_coord((origin.get('z', 0.0) - translation[2]) * scale)
        )
    else:
        # 没有归一化参数，原始数据是米，转为毫米
        origin_pt = (
            format_coord(origin.get('x', 0.0) * 1000),
            format_coord(origin.get('y', 0.0) * 1000),
            format_coord(origin.get('z', 0.0) * 1000)
        )
    
    x_dir = (
        format_coord(x_axis.get('x', 1.0)),
        format_coord(x_axis.get('y', 0.0)),
        format_coord(x_axis.get('z', 0.0))
    )
    
    normal_dir = (
        format_coord(z_axis.get('x', 0.0)),
        format_coord(z_axis.get('y', 0.0)),
        format_coord(z_axis.get('z', 1.0))
    )
    
    # Always use custom plane to correctly handle origin in 3D space
    return {
        'type': 'custom',
        'origin': origin_pt,
        'x_dir': x_dir,
        'normal': normal_dir
    }

def create_workplane_code(plane_info):
    """
    Creates CADQuery workplane code based on plane information.
    Returns a string that can be used in CADQuery code.
    """
    if isinstance(plane_info, str):
        # Standard plane
        return f"cq.Workplane('{plane_info}')"
    elif isinstance(plane_info, dict) and plane_info.get('type') == 'custom':
        # Custom plane
        # CADQuery Plane(origin, xDir, normal) - 注意参数顺序！
        origin = plane_info['origin']
        x_dir = plane_info['x_dir']
        normal = plane_info['normal']
        
        return f"cq.Workplane(cq.Plane(cq.Vector{origin}, cq.Vector{x_dir}, cq.Vector{normal}))"
    else:
        # Fallback
        return "cq.Workplane('XY')"

def get_loop_direction(sorted_curves):
    """
    Determines the direction (CW or CCW) of a closed loop of 2D curves
    using the Shoelace formula on its vertices.
    """
    if len(sorted_curves) < 2:
        return 'ccw' # Default for safety

    # 提取构成轮廓的所有顶点 (您的解析器已提供2D点)
    points_2d = [curve['start'] for curve in sorted_curves]
    
    # 应用鞋带公式 (Shoelace formula) 计算多边形的有向面积
    area = 0.0
    for i in range(len(points_2d)):
        p1 = points_2d[i]
        p2 = points_2d[(i + 1) % len(points_2d)] # 回绕到第一个点
        area += (p1[0] * p2[1] - p2[0] * p1[1])

    # 负面积表示顺时针(cw), 正面积表示逆时针(ccw)
    return 'cw' if area < 0 else 'ccw'

def parse_circle(curve_data):
    """Parse circle curve data"""
    center = curve_data.get('center_point', {})
    radius = curve_data.get('radius', 0.0)
    
    center_coords = (
        format_coord(center.get('x', 0.0)),
        format_coord(center.get('y', 0.0))
    )
    radius_coord = format_coord(radius)
    
    return {
        'type': 'circle',
        'center': center_coords,
        'radius': radius_coord,
        'curve': curve_data.get('curve') # 保留原始curve标签
    }

def parse_line(curve_data):
    """Parse line curve data"""
    start = curve_data.get('start_point', {})
    end = curve_data.get('end_point', {})
    
    start_coords = (
        format_coord(start.get('x', 0.0)),
        format_coord(start.get('y', 0.0))
    )
    end_coords = (
        format_coord(end.get('x', 0.0)),
        format_coord(end.get('y', 0.0))
    )
    
    return {
        'type': 'line',
        'start': start_coords,
        'end': end_coords,
        'curve': curve_data.get('curve') # 保留原始curve标签
    }

def parse_arc(curve_data):
    """Parse arc curve data"""
    center = curve_data.get('center_point', {})
    radius = curve_data.get('radius', 0.0)
    start_angle = curve_data.get('start_angle', 0.0)
    end_angle = curve_data.get('end_angle', 0.0)
    
    center_coords = (
        format_coord(center.get('x', 0.0)),
        format_coord(center.get('y', 0.0))
    )
    radius_coord = format_coord(radius)
    
    return {
        'type': 'arc',
        'center': center_coords,
        'radius': radius_coord,
        'start_angle': format_coord(math.degrees(start_angle)),
        'end_angle': format_coord(math.degrees(end_angle)),
        'curve': curve_data.get('curve') # 保留原始curve标签
    }

def collect_sketch_points(profiles_data):
    """
    收集sketch中的所有点位数据，用于归一化处理。
    
    Returns:
        points_list: 所有点的列表 [(x, y, z), ...]
    """
    points_list = []
    
    for profile_id, profile_data in profiles_data.items():
        loops = profile_data.get('loops', [])
        
        for loop in loops:
            curves = loop.get('profile_curves', [])
            
            for curve in curves:
                curve_type = curve.get('type', '')
                
                # 收集各种类型曲线的点位
                if curve_type in ['Arc3D', 'Line3D']:
                    if 'start_point' in curve:
                        pt = curve['start_point']
                        points_list.append([pt.get('x', 0.0), pt.get('y', 0.0), pt.get('z', 0.0)])
                    if 'end_point' in curve:
                        pt = curve['end_point']
                        points_list.append([pt.get('x', 0.0), pt.get('y', 0.0), pt.get('z', 0.0)])
                    if 'center_point' in curve:
                        pt = curve['center_point']
                        points_list.append([pt.get('x', 0.0), pt.get('y', 0.0), pt.get('z', 0.0)])
                
                # 备用逻辑
                elif 'start_point' in curve:
                    pt = curve['start_point']
                    points_list.append([pt.get('x', 0.0), pt.get('y', 0.0), pt.get('z', 0.0)])
                if 'end_point' in curve:
                    pt = curve['end_point']
                    points_list.append([pt.get('x', 0.0), pt.get('y', 0.0), pt.get('z', 0.0)])
                if 'center_point' in curve:
                    pt = curve['center_point']
                    points_list.append([pt.get('x', 0.0), pt.get('y', 0.0), pt.get('z', 0.0)])
    
    if not points_list:
        return np.array([])
    
    return np.array(points_list)


def parse_sketch_profiles(profiles_data):
    """(最终版) 只解析2D点位, 但为Arc3D增加了所有必要的附加信息。"""
    parsed_profiles = []
    
    for profile_id, profile_data in profiles_data.items():
        loops = profile_data.get('loops', [])
        parsed_loops = []
        
        for loop in loops:
            curves = loop.get('profile_curves', [])
            parsed_curves = []
            
            for curve in curves:
                curve_type = curve.get('type', '')

                if curve_type == 'Arc3D':
                    start_pt = curve.get('start_point', {})
                    end_pt = curve.get('end_point', {})
                    center_pt = curve.get('center_point', {})
                    radius = curve.get('radius', 0.0)
                    ref_vec = curve.get('reference_vector', {})
                    # Defer rounding: keep raw floats until final output to preserve EQUAL-constraint precision
                    parsed_curves.append({
                        'type': 'arc',
                        'start': [float(start_pt.get('x', 0.0)), float(start_pt.get('y', 0.0))],
                        'end': [float(end_pt.get('x', 0.0)), float(end_pt.get('y', 0.0))],
                        'center': [float(center_pt.get('x', 0.0)), float(center_pt.get('y', 0.0))],
                        'radius': float(radius),
                        'start_angle': curve.get('start_angle'),
                        'end_angle': curve.get('end_angle'),
                        'ref_vec': [ref_vec.get('x', 1.0), ref_vec.get('y', 0.0)], # 解析为2D向量
                        'curve': curve.get('curve') # 保留原始curve标签
                    })
                elif curve_type == 'Line3D':
                    start_pt = curve.get('start_point', {})
                    end_pt = curve.get('end_point', {})
                    # Defer rounding: keep raw floats until final output to preserve EQUAL-constraint precision
                    parsed_curves.append({
                        'type': 'line',
                        'start': [float(start_pt.get('x', 0.0)), float(start_pt.get('y', 0.0))],
                        'end': [float(end_pt.get('x', 0.0)), float(end_pt.get('y', 0.0))],
                        'curve': curve.get('curve') # 保留原始curve标签
                    })
                # 您的备用逻辑保持不变
                elif 'radius' in curve and curve.get('radius', 0) > 0:
                    if 'start_angle' in curve and 'end_angle' in curve:
                        parsed_curves.append(parse_arc(curve))
                    else:
                        parsed_curves.append(parse_circle(curve))
                elif 'start_point' in curve and 'end_point' in curve:
                    parsed_curves.append(parse_line(curve))
                else:
                    parsed_curves.append({'type': 'unknown', 'data': curve})
            
            parsed_loops.append({'curves': parsed_curves})
        
        parsed_profiles.append({'id': profile_id, 'loops': parsed_loops})
    
    return parsed_profiles

def parse_entities(entities_data):
    """Parse all entities from JSON data"""
    parsed_entities = {}
    
    for entity_id, entity_data in entities_data.items():
        entity_type = entity_data.get('type', '')
        entity_name = entity_data.get('name', entity_id)
        
        if entity_type == 'Sketch':
            profiles = entity_data.get('profiles', {})
            parsed_profiles = parse_sketch_profiles(profiles)
            
            parsed_entities[entity_id] = {
                'type': 'sketch',
                'name': entity_name,
                'profiles': parsed_profiles,
                # --- 关键修改：将transform信息也包含进来 ---
                'transform': entity_data.get('transform')
            }
        elif entity_type in ['Extrude', 'ExtrudeFeature']:
            # (此部分保持不变)
            depth = 0.0
            extent_one = entity_data.get('extent_one', {})
            if extent_one:
                distance_info = extent_one.get('distance', {})
                depth = distance_info.get('value', 0.0)
            if depth == 0.0:
                depth = entity_data.get('depth', 0.0)
                if depth == 0.0:
                    depth = entity_data.get('distance', 10.0)
            depth_coord = format_coord(depth)
            profiles_list = entity_data.get('profiles', [])
            profile_refs = []
            sketch_refs = []
            for profile_info in profiles_list:
                if isinstance(profile_info, dict):
                    profile_refs.append(profile_info.get('profile', ''))
                    sketch_refs.append(profile_info.get('sketch', ''))
            
            parsed_entities[entity_id] = {
                'type': 'extrude',
                'name': entity_name,
                'depth': depth_coord,
                'profile_refs': profile_refs,
                'sketch_refs': sketch_refs,
                'raw_data': entity_data
            }
        else:
            parsed_entities[entity_id] = {
                'type': entity_type.lower().replace('feature', ''),
                'name': entity_name,
                'data': entity_data
            }
    
    return parsed_entities


def generate_sketch_code(sketch_data, var_name, transform=None, constraint_info=None, norm_params=None, loop_counter_start=1):
    """
    生成 cq.Sketch 表达式，先构成loop，然后再组成sketch：
      - line -> segment
      - arc -> arc（使用起点、终点和中点）
      - circle -> circle
    
    Args:
        constraint_info: 约束信息，用于生成tag和约束JSON
        norm_params: 归一化参数 {'translation': array, 'scale': float}，在生成代码时应用
        loop_counter_start: 全局loop计数器起始值
    
    Returns:
        lines: 生成的代码行
        sketch_expr: sketch表达式
        loop_counter_end: 更新后的全局loop计数器值
    """
    lines = []
    loop_counter = loop_counter_start

    profiles = sketch_data.get('profiles', [])
    if not profiles:
        lines.append(f"{var_name} = cq.Sketch()")
        return lines
    
    # 所有平面现在都使用自定义Plane定义，origin在Plane中处理
    # 2D坐标不需要应用任何偏移，只需要统一精度
    
    def apply_transform(point):
        """统一坐标精度，对于Sketch内部的2D点，只应用缩放，不应用平移"""
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x = point[0]
            y = point[1]
            
            # 如果提供了归一化参数，应用归一化
            if norm_params:
                # 修正：局部坐标只需要缩放！位置由Workplane的Origin决定。
                scale = norm_params['scale']
                x = x * scale 
                y = y * scale
                # 注意：这里千万不要减 translation
            else:
                # 没有归一化，原始数据是米，转为毫米
                x = x * 1000
                y = y * 1000
            
            # 统一精度
            x = format_coord(x)
            y = format_coord(y)
            return [x, y]
        return point
    
    def apply_radius_transform(radius):
        """对半径应用归一化（只缩放，不平移），最终输出为毫米单位"""
        if norm_params:
            # 应用归一化缩放
            radius = radius * norm_params['scale']
        else:
            # 没有归一化，原始数据是米，转为毫米
            radius = radius * 1000
        return format_coord(radius)

    # flatten profiles -> loops
    loops = []
    for prof in profiles:
        prof_is_outer = prof.get('is_outer', False)
        for loop in prof.get('loops', []):
            lc = copy.deepcopy(loop)
            lc['_is_outer'] = loop.get('is_outer', prof_is_outer)
            loops.append(lc)

    if not loops:
        lines.append(f"{var_name} = cq.Sketch()")
        return lines

    # 为每个loop检查是否包含arc/circle（用于独立判断是否可以使用polygon优化）
    def loop_has_arc(loop):
        """检查单个loop是否包含圆弧或圆形"""
        curves = loop.get('curves') or loop.get('profile_curves') or []
        for curve in curves:
            ctype = (curve.get('type') or '').lower()
            if ctype in ('arc', 'arc3d', 'circle', 'circle3d'):
                return True
        return False

    # 按面积排序：大的loop作为外轮廓，小的作为内孔
    # 计算每个loop的近似面积（基于包围盒）
    def estimate_loop_area(loop):
        """估算loop的面积，用于排序"""
        curves = loop.get('curves') or loop.get('profile_curves') or []
        if not curves:
            return 0.0
        
        # 检查是否是单圆形loop（特殊情况）
        if len(curves) == 1:
            curve = curves[0]
            ctype = (curve.get('type') or '').lower()
            if ctype in ('circle', 'circle3d'):
                # 圆形：使用πr²作为面积估算
                r = curve.get('radius', 0.0)
                return 3.14159 * r * r
        
        # 收集所有点来计算包围盒
        all_x, all_y = [], []
        for curve in curves:
            # 支持解析后的数据结构
            if 'start' in curve:
                if isinstance(curve['start'], (list, tuple)):
                    all_x.append(curve['start'][0])
                    all_y.append(curve['start'][1])
            if 'end' in curve:
                if isinstance(curve['end'], (list, tuple)):
                    all_x.append(curve['end'][0])
                    all_y.append(curve['end'][1])
            
            # 支持原始 JSON 数据结构
            if 'start_point' in curve:
                pt = curve['start_point']
                if isinstance(pt, dict):
                    all_x.append(pt.get('x', 0.0))
                    all_y.append(pt.get('y', 0.0))
            if 'end_point' in curve:
                pt = curve['end_point']
                if isinstance(pt, dict):
                    all_x.append(pt.get('x', 0.0))
                    all_y.append(pt.get('y', 0.0))
        
        if not all_x:
            return 0.0
        
        # 使用包围盒面积作为估算
        width = max(all_x) - min(all_x)
        height = max(all_y) - min(all_y)
        return width * height
    
    # 按面积从大到小排序（大的在前，作为外轮廓）
    loops_sorted = sorted(loops, key=lambda l: -estimate_loop_area(l))

    # 先为每个loop生成单独的代码，然后再组合成sketch
    loop_vars = []
    
    # 按顺序处理每个 loop，为每个loop创建单独的变量
    for loop_idx, loop in enumerate(loops_sorted):
        curves = loop.get('curves') or loop.get('profile_curves') or []
        if not curves:
            continue

        loop_var = f"loop_{loop_counter}"
        loop_vars.append(loop_var)
        loop_counter += 1

        # 单圆特殊处理 - 使用 circle() 方法创建Face对象
        if len(curves) == 1 and curves[0].get('type', '').lower() == 'circle':
            c = curves[0]
            center = c.get('center') or c.get('center_point') or {'x': 0.0, 'y': 0.0}
            r = c.get('radius', 0.0)
            # 处理圆心坐标
            if isinstance(center, dict):
                center_coords = [center.get('x', 0.0), center.get('y', 0.0)]
            else:
                center_coords = [center[0], center[1]]
            
            # 应用transform偏移到圆心
            transformed_center = apply_transform(center_coords)
            transformed_radius = apply_radius_transform(r)
            # 使用统一的loop tag格式
            # tag_name = f"loop_{var_name}_{loop_idx}"
            tag_name = c.get('curve')
            
            # 使用 circle 方法创建完整圆形（Face对象）
            if abs(transformed_center[0]) < ZERO_TOLERANCE and abs(transformed_center[1]) < ZERO_TOLERANCE:
                # 圆心在原点，不需要移动
                lines.append(f"{loop_var} = cq.Sketch().circle({format_coord_str(transformed_radius)}, tag='{tag_name}')")
            else:
                # 使用push设置位置，避免moved()丢失tag
                lines.append(f"{loop_var} = cq.Sketch().push([({format_coord_str(transformed_center[0])}, {format_coord_str(transformed_center[1])})]).circle({format_coord_str(transformed_radius)}, tag='{tag_name}').reset()")
            continue

        # 排序 loop 曲线
        rem = copy.deepcopy(curves)
        sorted_curves = []
        tol = COORDINATE_TOLERANCE
        cur = rem.pop(0)
        sorted_curves.append(cur)
        last_pt = cur.get('end') or cur.get('start')

        while rem:
            found = False
            for idx, nc in enumerate(rem):
                s = nc.get('start'); e = nc.get('end')
                if s and last_pt and abs(s[0]-last_pt[0])<tol and abs(s[1]-last_pt[1])<tol:
                    # 直接按当前方向连接，不修改原始几何的起点/终点
                    cur = rem.pop(idx)
                    sorted_curves.append(cur)
                    last_pt = cur.get('end') or cur.get('start')
                    found = True
                    break
                elif e and last_pt and abs(e[0]-last_pt[0])<tol and abs(e[1]-last_pt[1])<tol:
                    # 反转方向以保持路径连贯，但不强行把起点改成 last_pt
                    cur = rem.pop(idx)
                    cur['start'], cur['end'] = cur.get('end'), cur.get('start')
                    sorted_curves.append(cur)
                    last_pt = cur.get('end') or cur.get('start')
                    found = True
                    break
            if not found:
                sorted_curves = []
                break

        if not sorted_curves:
            continue  # 跳过无法排序的 loop

        # 检查是否为简单的多边形（只有直线段）
        # 注意：只有当前loop中都没有arc时，才能使用polygon()
        # 因为polygon()会创建face而非wire，混用会导致assemble()失败
        all_lines = all((curve.get('type') or '').lower() in ('line', 'edge', 'line3d') for curve in sorted_curves)
        current_loop_has_arc = loop_has_arc(loop)
        
        if all_lines and len(sorted_curves) >= 3 and not current_loop_has_arc:
            # TODO: 线段-only 情况，可以尝试用 rect()，否则统一用隐式 segment 逻辑
            # if len(sorted_curves) == 4:
            #     # 先判断是否是轴对齐矩形
            #     points = []
            #     for curve in sorted_curves:
            #         start_pt = curve.get('start') or [0.0, 0.0]
            #         transformed_pt = apply_transform(start_pt)
            #         points.append(transformed_pt)
                
            #     def is_rect(pts):
            #         """简单判断是否为轴对齐矩形"""
            #         if len(pts) != 4:
            #             return False
            #         xs = [p[0] for p in pts]
            #         ys = [p[1] for p in pts]
            #         xs_sorted = sorted(xs)
            #         ys_sorted = sorted(ys)
            #         rect_tol = COORDINATE_TOLERANCE
            #         x_pairs = abs(xs_sorted[0] - xs_sorted[1]) < rect_tol and abs(xs_sorted[2] - xs_sorted[3]) < rect_tol
            #         y_pairs = abs(ys_sorted[0] - ys_sorted[1]) < rect_tol and abs(ys_sorted[2] - ys_sorted[3]) < rect_tol
            #         return x_pairs and y_pairs
                
            #     if is_rect(points):
            #         # 使用 rect()（矩形仍然保留优化）
            #         xs = [p[0] for p in points]
            #         ys = [p[1] for p in points]
            #         x_min, x_max = min(xs), max(xs)
            #         y_min, y_max = min(ys), max(ys)
            #         width = x_max - x_min
            #         height = y_max - y_min
            #         center_x = (x_min + x_max) / 2
            #         center_y = (y_min + y_max) / 2
                    
            #         tag_name = f"loop_{var_name}_{loop_idx}"
            #         if abs(center_x) < ZERO_TOLERANCE and abs(center_y) < ZERO_TOLERANCE:
            #             lines.append(f"{loop_var} = cq.Sketch().rect({format_coord_str(width)}, {format_coord_str(height)}, tag='{tag_name}')")
            #         else:
            #             lines.append(
            #                 f"{loop_var} = cq.Sketch().push([({format_coord_str(center_x)}, {format_coord_str(center_y)})])."
            #                 f"rect({format_coord_str(width)}, {format_coord_str(height)}, tag='{tag_name}').reset()"
            #             )
            #     else:
            #         # 非矩形四边形，使用隐式 segment 逻辑
            #         loop_parts = [f"{loop_var} = ("]
            #         loop_parts.append("cq.Sketch()")
                    
            #         current_pt = None
            #         for idx, curve in enumerate(sorted_curves):
            #             start_pt = curve.get('start') or [0.0, 0.0]
            #             end_pt = curve.get('end') or curve.get('start')
            #             a = apply_transform(start_pt)
            #             b = apply_transform(end_pt)
                        
            #             line_tag = curve.get('curve', f"{var_name}_loop_{loop_idx}_curve_{idx}")
                        
            #             if current_pt is None:
            #                 # 第一条线：显式给起点和终点
            #                 loop_parts.append(
            #                     f".segment(({format_coord_str(a[0])}, {format_coord_str(a[1])}), "
            #                     f"({format_coord_str(b[0])}, {format_coord_str(b[1])}), tag='{line_tag}')"
            #                 )
            #                 current_pt = b
            #             else:
            #                 # 后续线：只给终点，终点选择为“离 current_pt 更远的那个”
            #                 da = (a[0] - current_pt[0]) ** 2 + (a[1] - current_pt[1]) ** 2
            #                 db = (b[0] - current_pt[0]) ** 2 + (b[1] - current_pt[1]) ** 2
            #                 target = a if da > db else b
                            
            #                 loop_parts.append(
            #                     f".segment(({format_coord_str(target[0])}, {format_coord_str(target[1])}), tag='{line_tag}')"
            #                 )
            #                 current_pt = target
                    
            #         tag_name = f"loop_{var_name}_{loop_idx}"
            #         loop_parts.append(f".assemble(tag='{tag_name}')")
            #         loop_parts.append(")")
            #         lines.extend(loop_parts)
            # else:
                # 普通多边形（≥3条线段），统一使用两点式 segment + assemble
                loop_parts = [f"{loop_var} = ("]
                loop_parts.append("cq.Sketch()")
                
                current_pt = None
                for idx, curve in enumerate(sorted_curves):
                    start_pt = curve.get('start') or [0.0, 0.0]
                    end_pt = curve.get('end') or curve.get('start')
                    a = apply_transform(start_pt)
                    b = apply_transform(end_pt)
                    
                    line_tag = curve.get('curve', f"{var_name}_loop_{loop_idx}_curve_{idx}")
                    
                    if current_pt is None:
                        # 第一条线段：直接使用原始的起点和终点
                        loop_parts.append(
                            f".segment(({format_coord_str(a[0])}, {format_coord_str(a[1])}), "
                            f"({format_coord_str(b[0])}, {format_coord_str(b[1])}), tag='{line_tag}')"
                        )
                        current_pt = b
                    else:
                        # 后续线段：判断起点终点顺序，但都写成两点式
                        da = (a[0] - current_pt[0]) ** 2 + (a[1] - current_pt[1]) ** 2
                        db = (b[0] - current_pt[0]) ** 2 + (b[1] - current_pt[1]) ** 2
                        
                        if da > db:
                            # b更接近current_pt，说明需要反转：b -> a
                            loop_parts.append(
                                f".segment(({format_coord_str(b[0])}, {format_coord_str(b[1])}), "
                                f"({format_coord_str(a[0])}, {format_coord_str(a[1])}), tag='{line_tag}')"
                            )
                            current_pt = a
                        else:
                            # a更接近current_pt，正常顺序：a -> b
                            loop_parts.append(
                                f".segment(({format_coord_str(a[0])}, {format_coord_str(a[1])}), "
                                f"({format_coord_str(b[0])}, {format_coord_str(b[1])}), tag='{line_tag}')"
                            )
                            current_pt = b
                
                tag_name = f"loop_{var_name}_{loop_idx}"
                loop_parts.append(f".assemble(tag='{tag_name}')")
                loop_parts.append(")")
                lines.extend(loop_parts)
        else:
            # 使用 Edge-based API 构建路径，为每个loop单独创建sketch
            loop_parts = [f"{loop_var} = ("]
            loop_parts.append("cq.Sketch()")
            
            # 使用隐式路径逻辑：维护 current_pt，只传终点（首段仍需起点+终点）
            first_curve = True
            current_pt = None
            
            for curve in sorted_curves:
                ctype = (curve.get('type') or '').lower()
                start_pt = curve.get('start') or [0.0, 0.0]
                end_pt = curve.get('end') or curve.get('start')
                
                if ctype in ('line', 'edge', 'line3d'):
                    if end_pt:
                        a = apply_transform(start_pt)
                        b = apply_transform(end_pt)
                        
                        line_tag = curve.get('curve', f"{var_name}_loop_{loop_idx}_curve_{sorted_curves.index(curve)}")
                        
                        if current_pt is None:
                            # 第一条线段：直接使用原始的起点和终点
                            loop_parts.append(
                                f".segment(({format_coord_str(a[0])}, {format_coord_str(a[1])}), "
                                f"({format_coord_str(b[0])}, {format_coord_str(b[1])}), tag='{line_tag}')"
                            )
                            current_pt = b
                        else:
                            # 后续线段：判断起点终点顺序，但都写成两点式
                            da = (a[0] - current_pt[0]) ** 2 + (a[1] - current_pt[1]) ** 2
                            db = (b[0] - current_pt[0]) ** 2 + (b[1] - current_pt[1]) ** 2
                            
                            if da > db:
                                # b更接近current_pt，说明需要反转：b -> a
                                loop_parts.append(
                                    f".segment(({format_coord_str(b[0])}, {format_coord_str(b[1])}), "
                                    f"({format_coord_str(a[0])}, {format_coord_str(a[1])}), tag='{line_tag}')"
                                )
                                current_pt = a
                            else:
                                # a更接近current_pt，正常顺序：a -> b
                                loop_parts.append(
                                    f".segment(({format_coord_str(a[0])}, {format_coord_str(a[1])}), "
                                    f"({format_coord_str(b[0])}, {format_coord_str(b[1])}), tag='{line_tag}')"
                                )
                                current_pt = b
                
                elif ctype in ('arc', 'arc3d', 'circlearc'):
                    # 对于弧线，使用三点方式：起点、中点、终点
                    center = curve.get('center') or curve.get('center_point')
                    radius = curve.get('radius')
                    sa = curve.get('start_angle'); ea = curve.get('end_angle')
                    ref_vec = curve.get('ref_vec')
                    s = curve.get('start') or [0.0, 0.0]
                    e = curve.get('end') or [0.0, 0.0]
                    
                    # 使用JSON中的原始curve标签
                    arc_tag = curve.get('curve', f"{var_name}_loop_{loop_idx}_curve_{sorted_curves.index(curve)}")
                    
                    if center and radius and sa is not None and ea is not None:
                        # 由 start/end angle 与 ref_vec 生成一个 through 点（mid）
                        mid_ang = (sa + ea) / 2.0
                        # 如果有 ref_vec 则以其为基准旋转，否则用 (1,0)
                        rvx, rvy = (ref_vec[0], ref_vec[1]) if ref_vec else (1.0, 0.0)
                        mx = center[0] + (rvx * math.cos(mid_ang) - rvy * math.sin(mid_ang)) * radius
                        my = center[1] + (rvx * math.sin(mid_ang) + rvy * math.cos(mid_ang)) * radius
                        # 应用transform偏移
                        transformed_s = apply_transform(s)
                        transformed_e = apply_transform(e)
                        transformed_mid = apply_transform([mx, my])
                        
                        # 判断起点和终点的顺序：选择离 current_pt 更远的作为目标终点
                        if current_pt is not None:
                            ds = (transformed_s[0] - current_pt[0]) ** 2 + (transformed_s[1] - current_pt[1]) ** 2
                            de = (transformed_e[0] - current_pt[0]) ** 2 + (transformed_e[1] - current_pt[1]) ** 2
                            
                            if de < ds:
                                # e 更接近 current_pt，说明需要反转：e -> mid -> s
                                loop_parts.append(f".arc(({format_coord_str(transformed_e[0])}, {format_coord_str(transformed_e[1])}), ({format_coord_str(transformed_mid[0])}, {format_coord_str(transformed_mid[1])}), ({format_coord_str(transformed_s[0])}, {format_coord_str(transformed_s[1])}), tag='{arc_tag}')")
                                current_pt = transformed_s
                            else:
                                # s 更接近 current_pt，正常顺序：s -> mid -> e
                                loop_parts.append(f".arc(({format_coord_str(transformed_s[0])}, {format_coord_str(transformed_s[1])}), ({format_coord_str(transformed_mid[0])}, {format_coord_str(transformed_mid[1])}), ({format_coord_str(transformed_e[0])}, {format_coord_str(transformed_e[1])}), tag='{arc_tag}')")
                                current_pt = transformed_e
                        else:
                            # 第一条曲线，默认使用 s -> mid -> e
                            loop_parts.append(f".arc(({format_coord_str(transformed_s[0])}, {format_coord_str(transformed_s[1])}), ({format_coord_str(transformed_mid[0])}, {format_coord_str(transformed_mid[1])}), ({format_coord_str(transformed_e[0])}, {format_coord_str(transformed_e[1])}), tag='{arc_tag}')")
                            current_pt = transformed_e
                    # else:
                    #     # 退化：用 start/end 的中点稍作偏移作为 through 点
                    #     mx = 0.5*(s[0]+e[0]); my = 0.5*(s[1]+e[1])
                    #     # 应用transform偏移
                    #     transformed_s = apply_transform(s)
                    #     transformed_e = apply_transform(e)
                    #     transformed_mid = apply_transform([mx, my])
                    #     loop_parts.append(f".arc(({format_coord_str(transformed_s[0])}, {format_coord_str(transformed_s[1])}), ({format_coord_str(transformed_mid[0])}, {format_coord_str(transformed_mid[1])}), ({format_coord_str(transformed_e[0])}, {format_coord_str(transformed_e[1])}), tag='{arc_tag}')")

                elif ctype == 'circle':
                    # 在复杂形状中的圆，使用 circle() 方法创建Face对象
                    center = curve.get('center') or curve.get('center_point') or {'x': 0.0, 'y': 0.0}
                    r = curve.get('radius', 0.0)
                    if isinstance(center, dict):
                        center_coords = [center.get('x', 0.0), center.get('y', 0.0)]
                    else:
                        center_coords = [center[0], center[1]]
                    # 应用transform偏移
                    transformed_center = apply_transform(center_coords)
                    transformed_radius = apply_radius_transform(r)
                    
                    # 使用JSON中的原始curve标签（保持curve的原始标签，这是基础图元）
                    circle_tag = f"{var_name}_loop_{loop_idx}_curve_{sorted_curves.index(curve)}"
                    
                    if abs(transformed_center[0]) < ZERO_TOLERANCE and abs(transformed_center[1]) < ZERO_TOLERANCE:
                        # 圆心在原点，不需要移动
                        loop_parts.append(f"    .circle({format_coord_str(transformed_radius)}, tag='{circle_tag}')")
                    else:
                        # 使用push设置位置，避免moved()丢失tag
                        loop_parts.append(f"    .push([({format_coord_str(transformed_center[0])}, {format_coord_str(transformed_center[1])})]).circle({format_coord_str(transformed_radius)}, tag='{circle_tag}').reset()")
                else:
                    # 不认识的曲线类型，保守当作线段处理，采用两点式写法
                    if end_pt:
                        a = apply_transform(start_pt)
                        b = apply_transform(end_pt)
                        
                        if current_pt is None:
                            # 第一条线段：直接使用原始的起点和终点
                            loop_parts.append(
                                f".segment(({format_coord_str(a[0])}, {format_coord_str(a[1])}), "
                                f"({format_coord_str(b[0])}, {format_coord_str(b[1])}))"
                            )
                            current_pt = b
                        else:
                            # 后续线段：判断起点终点顺序，但都写成两点式
                            da = (a[0] - current_pt[0]) ** 2 + (a[1] - current_pt[1]) ** 2
                            db = (b[0] - current_pt[0]) ** 2 + (b[1] - current_pt[1]) ** 2
                            
                            if da > db:
                                # b更接近current_pt，说明需要反转：b -> a
                                loop_parts.append(
                                    f".segment(({format_coord_str(b[0])}, {format_coord_str(b[1])}), "
                                    f"({format_coord_str(a[0])}, {format_coord_str(a[1])}))"
                                )
                                current_pt = a
                            else:
                                # a更接近current_pt，正常顺序：a -> b
                                loop_parts.append(
                                    f".segment(({format_coord_str(a[0])}, {format_coord_str(a[1])}), "
                                    f"({format_coord_str(b[0])}, {format_coord_str(b[1])}))"
                                )
                                current_pt = b
                
                first_curve = False

            # 检查是否使用了 Edge-based API，只有这种情况才需要 assemble()
            has_edge_based_api = any('.segment(' in part or '.arc(' in part for part in loop_parts)
            if has_edge_based_api:
                # 为复杂形状添加tag
                tag_name = f"loop_{var_name}_{loop_idx}"
                loop_parts.append(f".assemble(tag='{tag_name}')")
            loop_parts.append(")")
            
            lines.extend(loop_parts)

    # 现在组合所有loop成最终的sketch表达式（不生成赋值语句）
    sketch_expr = None
    if loop_vars:
        # lines.append(f"# Profile '{var_name}' created from {len(loop_vars)} loop(s).")
        if len(loop_vars) == 1:
            # 只有一个loop，直接使用loop变量
            sketch_expr = loop_vars[0]
        else:
            # 多个loop，用括号包裹差集表达式
            sketch_expr = f"({' - '.join(loop_vars)})"
    else:
        lines.append(f"# FAILED to generate geometry for '{var_name}'.")
        sketch_expr = "cq.Sketch()"

    return lines, sketch_expr, loop_counter


def generate_cadquery_code(parsed_data, output_path):
    """
    Generate complete CADQuery script and constraints JSON.
    
    Args:
        parsed_data: Parsed CAD data
        output_path: Output path for Python script
        original_data: Original JSON data (needed for constraint extraction)
    """
    entities = parsed_data.get('entities', {})
    sequence = parsed_data.get('sequence', [])
    norm_params = parsed_data.get('normalization')

    lines = []
    loop_counter = 1  # 全局loop计数器
    # lines.append("# Auto-generated CADQuery script from JSON")
    lines.append("import cadquery as cq")
    # lines.append("import math")
    # lines.append("")
    
    # 添加归一化信息注释
    # if norm_params:
        # lines.append("# ===== DeepCAD Normalization Applied =====")
        # lines.append(f"# Translation: {norm_params['translation']}")
        # lines.append(f"# Scale factor: {norm_params['scale']:.10f}")
        # lines.append("# All coordinates have been normalized following DeepCAD (ICCV 2021)")
        # lines.append("# To denormalize: original = normalized / scale + translation")
    


    profile_sketch_vars = {}
    stored_profiles = {}

    # 不再生成初始 sketch 定义，直接在 feature construction 中生成需要的 sketch
    # 但仍需要建立 profile_sketch_vars 映射和存储 profile 数据
    for entity_id, entity_data in entities.items():
        if entity_data['type'] == 'sketch':
            transform = entity_data.get('transform')
            for profile in entity_data.get('profiles', []):
                profile_id = profile['id']
                var_name = f"sketch_{profile_id.replace('+', '_')}"
                profile_sketch_vars[profile_id] = var_name

                # 在存储的数据中也包含transform
                profile_data = {
                    'name': f"{entity_data['name']} - Profile {profile_id}",
                    'profiles': [profile],
                    'transform': transform
                }
                stored_profiles[profile_id] = profile_data

    sketch_usage_count = {var: 0 for var in profile_sketch_vars.values()}
    for step in sequence:
        if entities.get(step.get('entity'), {}).get('type') == 'extrude':
            entity_data = entities[step['entity']]
            for profile_ref in entity_data.get('profile_refs', []):
                if profile_ref in profile_sketch_vars:
                    sketch_var = profile_sketch_vars[profile_ref]
                    sketch_usage_count[sketch_var] += 1

    sketch_usage_current = {var: 0 for var in sketch_usage_count}
    body_counter = 0

    lines.append("# === 1. Sketch and Extrude ===")
    for i, step in enumerate(sequence):
        entity_id = step.get('entity', '')
        entity_data = entities.get(entity_id, {})

        if entity_data.get('type') == 'extrude':
            body_counter += 1
            body_var_name = f"body_{body_counter}"
            lines.append(f"# --- Body {body_counter} ---")

            profile_refs = entity_data.get('profile_refs', [])
            # 存储 tuple (sketch_var_name, profile_ref) 保证复制和映射不会丢失信息
            sketches_for_this_op = []
            for profile_ref in profile_refs:
                if profile_ref not in profile_sketch_vars:
                    continue

                initial_var = profile_sketch_vars[profile_ref]
                sketch_usage_current[initial_var] += 1

                # 每次都重新生成 sketch，使用原始名称，不需要复制逻辑
                sketches_for_this_op.append((initial_var, profile_ref))

            depth = entity_data.get('depth', 10.0)
            
            # 如果提供了归一化参数，深度也需要归一化；否则米转毫米，最终输出为毫米单位
            if norm_params:
                # 应用归一化：只需要缩放，不需要平移
                depth = depth * norm_params['scale']
            else:
                # 没有归一化，原始数据是米，转为毫米
                depth = depth * 1000
            
            if not sketches_for_this_op:
                continue

            # 检查所有相关 profile 是否在相同 transform/plane（否则无法合并为单一 Workplane）
            transforms = [stored_profiles.get(pref, {}).get('transform') for (_, pref) in sketches_for_this_op]
            same_plane = all(t == transforms[0] for t in transforms) if transforms else True

            raw = entity_data.get('raw_data', {}) or {}
            extent_type_raw = str(raw.get('extent_type', '')).lower()
            two_sided_flag = any(k in extent_type_raw for k in ('two', 'both', 'symmetric'))

            # helper: fallback two-side implementation for older CadQuery (if both=True unsupported)
            def emit_try_both(var_sketch, depth_val, out_var, transform=None):
                # 检查是否已经是 Workplane 对象（combined sketch）还是 Sketch 对象
                # combined sketch 的变量名格式为 sketch_N（N为数字）
                if re.match(r'^sketch_\d+$', var_sketch):
                    # 已经是 Workplane 对象，直接 extrude
                    lines.append(f"{out_var} = {var_sketch}.extrude({format_coord_str(depth_val)}, both=True).tag('{out_var}')")
                else:
                    # Sketch 对象或表达式，需要先放置到 Workplane 上才能 extrude
                    plane_info = get_workplane_from_transform(transform, norm_params=norm_params)
                    workplane_code = create_workplane_code(plane_info)
                    lines.append(f"{out_var} = {workplane_code}.placeSketch({var_sketch}).extrude({format_coord_str(depth_val)}, both=True)")
                return

            if same_plane:
                # 合并 profiles 的原始数据并一次生成 combined sketch
                combined_profiles = []
                combined_transform = None
                for (svar, pref) in sketches_for_this_op:
                    pdata = stored_profiles.get(pref)
                    if not pdata:
                        continue
                    # print(f"Merging profiles: {pdata.get('profiles', [])[0]['id']}")
                    combined_profiles.extend(pdata.get('profiles', []))
                    if combined_transform is None:
                        combined_transform = pdata.get('transform')

                # 生成每个 Sketch 对象，使用原始的 profile ID 作为变量名
                individual_sketch_exprs = []
                for profile in combined_profiles:
                    profile_id = profile['id']
                    sk_var = f"{profile_id.replace('+', '_')}"
                    code_lines, sketch_expr, loop_counter = generate_sketch_code({'profiles': [profile]}, sk_var, combined_transform, norm_params=norm_params, loop_counter_start=loop_counter)
                    lines.extend(code_lines)
                    individual_sketch_exprs.append(sketch_expr)
                    # lines.append("")

                # 合并所有 Sketch - 在 CadQuery 中需要将所有 sketch 放到同一个 workplane 上
                combined_sketch_var = f"sketch_{body_counter}"
                sketch_list = ', '.join(individual_sketch_exprs)
                # 根据 transform 确定正确的工作平面
                plane_info = get_workplane_from_transform(combined_transform, norm_params=norm_params)
                workplane_code = create_workplane_code(plane_info)
                lines.append(f"{combined_sketch_var} = {workplane_code}.placeSketch({sketch_list})")
                # lines.append("")


                if two_sided_flag:
                    emit_try_both(combined_sketch_var, depth, body_var_name, combined_transform)
                else:
                    lines.append(f"{body_var_name} = {combined_sketch_var}.extrude({format_coord_str(depth)}).tag('{body_var_name}')")
            else:
                # 不能合并为单面（不同 plane 或只有单个 profile），退回为：
                # 对每个 profile 单独 extrude，然后按 operation 顺序把它们合并为一个 body（临时生成多个临时 body）
                temp_vars = []
                for idx, (svar, pref) in enumerate(sketches_for_this_op):
                    tmp_var = f"{body_var_name}_part_{idx+1}"
                    pdata = stored_profiles.get(pref, {})
                    profile_transform = pdata.get('transform')
                    
                    # 生成 sketch 定义
                    code_lines, sketch_expr, loop_counter = generate_sketch_code(pdata, svar, profile_transform, norm_params=norm_params, loop_counter_start=loop_counter)
                    lines.extend(code_lines)
                    # lines.append("")
                    
                    if two_sided_flag:
                        emit_try_both(sketch_expr, depth, tmp_var, profile_transform)
                    else:
                        plane_info = get_workplane_from_transform(profile_transform, norm_params=norm_params)
                        workplane_code = create_workplane_code(plane_info)
                        lines.append(f"{tmp_var} = {workplane_code}.placeSketch({sketch_expr}).extrude({format_coord_str(depth)})")
                    temp_vars.append(tmp_var)
                # 把所有 temp_vars 用 union 合并为最终 body_var_name（简单合并，后续根据 operation 决定如何插入 assembly）
                if temp_vars:
                    # 以第一个为基础，将其余 union 上去
                    merge_chain = temp_vars[0]
                    for mv in temp_vars[1:]:
                        merge_chain = f"{merge_chain}.union({mv})"
                    lines.append(f"{body_var_name} = {merge_chain}")

            # 判断布尔操作类型并立即执行
            op_raw = str(raw.get('operation', '')).lower()
            if any(k in op_raw for k in ('cut', 'subtract', 'booleancut', 'remove')):
                op_type = 'cut'
            elif any(k in op_raw for k in ('intersect', 'and', 'common')):
                op_type = 'intersect'
            else:
                op_type = 'union'
            
            # 第一个body作为result，后续body根据operation进行布尔运算
            if body_counter == 1:
                lines.append(f"result = {body_var_name}")
            else:
                lines.append(f"result = result.{op_type}({body_var_name}.val())")



    # --- 导出 ---
    lines.append("# === 2. Export ===")
    if body_counter > 0:
        lines.append("cq.exporters.export(result, 'output.step')")
    else:
        lines.append("# No bodies to export")


    # # --- 导出 ---
    # lines.append("# === 3. Export ---")
    # lines.append("if result:")
    # lines.append("    cq.exporters.export(result, 'output.step')")
    # lines.append("    print('Exported output.step')")
    # lines.append("else:")
    # lines.append("    print('No result to export')")

    # 生成Python脚本
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_path

def process_json_file(input_path, out_prefix, apply_normalization=True, norm_factor=0.75):
    """
    Process a single JSON file
    
    Args:
        input_path: 输入JSON文件路径
        out_prefix: 输出文件前缀
        apply_normalization: 是否应用归一化（默认True）
        norm_factor: 归一化因子（默认0.75，遵循DeepCAD论文）
    """    
    print(f"Processing: {input_path}")
    
    # Load JSON data
    data = load_json(input_path)
    if data is None:
        return False
    
    # Step 1: 计算归一化参数（如果需要）
    norm_params = None
    if apply_normalization:
        print("\n=== Computing DeepCAD Normalization Parameters ===")
        
        # 从 JSON 的 properties.bounding_box 中获取
        bounding_box = data.get('properties', {}).get('bounding_box')
        
        if bounding_box and 'min_point' in bounding_box and 'max_point' in bounding_box:
            # 从 JSON 中直接读取 bounding box
            min_pt = bounding_box['min_point']
            max_pt = bounding_box['max_point']
            
            bbox_min = np.array([min_pt['x'], min_pt['y'], min_pt['z']])
            bbox_max = np.array([max_pt['x'], max_pt['y'], max_pt['z']])
            
            print(f"Original bounding box: min={bbox_min}, max={bbox_max}")
            
            # 计算归一化参数（与 normalize_cad_points 中的逻辑一致）
            # bbox_center = np.array([0.0, 0.0, 0.0]) DEEPCAD
            bbox_center = (bbox_min + bbox_max) / 2.0
            
            # 计算最大半轴长度（等同于 np.max(np.abs(centered_points))）
            # max_abs_distance = np.max(np.abs([bbox_min, bbox_max]))  DEEPCAD
            half_extent = (bbox_max - bbox_min) / 2.0
            max_abs_distance = np.max(half_extent)
            
            # 处理边界情况：bounding box 为点的情况
            if max_abs_distance < 1e-10:
                scale_factor = 1.0
            else:
                scale_factor = norm_factor / max_abs_distance
            
            # 保存归一化参数
            norm_params = {
                'translation': bbox_center,
                'scale': scale_factor
            }
            
            print(f"Normalization parameters:")
            print(f"  Translation: {bbox_center}")
            print(f"  Scale factor: {scale_factor:.6f}")
            print(f"  Target normalized range: ±{norm_factor}")
            print("=" * 50 + "\n")
        
    
    # Step 2: Parse entities (不应用归一化，保持原始数据)
    entities = parse_entities(data.get('entities', {}))
    
    # Create processed data structure
    processed_data = {
        'entities': entities,
        'sequence': data.get('sequence', []),
        'properties': data.get('properties', {}),
        'normalization': norm_params  # 保存归一化参数供后续使用
    }
    
    # Generate summary
    entity_types = defaultdict(int)
    for entity in entities.values():
        entity_types[entity['type']] += 1
    
    print(f"Found entities: {dict(entity_types)}")
    print(f"Sequence steps: {len(processed_data['sequence'])}")
    
    # Generate CADQuery code
    out_py = f"{out_prefix}.py"
    generate_cadquery_code(processed_data, out_py)
    print(f"Generated CADQuery script: {out_py}")
    
    return True

