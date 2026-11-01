#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADQuery低级脚本转高级语义特征脚本转换器 V2
简化版本，更易理解和维护
"""

import re
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


def format_coord_str(val):
    """
    格式化浮点数：截断到 6 位小数（不四舍五入），与 json2cadquery.py 一致。
    避免 EQUAL 约束的相近值被舍入到不同结果。
    """
    if val == 0:
        return "0"
    truncated = math.trunc(val * 1e6) / 1e6
    formatted = f"{truncated:.6f}".rstrip('0').rstrip('.')
    if formatted == '-0':
        return "0"
    return formatted


@dataclass
class Point:
    """2D点"""
    x: float
    y: float
    
    def distance(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class FeatureRecognizer:
    """识别几何特征"""
    
    @staticmethod
    def _merge_collinear_segments(segments: List) -> List:
        """合并连续的共线segment，去除冗余顶点"""
        if len(segments) < 2:
            return segments
        
        # 检查两个segment是否共线（连续且在同一直线上）
        def is_collinear(seg1, seg2, tolerance=1e-5):
            # seg1和seg2必须连续（seg1的end == seg2的start）
            if abs(seg1['end'][0] - seg2['start'][0]) > tolerance or \
               abs(seg1['end'][1] - seg2['start'][1]) > tolerance:
                return False
            
            # 检查三点共线
            p1 = seg1['start']
            p2 = seg1['end']  # = seg2['start']
            p3 = seg2['end']
            
            dx1 = p2[0] - p1[0]
            dy1 = p2[1] - p1[1]
            dx2 = p3[0] - p2[0]
            dy2 = p3[1] - p2[1]
            cross = dx1 * dy2 - dy1 * dx2
            return abs(cross) < tolerance
        
        merged = []
        i = 0
        while i < len(segments):
            current_seg = segments[i].copy()
            merged_tags = [current_seg.get('tag')] if current_seg.get('tag') else []
            
            # 尝试向前合并连续的共线segment
            j = i + 1
            while j < len(segments):
                next_seg = segments[j]
                
                if is_collinear(current_seg, next_seg):
                    # 合并：扩展current_seg到next_seg的终点
                    current_seg['end'] = next_seg['end']
                    if next_seg.get('tag'):
                        merged_tags.append(next_seg['tag'])
                    j += 1
                else:
                    break
            
            # Store all tags from merged collinear segments (each edge may consume >1 tag)
            current_seg['tags'] = merged_tags
            merged.append(current_seg)
            i = j
        
        return merged
    
    @staticmethod
    def _merge_cocircular_arcs(arcs: List) -> List:
        """合并连续的共圆arc（在同一圆上且连续的arc）"""
        if len(arcs) < 2:
            return arcs
        
        # 简化版：只检查连续性，暂不做复杂的共圆判断
        # 如果两个arc连续（第一个的end == 第二个的start），可以考虑合并
        # 但由于arc的表示复杂，这里先保持原样，后续可以扩展
        
        return arcs
    
    @staticmethod
    def recognize_from_segments_arcs(segments: List, arcs: List) -> Dict:
        """从segment和arc序列识别特征"""
        # 总是先合并连续的共线segments（对所有情况）
        original_seg_count = len(segments)
        if len(segments) > 1:
            segments = FeatureRecognizer._merge_collinear_segments(segments)
        
        # 总是先合并连续的共圆arcs（对所有情况）
        original_arc_count = len(arcs)
        if len(arcs) > 1:
            arcs = FeatureRecognizer._merge_cocircular_arcs(arcs)
        
        # 收集所有tags（含共线合并后的完整tags，每条边可能消耗多个tag）
        tags = []
        for seg in segments:
            if seg.get('tags'):
                tags.extend(seg['tags'])
            elif seg.get('tag'):
                tags.append(seg['tag'])
        for arc in arcs:
            if arc.get('tag'):
                tags.append(arc['tag'])
        
        # 圆角矩形：4个segment + 4个arc
        if len(segments) == 4 and len(arcs) == 4:
            if FeatureRecognizer._is_rounded_rect(segments, arcs):
                result = FeatureRecognizer._extract_rounded_rect_params(segments, arcs)
                # fillet的tags已经在_extract_rounded_rect_params中分别设置了
                # 不需要再设置统一的tags
                return result
        
        # 纯矩形：4个segment（含旋转矩形）
        if len(segments) == 4 and len(arcs) == 0:
            if FeatureRecognizer._is_rect(segments):
                result = FeatureRecognizer._extract_rect_params(segments)
                result['tags'] = tags
                return result
        
        # 槽：2个segment + 2个arc
        if len(segments) == 2 and len(arcs) == 2:
            if FeatureRecognizer._is_slot(segments, arcs):
                result = FeatureRecognizer._extract_slot_params(segments, arcs)
                result['tags'] = tags
                return result
        
        # 正多边形：所有边等长，只有segments
        if len(arcs) == 0 and len(segments) >= 3:
            if FeatureRecognizer._is_regular_polygon(segments):
                result = FeatureRecognizer._extract_regular_polygon_params(segments)
                result['tags'] = tags
                return result
        
        # 一般多边形：segments组成的封闭多边形
        if len(arcs) == 0 and len(segments) >= 3:
            result = FeatureRecognizer._extract_polygon_params(segments)
            result['tags'] = tags
            return result
        
        # 默认：未识别的复杂几何，保留原始的segments和arcs
        return {
            'type': 'unknown', 
            'segments': segments,  # 保留完整的segment数据
            'arcs': arcs,  # 保留完整的arc数据
            'tags': tags  # 保留tags
        }
    
    @staticmethod
    def _is_axis_aligned_rect(segments: List) -> bool:
        """检查是否是轴对齐的矩形（segments形成闭合的矩形侧边）"""
        horizontals = [s for s in segments if abs(s['start'][1] - s['end'][1]) < 1e-5]
        verticals = [s for s in segments if abs(s['start'][0] - s['end'][0]) < 1e-5]
        if len(horizontals) != 2 or len(verticals) != 2:
            return False

        tol = 1e-5

        # 两条水平边的所有端点x坐标应只有2个不同值（对应矩形左右各一）
        h_xs = [horizontals[0]['start'][0], horizontals[0]['end'][0],
                horizontals[1]['start'][0], horizontals[1]['end'][0]]
        unique_xs = []
        for x in h_xs:
            if not any(abs(x - ux) < tol for ux in unique_xs):
                unique_xs.append(x)
        if len(unique_xs) != 2:
            return False

        # 两条垂直边的所有端点y坐标应只有2个不同值（对应矩形上下各一）
        v_ys = [verticals[0]['start'][1], verticals[0]['end'][1],
                verticals[1]['start'][1], verticals[1]['end'][1]]
        unique_ys = []
        for y in v_ys:
            if not any(abs(y - uy) < tol for uy in unique_ys):
                unique_ys.append(y)
        if len(unique_ys) != 2:
            return False

        return True
    
    @staticmethod
    def _is_rect(segments: List) -> bool:
        """检查4条segments是否构成矩形（允许旋转）：连续闭合、相邻边垂直、对边等长"""
        if len(segments) != 4:
            return False

        tol = 1e-5

        # 检查连续闭合：每段终点 == 下一段起点
        for i in range(4):
            s1 = segments[i]
            s2 = segments[(i + 1) % 4]
            if abs(s1['end'][0] - s2['start'][0]) > tol or abs(s1['end'][1] - s2['start'][1]) > tol:
                return False

        def vec(seg):
            return (seg['end'][0] - seg['start'][0], seg['end'][1] - seg['start'][1])

        def length(seg):
            dx, dy = vec(seg)
            return math.hypot(dx, dy)

        vecs = [vec(s) for s in segments]
        lens = [length(s) for s in segments]

        # 检查所有段长度 > 0
        if any(l < 1e-5 for l in lens):
            return False

        # 相邻边垂直（点积归一化后 ≈ 0）
        for i in range(4):
            vx0, vy0 = vecs[i]
            vx1, vy1 = vecs[(i + 1) % 4]
            dot = (vx0 * vx1 + vy0 * vy1) / (lens[i] * lens[(i + 1) % 4])
            if abs(dot) > 0.01:
                return False

        # 对边等长
        if abs(lens[0] - lens[2]) > tol or abs(lens[1] - lens[3]) > tol:
            return False

        return True

    @staticmethod
    def _is_rounded_rect(segments: List, arcs: List) -> bool:
        """检查是否是圆角矩形"""
        return FeatureRecognizer._is_axis_aligned_rect(segments)
    
    @staticmethod
    def _is_slot(segments: List, arcs: List) -> bool:
        """检查是否是槽：两段等长且平行（反向），且两端弧中点在线段轴外侧"""
        if len(segments) != 2:
            return False

        s1, s2 = segments[0], segments[1]
        s1_len = Point(*s1['start']).distance(Point(*s1['end']))
        s2_len = Point(*s2['start']).distance(Point(*s2['end']))
        if abs(s1_len - s2_len) > 1e-5:
            return False
        if s1_len < 1e-5:
            return False

        # 两段方向向量的叉积绝对值（归一化后）应接近 0（平行或反向平行）
        dx1 = s1['end'][0] - s1['start'][0]
        dy1 = s1['end'][1] - s1['start'][1]
        dx2 = s2['end'][0] - s2['start'][0]
        dy2 = s2['end'][1] - s2['start'][1]
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) / (s1_len * s2_len) > 0.01:  # 允许约 0.57° 偏差
            return False

        # 弧中点必须在线段端点区间之外（外侧），才是标准 slot 的端盖
        # 取 s1 的方向为槽轴方向，投影所有点
        ux, uy = dx1 / s1_len, dy1 / s1_len  # 单位向量

        def proj(pt):
            return pt[0] * ux + pt[1] * uy

        seg_projs = [proj(s1['start']), proj(s1['end']),
                     proj(s2['start']), proj(s2['end'])]
        p_min = min(seg_projs)
        p_max = max(seg_projs)

        arc_mid_projs = [proj(arc['mid']) for arc in arcs]
        # 每个弧中点必须落在 [p_min, p_max] 之外（外侧端盖）
        if not all(mp < p_min - 1e-5 or mp > p_max + 1e-5 for mp in arc_mid_projs):
            return False
        # 且两端各有一个弧（一左一右）
        if not any(mp < p_min - 1e-5 for mp in arc_mid_projs):
            return False
        if not any(mp > p_max + 1e-5 for mp in arc_mid_projs):
            return False

        # 每段弧必须是半圆端盖：弧中点到弦中点的距离 == 弦长 / 2
        # 这排除弧是大圆弧（如以形状中心为圆心）而非标准端盖的情形
        import math
        for arc in arcs:
            ax, ay = arc['start']
            bx, by = arc['end']
            mx, my = arc['mid']
            chord_mid_x = (ax + bx) / 2.0
            chord_mid_y = (ay + by) / 2.0
            chord_len = math.hypot(bx - ax, by - ay)
            expected_r = chord_len / 2.0
            actual_dist = math.hypot(mx - chord_mid_x, my - chord_mid_y)
            if abs(actual_dist - expected_r) > 1e-5 * max(expected_r, 1e-5):
                return False

        return True
    
    @staticmethod
    def _extract_rect_params(segments: List) -> Dict:
        """
        提取矩形参数：
        - 对轴对齐矩形：width = x方向长度, height = y方向长度（用bbox）
        - 对旋转矩形：width/height 取相邻两条边长，angle 取一条边方向
        """
        # 收集顶点（start点通常足够；这里也把end点加上以更稳）
        pts = []
        for s in segments:
            pts.append(s["start"])
            pts.append(s["end"])

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 中心点：bbox中心对矩形最稳（比顶点平均更不受顺序影响）
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        # 判断是否轴对齐：所有边要么水平要么垂直
        is_axis_aligned = FeatureRecognizer._is_axis_aligned_rect(segments)

        if is_axis_aligned:
            width = max_x - min_x   # x方向长度
            height = max_y - min_y  # y方向长度
            return {
                "type": "rect",
                "center": (center_x, center_y),
                "width": width,
                "height": height
            }

        # -------- 非轴对齐：估计旋转角 + 宽高 --------
        # 找到两条不共线的边（矩形相邻边）
        def seg_vec(seg):
            return (seg["end"][0] - seg["start"][0], seg["end"][1] - seg["start"][1])

        def seg_len(seg):
            dx, dy = seg_vec(seg)
            return math.hypot(dx, dy)

        # 取第一条边作为参考方向
        s0 = segments[0]
        dx0, dy0 = seg_vec(s0)
        len0 = math.hypot(dx0, dy0)
        if len0 < 1e-5:
            # 退化情况：回退用bbox
            width = max_x - min_x
            height = max_y - min_y
            return {
                "type": "rect",
                "center": (center_x, center_y),
                "width": width,
                "height": height
            }

        # angle：让width沿着s0方向
        angle = math.degrees(math.atan2(dy0, dx0))

        # 找一条与s0不平行的边作为height方向
        # 用叉积判断是否平行
        width_len = seg_len(s0)
        height_len = None
        vx0, vy0 = dx0, dy0

        for s in segments[1:]:
            vx, vy = seg_vec(s)
            # 叉积接近0 => 平行
            cross = vx0 * vy - vy0 * vx
            if abs(cross) > 1e-5:
                height_len = seg_len(s)
                break

        if height_len is None:
            # 如果没找到（异常），回退bbox
            width = max_x - min_x
            height = max_y - min_y
            return {
                "type": "rect",
                "center": (center_x, center_y),
                "width": width,
                "height": height
            }

        # 标准化角度到 (-180, 180]
        while angle > 180:
            angle -= 360
        while angle <= -180:
            angle += 360

        return {
            "type": "rect",
            "center": (center_x, center_y),
            "width": width_len,
            "height": height_len,
            "angle": angle
        }

    
    @staticmethod
    def _extract_rounded_rect_params(segments: List, arcs: List) -> Dict:
        """提取圆角矩形参数"""
        points = [s['start'] for s in segments]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 计算圆角半径（从arc获取）
        # 简化计算：使用第一个arc的近似半径
        arc1 = arcs[0]
        p1 = Point(*arc1['start'])
        p2 = Point(*arc1['mid'])
        p3 = Point(*arc1['end'])
        
        # 使用简单的近似：圆角半径约等于弦长的一半
        radius = p1.distance(p3) / math.sqrt(2)
        
        # 分别收集segment和arc的tags（含共线合并后的完整tags）
        segment_tags = []
        for s in segments:
            if s.get('tags'):
                segment_tags.extend(s['tags'])
            elif s.get('tag'):
                segment_tags.append(s['tag'])
        arc_tags = [a.get('tag') for a in arcs if a.get('tag')]
        
        return {
            'type': 'fillet',
            'center': ((min_x + max_x) / 2, (min_y + max_y) / 2),
            'width': max_x - min_x,
            'height': max_y - min_y,
            'radius': radius,
            'segment_tags': segment_tags,
            'arc_tags': arc_tags
        }
    
    @staticmethod
    def _extract_slot_params(segments: List, arcs: List) -> Dict:
        """提取槽参数"""
        # 槽长度是segment的长度
        s1 = segments[0]
        length = Point(*s1['start']).distance(Point(*s1['end']))
        
        # 半径从arc获取
        arc1 = arcs[0]
        p1 = Point(*arc1['start'])
        p3 = Point(*arc1['end'])
        diameter = p1.distance(p3)
        
        # 中心点
        all_points = [s['start'] for s in segments] + [s['end'] for s in segments]
        center_x = sum(p[0] for p in all_points) / len(all_points)
        center_y = sum(p[1] for p in all_points) / len(all_points)
        
        # 槽的方向角（segment 方向），度；槽 180° 对称，归一化到 [0, 180)
        dx = s1['end'][0] - s1['start'][0]
        dy = s1['end'][1] - s1['start'][1]
        angle_deg = math.degrees(math.atan2(dy, dx))
        angle_norm = angle_deg % 180.0  # [0, 180)
        
        return {
            'type': 'slot',
            'center': (center_x, center_y),
            'length': length,
            'diameter': diameter,
            'angle': angle_norm
        }
    
    @staticmethod
    def _is_regular_polygon(segments: List) -> bool:
        if len(segments) < 3:
            return False

        # 收集去重点集（用端点更稳）
        pts = []
        for s in segments:
            pts.append(s["start"])
            pts.append(s["end"])

        # 去重（避免重复点干扰）
        uniq = []
        for p in pts:
            if not any(abs(p[0]-q[0]) < 1e-5 and abs(p[1]-q[1]) < 1e-5 for q in uniq):
                uniq.append(p)

        n = len(segments)
        if len(uniq) != n:
            # 顶点数不匹配就别判 regular
            return False

        # 1) 等边
        lens = [Point(*s["start"]).distance(Point(*s["end"])) for s in segments]
        avg = sum(lens)/len(lens)
        if avg < 1e-5:
            return False
        if max(abs(l-avg) for l in lens) > avg * 0.02:  # 2% 比 5% 更合理，但你可调
            return False

        # 2) 共圆（顶点到中心距离几乎相等）
        cx = sum(p[0] for p in uniq)/len(uniq)
        cy = sum(p[1] for p in uniq)/len(uniq)
        rs = [Point(cx, cy).distance(Point(*p)) for p in uniq]
        ravg = sum(rs)/len(rs)
        if ravg < 1e-5:
            return False
        if max(abs(r-ravg) for r in rs) > ravg * 0.02:
            return False

        # 3) 等角：按极角排序，相邻角差接近 2π/n
        angles = sorted(math.atan2(p[1]-cy, p[0]-cx) for p in uniq)
        # 补上闭环差值
        diffs = [(angles[(i+1)%n] - angles[i]) % (2*math.pi) for i in range(n)]
        target = 2*math.pi/n
        if max(abs(d-target) for d in diffs) > 0.10 * target:  # 允许 10% 波动，可调
            return False

        return True

    
    @staticmethod
    def _extract_regular_polygon_params(segments: List) -> Dict:
        """提取正多边形参数（支持旋转）"""
        # 计算中心点（所有顶点的平均）
        points = [s['start'] for s in segments]
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        center = Point(center_x, center_y)
        
        # 计算外接圆半径（中心到顶点的距离）
        radius = sum(center.distance(Point(*p)) for p in points) / len(points)
        
        # 计算旋转角度
        n = len(segments)
        
        # 计算第一个顶点相对于中心的角度
        first_point = Point(*points[0])
        first_angle = math.atan2(first_point.y - center_y, first_point.x - center_x) * 180 / math.pi
        
        # CadQuery regularPolygon(r, n, angle=0) 默认一个顶点在 90°（y 轴正方向，即顶点朝上）
        # 对所有 n 实测均如此：vertices at {90+i*360/n for i in range(n)}
        default_first_angle = 90.0
        
        # 计算相对于默认方向的旋转角度
        angle = first_angle - default_first_angle
        
        # 对于正多边形，旋转 360/n 度后形状相同，将角度归一化到 [0, 360/n)
        angle_step = 360.0 / n
        angle = angle % angle_step
        
        # 如果角度接近 0 或 angle_step，归零
        if abs(angle) < 0.1 or abs(angle - angle_step) < 0.1:
            angle = 0
        
        result = {
            'type': 'regularPolygon',
            'center': (center_x, center_y),
            'radius': radius,
            'sides': len(segments)
        }
        
        # 如果角度不是0，添加angle参数
        if abs(angle) > 0.1:
            result['angle'] = angle
        
        return result
    
    @staticmethod
    def _extract_polygon_params(segments: List) -> Dict:
        """提取一般多边形参数"""
        # 收集所有顶点（绝对坐标）
        points_abs = [s['start'] for s in segments]
        
        return {
            'type': 'polygon',
            'points': points_abs  # 使用绝对坐标
        }


class ScriptParser:
    """解析CADQuery脚本"""
    
    def __init__(self, content: str):
        self.content = content
        self.loops = {}
    
    def _get_defined_loop_names(self):
        # """从原始脚本中提取所有被定义的loop名称（loop_X = ...）"""
        # # 匹配所有 loop_数字 = 的定义
        # names = re.findall(r'(loop_\d+)\s*=', self.content)
        # # 保持出现顺序、去重
        # seen = set()
        # unique = []
        # for n in names:
        #     if n not in seen:
        #         seen.add(n)
        #         unique.append(n)
        # return unique
        """根据数据生成规律（json2cadquery 全局连续计数），取最后一个 loop 的编号 N，则共有 N 个 loop：loop_1 .. loop_N"""
        # 匹配所有 loop_数字 = 的定义，取最大编号
        numbers = re.findall(r'loop_(\d+)\s*=', self.content)
        if not numbers:
            return []
        max_num = max(int(n) for n in numbers)
        return [f"loop_{i}" for i in range(1, max_num + 1)]
    
    def parse(self):
        """解析所有loop定义"""
        defined_loop_names = self._get_defined_loop_names()
        
        # 解析circle类型的loop（带push的）
        self._parse_circles()
        
        # 解析segment+arc类型的loop（复杂格式）及简单circle
        self._parse_sketch_loops()
        
        parsed_names = set(self.loops.keys())
        defined_set = set(defined_loop_names)
        missing = defined_set - parsed_names
        
        if missing:
            raise ValueError(
                f"未处理全部loop：原始脚本共定义 {len(defined_set)} 个loop "
                f"( {', '.join(sorted(defined_set))} )，仅解析出 {len(parsed_names)} 个。"
                f"未处理的loop: {', '.join(sorted(missing))}"
            )
        
        return self.loops
    
    def _parse_circles(self):
        """解析圆形loop"""
        pattern = r'(loop_\d+)\s*=\s*cq\.Sketch\(\)\.push\(\[\(([^)]+)\)\]\)\.circle\(([^,)]+)(?:,\s*tag=[\'"]([^\'"]+)[\'"])?\)'
        
        for match in re.finditer(pattern, self.content):
            loop_name = match.group(1)
            center = tuple(float(x.strip()) for x in match.group(2).split(','))
            radius = float(match.group(3))
            tag = match.group(4) if match.group(4) else None
            
            self.loops[loop_name] = {
                'type': 'circle',
                'center': center,
                'radius': radius,
                'tags': [tag] if tag else [],
                'original': match.group(0)
            }
    
    def _parse_sketch_loops(self):
        """解析所有类型的loop"""
        # 1. 匹配复杂的loop定义（带括号和assemble）
        complex_pattern = r'(loop_\d+)\s*=\s*\(\s*(cq\.Sketch\(\).*?\.assemble\([^)]*\))\s*\)'
        
        for match in re.finditer(complex_pattern, self.content, re.DOTALL):
            loop_name = match.group(1)
            loop_body = match.group(2)
            
            # 提取segments和arcs（保持顺序）
            elements = self._extract_elements_ordered(loop_body)
            segments = [e for e in elements if e['type'] == 'segment']
            arcs = [e for e in elements if e['type'] == 'arc']
            
            # 识别特征
            feature = FeatureRecognizer.recognize_from_segments_arcs(segments, arcs)
            
            self.loops[loop_name] = {
                **feature,
                'segments': segments,
                'arcs': arcs,
                'elements': elements,  # 保留原始顺序的所有元素
                'original': match.group(0)
            }
        
        # 2. 匹配简单的loop定义（如circle等直接调用）
        # 格式: loop_X = cq.Sketch().circle(radius, tag='...')
        simple_pattern = r'(loop_\d+)\s*=\s*cq\.Sketch\(\)\.(circle)\(([^)]+)\)'
        
        for match in re.finditer(simple_pattern, self.content):
            loop_name = match.group(1)
            method_name = match.group(2)
            args_str = match.group(3)
            
            # 跳过已经处理过的loop
            if loop_name in self.loops:
                continue
            
            # 解析circle
            if method_name == 'circle':
                # 提取radius和tag
                # 格式: radius 或 radius, tag='xxx'
                args_pattern = r'([\d.]+)(?:,\s*tag=[\'"]([^\'"]+)[\'"])?'
                args_match = re.match(args_pattern, args_str.strip())
                if args_match:
                    radius = float(args_match.group(1))
                    tag = args_match.group(2)
                    self.loops[loop_name] = {
                        'type': 'circle',
                        'radius': radius,
                        'center': (0, 0),  # circle默认中心在原点
                        'tags': [tag] if tag else [],
                        'original': match.group(0)
                    }
            
            # 可以继续添加其他简单格式的解析（rect、slot等）
    
    def _extract_elements_ordered(self, content: str) -> List[Dict]:
        """按出现顺序提取所有segment和arc"""
        elements = []
        
        # 找到所有segment和arc，保持它们的出现顺序
        # 使用正则表达式找到所有匹配，然后按位置排序
        
        # 找segment
        seg_pattern = r'\.segment\(\(([^)]+)\),\s*\(([^)]+)\)(?:,\s*tag=[\'"]([^\'"]+)[\'"])?\)'
        for match in re.finditer(seg_pattern, content):
            start = tuple(float(x.strip()) for x in match.group(1).split(','))
            end = tuple(float(x.strip()) for x in match.group(2).split(','))
            tag = match.group(3) if match.group(3) else None
            elements.append({
                'type': 'segment',
                'start': start,
                'end': end,
                'tag': tag,
                'pos': match.start()  # 在文本中的位置
            })
        
        # 找arc
        arc_pattern = r'\.arc\(\(([^)]+)\),\s*\(([^)]+)\),\s*\(([^)]+)\)(?:,\s*tag=[\'"]([^\'"]+)[\'"])?\)'
        for match in re.finditer(arc_pattern, content):
            start = tuple(float(x.strip()) for x in match.group(1).split(','))
            mid = tuple(float(x.strip()) for x in match.group(2).split(','))
            end = tuple(float(x.strip()) for x in match.group(3).split(','))
            tag = match.group(4) if match.group(4) else None
            elements.append({
                'type': 'arc',
                'start': start,
                'mid': mid,
                'end': end,
                'tag': tag,
                'pos': match.start()
            })
        
        # 按位置排序
        elements.sort(key=lambda x: x['pos'])
        
        return elements
    
    def _extract_segments(self, content: str) -> List[Dict]:
        """提取segment"""
        segments = []
        pattern = r'\.segment\(\(([^)]+)\),\s*\(([^)]+)\)(?:,\s*tag=[\'"]([^\'"]+)[\'"])?\)'
        
        for match in re.finditer(pattern, content):
            start = tuple(float(x.strip()) for x in match.group(1).split(','))
            end = tuple(float(x.strip()) for x in match.group(2).split(','))
            tag = match.group(3) if match.group(3) else None
            segments.append({'start': start, 'end': end, 'tag': tag})
        
        return segments
    
    def _extract_arcs(self, content: str) -> List[Dict]:
        """提取arc"""
        arcs = []
        pattern = r'\.arc\(\(([^)]+)\),\s*\(([^)]+)\),\s*\(([^)]+)\)(?:,\s*tag=[\'"]([^\'"]+)[\'"])?\)'
        
        for match in re.finditer(pattern, content):
            start = tuple(float(x.strip()) for x in match.group(1).split(','))
            mid = tuple(float(x.strip()) for x in match.group(2).split(','))
            end = tuple(float(x.strip()) for x in match.group(3).split(','))
            tag = match.group(4) if match.group(4) else None
            arcs.append({'start': start, 'mid': mid, 'end': end, 'tag': tag})
        
        return arcs


class HighLevelCodeGenerator:
    """生成高级语义代码"""

    _VAR_LOOP_TYPES = ('circle', 'rect', 'fillet', 'slot', 'regularPolygon', 'polygon')

    def __init__(self, loops: Dict):
        self.loops = loops
        self.variables = []
        self.arrays = self._detect_arrays()
        self._array_center_var_counter = 0
        self._array_center_vars: Dict[Tuple[str, ...], str] = {}  # (loop_name, ...) -> var_name
        # loop_name -> var_N, N starts from 1 for first variable-holding loop
        var_loops = sorted(
            (n for n, l in loops.items() if l.get('type') in self._VAR_LOOP_TYPES),
            key=lambda n: int(n.replace('loop_', ''))
        )
        self._loop_to_var_prefix = {name: f"var_{i}" for i, name in enumerate(var_loops, 1)}

    def _get_var_prefix(self, loop_name: str) -> str:
        """Return var_1, var_2, ... for loops with variables; fallback for unknown."""
        return self._loop_to_var_prefix.get(loop_name, loop_name.replace('loop_', 'var_'))
    
    @staticmethod
    def _is_regular_array_pattern(centers: List[Tuple[float, float]], tol: float = 1e-5) -> bool:
        """
        Check if centers form a CAD-style regular array pattern:
        - Linear/rectangular: collinear with equal spacing, or 2D grid
        - Circular: points on a circle with equal angular spacing
        Returns False for arbitrary positions.
        """
        if len(centers) < 3:
            return False
        pts = [(float(c[0]), float(c[1])) for c in centers]

        # --- Linear pattern: collinear + equal spacing ---
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def collinear():
            o = pts[0]
            for p in pts[2:]:
                if abs(cross(o, pts[1], p)) > tol * (1 + abs(cross(o, pts[1], p))):
                    return False
            return True

        if collinear():
            # Sort by projection onto line direction
            dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
            L = math.sqrt(dx * dx + dy * dy)
            if L < tol:
                return False
            ux, uy = dx / L, dy / L
            proj = [(ux * (p[0] - pts[0][0]) + uy * (p[1] - pts[0][1]), p) for p in pts]
            proj.sort(key=lambda x: x[0])
            gaps = [proj[i + 1][0] - proj[i][0] for i in range(len(proj) - 1)]
            if gaps and all(abs(g - gaps[0]) <= tol * (1 + abs(gaps[0])) for g in gaps):
                return True

        # --- Circular pattern: same radius from centroid, equal angular spacing ---
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        radii = [math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in pts]
        R_mean = sum(radii) / len(radii)
        if R_mean < tol:
            return False
        if any(abs(r - R_mean) > tol * (1 + R_mean) for r in radii):
            return False
        angles = [math.atan2(p[1] - cy, p[0] - cx) for p in pts]
        angles.sort()
        gaps = []
        for i in range(len(angles)):
            d = angles[(i + 1) % len(angles)] - angles[i]
            if d < 0:
                d += 2 * math.pi
            gaps.append(d)
        gap_ref = 2 * math.pi / len(pts)
        if all(abs(g - gap_ref) <= tol for g in gaps):
            return True

        # --- 2D rectangular grid: unique x/y with tolerance, equal dx/dy ---
        def unique_sorted(vals, t=tol):
            out = []
            for v in sorted(vals):
                if not out or abs(v - out[-1]) > t * (1 + abs(v)):
                    out.append(v)
            return out

        xs = unique_sorted(p[0] for p in pts)
        ys = unique_sorted(p[1] for p in pts)
        if len(xs) >= 2 and len(ys) >= 2 and len(pts) == len(xs) * len(ys):
            dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            dys = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            if (all(abs(d - dxs[0]) <= tol * (1 + abs(dxs[0])) for d in dxs) and
                    all(abs(d - dys[0]) <= tol * (1 + abs(dys[0])) for d in dys)):
                return True

        return False

    def _detect_arrays(self) -> List[Dict]:
        """检测阵列模式（相似特征 + 位置符合 CAD 阵列规则：线性/环形/网格）"""
        arrays = []
        used = set()

        loop_names = list(self.loops.keys())
        for i, name1 in enumerate(loop_names):
            if name1 in used:
                continue

            loop1 = self.loops[name1]
            similar = [name1]

            for name2 in loop_names[i + 1:]:
                if name2 in used:
                    continue

                loop2 = self.loops[name2]
                if self._is_similar(loop1, loop2):
                    similar.append(name2)
                    used.add(name2)

            if len(similar) > 1:
                used.add(name1)
                centers = []
                for name in similar:
                    loop = self.loops.get(name)
                    if loop and 'center' in loop:
                        c = loop['center']
                        centers.append((c[0], c[1]) if isinstance(c, (list, tuple)) else (float(c), 0.0))
                is_regular = self._is_regular_array_pattern(centers) if len(centers) == len(similar) else False
                arrays.append({
                    'loops': similar,
                    'type': loop1['type'],
                    'base_loop': loop1,
                    'is_regular_pattern': is_regular,
                })
        return arrays
    
    @staticmethod
    def _angles_equal_360(a1: float, a2: float, tol_deg: float = 1.0) -> bool:
        """判断两角度是否相等（考虑 360° 周期，归一化到 (-180, 180] 比较）"""
        def norm(x):
            while x > 180:
                x -= 360
            while x <= -180:
                x += 360
            return x
        d = abs(norm(a1) - norm(a2))
        if d > 180:
            d = 360 - d
        return d < tol_deg

    @staticmethod
    def _slot_angles_equal(a1: float, a2: float, tol_deg: float = 1.0) -> bool:
        """槽方向角相等（槽 180° 对称，两角已为 [0, 180) 或在此比较）"""
        # 归一化到 [0, 180)
        a1 = (a1 % 360) % 180
        a2 = (a2 % 360) % 180
        return abs(a1 - a2) < tol_deg

    def _is_similar(self, loop1: Dict, loop2: Dict, tolerance=1e-5, angle_tolerance_deg=0.1) -> bool:
        """判断两个 loop 是否相似：几何参数一致且方向/角度相同（同向可重叠，非旋转关系）"""
        if loop1['type'] != loop2['type']:
            return False
        
        ftype = loop1['type']
        
        if ftype == 'circle':
            # 圆无方向，只比半径
            return abs(loop1['radius'] - loop2['radius']) < tolerance
        
        elif ftype == 'rect':
            if (abs(loop1['width'] - loop2['width']) >= tolerance or
                    abs(loop1['height'] - loop2['height']) >= tolerance):
                return False
            a1 = loop1.get('angle', 0)
            a2 = loop2.get('angle', 0)
            return self._angles_equal_360(a1, a2, angle_tolerance_deg)
        
        elif ftype == 'fillet':
            if (abs(loop1['width'] - loop2['width']) >= tolerance or
                    abs(loop1['height'] - loop2['height']) >= tolerance or
                    abs(loop1['radius'] - loop2['radius']) >= tolerance):
                return False
            a1 = loop1.get('angle', 0)
            a2 = loop2.get('angle', 0)
            return self._angles_equal_360(a1, a2, angle_tolerance_deg)
        
        elif ftype == 'slot':
            if (abs(loop1['length'] - loop2['length']) >= tolerance or
                    abs(loop1['diameter'] - loop2['diameter']) >= tolerance):
                return False
            a1 = loop1.get('angle', 0)
            a2 = loop2.get('angle', 0)
            return self._slot_angles_equal(a1, a2, angle_tolerance_deg)
        
        return False
    
    # def generate_loop_code(self, loop_name: str, in_array: bool = False) -> str:
    #     """为指定loop生成高级代码（已废弃，保留兼容性）"""
    #     loop = self.loops[loop_name]
    #     var_prefix = loop_name.replace('loop_', 'var_')
        
    #     if loop['type'] == 'circle':
    #         self.variables.append(f"{var_prefix}_center = {loop['center']}")
    #         self.variables.append(f"{var_prefix}_radius = {loop['radius']:.6f}")
    #         return f"cq.Sketch().push([{var_prefix}_center]).circle({var_prefix}_radius)"
        
    #     elif loop['type'] == 'rect':
    #         self.variables.append(f"{var_prefix}_center = {self._format_point(loop['center'])}")
    #         self.variables.append(f"{var_prefix}_width = {loop['width']:.6f}")
    #         self.variables.append(f"{var_prefix}_height = {loop['height']:.6f}")
    #         return f"cq.Sketch().push([{var_prefix}_center]).rect({var_prefix}_width, {var_prefix}_height)"
        
    #     elif loop['type'] == 'fillet':
    #         self.variables.append(f"{var_prefix}_center = {self._format_point(loop['center'])}")
    #         self.variables.append(f"{var_prefix}_width = {loop['width']:.6f}")
    #         self.variables.append(f"{var_prefix}_height = {loop['height']:.6f}")
    #         self.variables.append(f"{var_prefix}_fillet_radius = {loop['radius']:.6f}")
    #         return f"cq.Sketch().push([{var_prefix}_center]).rect({var_prefix}_width, {var_prefix}_height).reset().vertices().fillet({var_prefix}_fillet_radius).assemble()"
        
    #     elif loop['type'] == 'slot':
    #         self.variables.append(f"{var_prefix}_center = {self._format_point(loop['center'])}")
    #         self.variables.append(f"{var_prefix}_length = {loop['length']:.6f}")
    #         self.variables.append(f"{var_prefix}_diameter = {loop['diameter']:.6f}")
    #         return f"cq.Sketch().push([{var_prefix}_center]).slot({var_prefix}_length, {var_prefix}_diameter)"
        
    #     else:
    #         return loop['original']
    
    def generate_sketch_method(self, loop_name: str, mode: Optional[str] = None) -> str:
        """生成sketch的方法调用（不包含cq.Sketch()）"""
        loop = self.loops[loop_name]
        var_prefix = self._get_var_prefix(loop_name)
        
        # 确保变量已定义（用 var_prefix+"_" 避免 "var_1" 误匹配 "var_10_center" 等子串）
        # When center is at origin (0,0), skip redundant center variable
        if not any((var_prefix + "_") in v for v in self.variables):
            if loop['type'] == 'circle':
                if not self._is_origin_center(loop.get('center')):
                    self.variables.append(f"{var_prefix}_center = {loop['center']}")
                self.variables.append(f"{var_prefix}_radius = {format_coord_str(loop['radius'])}")
            
            elif loop['type'] == 'rect':
                if not self._is_origin_center(loop.get('center')):
                    self.variables.append(f"{var_prefix}_center = {self._format_point(loop['center'])}")
                self.variables.append(f"{var_prefix}_width = {format_coord_str(loop['width'])}")
                self.variables.append(f"{var_prefix}_height = {format_coord_str(loop['height'])}")
                if 'angle' in loop:
                    self.variables.append(f"{var_prefix}_angle = {format_coord_str(loop['angle'])}")
            
            elif loop['type'] == 'fillet':
                if not self._is_origin_center(loop.get('center')):
                    self.variables.append(f"{var_prefix}_center = {self._format_point(loop['center'])}")
                self.variables.append(f"{var_prefix}_width = {format_coord_str(loop['width'])}")
                self.variables.append(f"{var_prefix}_height = {format_coord_str(loop['height'])}")
                self.variables.append(f"{var_prefix}_fillet_radius = {format_coord_str(loop['radius'])}")
                if 'angle' in loop:
                    self.variables.append(f"{var_prefix}_angle = {format_coord_str(loop['angle'])}")
            
            elif loop['type'] == 'slot':
                if not self._is_origin_center(loop.get('center')):
                    self.variables.append(f"{var_prefix}_center = {self._format_point(loop['center'])}")
                self.variables.append(f"{var_prefix}_length = {format_coord_str(loop['length'])}")
                self.variables.append(f"{var_prefix}_diameter = {format_coord_str(loop['diameter'])}")
                slot_angle = loop.get('angle', 0)
                if abs(slot_angle) > 0.1:
                    self.variables.append(f"{var_prefix}_angle = {format_coord_str(slot_angle)}")
            
            elif loop['type'] == 'regularPolygon':
                if not self._is_origin_center(loop.get('center')):
                    self.variables.append(f"{var_prefix}_center = {self._format_point(loop['center'])}")
                self.variables.append(f"{var_prefix}_radius = {format_coord_str(loop['radius'])}")
                self.variables.append(f"{var_prefix}_sides = {loop['sides']}")
                if 'angle' in loop:
                    self.variables.append(f"{var_prefix}_angle = {format_coord_str(loop['angle'])}")
            
            elif loop['type'] == 'polygon':
                points_str = ', '.join([self._format_point(p) for p in loop['points']])
                self.variables.append(f"{var_prefix}_points = [{points_str}]")
            
            elif loop['type'] == 'unknown':
                # unknown类型不生成变量，直接使用原始值
                pass
        
        # 生成方法调用
        mode_str = f", mode='{mode}'" if mode else ""
        # When center is at origin, Sketch defaults to origin - omit push entirely
        at_origin = self._is_origin_center(loop.get('center'))
        push_prefix = "" if at_origin else f".push([{var_prefix}_center])"
        
        if loop['type'] == 'circle':
            return f"{push_prefix}.circle({var_prefix}_radius{mode_str}).reset()"
        
        elif loop['type'] == 'rect':
            angle_str = f", angle={var_prefix}_angle" if 'angle' in loop else ""
            return f"{push_prefix}.rect({var_prefix}_width, {var_prefix}_height{angle_str}{mode_str}).reset()"
        
        elif loop['type'] == 'fillet':
            # 圆角矩形需要先创建矩形，然后fillet
            angle_str = f", angle={var_prefix}_angle" if 'angle' in loop else ""
            if mode:
                return f"{push_prefix}.rect({var_prefix}_width, {var_prefix}_height{angle_str}, mode='{mode}').reset().vertices().fillet({var_prefix}_fillet_radius).reset()"
            else:
                return f"{push_prefix}.rect({var_prefix}_width, {var_prefix}_height{angle_str}).reset().vertices().fillet({var_prefix}_fillet_radius).reset()"
        
        elif loop['type'] == 'slot':
            angle_str = f", angle={var_prefix}_angle" if abs(loop.get('angle', 0)) > 0.1 else ""
            return f"{push_prefix}.slot({var_prefix}_length, {var_prefix}_diameter{angle_str}{mode_str}).reset()"
        
        elif loop['type'] == 'regularPolygon':
            angle_str = f", angle={var_prefix}_angle" if 'angle' in loop else ""
            return f"{push_prefix}.regularPolygon({var_prefix}_radius, {var_prefix}_sides{angle_str}{mode_str}).reset()"
        
        elif loop['type'] == 'polygon':
            # polygon使用绝对坐标点列表
            if mode:
                return f".polygon({var_prefix}_points, mode='{mode}')"
            else:
                return f".polygon({var_prefix}_points)"
        
        elif loop['type'] == 'unknown':
            # 未识别的复杂几何，保留原始segment/arc代码（带tag）
            # 返回特殊格式：MULTILINE:<line1>|<line2>|...
            # 这样在_process_bodies中可以识别并展开成多行
            parts = []
            
            # 使用elements保持原始顺序（segment和arc混合）
            if 'elements' in loop and isinstance(loop['elements'], list):
                for elem in loop['elements']:
                    if elem['type'] == 'segment':
                        start = self._format_point(elem['start'])
                        end = self._format_point(elem['end'])
                        parts.append(f".segment({start}, {end})")
                    
                    elif elem['type'] == 'arc':
                        start = self._format_point(elem['start'])
                        mid = self._format_point(elem['mid'])
                        end = self._format_point(elem['end'])
                        parts.append(f".arc({start}, {mid}, {end})")
            
            # 注意：不在这里添加.assemble()
            # 因为连续的unknown loop，只有最后一个才需要.assemble()
            # 这个逻辑在_process_bodies中处理
            
            # 返回多行标记；对于内孔（mode='s'）用特殊前缀，之后在 assemble 处加 mode='s'
            prefix = "MULTILINE_S:" if mode == 's' else "MULTILINE:"
            return prefix + "|".join(parts)
        
        else:
            # 完全未知的情况
            return f"  # 未识别的几何: {loop_name}"
    
    def _format_point(self, point: Tuple[float, float]) -> str:
        """格式化点坐标（不补足小数位，与 json2cadquery 一致）"""
        return f"({format_coord_str(point[0])}, {format_coord_str(point[1])})"

    def _is_origin_center(self, center) -> bool:
        """Check if center is at origin (0, 0). Skip redundant center variable when true."""
        if center is None:
            return False
        cx, cy = center[0], center[1] if len(center) >= 2 else 0.0
        return abs(cx) < 1e-9 and abs(cy) < 1e-9
    
    def get_variables(self) -> str:
        """获取所有变量定义，按 var_N 的自然数顺序排列（var_2, var_3, var_4, ...）"""
        def sort_key(line: str) -> tuple:
            # var_N_property -> (0, N, property); array_centers_N -> (1, N, '')
            m = re.match(r'^(var_(\d+)_(\w+))\s*=', line)
            if m:
                return (0, int(m.group(2)), m.group(3))
            m = re.match(r'^(array_centers_(\d+))\s*=', line)
            if m:
                return (1, int(m.group(2)), '')
            return (2, 0, line)  # other vars last, preserve order
        sorted_vars = sorted(self.variables, key=sort_key)
        return '\n'.join(sorted_vars)
    
    def get_array_info(self) -> str:
        """获取阵列信息（含阵列数量统计）"""
        if not self.arrays:
            return ""
        used_as_array = [a for a in self.arrays if len(a['loops']) >= 3 and a.get('is_regular_pattern', False)]
        info = ["# 检测到的阵列:"]
        for i, array in enumerate(self.arrays, 1):
            loops_str = ', '.join(array['loops'])
            n = len(array['loops'])
            reg = " [规则阵列]" if array.get('is_regular_pattern', False) else " [非规则，不合并]"
            info.append(f"#   阵列{i}: {loops_str} - {array['type']} x {n}{reg}")
        info.append(f"# 共 {len(self.arrays)} 个相似组，其中 {len(used_as_array)} 个用于阵列生成（3+ 重复且位置规则）")
        return '\n'.join(info)

    def get_loop_to_array(self) -> Dict[str, Tuple[int, Dict]]:
        """返回 loop_name -> (array_index, array_dict)，仅包含长度>=3 且位置符合规则的阵列"""
        result = {}
        for idx, arr in enumerate(self.arrays):
            if len(arr['loops']) >= 3 and arr.get('is_regular_pattern', False):
                for name in arr['loops']:
                    result[name] = (idx, arr)
        return result

    def generate_sketch_method_array(
        self, array_loop_names: List[str], mode: Optional[str] = None
    ) -> Optional[str]:
        """
        当 3 个及以上相同几何、同向的 loop 成组时，生成单次 .push(中心点列表).shape(...).reset() 阵列代码。
        会确保 base loop 变量已定义，并可能新增 array_centers_* 变量。
        不支持的类型（unknown/polygon 等）返回 None，调用方按单 loop 逐个输出。
        """
        if len(array_loop_names) < 3:
            return None
        base_loop = self.loops.get(array_loop_names[0])
        if not base_loop:
            return None
        ftype = base_loop.get('type')
        if ftype not in ('circle', 'rect', 'fillet', 'slot', 'regularPolygon'):
            return None
        # 确保 base 形状变量已定义（用第一个 loop 的 prefix）
        self.generate_sketch_method(array_loop_names[0], mode=mode)
        centers = []
        for name in array_loop_names:
            loop = self.loops.get(name)
            if not loop or 'center' not in loop:
                return None
            centers.append(loop['center'])
        key = tuple(array_loop_names)
        if key not in self._array_center_vars:
            var_name = f"array_centers_{self._array_center_var_counter}"
            self._array_center_var_counter += 1
            self._array_center_vars[key] = var_name
            centers_str = ', '.join(self._format_point(c) for c in centers)
            self.variables.append(f"{var_name} = [{centers_str}]")
        var_name = self._array_center_vars[key]
        var_prefix = self._get_var_prefix(array_loop_names[0])
        mode_str = f", mode='{mode}'" if mode else ""
        if ftype == 'circle':
            return f".push({var_name}).circle({var_prefix}_radius{mode_str}).reset()"
        if ftype == 'rect':
            angle_str = f", angle={var_prefix}_angle" if 'angle' in base_loop else ""
            return f".push({var_name}).rect({var_prefix}_width, {var_prefix}_height{angle_str}{mode_str}).reset()"
        if ftype == 'fillet':
            angle_str = f", angle={var_prefix}_angle" if 'angle' in base_loop else ""
            if mode:
                return f".push({var_name}).rect({var_prefix}_width, {var_prefix}_height{angle_str}, mode='{mode}').reset().vertices().fillet({var_prefix}_fillet_radius).reset()"
            return f".push({var_name}).rect({var_prefix}_width, {var_prefix}_height{angle_str}).reset().vertices().fillet({var_prefix}_fillet_radius).reset()"
        if ftype == 'slot':
            angle_str = f", angle={var_prefix}_angle" if abs(base_loop.get('angle', 0)) > 0.1 else ""
            return f".push({var_name}).slot({var_prefix}_length, {var_prefix}_diameter{angle_str}{mode_str}).reset()"
        if ftype == 'regularPolygon':
            angle_str = f", angle={var_prefix}_angle" if 'angle' in base_loop else ""
            return f".push({var_name}).regularPolygon({var_prefix}_radius, {var_prefix}_sides{angle_str}{mode_str}).reset()"
        return None


class ScriptConverter:
    """主转换器"""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        with open(input_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

    @staticmethod
    def _compute_loop_aliases(loops: Dict) -> Dict[str, str]:
        """检测 tags 完全相同的 loop，后出现的复用先出现的变量名。
        返回 {alias_loop_name: canonical_loop_name}。
        """
        loop_to_alias: Dict[str, str] = {}
        tag_to_first: Dict[frozenset, str] = {}
        for loop_name, loop_data in loops.items():
            tags = loop_data.get('tags')
            if not tags:
                continue
            key = frozenset(tags)
            if key in tag_to_first:
                loop_to_alias[loop_name] = tag_to_first[key]
            else:
                tag_to_first[key] = loop_name
        return loop_to_alias
    
    def convert(self, output_path: str, silent: bool = False):
        """执行转换
        
        Args:
            output_path: 输出文件路径
            silent: 是否静默模式（不打印信息）
        """
        # 1. 解析
        self.parser = ScriptParser(self.content)
        loops = self.parser.parse()
        
        if not silent:
            print(f"\n解析到 {len(loops)} 个loops:")
            for name, loop in loops.items():
                print(f"  - {name}: {loop['type']}")
        
        # 2. 生成高级代码
        generator = HighLevelCodeGenerator(loops)

        # 预计算 loop 别名（tag 完全相同的 loop 复用已有变量）
        loop_aliases = self._compute_loop_aliases(loops)

        # 预生成所有 loop 的变量（跳过别名 loop，避免重复定义 var_X_*）
        for loop_name in loops:
            if loop_name not in loop_aliases:
                generator.generate_sketch_method(loop_name)
        # 预注册所有阵列 run 的变量（3+ 重复用阵列，需提前生成 array_centers_*）
        self._register_array_variables(generator, loops)

        # 3. 构建输出脚本
        lines = []
        lines.append("import cadquery as cq")
        lines.append("# === 1. Sketch and Extrude ===")

        # 生成 tag 映射注释：tag -> loop_X（仅对 canonical loop）
        for loop_name, loop_data in loops.items():
            if loop_name in loop_aliases:
                continue
            ltype = loop_data.get('type')
            if ltype == 'fillet':
                seg_tags = loop_data.get('segment_tags', [])
                arc_tags = loop_data.get('arc_tags', [])
                all_tags = [t for t in seg_tags + arc_tags if t]
                if all_tags:
                    tags_str = ', '.join(f"'{t}'" for t in all_tags)
                    lines.append(f"# {tags_str} -> {loop_name}")
            elif 'tags' in loop_data and loop_data['tags']:
                tags_str = ', '.join(f"'{t}'" for t in loop_data['tags'] if t)
                if tags_str:
                    lines.append(f"# {tags_str} -> {loop_name}")

        vars_text = generator.get_variables()
        if vars_text:
            lines.append(vars_text)

        # 4. 处理每个body
        self._process_bodies(lines, loops, generator, loop_aliases)

        lines.append("# === 2. Export ===")
        lines.append("cq.exporters.export(result, 'output.step')")
        
        # 5. 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        if not silent:
            print(f"\n✓ 转换完成: {output_path}")
        return output_path
    
    def _extract_place_sketch_args(self, body_block: str) -> str:
        """从 body 文本中提取 placeSketch(...) 的完整参数（平衡括号）"""
        start_marker = '.placeSketch('
        pos = body_block.find(start_marker)
        if pos == -1:
            return ''
        start = pos + len(start_marker)
        depth = 1
        i = start
        while i < len(body_block) and depth > 0:
            if body_block[i] == '(':
                depth += 1
            elif body_block[i] == ')':
                depth -= 1
            i += 1
        return body_block[start:i-1].strip()
    
    def _split_sketch_args(self, args_str: str) -> List[str]:
        """按顶层逗号分割 placeSketch 参数，保留括号内内容不分割。返回有序参数列表。"""
        args = []
        current = []
        depth = 0
        for i, c in enumerate(args_str):
            if c == '(':
                depth += 1
                current.append(c)
            elif c == ')':
                depth -= 1
                current.append(c)
            elif c == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(c)
        if current:
            args.append(''.join(current).strip())
        return args
    
    def _parse_one_sketch_arg(self, arg_str: str) -> tuple:
        """解析单个 placeSketch 参数，返回 (main_loops, hole_loops)。"""
        arg_str = arg_str.strip()
        main_loops = []
        hole_loops = []
        if arg_str.startswith('(') and ')' in arg_str:
            inner = arg_str.strip('()').strip()
            parts = re.split(r'([+-])', inner)
            parts = [p.strip() for p in parts if p.strip()]
            current_mode = '+'
            for part in parts:
                if part in ['+', '-']:
                    current_mode = part
                elif part.startswith('loop_'):
                    if current_mode == '-':
                        hole_loops.append(part)
                    else:
                        main_loops.append(part)
        else:
            if arg_str.startswith('loop_'):
                main_loops.append(arg_str)
        return (main_loops, hole_loops)
    
    def _compute_array_runs(
        self,
        loop_names: List[str],
        loop_to_array: Dict[str, Tuple[int, Dict]],
        loops: Dict,
    ) -> List[Tuple[int, int, bool]]:
        """
        将 loop 列表按「阵列 run」分组。返回 [(start, end, is_array_run), ...]，
        is_array_run 为 True 表示 loop_names[start:end] 为同一阵列且长度>=3。
        """
        runs = []
        i = 0
        while i < len(loop_names):
            name = loop_names[i]
            if name not in loops:
                runs.append((i, i + 1, False))
                i += 1
                continue
            arr_info = loop_to_array.get(name)
            if arr_info is None or len(arr_info[1]['loops']) < 3:
                runs.append((i, i + 1, False))
                i += 1
                continue
            arr_idx, arr = arr_info[0], arr_info[1]
            j = i
            while j < len(loop_names) and loop_names[j] in loop_to_array:
                ai = loop_to_array[loop_names[j]]
                if ai[0] != arr_idx:
                    break
                j += 1
            if j - i >= 3:
                runs.append((i, j, True))
            else:
                for k in range(i, j):
                    runs.append((k, k + 1, False))
            i = j
        return runs

    def _register_array_variables(self, generator: HighLevelCodeGenerator, loops: Dict):
        """扫描所有 body 的 placeSketch 参数，为每个长度>=3 的阵列 run 生成变量（array_centers_*）。"""
        loop_to_array = generator.get_loop_to_array()
        body_pattern = r'# --- Body (\d+) ---'
        for match in re.finditer(body_pattern, self.content):
            body_id = match.group(1)
            block_start = match.start()
            next_body = self.content.find('# --- Body ', block_start + 10)
            next_export = self.content.find('# === 2. Export', block_start)
            end = len(self.content)
            if next_body != -1:
                end = min(end, next_body)
            if next_export != -1:
                end = min(end, next_export)
            body_block = self.content[block_start:end]
            sketch_content = self._extract_place_sketch_args(body_block)
            if not sketch_content:
                continue
            arg_list = self._split_sketch_args(sketch_content)
            sketch_items = [self._parse_one_sketch_arg(a) for a in arg_list]
            for main_loops, hole_loops in sketch_items:
                for start, end, is_arr in self._compute_array_runs(main_loops, loop_to_array, loops):
                    if is_arr:
                        generator.generate_sketch_method_array(main_loops[start:end], mode=None)
                for start, end, is_arr in self._compute_array_runs(hole_loops, loop_to_array, loops):
                    if is_arr:
                        generator.generate_sketch_method_array(hole_loops[start:end], mode=None)
    
    def _emit_sketch_var_from_chain(self, lines: List[str], sketch_var: str, chain: str):
        """根据 generate_sketch_method 返回的 chain 生成 sketch 变量定义。
        chain 以 '_DIRECT_:' 开头时直接把冒号后内容作为完整表达式输出。
        与原始脚本保持一致：多行块内不缩进，紧贴左侧。
        """
        if chain.startswith("_DIRECT_:"):
            lines.append(f"{sketch_var} = {chain[len('_DIRECT_:'):]}")
        elif chain.startswith("MULTILINE:") or chain.startswith("MULTILINE_S:"):
            parts = chain.split(":", 1)[1].split("|")
            lines.append(f"{sketch_var} = (")
            lines.append("cq.Sketch()")
            for part in parts:
                lines.append(part)
            lines.append(".assemble()")
            lines.append(")")
        else:
            lines.append(f"{sketch_var} = cq.Sketch(){chain}")

    def _process_bodies(self, lines: List[str], loops: Dict,
                        generator: HighLevelCodeGenerator,
                        loop_aliases: Dict[str, str] = None):
        """处理所有body。loop_aliases: {alias_loop -> canonical_loop}"""
        body_pattern = r'# --- Body (\d+) ---'
        defined_sketch_vars: set = set()
        if loop_aliases is None:
            loop_aliases = {}

        body_num = 1
        for match in re.finditer(body_pattern, self.content):
            body_id = match.group(1)
            block_start = match.start()
            next_body = self.content.find('# --- Body ', block_start + 10)
            next_export = self.content.find('# === 2. Export', block_start)
            end = len(self.content)
            if next_body != -1:
                end = min(end, next_body)
            if next_export != -1:
                end = min(end, next_export)
            body_block = self.content[block_start:end]

            sketch_content = self._extract_place_sketch_args(body_block)
            extrude_match = re.search(r'\.extrude\(([^)]+)\)', body_block)
            extrude_part = extrude_match.group(0) if extrude_match else '.extrude(0.1)'

            lines.append(f"# --- Body {body_id} ---")

            if not sketch_content:
                body_num += 1
                continue

            # 按顺序解析 placeSketch 的每个参数
            arg_list = self._split_sketch_args(sketch_content)
            sketch_items = [self._parse_one_sketch_arg(a) for a in arg_list]
            loop_to_array = generator.get_loop_to_array()

            loop_names = []
            for mains, holes in sketch_items:
                loop_names.extend(mains)
                loop_names.extend(holes)

            if not loop_names:
                body_num += 1
                continue

            # 提取workplane（平衡括号）
            workplane_def = ''
            wp_start = body_block.find('cq.Workplane(')
            if wp_start != -1:
                frag = body_block[wp_start:]
                depth = 0
                i = len('cq.Workplane')
                while i < len(frag):
                    if frag[i] == '(':
                        depth += 1
                    elif frag[i] == ')':
                        depth -= 1
                        if depth == 0:
                            workplane_def = frag[:i+1]
                            break
                    i += 1
            plane_match = re.search(r'cq\.Plane\(', workplane_def)
            if plane_match:
                start_pos = workplane_def.find('cq.Plane(')
                if start_pos != -1:
                    paren_count = 0
                    i = start_pos + len('cq.Plane')
                    while i < len(workplane_def):
                        if workplane_def[i] == '(':
                            paren_count += 1
                        elif workplane_def[i] == ')':
                            paren_count -= 1
                            if paren_count == 0:
                                plane_def = workplane_def[start_pos:i+1]
                                break
                        i += 1
                    else:
                        plane_def = workplane_def
                else:
                    plane_def = workplane_def
            else:
                plane_def = workplane_def

            # 为每个 sketch_item 生成独立的 loop 变量，重建 placeSketch 参数
            place_sketch_args = []

            for main_loops, hole_loops in sketch_items:
                main_runs = self._compute_array_runs(main_loops, loop_to_array, loops)
                hole_runs = self._compute_array_runs(hole_loops, loop_to_array, loops)

                main_var_names = []
                hole_var_names = []

                def resolve_loop(name):
                    """若 name 是别名，返回其 canonical 名"""
                    return loop_aliases.get(name, name)

                def emit_loop_var(loop_name):
                    """为 loop_name 生成变量（若是别名则跳过，直接返回 canonical 名）"""
                    canonical = resolve_loop(loop_name)
                    if canonical not in defined_sketch_vars:
                        chain = generator.generate_sketch_method(canonical)
                        self._emit_sketch_var_from_chain(lines, canonical, chain)
                        defined_sketch_vars.add(canonical)
                    return canonical

                # 生成主轮廓 loop 变量
                for start, end, is_arr in main_runs:
                    run_loops = main_loops[start:end]
                    if is_arr:
                        loop_var = run_loops[0]
                        if loop_var not in defined_sketch_vars:
                            array_chain = generator.generate_sketch_method_array(run_loops)
                            if array_chain:
                                self._emit_sketch_var_from_chain(
                                    lines, loop_var, f"_DIRECT_:cq.Sketch(){array_chain}"
                                )
                                defined_sketch_vars.add(loop_var)
                        main_var_names.append(loop_var)
                    else:
                        for loop_name in run_loops:
                            if loop_name not in loops:
                                continue
                            main_var_names.append(emit_loop_var(loop_name))

                # 生成内孔 loop 变量
                for start, end, is_arr in hole_runs:
                    run_loops = hole_loops[start:end]
                    if is_arr:
                        loop_var = run_loops[0]
                        if loop_var not in defined_sketch_vars:
                            array_chain = generator.generate_sketch_method_array(run_loops)
                            if array_chain:
                                self._emit_sketch_var_from_chain(
                                    lines, loop_var, f"_DIRECT_:cq.Sketch(){array_chain}"
                                )
                                defined_sketch_vars.add(loop_var)
                        hole_var_names.append(loop_var)
                    else:
                        for loop_name in run_loops:
                            if loop_name not in loops:
                                continue
                            hole_var_names.append(emit_loop_var(loop_name))

                # 构建本 item 的 placeSketch 参数表达式
                if main_var_names and hole_var_names:
                    outer_str = main_var_names[0] if len(main_var_names) == 1 else f"({'+ '.join(main_var_names)})"
                    holes_str = ' - '.join(hole_var_names)
                    place_sketch_args.append(f"({outer_str} - {holes_str})")
                elif main_var_names:
                    if len(main_var_names) == 1:
                        place_sketch_args.append(main_var_names[0])
                    else:
                        place_sketch_args.append(f"({'+ '.join(main_var_names)})")
                elif hole_var_names:
                    place_sketch_args.extend(hole_var_names)

            place_sketch_str = ', '.join(place_sketch_args)

            extrude_match2 = re.search(r'extrude\(([^)]+)\)', extrude_part)
            extrude_amount = extrude_match2.group(1) if extrude_match2 else "0.1"

            lines.append(f"sketch_{body_id} = cq.Workplane({plane_def}).placeSketch({place_sketch_str})")
            lines.append(f"body_{body_id} = sketch_{body_id}.extrude({extrude_amount})")

            if body_num == 1:
                lines.append(f"result = body_{body_id}")
            else:
                operation = 'union'
                op_pattern = rf'result\s*=\s*result\.(union|cut|intersect)\(body_{body_id}\.val\(\)\)'
                op_match = re.search(op_pattern, self.content)
                if op_match:
                    operation = op_match.group(1)
                lines.append(f"result = result.{operation}(body_{body_id}.val())")

            body_num += 1




class HighLevelInlineApiGenerator(HighLevelCodeGenerator):
    """Same recognition as HighLevelCodeGenerator; output uses inline args only (no var_* lines)."""

    @staticmethod
    def _resolved_center(loop: Dict) -> tuple:
        c = loop.get("center")
        if c is None:
            return (0.0, 0.0)
        return (float(c[0]), float(c[1]))

    def generate_sketch_method(self, loop_name: str, mode: Optional[str] = None) -> str:
        loop = self.loops[loop_name]
        mode_str = f", mode='{mode}'" if mode else ""
        center = self._resolved_center(loop)
        at_origin = self._is_origin_center(center)
        push_prefix = "" if at_origin else f".push([{self._format_point(center)}])"

        if loop["type"] == "circle":
            return f"{push_prefix}.circle({format_coord_str(loop['radius'])}{mode_str}).reset()"

        if loop["type"] == "rect":
            angle_str = f", angle={format_coord_str(loop['angle'])}" if "angle" in loop else ""
            return (
                f"{push_prefix}.rect({format_coord_str(loop['width'])}, "
                f"{format_coord_str(loop['height'])}{angle_str}{mode_str}).reset()"
            )

        if loop["type"] == "fillet":
            angle_str = f", angle={format_coord_str(loop['angle'])}" if "angle" in loop else ""
            fr = format_coord_str(loop["radius"])
            if mode:
                return (
                    f"{push_prefix}.rect({format_coord_str(loop['width'])}, "
                    f"{format_coord_str(loop['height'])}{angle_str}, mode='{mode}').reset().vertices().fillet({fr}).reset()"
                )
            return (
                f"{push_prefix}.rect({format_coord_str(loop['width'])}, "
                f"{format_coord_str(loop['height'])}{angle_str}).reset().vertices().fillet({fr}).reset()"
            )

        if loop["type"] == "slot":
            angle_str = f", angle={format_coord_str(loop['angle'])}" if abs(loop.get("angle", 0)) > 0.1 else ""
            return (
                f"{push_prefix}.slot({format_coord_str(loop['length'])}, "
                f"{format_coord_str(loop['diameter'])}{angle_str}{mode_str}).reset()"
            )

        if loop["type"] == "regularPolygon":
            angle_str = f", angle={format_coord_str(loop['angle'])}" if "angle" in loop else ""
            return (
                f"{push_prefix}.regularPolygon({format_coord_str(loop['radius'])}, "
                f"{loop['sides']}{angle_str}{mode_str}).reset()"
            )

        if loop["type"] == "polygon":
            points_str = ", ".join([self._format_point(p) for p in loop["points"]])
            if mode:
                return f".polygon([{points_str}], mode='{mode}')"
            return f".polygon([{points_str}])"

        if loop["type"] == "unknown":
            parts = []
            if "elements" in loop and isinstance(loop["elements"], list):
                for elem in loop["elements"]:
                    if elem["type"] == "segment":
                        start = self._format_point(elem["start"])
                        end = self._format_point(elem["end"])
                        parts.append(f".segment({start}, {end})")
                    elif elem["type"] == "arc":
                        start = self._format_point(elem["start"])
                        mid = self._format_point(elem["mid"])
                        end = self._format_point(elem["end"])
                        parts.append(f".arc({start}, {mid}, {end})")
            prefix = "MULTILINE_S:" if mode == "s" else "MULTILINE:"
            return prefix + "|".join(parts)

        return f"  # Unrecognized geometry: {loop_name}"

    def generate_sketch_method_array(
        self, array_loop_names: List[str], mode: Optional[str] = None
    ) -> Optional[str]:
        if len(array_loop_names) < 3:
            return None
        base_loop = self.loops.get(array_loop_names[0])
        if not base_loop:
            return None
        ftype = base_loop.get("type")
        if ftype not in ("circle", "rect", "fillet", "slot", "regularPolygon"):
            return None
        centers = []
        for name in array_loop_names:
            loop = self.loops.get(name)
            if not loop or "center" not in loop:
                return None
            centers.append(loop["center"])
        centers_list = f"[{', '.join(self._format_point(c) for c in centers)}]"
        mode_str = f", mode='{mode}'" if mode else ""
        if ftype == "circle":
            return f".push({centers_list}).circle({format_coord_str(base_loop['radius'])}{mode_str}).reset()"
        if ftype == "rect":
            angle_str = f", angle={format_coord_str(base_loop['angle'])}" if "angle" in base_loop else ""
            return (
                f".push({centers_list}).rect({format_coord_str(base_loop['width'])}, "
                f"{format_coord_str(base_loop['height'])}{angle_str}{mode_str}).reset()"
            )
        if ftype == "fillet":
            angle_str = f", angle={format_coord_str(base_loop['angle'])}" if "angle" in base_loop else ""
            fr = format_coord_str(base_loop["radius"])
            if mode:
                return (
                    f".push({centers_list}).rect({format_coord_str(base_loop['width'])}, "
                    f"{format_coord_str(base_loop['height'])}{angle_str}, mode='{mode}').reset().vertices().fillet({fr}).reset()"
                )
            return (
                f".push({centers_list}).rect({format_coord_str(base_loop['width'])}, "
                f"{format_coord_str(base_loop['height'])}{angle_str}).reset().vertices().fillet({fr}).reset()"
            )
        if ftype == "slot":
            angle_str = f", angle={format_coord_str(base_loop['angle'])}" if abs(base_loop.get("angle", 0)) > 0.1 else ""
            return (
                f".push({centers_list}).slot({format_coord_str(base_loop['length'])}, "
                f"{format_coord_str(base_loop['diameter'])}{angle_str}{mode_str}).reset()"
            )
        if ftype == "regularPolygon":
            angle_str = f", angle={format_coord_str(base_loop['angle'])}" if "angle" in base_loop else ""
            return (
                f".push({centers_list}).regularPolygon({format_coord_str(base_loop['radius'])}, "
                f"{base_loop['sides']}{angle_str}{mode_str}).reset()"
            )
        return None

class ScriptConverterHighApi(ScriptConverter):
    """Like ScriptConverter but uses HighLevelInlineApiGenerator (no var_* / array_centers_*)."""

    def convert(self, output_path: str, silent: bool = False) -> str:
        self.parser = ScriptParser(self.content)
        loops = self.parser.parse()

        if not silent:
            print(f"\n解析到 {len(loops)} 个loops:")
            for name, loop in loops.items():
                print(f"  - {name}: {loop['type']}")

        generator = HighLevelInlineApiGenerator(loops)
        loop_aliases = self._compute_loop_aliases(loops)

        for loop_name in loops:
            if loop_name not in loop_aliases:
                generator.generate_sketch_method(loop_name)
        self._register_array_variables(generator, loops)

        lines = []
        lines.append("import cadquery as cq")
        lines.append("# === 1. Sketch and Extrude === (high-level API, inline params)")

        for loop_name, loop_data in loops.items():
            if loop_name in loop_aliases:
                continue
            ltype = loop_data.get("type")
            if ltype == "fillet":
                seg_tags = loop_data.get("segment_tags", [])
                arc_tags = loop_data.get("arc_tags", [])
                all_tags = [t for t in seg_tags + arc_tags if t]
                if all_tags:
                    tags_str = ", ".join(f"'{t}'" for t in all_tags)
                    lines.append(f"# {tags_str} -> {loop_name}")
            elif "tags" in loop_data and loop_data["tags"]:
                tags_str = ", ".join(f"'{t}'" for t in loop_data["tags"] if t)
                if tags_str:
                    lines.append(f"# {tags_str} -> {loop_name}")

        vars_text = generator.get_variables()
        if vars_text:
            lines.append(vars_text)

        self._process_bodies(lines, loops, generator, loop_aliases)

        lines.append("# === 2. Export ===")
        lines.append("cq.exporters.export(result, 'output.step')")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if not silent:
            print(f"\n✓ 转换完成 (inline high API): {output_path}")
        return output_path


def lift_program(primitive_code: str, constraints: dict, working_root):
    """Recognize geometry, apply source-evidence gates, then validate both targets."""
    import copy
    from .gate import Evidence, tags_of
    from .validation import execute_code, geometry_consistent, strip_exports

    primitive_code = strip_exports(primitive_code)
    baseline = execute_code(primitive_code, working_root)
    if not baseline['ok']:
        raise ValueError('Primitive reconstruction does not execute into a valid solid.')
    parser = ScriptParser(primitive_code)
    try:
        loops = parser.parse()
    except Exception as error:
        return primitive_code, primitive_code, {
            'decisions': [], 'fallback': 'unsupported_parser:' + type(error).__name__,
            'baseline': baseline,
        }
    evidence = Evidence(loops, constraints, FeatureRecognizer)
    decisions = []
    for name, loop in loops.items():
        try:
            accepted, reason = evidence.check(loop)
        except (ValueError, KeyError, ZeroDivisionError):
            accepted, reason = False, 'ambiguous_or_degenerate_evidence'
        decisions.append({'loop': name, 'candidate': loop['type'], 'accepted': accepted, 'reason': reason})
        if not accepted:
            loop['tags'] = sorted(tags_of(loop))
            loop['type'] = 'unknown'

    def emit(inline):
        base = HighLevelInlineApiGenerator if inline else HighLevelCodeGenerator

        class GatedGenerator(base):
            def _detect_arrays(self):
                return [array for array in super()._detect_arrays() if evidence.array_supported(array)]

        candidate_loops = copy.deepcopy(loops)
        generator = GatedGenerator(candidate_loops)
        converter = ScriptConverter.__new__(ScriptConverter)
        converter.content = primitive_code
        aliases = converter._compute_loop_aliases(candidate_loops)
        for name in candidate_loops:
            if name not in aliases:
                generator.generate_sketch_method(name)
        converter._register_array_variables(generator, candidate_loops)
        lines = ['import cadquery as cq']
        for name, loop in candidate_loops.items():
            if name not in aliases and tags_of(loop):
                lines.append('# ' + ', '.join(repr(t) for t in sorted(tags_of(loop))) + ' -> ' + name)
        variables = generator.get_variables()
        if variables:
            lines.append(variables)
        converter._process_bodies(lines, candidate_loops, generator, aliases)
        return '\n'.join(lines) + '\n'

    try:
        parameterized, semantic = emit(False), emit(True)
    except Exception as error:
        return primitive_code, primitive_code, {
            'decisions': decisions, 'fallback': 'unsupported_generator:' + type(error).__name__,
            'baseline': baseline,
        }
    checks = [execute_code(code, working_root) for code in [semantic, parameterized]]
    if not all(geometry_consistent(baseline, check) for check in checks):
        return primitive_code, primitive_code, {'decisions': decisions, 'fallback': 'stage_i_geometry_failed', 'baseline': baseline}
    return semantic, parameterized, {'decisions': decisions, 'fallback': None, 'baseline': checks[1]}
