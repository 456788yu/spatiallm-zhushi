import os  # 操作系统接口
import argparse  # 命令行参数解析
import math  # 数学运算
import csv  # CSV文件处理
import itertools  # 迭代工具
import logging  # 日志记录
from typing import Dict, List  # 类型注解
from collections import defaultdict  # 默认字典
from dataclasses import dataclass  # 数据类装饰器

import torch  # PyTorch深度学习框架
import pandas as pd  # 数据分析库
import numpy as np  # 数值计算库
from scipy.optimize import linear_sum_assignment  # 线性分配问题求解
from shapely import Polygon, LineString, polygonize, polygonize_full, make_valid  # 几何图形处理
from bbox import BBox3D  # 3D边界框
from bbox.metrics import iou_3d  # 3D IoU计算
from terminaltables import AsciiTable  # 终端表格显示

from spatiallm import Layout  # 空间布局类
from spatiallm.layout.entity import Wall, Door, Window, Bbox  # 布局实体类

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 常量定义
ZERO_TOLERANCE = 1e-6  # 零值容忍度
LARGE_COST_VALUE = 1e6  # 大成本值(用于优化问题)
# 常规物体类别列表
OBJECTS = [
    "curtain", "nightstand", "chandelier", "wardrobe", "bed", 
    "sofa", "chair", "cabinet", "dining table", "plants",
    "tv cabinet", "coffee table", "side table", "air conditioner", "dresser"
]
# 薄型物体类别列表
THIN_OBJECTS = ["painting", "carpet", "tv", "door", "window"]

@dataclass
class EvalTuple:
    """评估结果数据类，包含TP、预测数和真实数"""
    tp: int  # 真正例数
    num_pred: int  # 预测总数
    num_gt: int  # 真实总数

    @property
    def precision(self):
        """计算精确率"""
        return self.tp / self.num_pred if self.num_pred > 0 else 0

    @property
    def recall(self):
        """计算召回率"""
        return self.tp / self.num_gt if self.num_gt > 0 else 0

    @property
    def f1(self):
        """计算F1分数"""
        return (
            (2 * self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0 else 0
        )

    @property
    def masked(self):
        """检查是否被掩码(无预测和真实值)"""
        return self.num_pred == 0 and self.num_gt == 0


def calc_poly_iou(poly1, poly2):
    """计算两个多边形的IoU"""
    if poly1.intersects(poly2):  # 如果有交集
        inter_area = poly1.intersection(poly2).area  # 交集面积
        union_area = poly1.union(poly2).area  # 并集面积
        poly_iou = inter_area / union_area if union_area > 0 else 0
    else:
        poly_iou = 0
    return poly_iou


def construct_polygon(lines: List[LineString]):
    """从线集合构造多边形"""
    try:
        poly = polygonize(lines)  # 尝试构造多边形
        if poly.is_empty:  # 如果构造失败
            candidates = []
            for p in polygonize_full(lines):  # 尝试完整构造
                if p.is_empty:
                    continue

                candidate = p.geoms[0]
                if isinstance(candidate, Polygon):
                    candidates.append(candidate)
                elif isinstance(candidate, LineString):
                    candidates.append(Polygon(candidate))  # 将线转为多边形
                else:
                    log.warning(f"不支持的类型 {candidate.geom_type}")

            # 按面积排序选择最大的有效多边形
            candidates.sort(key=lambda x: x.area, reverse=True)
            poly = candidates[0]
            if not poly.is_valid:
                poly = make_valid(poly)  # 修复无效多边形
        return poly
    except Exception as e:
        log.error(f"从线构造多边形失败 {lines}", e)
        return Polygon()  # 返回空多边形

def read_label_mapping(label_path: str, label_from="spatiallm59", label_to="spatiallm18"):
    """读取标签映射文件"""
    assert os.path.isfile(label_path), f"标签映射文件 {label_path} 不存在"
    mapping = dict()
    with open(label_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")  # TSV文件读取
        for row in reader:
            label_from_value = row[label_from]
            label_to_value = row[label_to]
            if label_from_value == "" or label_to_value == "":
                continue
            mapping[label_from_value] = label_to_value  # 构建映射字典
    return mapping


def assign_class_map(entities: List[Bbox], class_map=Dict[str, str]):
    """根据类别映射表转换实体类别
    
    参数:
        entities: 边界框实体列表
        class_map: 类别映射字典
        
    返回:
        转换类别后的实体列表
    """
    res_entities = list()
    for entity in entities:
        # 将下划线替换为空格并查找映射
        mapping_to_class = class_map.get(entity.class_name.replace("_", " "))
        if mapping_to_class:
            entity.class_name = mapping_to_class  # 更新类别名称
            res_entities.append(entity)
    return res_entities


def get_entity_class(entity):
    """获取实体类别名称的通用方法
    
    参数:
        entity: 实体对象
        
    返回:
        实体类别名称字符串
    """
    try:
        return entity.class_name  # 尝试获取class_name属性
    except:
        return entity.entity_label  # 回退到entity_label属性


def get_BBox3D(entity: Bbox):
    """将Bbox实体转换为BBox3D对象
    
    参数:
        entity: Bbox实体对象
        
    返回:
        BBox3D对象
    """
    return BBox3D(
        entity.position_x,  # X坐标
        entity.position_y,  # Y坐标
        entity.position_z,  # Z坐标
        entity.scale_x,     # X方向尺寸
        entity.scale_y,     # Y方向尺寸
        entity.scale_z,     # Z方向尺寸
        euler_angles=[0, 0, entity.angle_z],  # 旋转角度(仅Z轴)
        is_center=True,     # 表示位置是中心点
    )


def calc_bbox_tp(
    pred_entities: List[Bbox], gt_entities: List[Bbox], iou_threshold: float = 0.25
):
    """计算边界框的匹配结果(真正例数)
    
    参数:
        pred_entities: 预测实体列表
        gt_entities: 真实实体列表
        iou_threshold: IoU阈值
        
    返回:
        EvalTuple对象(包含TP、预测数、真实数)
    """
    num_pred = len(pred_entities)
    num_gt = len(gt_entities)
    if num_pred == 0 or num_gt == 0:  # 边界情况处理
        return EvalTuple(0, num_pred, num_gt)

    # 计算所有预测和真实框之间的IoU矩阵
    iou_matrix = torch.as_tensor(
        [
            iou_3d(bbox_1, bbox_2)
            for bbox_1, bbox_2 in itertools.product(
                [get_BBox3D(entity) for entity in pred_entities],
                [get_BBox3D(entity) for entity in gt_entities],
            )
        ]
    ).resize(num_pred, num_gt)

    # 构建成本矩阵(大值表示不匹配)
    cost_matrix = torch.full((num_pred, num_gt), LARGE_COST_VALUE)
    cost_matrix[iou_matrix > iou_threshold] = -1  # IoU大于阈值的位置设为-1

    # 使用匈牙利算法进行最优匹配
    indices = linear_sum_assignment(cost_matrix.numpy())

    # 计算匹配对的IoU值
    tp_percent = iou_matrix[
        torch.as_tensor(indices[0], dtype=torch.int64),
        torch.as_tensor(indices[1], dtype=torch.int64),
    ]
    tp = torch.sum(tp_percent >= iou_threshold).item()  # 统计真正例数

    return EvalTuple(tp, num_pred, num_gt)


def is_valid_dw(entity: Door | Window, wall_id_lookup: Dict[int, Wall]):
    """检查门窗实体是否有效(是否附着在有效墙体上)
    
    参数:
        entity: 门窗实体
        wall_id_lookup: 墙体ID查找字典
        
    返回:
        bool: 是否有效
    """
    attach_wall = wall_id_lookup.get(entity.id, None)
    if attach_wall is None:  # 没有关联墙体
        return False

    # 计算墙体在X和Y方向的延伸范围
    wall_extent_x = max(
        max(attach_wall.ax, attach_wall.bx) - min(attach_wall.ax, attach_wall.bx), 0
    )
    wall_extent_y = max(
        max(attach_wall.ay, attach_wall.by) - min(attach_wall.ay, attach_wall.by), 0
    )
    # 检查墙体是否有有效尺寸
    return wall_extent_x > ZERO_TOLERANCE or wall_extent_y > ZERO_TOLERANCE


def get_corners(entity: Door | Window | Bbox, wall_id_lookup: Dict[int, Wall]):
    """获取实体的角点坐标
    
    参数:
        entity: 门、窗或边界框实体
        wall_id_lookup: 墙体ID查找字典
        
    返回:
        4个角点的3D坐标数组，形状为(4,3)
    """
    if isinstance(entity, (Door, Window)):
        # 处理门窗实体
        attach_wall = wall_id_lookup.get(entity.id, None)
        if attach_wall is None:
            log.error(f"{entity} 未找到关联墙体")
            return

        # 计算墙体方向向量
        wall_start = np.array([attach_wall.ax, attach_wall.ay])
        wall_end = np.array([attach_wall.bx, attach_wall.by])
        wall_length = np.linalg.norm(wall_end - wall_start)
        wall_xy_unit_vec = (wall_end - wall_start) / wall_length
        wall_xy_unit_vec = np.nan_to_num(wall_xy_unit_vec, nan=0)  # 处理零长度墙体

        # 计算门窗角点
        door_center = np.array(
            [entity.position_x, entity.position_y, entity.position_z]
        )
        offset = 0.5 * np.concatenate(
            [wall_xy_unit_vec * entity.width, np.array([entity.height])]
        )
        door_start_xyz = door_center - offset
        door_end_xyz = door_center + offset

        # 返回4个角点坐标
        return np.array([
            [door_start_xyz[0], door_start_xyz[1], door_start_xyz[2]],  # 左下前
            [door_end_xyz[0], door_end_xyz[1], door_start_xyz[2]],    # 右下前 
            [door_end_xyz[0], door_end_xyz[1], door_end_xyz[2]],      # 右后上
            [door_start_xyz[0], door_start_xyz[1], door_end_xyz[2]]   # 左后上
        ])
        
    elif isinstance(entity, Bbox):
        # 处理边界框实体
        bbox_points = get_BBox3D(entity).p  # 获取8个顶点
        
        # 找到最短边作为基准面
        scale_key = ["scale_x", "scale_y", "scale_z"]
        match min(scale_key, key=lambda k: abs(getattr(entity, k))):
            case "scale_x":  # X方向最短
                return np.array([
                    (bbox_points[0] + bbox_points[1]) / 2,  # 前下边中点
                    (bbox_points[2] + bbox_points[3]) / 2,  # 前上边中点
                    (bbox_points[6] + bbox_points[7]) / 2,  # 后上边中点
                    (bbox_points[4] + bbox_points[5]) / 2   # 后下边中点
                ])
            case "scale_y":  # Y方向最短
                return np.array([
                    (bbox_points[0] + bbox_points[3]) / 2,  # 左边中点
                    (bbox_points[4] + bbox_points[7]) / 2,  # 后边中点
                    (bbox_points[5] + bbox_points[6]) / 2,  # 右边中点
                    (bbox_points[1] + bbox_points[2]) / 2   # 前边中点
                ])
            case "scale_z":  # Z方向最短
                return np.array([
                    (bbox_points[0] + bbox_points[4]) / 2,  # 下边中点
                    (bbox_points[1] + bbox_points[5]) / 2,  # 前边中点
                    (bbox_points[2] + bbox_points[6]) / 2,  # 上边中点
                    (bbox_points[3] + bbox_points[7]) / 2   # 后边中点
                ])
            case _:
                log.error(f"无法识别的属性 {entity}")
                return


def are_planes_parallel_and_close(
    corners_1: np.ndarray,
    corners_2: np.ndarray,
    parallel_tolerance: float,
    dist_tolerance: float,
):
    """检查两个平面是否平行且接近
    
    参数:
        corners_1: 第一个平面的4个角点
        corners_2: 第二个平面的4个角点
        parallel_tolerance: 平行度容忍阈值
        dist_tolerance: 距离容忍阈值
        
    返回:
        bool: 是否平行且接近
    """
    # 计算两个平面的法向量
    p1, p2, p3, _ = corners_1
    q1, q2, q3, _ = corners_2
    n1 = np.cross(np.subtract(p2, p1), np.subtract(p3, p1))  # 平面1法向量
    n2 = np.cross(np.subtract(q2, q1), np.subtract(q3, q1))  # 平面2法向量
    
    # 检查法向量有效性
    n1_length = np.linalg.norm(n1)
    n2_length = np.linalg.norm(n2)
    assert (n1_length * n2_length > ZERO_TOLERANCE), f"无效的平面角点: {corners_1}, {corners_2}"

    # 计算平行度和距离
    cross_norm = np.linalg.norm(np.cross(n1, n2))  # 叉积范数衡量平行度
    normalized_cross = cross_norm / (n1_length * n2_length)  # 归一化
    
    distance = np.dot(np.subtract(q1, p1), n1) / n1_length  # 平面间距离
    
    # 判断是否平行且接近
    return (
        normalized_cross < parallel_tolerance  # 平行检查
        and distance < dist_tolerance         # 距离检查
    )


def calc_thin_bbox_iou_2d(
    corners_1: np.ndarray,
    corners_2: np.ndarray,
    parallel_tolerance: float,
    dist_tolerance: float,
):
    """计算薄型物体(如门、窗)的2D IoU
    
    参数:
        corners_1: 第一个物体的4个角点坐标
        corners_2: 第二个物体的4个角点坐标  
        parallel_tolerance: 平面平行度容忍阈值
        dist_tolerance: 平面距离容忍阈值
        
    返回:
        2D IoU值(0-1之间)
    """
    # 首先检查两个平面是否平行且接近
    if are_planes_parallel_and_close(corners_1, corners_2, parallel_tolerance, dist_tolerance):
        # 提取参考点
        p1, p2, _, p4 = corners_1
        
        # 计算投影基向量
        v1 = np.subtract(p2, p1)  # 第一条边向量
        v2 = np.subtract(p4, p1)  # 第二条边向量
        
        # 构建正交基
        basis1 = v1 / np.linalg.norm(v1)  # 主基向量
        basis1_orth = v2 - np.dot(v2, basis1) * basis1  # 正交分量
        basis2 = basis1_orth / np.linalg.norm(basis1_orth)  # 次基向量

        # 将3D点投影到2D平面
        projected_corners_1 = [
            [np.dot(np.subtract(point, p1), basis1),  # 投影到basis1
             np.dot(np.subtract(point, p1), basis2)]   # 投影到basis2
            for point in corners_1
        ]
        projected_corners_2 = [
            [np.dot(np.subtract(point, p1), basis1),
             np.dot(np.subtract(point, p1), basis2)]
            for point in corners_2
        ]
        
        # 创建2D多边形并计算IoU
        box1 = Polygon(projected_corners_1)
        box2 = Polygon(projected_corners_2)
        
        return calc_poly_iou(box1, box2)
    else:
        return 0  # 平面不平行或距离过远


def calc_thin_bbox_tp(
    pred_entities: List[Door | Window | Bbox],
    gt_entities: List[Door | Window | Bbox],
    pred_wall_id_lookup: Dict[int, Wall],
    gt_wall_id_lookup: Dict[int, Wall],
    iou_threshold: float = 0.25,
    parallel_tolerance: float = math.sin(math.radians(5)),  # 5度容忍
    dist_tolerance: float = 0.2,  # 20cm距离容忍
):
    """计算薄型物体的匹配结果(真正例数)
    
    参数:
        pred_entities: 预测实体列表
        gt_entities: 真实实体列表
        pred_wall_id_lookup: 预测墙体ID映射
        gt_wall_id_lookup: 真实墙体ID映射
        iou_threshold: IoU阈值
        parallel_tolerance: 平行度容忍
        dist_tolerance: 距离容忍
        
    返回:
        EvalTuple对象(包含TP、预测数、真实数)
    """
    num_pred = len(pred_entities)
    num_gt = len(gt_entities)
    
    # 边界情况处理
    if num_pred == 0 or num_gt == 0:
        return EvalTuple(0, num_pred, num_gt)

    # 计算所有预测和真实框之间的IoU矩阵
    iou_matrix = torch.as_tensor(
        [
            calc_thin_bbox_iou_2d(
                corners_1, corners_2, parallel_tolerance, dist_tolerance
            )
            for corners_1, corners_2 in itertools.product(
                [get_corners(entity, pred_wall_id_lookup) for entity in pred_entities],
                [get_corners(entity, gt_wall_id_lookup) for entity in gt_entities],
            )
        ]
    ).resize(num_pred, num_gt)

    # 构建成本矩阵(大值表示不匹配)
    cost_matrix = torch.full((num_pred, num_gt), LARGE_COST_VALUE)
    cost_matrix[iou_matrix > iou_threshold] = -1  # IoU大于阈值的位置设为-1

    # 使用匈牙利算法进行最优匹配
    indices = linear_sum_assignment(cost_matrix.numpy())

    # 计算匹配对的IoU值
    tp_percent = iou_matrix[
        torch.as_tensor(indices[0], dtype=torch.int64),
        torch.as_tensor(indices[1], dtype=torch.int64),
    ]
    tp = torch.sum(tp_percent >= iou_threshold).item()  # 统计真正例数

    return EvalTuple(tp, num_pred, num_gt)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser("SpatialLM evaluation script")
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="包含id,pcd,layout列的元数据CSV文件",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="真实布局txt文件目录路径",
    )
    parser.add_argument(
        "--pred_dir", 
        type=str,
        required=True,
        help="预测布局txt文件目录路径",
    )
    parser.add_argument(
        "--label_mapping",
        type=str,
        required=True,
        help="标签映射文件路径",
    )
    args = parser.parse_args()

    # 读取元数据文件
    df = pd.read_csv(args.metadata)
    scene_id_list = df["id"].tolist()  # 获取场景ID列表
    class_map = read_label_mapping(args.label_mapping)  # 读取标签映射

    # 初始化评估数据结构
    floorplan_ious = list()  # 存储各场景的平面图IoU
    classwise_eval_tuples: Dict[str, List[EvalTuple]] = defaultdict(list)  # 按类别存储评估结果

    # 遍历每个场景进行评估
    for scene_id in scene_id_list:
        log.info(f"Evaluating scene {scene_id}")
        
        # 加载预测和真实布局文件
        with open(os.path.join(args.pred_dir, f"{scene_id}.txt"), "r") as f:
            pred_layout = Layout(f.read())
        with open(os.path.join(args.gt_dir, f"{scene_id}.txt"), "r") as f:
            gt_layout = Layout(f.read())
        
        # 应用类别映射
        pred_layout.bboxes = assign_class_map(pred_layout.bboxes, class_map)
        gt_layout.bboxes = assign_class_map(gt_layout.bboxes, class_map)

        # 1. 计算平面图IoU (基于墙体)
        pred_poly = construct_polygon(
            [LineString([(w.ax, w.ay), (w.bx, w.by)]) for w in pred_layout.walls]
        )
        gt_poly = construct_polygon(
            [LineString([(w.ax, w.ay), (w.bx, w.by)]) for w in gt_layout.walls]
        )
        floorplan_ious.append(calc_poly_iou(pred_poly, gt_poly))

        # 2. 评估常规物体(使用3D IoU)
        pred_normal_objects = [b for b in pred_layout.bboxes if b.class_name in OBJECTS]
        gt_normal_objects = [b for b in gt_layout.bboxes if b.class_name in OBJECTS]
        
        for class_name in OBJECTS:
            classwise_eval_tuples[class_name].append(
                calc_bbox_tp(
                    pred_entities=[b for b in pred_normal_objects if get_entity_class(b) == class_name],
                    gt_entities=[b for b in gt_normal_objects if get_entity_class(b) == class_name],
                )
            )

        # 3. 评估薄型物体(使用2D IoU)
        pred_thin_objects = [b for b in pred_layout.bboxes if b.class_name in THIN_OBJECTS]
        gt_thin_objects = [b for b in gt_layout.bboxes if b.class_name in THIN_OBJECTS]
        
        # 创建墙体ID查找表
        pred_wall_id_lookup = {w.id: w for w in pred_layout.walls}
        gt_wall_id_lookup = {w.id: w for w in gt_layout.walls}
        
        # 添加有效的门窗实体
        pred_thin_objects += [e for e in pred_layout.doors + pred_layout.windows if is_valid_dw(e, pred_wall_id_lookup)]
        gt_thin_objects += [e for e in gt_layout.doors + gt_layout.windows if is_valid_dw(e, gt_wall_id_lookup)]

        for class_name in THIN_OBJECTS:
            classwise_eval_tuples[class_name].append(
                calc_thin_bbox_tp(
                    pred_entities=[b for b in pred_thin_objects if get_entity_class(b) == class_name],
                    gt_entities=[b for b in gt_thin_objects if get_entity_class(b) == class_name],
                    pred_wall_id_lookup=pred_wall_id_lookup,
                    gt_wall_id_lookup=gt_wall_id_lookup,
                )
            )

    # 打印评估结果表格

    # 1. 平面图IoU结果
    headers = ["Floorplan", "mean IoU"]
    table_data = [headers]
    table_data += [["wall", np.mean(floorplan_ious)]]
    print("\n" + AsciiTable(table_data).table)

    # 2. 常规物体F1分数结果
    headers = ["Objects", "F1 @.25 IoU"]
    table_data = [headers]
    for class_name in OBJECTS:
        tuples = classwise_eval_tuples[class_name]
        f1 = np.ma.masked_where([t.masked for t in tuples], [t.f1 for t in tuples]).mean()
        table_data.append([class_name, f1])
    print("\n" + AsciiTable(table_data).table)

    # 3. 薄型物体F1分数结果
    headers = ["Thin Objects", "F1 @.25 IoU"]
    table_data = [headers]
    for class_name in THIN_OBJECTS:
        tuples = classwise_eval_tuples[class_name]
        f1 = np.ma.masked_where([t.masked for t in tuples], [t.f1 for t in tuples]).mean()
        table_data.append([class_name, f1])
    print("\n" + AsciiTable(table_data).table)
