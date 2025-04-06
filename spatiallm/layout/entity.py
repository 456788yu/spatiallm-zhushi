# Copyright (c) Manycore Tech Inc. and affiliates.
# All rights reserved.

"""
This code is derived from the SceneScript language sequence and entity parameters.

Reference: https://github.com/facebookresearch/scenescript/blob/main/src/data/language_sequence.py
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R  # 用于3D旋转操作

# 归一化预设参数
NORMALIZATION_PRESET = {
    "world": (0.0, 32.0),    # 世界坐标范围
    "height": (0.0, 25.6),   # 高度范围
    "width": (0.0, 25.6),    # 宽度范围
    "scale": (0.0, 20.0),    # 缩放范围
    "angle": (-6.2832, 6.2832),  # 角度范围(-2π到2π)
    "num_bins": 640,         # 离散化分箱数
}

@dataclass
class Wall:
    """墙体数据类，表示3D空间中的一面墙"""
    id: int          # 墙体唯一标识
    ax: float        # 起点x坐标
    ay: float        # 起点y坐标
    az: float        # 起点z坐标
    bx: float        # 终点x坐标
    by: float        # 终点y坐标
    bz: float        # 终点z坐标
    height: float    # 墙高
    thickness: float # 墙厚
    entity_label: str = "wall"  # 实体标签

    def __post_init__(self):
        """初始化后处理，确保数据类型正确"""
        self.id = int(self.id)
        self.ax = float(self.ax)
        self.ay = float(self.ay)
        self.az = float(self.az)
        self.bx = float(self.bx)
        self.by = float(self.by)
        self.bz = float(self.bz)
        self.height = float(self.height)
        self.thickness = float(self.thickness)

    def rotate(self, angle: float):
        """绕Z轴旋转墙体
        
        参数:
            angle: 旋转角度(弧度)
        """
        wall_start = np.array([self.ax, self.ay, self.az])
        wall_end = np.array([self.bx, self.by, self.bz])
        rotmat = R.from_rotvec([0, 0, angle]).as_matrix()  # 创建绕Z轴旋转矩阵
        wall_start = rotmat @ wall_start  # 旋转起点
        wall_end = rotmat @ wall_end      # 旋转终点

        # 更新坐标
        self.ax = wall_start[0]
        self.ay = wall_start[1]
        self.az = wall_start[2]
        self.bx = wall_end[0]
        self.by = wall_end[1]
        self.bz = wall_end[2]

    def translate(self, translation: np.ndarray):
        """平移墙体
        
        参数:
            translation: 3D平移向量[x, y, z]
        """
        self.ax += translation[0]
        self.ay += translation[1]
        self.az += translation[2]
        self.bx += translation[0]
        self.by += translation[1]
        self.bz += translation[2]

    def scale(self, scaling: float):
        """缩放墙体尺寸
        
        参数:
            scaling: 缩放系数
        """
        self.height *= scaling
        self.thickness *= scaling
        self.ax *= scaling
        self.ay *= scaling
        self.az *= scaling
        self.bx *= scaling
        self.by *= scaling
        self.bz *= scaling

    def normalize_and_discretize(self):
        """归一化并离散化墙体参数"""
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]
        num_bins = NORMALIZATION_PRESET["num_bins"]

        # 归一化到[0, num_bins]范围
        self.height = (self.height - height_min) / (height_max - height_min) * num_bins
        self.thickness = (self.thickness - height_min) / (height_max - height_min) * num_bins
        self.ax = (self.ax - world_min) / (world_max - world_min) * num_bins
        self.ay = (self.ay - world_min) / (world_max - world_min) * num_bins
        self.az = (self.az - world_min) / (world_max - world_min) * num_bins
        self.bx = (self.bx - world_min) / (world_max - world_min) * num_bins
        self.by = (self.by - world_min) / (world_max - world_min) * num_bins
        self.bz = (self.bz - world_min) / (world_max - world_min) * num_bins

        # 确保值在有效范围内
        self.height = np.clip(self.height, 0, num_bins - 1)
        self.thickness = np.clip(self.thickness, 0, num_bins - 1)
        self.ax = np.clip(self.ax, 0, num_bins - 1)
        self.ay = np.clip(self.ay, 0, num_bins - 1)
        self.az = np.clip(self.az, 0, num_bins - 1)
        self.bx = np.clip(self.bx, 0, num_bins - 1)
        self.by = np.clip(self.by, 0, num_bins - 1)
        self.bz = np.clip(self.bz, 0, num_bins - 1)

    def undiscretize_and_unnormalize(self):
        """将离散化和归一化的墙体参数还原为原始尺度
        
        执行两个步骤:
        1. 反离散化: 将离散化的bin值转换回[0,1]范围的连续值
        2. 反归一化: 将[0,1]范围的值还原到原始物理尺度
        """
        # 获取归一化参数
        num_bins = NORMALIZATION_PRESET["num_bins"]
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]

        # 1. 反离散化 - 将bin值转换为[0,1]范围的连续值
        self.height = self.height / num_bins
        self.thickness = self.thickness / num_bins
        self.ax = self.ax / num_bins
        self.ay = self.ay / num_bins
        self.az = self.az / num_bins
        self.bx = self.bx / num_bins
        self.by = self.by / num_bins
        self.bz = self.bz / num_bins

        # 2. 反归一化 - 将[0,1]范围的值还原到原始物理尺度
        self.height = self.height * (height_max - height_min) + height_min
        self.thickness = self.thickness * (height_max - height_min) + height_min
        self.ax = self.ax * (world_max - world_min) + world_min
        self.ay = self.ay * (world_max - world_min) + world_min
        self.az = self.az * (world_max - world_min) + world_min
        self.bx = self.bx * (world_max - world_min) + world_min
        self.by = self.by * (world_max - world_min) + world_min
        self.bz = self.bz * (world_max - world_min) + world_min

    def to_language_string(self):
        """将墙体对象转换为可读的语言字符串
        
        返回格式示例:
        wall_0=Wall(1.2,3.4,5.6,7.8,9.0,2.1,3.0,0.2)
        
        返回:
            描述墙体的标准化字符串
        """
        capitalized_label = self.entity_label.capitalize()  # 首字母大写
        language_string = (
            f"{self.entity_label}_{self.id}="  # 实体类型和ID
            f"{capitalized_label}("  # 类名
            f"{self.ax},{self.ay},{self.az},"  # 起点坐标
            f"{self.bx},{self.by},{self.bz},"  # 终点坐标
            f"{self.height},{self.thickness})"  # 高度和厚度
        )
        return language_string


@dataclass
class Door:
    """门的数据类，表示3D空间中的一扇门"""
    id: int           # 门的唯一标识
    wall_id: int      # 所属墙体的ID
    position_x: float # 中心点x坐标
    position_y: float # 中心点y坐标
    position_z: float # 中心点z坐标
    width: float      # 门宽
    height: float     # 门高
    entity_label: str = "door"  # 实体标签

    def __post_init__(self):
        """初始化后处理，确保数据类型正确"""
        self.id = int(self.id)
        self.wall_id = int(self.wall_id)
        self.position_x = float(self.position_x)
        self.position_y = float(self.position_y)
        self.position_z = float(self.position_z)
        self.width = float(self.width)
        self.height = float(self.height)

    def rotate(self, angle: float):
        """绕Z轴旋转门
        
        参数:
            angle: 旋转角度(弧度)
        """
        center = np.array([self.position_x, self.position_y, self.position_z])
        rotmat = R.from_rotvec([0, 0, angle]).as_matrix()  # 创建绕Z轴旋转矩阵
        new_center = rotmat @ center  # 旋转中心点

        # 更新坐标
        self.position_x = new_center[0]
        self.position_y = new_center[1]
        self.position_z = new_center[2]

    def translate(self, translation: np.ndarray):
        """平移门
        
        参数:
            translation: 3D平移向量[x, y, z]
        """
        self.position_x += translation[0]
        self.position_y += translation[1]
        self.position_z += translation[2]

    def scale(self, scaling: float):
        """缩放门尺寸
        
        参数:
            scaling: 缩放系数
        """
        self.width *= scaling
        self.height *= scaling
        self.position_x *= scaling
        self.position_y *= scaling
        self.position_z *= scaling

    def normalize_and_discretize(self):
        """归一化并离散化门参数"""
        width_min, width_max = NORMALIZATION_PRESET["width"]
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]
        num_bins = NORMALIZATION_PRESET["num_bins"]

        # 归一化到[0, num_bins]范围
        self.width = (self.width - width_min) / (width_max - width_min) * num_bins
        self.height = (self.height - height_min) / (height_max - height_min) * num_bins
        self.position_x = (self.position_x - world_min) / (world_max - world_min) * num_bins
        self.position_y = (self.position_y - world_min) / (world_max - world_min) * num_bins
        self.position_z = (self.position_z - world_min) / (world_max - world_min) * num_bins

        # 确保值在有效范围内
        self.width = np.clip(self.width, 0, num_bins - 1)
        self.height = np.clip(self.height, 0, num_bins - 1)
        self.position_x = np.clip(self.position_x, 0, num_bins - 1)
        self.position_y = np.clip(self.position_y, 0, num_bins - 1)
        self.position_z = np.clip(self.position_z, 0, num_bins - 1)

    def undiscretize_and_unnormalize(self):
        """将离散化和归一化的门参数还原为原始尺度"""
        num_bins = NORMALIZATION_PRESET["num_bins"]
        width_min, width_max = NORMALIZATION_PRESET["width"]
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]

        # 反离散化 - 将bin值转换为[0,1]范围的连续值
        self.width = self.width / num_bins
        self.height = self.height / num_bins
        self.position_x = self.position_x / num_bins
        self.position_y = self.position_y / num_bins
        self.position_z = self.position_z / num_bins

        # 反归一化 - 将[0,1]范围的值还原到原始物理尺度
        self.width = self.width * (width_max - width_min) + width_min
        self.height = self.height * (height_max - height_min) + height_min
        self.position_x = self.position_x * (world_max - world_min) + world_min
        self.position_y = self.position_y * (world_max - world_min) + world_min
        self.position_z = self.position_z * (world_max - world_min) + world_min

    def to_language_string(self):
        """将门对象转换为可读的语言字符串
        
        返回格式示例:
        door_0=Door(wall_1,1.2,3.4,5.6,0.9,2.1)
        
        返回:
            描述门的标准化字符串
        """
        capitalized_label = self.entity_label.capitalize()  # 首字母大写
        self.id = self.id % 1000  # 限制ID范围
        language_string = (
            f"{self.entity_label}_{self.id}="  # 实体类型和ID
            f"{capitalized_label}("  # 类名
            f"wall_{self.wall_id},"  # 所属墙体ID
            f"{self.position_x},{self.position_y},{self.position_z},"  # 位置坐标
            f"{self.width},{self.height})"  # 宽度和高度
        )
        return language_string


@dataclass
class Window(Door):
    """窗的数据类，继承自Door类"""
    entity_label: str = "window"  # 实体标签设置为"window"


@dataclass 
class Bbox:
    """3D边界框数据类，表示场景中的物体"""
    id: int           # 边界框唯一标识
    class_name: str   # 物体类别名称
    position_x: float # 中心点x坐标
    position_y: float # 中心点y坐标
    position_z: float # 中心点z坐标
    angle_z: float    # 绕Z轴旋转角度(弧度)
    scale_x: float    # X方向尺寸
    scale_y: float    # Y方向尺寸
    scale_z: float    # Z方向尺寸
    entity_label: str = "bbox"  # 实体标签

    def __post_init__(self):
        """初始化后处理，确保数据类型正确"""
        self.id = int(self.id)
        self.class_name = str(self.class_name)
        self.position_x = float(self.position_x)
        self.position_y = float(self.position_y)
        self.position_z = float(self.position_z)
        self.angle_z = float(self.angle_z)
        self.scale_x = abs(float(self.scale_x))  # 确保尺寸为正
        self.scale_y = abs(float(self.scale_y))
        self.scale_z = abs(float(self.scale_z))

    def rotate(self, angle: float):
        """绕Z轴旋转边界框
        
        参数:
            angle: 旋转角度(弧度)
        """
        # 计算增强旋转矩阵和边界框原始旋转矩阵
        augment_rot_mat = R.from_rotvec([0, 0, angle]).as_matrix()
        bbox_rot_mat = R.from_rotvec([0, 0, self.angle_z]).as_matrix()
        
        # 计算新的旋转矩阵和欧拉角
        new_bbox_rot_mat = augment_rot_mat @ bbox_rot_mat
        new_angle_z = R.from_matrix(new_bbox_rot_mat).as_euler("ZYX")[0]
        
        # 将角度限制在[-π, π)范围内
        new_angle_z = (new_angle_z + np.pi) % (2 * np.pi) - np.pi

        # 处理对称性(如果x和y尺寸接近则对称性更高)
        symmetry = np.pi  # 默认π对称(180度)
        if np.isclose(self.scale_x, self.scale_y, atol=1e-3):
            symmetry = np.pi / 2  # 如果x和y尺寸接近，则为π/2对称(90度)
        new_angle_z = (new_angle_z + np.pi) % symmetry - np.pi
        self.angle_z = new_angle_z

        # 旋转中心点
        bbox_center = np.array([self.position_x, self.position_y, self.position_z])
        bbox_center = augment_rot_mat @ bbox_center
        self.position_x = bbox_center[0]
        self.position_y = bbox_center[1]
        self.position_z = bbox_center[2]

    def translate(self, translation: np.ndarray):
        """平移边界框
        
        参数:
            translation: 3D平移向量[x, y, z]
        """
        self.position_x += translation[0]
        self.position_y += translation[1]
        self.position_z += translation[2]

    def scale(self, scaling: float):
        """缩放边界框尺寸
        
        参数:
            scaling: 缩放系数
        """
        self.scale_x *= scaling
        self.scale_y *= scaling
        self.scale_z *= scaling
        self.position_x *= scaling
        self.position_y *= scaling
        self.position_z *= scaling

    def normalize_and_discretize(self):
        """归一化并离散化边界框参数"""
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_min, scale_max = NORMALIZATION_PRESET["scale"]
        angle_min, angle_max = NORMALIZATION_PRESET["angle"]
        num_bins = NORMALIZATION_PRESET["num_bins"]

        # 归一化到[0, num_bins]范围
        self.position_x = (self.position_x - world_min) / (world_max - world_min) * num_bins
        self.position_y = (self.position_y - world_min) / (world_max - world_min) * num_bins
        self.position_z = (self.position_z - world_min) / (world_max - world_min) * num_bins
        self.angle_z = (self.angle_z - angle_min) / (angle_max - angle_min) * num_bins
        self.scale_x = (self.scale_x - scale_min) / (scale_max - scale_min) * num_bins
        self.scale_y = (self.scale_y - scale_min) / (scale_max - scale_min) * num_bins
        self.scale_z = (self.scale_z - scale_min) / (scale_max - scale_min) * num_bins

        # 确保值在有效范围内
        self.position_x = np.clip(self.position_x, 0, num_bins - 1)
        self.position_y = np.clip(self.position_y, 0, num_bins - 1)
        self.position_z = np.clip(self.position_z, 0, num_bins - 1)
        self.angle_z = np.clip(self.angle_z, 0, num_bins - 1)
        self.scale_x = np.clip(self.scale_x, 0, num_bins - 1)
        self.scale_y = np.clip(self.scale_y, 0, num_bins - 1)
        self.scale_z = np.clip(self.scale_z, 0, num_bins - 1)

    def undiscretize_and_unnormalize(self):
        """将离散化和归一化的边界框参数还原为原始尺度"""
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_min, scale_max = NORMALIZATION_PRESET["scale"]
        angle_min, angle_max = NORMALIZATION_PRESET["angle"]
        num_bins = NORMALIZATION_PRESET["num_bins"]

        # 反离散化 - 将bin值转换为[0,1]范围的连续值
        self.position_x = self.position_x / num_bins
        self.position_y = self.position_y / num_bins
        self.position_z = self.position_z / num_bins
        self.angle_z = self.angle_z / num_bins
        self.scale_x = self.scale_x / num_bins
        self.scale_y = self.scale_y / num_bins
        self.scale_z = self.scale_z / num_bins

        # 反归一化 - 将[0,1]范围的值还原到原始物理尺度
        self.position_x = self.position_x * (world_max - world_min) + world_min
        self.position_y = self.position_y * (world_max - world_min) + world_min
        self.position_z = self.position_z * (world_max - world_min) + world_min
        self.angle_z = self.angle_z * (angle_max - angle_min) + angle_min
        self.scale_x = self.scale_x * (scale_max - scale_min) + scale_min
        self.scale_y = self.scale_y * (scale_max - scale_min) + scale_min
        self.scale_z = self.scale_z * (scale_max - scale_min) + scale_min

    def to_language_string(self):
        """将边界框对象转换为可读的语言字符串
        
        返回格式示例:
        bbox_0=Bbox(table,1.2,3.4,5.6,0.1,0.8,0.8,1.2)
        
        返回:
            描述边界框的标准化字符串
        """
        capitalized_label = self.entity_label.capitalize()  # 首字母大写
        self.id = self.id % 1000  # 限制ID范围
        language_string = (
            f"{self.entity_label}_{self.id}="  # 实体类型和ID
            f"{capitalized_label}("  # 类名
            f"{self.class_name},"  # 物体类别
            f"{self.position_x},{self.position_y},{self.position_z},"  # 位置坐标
            f"{self.angle_z},"  # 旋转角度
            f"{self.scale_x},{self.scale_y},{self.scale_z})"  # 尺寸
        )
        return language_string