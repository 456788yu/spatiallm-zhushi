# 导入numpy数值计算库
import numpy as np
# 从scipy导入旋转处理类
from scipy.spatial.transform import Rotation as R
# 从自定义模块导入墙体、门窗等类及标准化预设
from spatiallm.layout.entity import Wall, Door, Window, Bbox, NORMALIZATION_PRESET


# 定义房屋布局类
class Layout:
    # 初始化方法（可选字符串输入）
    def __init__(self, str: str = None):
        self.walls = []    # 存储墙体的列表
        self.doors = []    # 存储门的列表  
        self.windows = []  # 存储窗户的列表
        self.bboxes = []   # 存储边界框的列表

        # 如果提供字符串则直接解析
        if str:
            self.from_str(str)

    # 静态方法：获取网格尺寸
    @staticmethod
    def get_grid_size():
        # 从预设获取世界坐标范围
        world_min, world_max = NORMALIZATION_PRESET["world"]
        # 计算单个网格的尺寸（范围/网格数）
        return (world_max - world_min) / NORMALIZATION_PRESET["num_bins"]

    # 静态方法：获取网格数量
    @staticmethod
    def get_num_bins():
        # 直接返回预设的网格数量
        return NORMALIZATION_PRESET["num_bins"]

    # 从字符串解析布局数据
    def from_str(self, s: str):
        # 去除开头空行
        s = s.lstrip("\n")
        # 按行分割字符串
        lines = s.split("\n")
        # 已存在墙体的ID查询表
        existing_walls = []
        # 逐行解析
        for line in lines:
            try:
                # 提取标签部分（等号前）
                label = line.split("=")[0]
                # 从标签提取实体ID（格式：类型_ID）
                entity_id = int(label.split("_")[1])
                # 提取实体类型（如wall/door等）
                entity_label = label.split("_")[0]

                # 定位参数括号位置
                start_pos = line.find("(")
                end_pos = line.find(")")
                # 提取括号内参数并分割
                params = line[start_pos + 1 : end_pos].split(",")

                # 处理墙体类型
                if entity_label == Wall.entity_label:
                    # 墙体参数定义
                    wall_args = [
                        "ax", "ay", "az",  # 起点坐标
                        "bx", "by", "bz",  # 终点坐标
                        "height",    # 高度
                        "thickness", # 厚度
                    ]
                    # 创建参数字典
                    wall_params = dict(zip(wall_args, params[0:8]))
                    # 实例化墙体对象
                    entity = Wall(id=entity_id, **wall_params)
                    # 记录已存在墙体ID
                    existing_walls.append(entity_id)
                    # 添加到墙体列表
                    self.walls.append(entity)
                # 处理门类型
                elif entity_label == Door.entity_label:
                    # 获取所属墙体ID
                    wall_id = int(params[0].split("_")[1])
                    # 校验墙体是否存在
                    if wall_id not in existing_walls:
                        continue

                    # 门参数定义
                    door_args = [
                        "position_x", "position_y", "position_z",  # 位置
                        "width", "height",  # 尺寸
                    ]
                    # 创建参数字典
                    door_params = dict(zip(door_args, params[1:6]))
                    # 实例化门对象
                    entity = Door(
                        id=entity_id,
                        wall_id=wall_id,
                        **door_params,
                    )
                    # 添加到门列表
                    self.doors.append(entity)
                # 处理窗户类型
                elif entity_label == Window.entity_label:
                    # 获取所属墙体ID
                    wall_id = int(params[0].split("_")[1])
                    # 校验墙体是否存在
                    if wall_id not in existing_walls:
                        continue

                    # 窗户参数定义
                    window_args = [
                        "position_x", "position_y", "position_z",  # 位置
                        "width", "height",  # 尺寸
                    ]
                    # 创建参数字典
                    window_params = dict(zip(window_args, params[1:6]))
                    # 实例化窗户对象
                    entity = Window(
                        id=entity_id,
                        wall_id=wall_id,
                        **window_params,
                    )
                    # 添加到窗户列表
                    self.windows.append(entity)
                # 处理边界框类型
                elif entity_label == Bbox.entity_label:
                    # 获取物体类别名
                    class_name = params[0]
                    # 边界框参数定义
                    bbox_args = [
                        "position_x", "position_y", "position_z",  # 位置
                        "angle_z",              # Z轴旋转角度
                        "scale_x", "scale_y", "scale_z",  # 缩放比例
                    ]
                    # 创建参数字典
                    bbox_params = dict(zip(bbox_args, params[1:8]))
                    # 实例化边界框对象
                    entity = Bbox(
                        id=entity_id,
                        class_name=class_name,
                        **bbox_params,
                    )
                    # 添加到边界框列表
                    self.bboxes.append(entity)
            # 忽略解析异常的行
            except Exception as e:
                continue

    # 将布局实体转换为边界框表示
    def to_boxes(self):
        # 初始化边界框列表
        boxes = []
        # 创建墙体信息查找表
        lookup = {}
        
        # 处理所有墙体
        for wall in self.walls:
            # 暂时假设墙体厚度为0.0
            thickness = 0.0
            # 获取墙体两个端点的坐标
            corner_a = np.array([wall.ax, wall.ay, wall.az])
            corner_b = np.array([wall.bx, wall.by, wall.bz])
            # 计算墙体长度（两点间距离）
            length = np.linalg.norm(corner_a - corner_b)
            # 计算墙体方向向量
            direction = corner_b - corner_a
            # 计算墙体在XY平面的旋转角度
            angle = np.arctan2(direction[1], direction[0])
            # 将墙体信息和角度存入查找表
            lookup[wall.id] = {"wall": wall, "angle": angle}

            # 计算墙体中心点（考虑高度）
            center = (corner_a + corner_b) * 0.5 + np.array([0, 0, 0.5 * wall.height])
            # 设置边界框尺寸（长度、厚度、高度）
            scale = np.array([length, thickness, wall.height])
            # 创建旋转矩阵（绕Z轴旋转）
            rotation = R.from_rotvec([0, 0, angle]).as_matrix()
            # 构建边界框字典
            box = {
                "id": wall.id,  # 使用墙体ID
                "class": Wall.entity_label,  # 类别标签
                "label": Wall.entity_label,  # 显示标签
                "center": center,  # 中心位置
                "rotation": rotation,  # 旋转矩阵
                "scale": scale,  # 尺寸
            }
            # 添加到边界框列表
            boxes.append(box)

        # 处理门和窗户
        for fixture in self.doors + self.windows:
            # 获取所属墙体ID
            wall_id = fixture.wall_id
            # 从查找表获取墙体信息
            wall_info = lookup.get(wall_id, None)
            # 如果墙体不存在则跳过
            if wall_info is None:
                continue

            # 提取墙体信息和角度
            wall = wall_info["wall"]
            angle = wall_info["angle"]
            # 使用实际墙体厚度
            thickness = wall.thickness

            # 获取门窗中心位置
            center = np.array(
                [fixture.position_x, fixture.position_y, fixture.position_z]
            )
            # 设置边界框尺寸（宽度、厚度、高度）
            scale = np.array([fixture.width, thickness, fixture.height])
            # 创建旋转矩阵（与墙体相同角度）
            rotation = R.from_rotvec([0, 0, angle]).as_matrix()
            # 设置ID前缀（门用1000，窗用2000）
            class_prefix = 1000 if fixture.entity_label == Door.entity_label else 2000
            # 构建边界框字典
            box = {
                "id": fixture.id + class_prefix,  # 添加前缀的ID
                "class": fixture.entity_label,  # 类别标签
                "label": fixture.entity_label,  # 显示标签
                "center": center,  # 中心位置
                "rotation": rotation,  # 旋转矩阵
                "scale": scale,  # 尺寸
            }
            # 添加到边界框列表
            boxes.append(box)

        # 处理其他边界框
        for bbox in self.bboxes:
            # 获取中心位置
            center = np.array([bbox.position_x, bbox.position_y, bbox.position_z])
            # 获取尺寸
            scale = np.array([bbox.scale_x, bbox.scale_y, bbox.scale_z])
            # 创建旋转矩阵（绕Z轴旋转）
            rotation = R.from_rotvec([0, 0, bbox.angle_z]).as_matrix()
            # 获取类别名称
            class_name = bbox.class_name
            # 构建边界框字典
            box = {
                "id": bbox.id + 3000,  # 添加3000前缀的ID
                "class": Bbox.entity_label,  # 类别标签
                "label": class_name,  # 显示标签
                "center": center,  # 中心位置
                "rotation": rotation,  # 旋转矩阵
                "scale": scale,  # 尺寸
            }
            # 添加到边界框列表
            boxes.append(box)

        # 返回所有边界框
        return boxes

    # 获取所有实体（墙、门、窗、边界框）
    def get_entities(self):
        return self.walls + self.doors + self.windows + self.bboxes

    # 对所有实体进行标准化和离散化处理
    def normalize_and_discretize(self):
        for entity in self.get_entities():
            entity.normalize_and_discretize()

    # 对所有实体进行反离散化和反标准化处理
    def undiscretize_and_unnormalize(self):
        for entity in self.get_entities():
            entity.undiscretize_and_unnormalize()

    # 对所有实体进行平移变换
    def translate(self, translation: np.ndarray):
        for entity in self.get_entities():
            entity.translate(translation)

    # 对所有实体进行旋转变换
    def rotate(self, angle: float):
        for entity in self.get_entities():
            entity.rotate(angle)

    # 对所有实体进行缩放变换
    def scale(self, scale: float):
        for entity in self.get_entities():
            entity.scale(scale)

    # 将布局转换为语言描述字符串
    def to_language_string(self):
        # 收集所有实体的字符串表示
        entity_strings = []
        for entity in self.get_entities():
            entity_strings.append(entity.to_language_string())
        # 用换行符连接所有字符串
        return "\n".join(entity_strings)
