# 导入必要的库
import logging  # 日志记录
import numpy as np  # 数值计算
import open3d as o3d  # 3D点云处理库

# 从项目中导入注册表类
from spatiallm.pcd.registry import Registry

# 创建名为"transforms"的注册表实例
TRANSFORMS = Registry("transforms")
# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

"""
3D点云预处理模块

参考: PointCept项目

@misc{pointcept2023,
    title={Pointcept: 点云感知研究的代码库},
    author={Pointcept贡献者},
    year={2023}
}
"""


# 组合变换类，用于按顺序应用多个变换
class Compose(object):
    def __init__(self, cfg=None):
        # 初始化配置，默认为空列表
        self.cfg = cfg if cfg is not None else []
        self.transforms = []  # 存储变换操作的列表
        # 根据配置构建变换序列
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    # 调用时依次应用所有变换
    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


# 注册正平移变换
@TRANSFORMS.register_module()
class PositiveShift(object):
    # 将坐标平移到正数区域
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # 计算最小坐标值
            coord_min = np.min(data_dict["coord"], 0)
            # 平移所有坐标
            data_dict["coord"] -= coord_min
        return data_dict


# 注册颜色归一化变换
@TRANSFORMS.register_module()
class NormalizeColor(object):
    # 将颜色值归一化到[-1,1]范围
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # 执行归一化 (0-255) -> (-1,1)
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict


# 注册网格采样变换类
@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,  # 网格大小
        hash_type="fnv",  # 哈希类型(fnv或ravel)
        mode="train",  # 模式(train/test)
        keys=("coord", "color", "normal", "segment"),  # 需要处理的键
        return_inverse=False,  # 是否返回逆映射
        return_grid_coord=False,  # 是否返回网格坐标
        return_min_coord=False,  # 是否返回最小坐标
        return_displacement=False,  # 是否返回位移
        project_displacement=False,  # 是否投影位移
        max_grid_coord=None,  # 最大网格坐标限制
    ):
        self.grid_size = grid_size
        # 选择哈希函数
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]  # 验证模式有效性
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        self.max_grid_coord = max_grid_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()  # 确保存在坐标数据
        
        # 坐标缩放和网格化
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)  # 计算网格坐标
        min_coord = grid_coord.min(0)  # 计算最小坐标
        grid_coord -= min_coord  # 坐标平移
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)  # 还原到原始尺度
        
        # 应用最大网格坐标限制
        if self.max_grid_coord is not None:
            grid_coord = np.clip(grid_coord, 0, self.max_grid_coord - 1)
            
        # 计算哈希键并排序
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        # 训练模式处理
        if self.mode == "train":
            # 随机采样每个网格中的点
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            
            # 处理带标签数据的采样
            if "sampled_index" in data_dict:
                # 确保标记点被采样(针对ScanNet数据集)
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
                
            # 处理可选输出
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                # 计算位移(相对于网格中心)
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5]
                if self.project_displacement:
                    # 沿法线方向投影位移
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                
            # 采样指定键的数据
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        # 测试模式处理
        elif self.mode == "test":
            # 取网格中心点(非随机)
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1]) + (count.max() // 2) % count
            )
            idx_part = idx_sort[idx_select]
            data_part = dict(index=idx_part)  # 创建部分数据字典
            
            # 处理可选输出
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_part["grid_coord"] = grid_coord[idx_part]
            if self.return_min_coord:
                data_part["min_coord"] = min_coord.reshape([1, 3])
                
            # 复制所有数据(部分采样)
            for key in data_dict.keys():
                if key in self.keys:
                    data_part[key] = data_dict[key][idx_part]
                else:
                    data_part[key] = data_dict[key]
            return data_part
        else:
            raise NotImplementedError  # 未知模式

    @staticmethod
    def ravel_hash_vec(arr):
        """
        对坐标进行ravel哈希处理(减去最小坐标后)
        输入: 
            arr - 二维坐标数组
        返回:
            哈希键值数组
        """
        assert arr.ndim == 2  # 确保输入是二维数组
        arr = arr.copy()  # 创建副本避免修改原数组
        arr -= arr.min(0)  # 减去最小坐标
        arr = arr.astype(np.uint64, copy=False)  # 转换为uint64类型
        arr_max = arr.max(0).astype(np.uint64) + 1  # 计算每维最大值+1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)  # 初始化哈希键
        # Fortran风格索引计算
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]  # 累加当前维度值
            keys *= arr_max[j + 1]  # 乘以下一维度的范围
        keys += arr[:, -1]  # 加上最后一维的值
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A哈希算法实现
        输入:
            arr - 二维坐标数组
        返回:
            哈希键值数组
        """
        assert arr.ndim == 2  # 确保输入是二维数组
        # 先取整处理负坐标
        arr = arr.copy()  # 创建副本
        arr = arr.astype(np.uint64, copy=False)  # 转换为uint64类型
        # 初始化哈希值(使用FNV偏移基础值)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        # 对每个维度进行FNV哈希计算
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)  # FNV质数
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])  # 异或操作
        return hashed_arr


# 从文件加载点云数据
def load_o3d_pcd(file_path: str):
    """
    加载Open3D点云文件
    参数:
        file_path - 文件路径
    返回:
        Open3D点云对象
    """
    return o3d.io.read_point_cloud(file_path)


# 从Open3D点云获取坐标和颜色
def get_points_and_colors(pcd: o3d.geometry.PointCloud):
    """
    提取点云坐标和颜色
    参数:
        pcd - Open3D点云对象
    返回:
        points - 坐标数组
        colors - 颜色数组(0-255)
    """
    points = np.asarray(pcd.points)  # 获取坐标
    colors = np.zeros_like(points, dtype=np.uint8)  # 初始化颜色数组
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)  # 获取颜色
        if colors.shape[1] == 4:  # 如果有alpha通道
            colors = colors[:, :3]  # 去除alpha通道
        if colors.max() < 1.1:  # 如果颜色值在0-1范围
            colors = (colors * 255).astype(np.uint8)  # 转换为0-255
    return points, colors


# 点云预处理
def cleanup_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.02,  # 体素大小
    num_nb: int = 3,  # 邻域点数阈值
    radius: float = 0.05,  # 搜索半径
):
    """
    点云清理预处理
    参数:
        pcd - Open3D点云对象
        voxel_size - 下采样体素大小
        num_nb - 邻域点数阈值
        radius - 邻域搜索半径
    返回:
        处理后的点云
    """
    # 体素下采样
    pcd = pcd.voxel_down_sample(voxel_size)
    # 半径离群点去除
    pcd, _ = pcd.remove_radius_outlier(num_nb, radius)
    return pcd
