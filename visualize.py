import argparse  # 命令行参数解析
import numpy as np  # 数值计算库
import rerun as rr  # 可视化库
import rerun.blueprint as rrb  # 可视化蓝图

from spatiallm import Layout  # 空间布局类
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors  # 点云处理工具

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser("Layout Visualization with rerun")
    parser.add_argument(
        "-p",
        "--point_cloud",
        type=str,
        required=True,
        help="输入点云文件路径",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        required=True,
        help="布局txt文件路径",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.01,
        help="可视化点的半径",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=1000000,
        help="可视化点的最大数量",
    )
    rr.script_add_args(parser)  # 添加rerun专用参数
    args = parser.parse_args()

    # 读取布局文件内容
    with open(args.layout, "r") as f:
        layout_content = f.read()

    # 加载点云数据
    pcd = load_o3d_pcd(args.point_cloud)
    points, colors = get_points_and_colors(pcd)

    # 解析布局内容
    layout = Layout(layout_content)
    floor_plan = layout.to_boxes()  # 转换为3D盒子表示

    # 初始化rerun可视化
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="3D", origin="/world", background=[255, 255, 255]),
        collapse_panels=True,
    )
    rr.script_setup(args, "rerun_spatiallm", default_blueprint=blueprint)

    # 设置世界坐标系
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # 随机采样点云用于可视化
    point_indices = np.arange(points.shape[0])
    np.random.shuffle(point_indices)
    point_indices = point_indices[: args.max_points]
    points = points[point_indices]
    colors = colors[point_indices]
    
    # 记录点云数据
    rr.log(
        "world/points",
        rr.Points3D(
            positions=points,
            colors=colors,
            radii=args.radius,
        ),
        static=True,
    )

    # 动画展示布局生成过程
    num_entities = len(floor_plan)
    seconds = 0.5  # 每帧间隔时间
    for ti in range(num_entities + 1):
        sub_floor_plan = floor_plan[:ti]  # 获取当前帧要显示的实体

        rr.set_time_seconds("time_sec", ti * seconds)  # 设置当前时间
        for box in sub_floor_plan:
            uid = box["id"]  # 实体唯一ID
            group = box["class"]  # 实体类别
            label = box["label"]  # 实体标签

            # 记录3D盒子
            rr.log(
                f"world/pred/{group}/{uid}",
                rr.Boxes3D(
                    centers=box["center"],  # 盒子中心
                    half_sizes=0.5 * box["scale"],  # 半尺寸
                    labels=label,  # 标签
                ),
                rr.InstancePoses3D(mat3x3=box["rotation"]),  # 旋转矩阵
                static=False,  # 非静态对象
            )
    rr.script_teardown(args)  # 结束可视化