import os  # 操作系统接口
import glob  # 文件路径匹配
import argparse  # 命令行参数解析

import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条显示
from threading import Thread  # 多线程支持
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace模型和分词器
from transformers import TextIteratorStreamer  # 文本流式输出

from spatiallm import Layout  # 空间布局类
from spatiallm import SpatialLMLlamaForCausalLM, SpatialLMQwenForCausalLM  # 空间语言模型
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose  # 点云处理工具


def preprocess_point_cloud(points, colors, grid_size, num_bins):
    """预处理点云数据
    
    参数:
        points: 点云坐标数组
        colors: 点云颜色数组
        grid_size: 网格采样大小
        num_bins: 最大网格坐标数
    
    返回:
        处理后的点云张量
    """
    transform = Compose(  # 组合变换
        [
            dict(type="PositiveShift"),  # 正平移变换
            dict(type="NormalizeColor"),  # 颜色归一化
            dict(  # 网格采样
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(  # 应用变换
        {
            "name": "pcd",
            "coord": points.copy(),  # 坐标深拷贝
            "color": colors.copy(),  # 颜色深拷贝
        }
    )
    coord = point_cloud["grid_coord"]  # 获取网格坐标
    xyz = point_cloud["coord"]  # 获取原始坐标
    rgb = point_cloud["color"]  # 获取颜色
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)  # 合并特征
    return torch.as_tensor(np.stack([point_cloud], axis=0))  # 转换为张量


def generate_layout(
    model,  # 语言模型
    point_cloud,  # 点云数据
    tokenizer,  # 分词器
    code_template_file,  # 代码模板文件路径
    top_k=10,  # 采样top-k参数
    top_p=0.95,  # 采样top-p参数
    temperature=0.6,  # 温度参数
    num_beams=1,  # beam search参数
    max_new_tokens=4096,  # 最大生成token数
):
    """生成空间布局
    
    返回:
        Layout对象
    """
    # 加载代码模板
    with open(code_template_file, "r") as f:
        code_template = f.read()

    prompt = f"<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {code_template}"

    # 准备对话数据(根据不同模型类型)
    if model.config.model_type == SpatialLMLlamaForCausalLM.config_class.model_type:
        conversation = [{"role": "user", "content": prompt}]
    elif model.config.model_type == SpatialLMQwenForCausalLM.config_class.model_type:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")

    # 应用对话模板并转换为tensor
    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)  # 移动到模型所在设备

    # 创建文本流式输出器
    streamer = TextIteratorStreamer(
        tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )

    # 准备生成参数
    generate_kwargs = dict(
        {"input_ids": input_ids, "point_clouds": point_cloud},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )
    
    # 启动生成线程
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # 流式输出生成结果
    print("Generating layout...\n")
    generate_texts = []
    for text in streamer:
        generate_texts.append(text)
        print(text, end="", flush=True)
    print("\nDone!")

    # 解析生成结果
    layout_str = "".join(generate_texts)
    layout = Layout(layout_str)
    layout.undiscretize_and_unnormalize()  # 反离散化和反归一化
    return layout


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser("SpatialLM推理脚本")
    parser.add_argument(
        "-p",
        "--point_cloud",
        type=str,
        required=True,
        help="输入点云文件路径或包含多个点云文件的文件夹路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="输出布局txt文件路径或保存多个布局txt文件的文件夹路径",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="manycore-research/SpatialLM-Llama-1B",
        help="模型检查点路径",
    )
    parser.add_argument(
        "-t",
        "--code_template_file",
        type=str,
        default="code_template.txt",
        help="代码模板文件路径",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="top-k采样中保留的最高概率词汇token数量",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="保留概率和达到top_p的最小最高概率token集合",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="用于调整下一个token概率的值",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="beam search的beam数量",
    )
    args = parser.parse_args()

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to("cuda")  # 将模型移动到GPU
    model.set_point_backbone_dtype(torch.float32)  # 设置点云骨干网络数据类型
    model.eval()  # 设置为评估模式

    # 检查输入是单个点云文件还是包含多个点云文件的文件夹
    if os.path.isfile(args.point_cloud):
        point_cloud_files = [args.point_cloud]
    else:
        point_cloud_files = glob.glob(os.path.join(args.point_cloud, "*.ply"))  # 获取所有ply文件

    # 处理每个点云文件
    for point_cloud_file in tqdm(point_cloud_files):
        # 加载并清理点云
        point_cloud = load_o3d_pcd(point_cloud_file)
        point_cloud = cleanup_pcd(point_cloud)
        points, colors = get_points_and_colors(point_cloud)
        min_extent = np.min(points, axis=0)  # 计算最小坐标值

        # 预处理点云为张量特征
        grid_size = Layout.get_grid_size()  # 获取网格大小
        num_bins = Layout.get_num_bins()  # 获取网格数量
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

        # 生成布局
        layout = generate_layout(
            model,
            input_pcd,
            tokenizer,
            args.code_template_file,
            args.top_k,
            args.top_p,
            args.temperature,
            args.num_beams,
        )
        layout.translate(min_extent)  # 平移布局到原始位置
        pred_language_string = layout.to_language_string()  # 转换为语言字符串

        # 检查输出路径是文件还是目录
        if os.path.splitext(args.output)[-1]:  # 如果是文件
            with open(args.output, "w") as f:
                f.write(pred_language_string)
        else:  # 如果是目录
            output_filename = os.path.basename(point_cloud_file).replace(".ply", ".txt")
            os.makedirs(args.output, exist_ok=True)  # 创建输出目录
            with open(os.path.join(args.output, output_filename), "w") as f:
                f.write(pred_language_string)
