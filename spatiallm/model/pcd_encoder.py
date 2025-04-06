# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SceneScript Point Cloud Encoder

Reference: https://github.com/facebookresearch/scenescript/blob/main/src/networks/encoder.py

@inproceedings{avetisyan2024scenescript,
    title       = {SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model},
    author      = {Avetisyan, Armen and Xie, Christopher and Howard-Jenkins, Henry and Yang, Tsun-Yi and Aroudj, Samir and Patra, Suvam and Zhang, Fuyang and Frost, Duncan and Holland, Luke and Orme, Campbell and Engel, Jakob and Miller, Edward and Newcombe, Richard and Balntas, Vasileios},
    booktitle   = {European Conference on Computer Vision (ECCV)},
    year        = {2024},
}
"""

# 导入PyTorch库
import torch
# 导入torchsparse稀疏张量库
import torchsparse
# 导入torchsparse的神经网络模块
import torchsparse.nn as spnn
# 导入einops的repeat函数用于张量操作
from einops import repeat

# 从PyTorch导入神经网络模块
from torch import nn
# 导入PyTorch的功能性函数
from torch.nn import functional as F


# 定义创建3D稀疏卷积的函数
def make_conv3d_sparse(
    channels_in,       # 输入通道数
    channels_out,      # 输出通道数
    kernel_size=3,     # 卷积核大小，默认为3
    num_groups=8,      # 分组数，默认为8
    activation=spnn.ReLU,  # 激活函数，默认为ReLU
):
    # 确保分组数不超过输出通道数
    num_groups = min(num_groups, channels_out)
    # 构建顺序网络模块
    block = nn.Sequential(
        # 3D稀疏卷积层
        spnn.Conv3d(channels_in, channels_out, kernel_size=kernel_size, stride=1),
        # 分组归一化层
        spnn.GroupNorm(num_groups, channels_out),
        # 激活函数
        activation(inplace=True),
    )
    return block


# 定义创建下采样3D稀疏卷积的函数
def make_conv3d_downscale_sparse(
    channels_in,       # 输入通道数
    channels_out,      # 输出通道数
    num_groups=8,      # 分组数，默认为8
    activation=spnn.ReLU,  # 激活函数，默认为ReLU
):
    # 确保分组数不超过输出通道数
    num_groups = min(num_groups, channels_out)
    # 构建顺序网络模块
    block = nn.Sequential(
        # 3D稀疏卷积层，使用2x2核和步长2实现下采样
        spnn.Conv3d(channels_in, channels_out, kernel_size=2, stride=2),
        # 分组归一化层
        spnn.GroupNorm(num_groups, channels_out),
        # 激活函数
        activation(inplace=True),
    )
    return block


# 定义稀疏残差块类
class ResBlockSparse(nn.Module):
    def __init__(
        self,
        channels,        # 输入输出通道数
        num_groups=8,    # 分组数，默认为8
        activation=spnn.ReLU,  # 激活函数，默认为ReLU
    ):
        super().__init__()

        # 第一个3D稀疏卷积块
        self.block0 = make_conv3d_sparse(
            channels, channels, num_groups=num_groups, activation=activation
        )
        # 第二个3D稀疏卷积块
        self.block1 = make_conv3d_sparse(
            channels, channels, num_groups=num_groups, activation=activation
        )

    def forward(self, x):
        # 前向传播：两个卷积块后加上残差连接
        out = self.block0(x)
        out = self.block1(out)
        return x + out


# 定义批处理稀疏张量的索引函数
def index_batched_sparse_tensor(sparse_tensor, index):
    """

        参数：
            sparse_tensor: 由 torchsparse.utils.collate.sparse_collate() 
                生成的批处理稀疏张量。
            index: 整数，指定要提取的样本在批次中的索引。

        返回：
            去除批次维度的 torchsparse.SparseTensor 单个样本。
    """
    # 创建批处理掩码，选择指定索引的批次
    batch_mask = sparse_tensor.C[:, 0] == index
    # 获取对应批次的坐标（去掉批次维度）
    coords = sparse_tensor.C[batch_mask, 1:]  # Get rid of batch dim
    # 获取对应批次的特征
    feats = sparse_tensor.F[batch_mask]
    # 返回新的稀疏张量
    return torchsparse.SparseTensor(
        coords=coords,
        feats=feats,
        stride=sparse_tensor.s,
    )


def sparse_uncollate(sparse_tensor):
    """将批处理的 torchsparse.SparseTensor 解批处理为单个样本列表

    Args:
        sparse_tensor: 由 torchsparse.utils.collate.sparse_collate()
            生成的批处理稀疏张量

    Returns:
        包含单个样本的 torchsparse.SparseTensor 列表
    """
    # 计算批次大小（根据坐标张量中的批次索引最大值+1）
    batch_size = sparse_tensor.C[:, 0].max() + 1

    # 初始化返回列表
    sparse_tensor_list = []
    # 遍历每个批次索引
    for b in range(batch_size):
        # 使用index_batched_sparse_tensor函数提取单个样本
        sparse_tensor_list.append(index_batched_sparse_tensor(sparse_tensor, b))
    return sparse_tensor_list


def vox_to_sequence(sparse_tensor):
    """将稀疏点云转换为序列格式，用于Transformer等模型

    Args:
        sparse_tensor: torchsparse.SparseTensor 输入稀疏张量

    Returns:
        包含以下键的字典:
            seq: [B, maxlen, C] 特征序列张量
            coords: [B, maxlen, 3] 坐标序列张量（用于Transformer的位置嵌入）
            mask: [B, maxlen] 布尔掩码张量（用于Transformer的注意力掩码）
    """
    # 首先解批处理为单个样本列表
    sparse_tensor_list = sparse_uncollate(sparse_tensor)
    # 获取批次大小
    batch_size = len(sparse_tensor_list)
    # 获取特征维度
    channels = sparse_tensor_list[0].F.shape[-1]
    # 计算批次中最长序列长度（最大点数）
    maxlen = max([x.C.shape[0] for x in sparse_tensor_list])

    # 初始化输出张量
    seq = torch.zeros(
        (batch_size, maxlen, channels),
        dtype=sparse_tensor.F.dtype,
        device=sparse_tensor.F.device,
    )
    coords = torch.zeros(
        (batch_size, maxlen, 3),
        dtype=sparse_tensor.C.dtype,
        device=sparse_tensor.C.device,
    )
    mask = torch.ones(
        (batch_size, maxlen),
        dtype=torch.bool,
        device=sparse_tensor.F.device,
    )
    
    # 遍历每个样本
    for i in range(batch_size):
        sparse_tensor = sparse_tensor_list[i]

        # 获取坐标（已按stride缩放，范围为{0, 1, ...}）
        coords_i = sparse_tensor.C  # [N_points_i, 3]
        coords_num = coords_i.shape[0]
        assert coords_num <= maxlen, f"coords_num: {coords_num} is too high..."

        # 获取特征
        feats = sparse_tensor.F  # [N_points_i, C]

        # 计算需要填充的长度
        pad = maxlen - coords_num

        # 对特征进行填充（转置-填充-转回）
        feats = F.pad(feats.T, (0, pad), value=0).T  # [maxlen, C]
        seq[i] = feats

        # 对坐标进行填充
        coords_i = F.pad(coords_i.T, (0, pad), value=0).T  # [maxlen, 3]
        coords[i] = coords_i

        # 设置掩码（有效点为False，填充点为True）
        mask[i, :coords_num] = False

    return {
        "seq": seq,
        "coords": coords,
        "mask": mask,
    }


def fourier_encode_vector(vec, num_bands=10, sample_rate=60):
    """对向量进行傅里叶编码（位置编码）

    Args:
        vec: [B, N, D] 输入向量
        num_bands: int 频带数量
        sample_rate: int 采样率

    Returns:
        [B, N, (2 * num_bands + 1) * D] 编码后的向量
    """
    # 获取输入维度
    b, n, d = vec.shape
    # 生成采样频率（1到sample_rate/2之间的线性间隔）
    samples = torch.linspace(1, sample_rate / 2, num_bands).to(vec.device) * torch.pi
    # 计算正弦和余弦分量
    sines = torch.sin(samples[None, None, :, None] * vec[:, :, None, :])
    cosines = torch.cos(samples[None, None, :, None] * vec[:, :, None, :])

    # 合并正弦和余弦编码
    encoding = torch.stack([sines, cosines], dim=3).reshape(b, n, 2 * num_bands, d)
    # 添加原始向量
    encoding = torch.cat([vec[:, :, None, :], encoding], dim=2)
    # 展平最后两个维度
    return encoding.flatten(2)
class ResNet3DSparse(nn.Module):
    def __init__(self, dim_in, dim_out, layers):
        super().__init__()

        # 初始化网络结构
        self.stem = nn.Sequential(
            # 第一层3D稀疏卷积（使用较大的7x7x7核）
            make_conv3d_sparse(dim_in, layers[0], kernel_size=7),
            # 紧接着一个残差块
            ResBlockSparse(layers[0]),
        )

        # 构建下采样模块（层数比layers参数少1）
        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(
                nn.Sequential(
                    # 下采样卷积（将分辨率减半）
                    make_conv3d_downscale_sparse(layers[i], layers[i + 1]),
                    # 两个连续的残差块
                    ResBlockSparse(layers[i + 1]),
                    ResBlockSparse(layers[i + 1]),
                )
            )
        # 将所有下采样块组合成顺序模块
        self.blocks = nn.Sequential(*blocks)

        # 瓶颈层（全连接网络）
        self.bottleneck = nn.Sequential(
            # 第一个全连接层（扩展维度）
            nn.Linear(layers[-1], 2 * layers[-1]),
            # 分组归一化（默认8组）
            nn.GroupNorm(8, 2 * layers[-1]),
            # ReLU激活
            nn.ReLU(inplace=True),
            # 最终投影到目标维度
            nn.Linear(2 * layers[-1], dim_out),
        )

    def forward(self, x):
        # 前向传播流程
        out = self.stem(x)  # 经过stem模块
        out = self.blocks(out)  # 经过所有下采样块
        out.F = self.bottleneck(out.F)  # 只对特征应用瓶颈层（保持坐标不变）
        return out


class PointCloudEncoder(nn.Module):
    def __init__(
        self,
        input_channels,  # 输入特征维度
        d_model,         # 模型隐藏层维度
        conv_layers,     # 卷积层通道数列表
        num_bins,        # 体素网格的初始分辨率
    ):
        """点云编码器网络

        Args:
            input_channels: 输入特征维度
            d_model: Transformer隐藏层维度
            conv_layers: 卷积各层通道数列表
            num_bins: 初始体素化时的网格分辨率
        """
        
        
        super().__init__()

        # 3D稀疏ResNet主干网络
        self.sparse_resnet = ResNet3DSparse(
            dim_in=input_channels,
            dim_out=d_model,
            layers=conv_layers,
        )
        # 计算下采样次数（根据卷积层数）
        downconvs = len(conv_layers) - 1
        # 计算分辨率缩减倍数（2^下采样次数）
        res_reduction = 2**downconvs
        # 计算最终的特征图分辨率
        self.reduced_grid_size = int(num_bins / res_reduction)
        # 输入投影层（将特征+位置编码映射到d_model）
        self.input_proj = nn.Linear(d_model + 63, d_model)  # 63是傅里叶编码的维度

        # 遗留参数（早期版本使用的额外嵌入）
        self.extra_embedding = nn.Parameter(torch.empty(d_model).normal_(std=0.02))

    def forward(self, point_cloud: torchsparse.SparseTensor):
        """前向传播

        Args:
            point_cloud: 输入稀疏点云张量

        Returns: 包含以下键的字典:
            context: [B, maxlen, d_model] 上下文特征
            context_mask: [B, maxlen] 掩码（True表示忽略）
        """
        # 通过3D稀疏ResNet提取特征
        outputs = self.sparse_resnet(point_cloud)
        # 将稀疏张量转换为序列格式
        outputs = vox_to_sequence(outputs)

        # 获取特征序列和掩码
        context = outputs["seq"]
        context_mask = outputs["mask"]

        # 处理坐标信息
        coords = outputs["coords"]
        # 坐标归一化到[0,1]范围
        coords_normalised = coords / (self.reduced_grid_size - 1)
        # 对归一化坐标进行傅里叶编码
        encoded_coords = fourier_encode_vector(coords_normalised)

        # 合并特征和位置编码
        context = torch.cat([context, encoded_coords], dim=-1)
        # 通过线性层投影到目标维度
        context = self.input_proj(context)

        # 添加遗留参数（早期版本使用的额外嵌入）
        context = context + repeat(self.extra_embedding, "d -> 1 1 d")

        return {
            "context": context,        # 上下文特征序列
            "context_mask": context_mask,  # 序列掩码
        }
        
        