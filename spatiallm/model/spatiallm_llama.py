# 导入必要的模块和库
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import torchsparse
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers import (
    LlamaModel,
    LlamaForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
)
from torchsparse.utils.collate import sparse_collate
from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig

# 定义忽略索引常量
IGNORE_INDEX = -100
# 获取logger实例
logger = logging.get_logger(__name__)


# 定义点云骨干网络类型的枚举
class PointBackboneType(Enum):
    SCENESCRIPT = "scenescript"  # 场景脚本类型的点云处理网络


# 扩展Llama配置类以支持空间语言模型
class SpatialLMLlamaConfig(LlamaConfig):
    model_type = "spatiallm_llama"  # 模型类型标识


# 基于LlamaForCausalLM的空间语言模型类
class SpatialLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = SpatialLMLlamaConfig  # 指定配置类

    def __init__(self, config):
        # 初始化父类LlamaForCausalLM
        super().__init__(config)
        # 初始化Llama模型
        self.model = LlamaModel(config)
        # 设置词汇表大小
        self.vocab_size = config.vocab_size
        # 初始化语言模型头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化点云处理相关组件
        self.point_backbone_type = PointBackboneType(config.point_backbone)
        self.point_backbone = None
        point_config = config.point_config
        # 根据配置选择点云处理骨干网络
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            from spatiallm.model.pcd_encoder import PointCloudEncoder

            # 初始化场景脚本点云编码器
            self.point_backbone = PointCloudEncoder(
                input_channels=point_config["input_channels"],
                d_model=point_config["embed_channels"],
                conv_layers=point_config["conv_layers"],
                num_bins=point_config["num_bins"],
            )
            embed_channels = point_config["embed_channels"]
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

        # 点云特征投影层
        self.point_proj = nn.Linear(embed_channels, config.hidden_size)

        # 设置点云相关的特殊token ID
        self.point_start_token_id = self.config.point_start_token_id
        self.point_end_token_id = self.config.point_end_token_id
        self.point_token_id = self.config.point_token_id

        # 初始化权重并应用最终处理
        self.post_init()

    def forward_point_cloud(self, point_cloud, device, dtype):
        # 点云前向传播处理
        # point_cloud形状: (n_points, n_features)
        
        # 将点云骨干网络转为float32精度
        self.point_backbone.to(torch.float32)
        # 检测并过滤包含NaN值的点
        nan_mask = torch.isnan(point_cloud).any(dim=1)
        point_cloud = point_cloud[~nan_mask]
        # 分离坐标和特征
        coords = point_cloud[:, :3].int()
        feats = point_cloud[:, 3:].float()
        
        # 根据点云骨干网络类型处理
        if self.point_backbone_type == PointBackboneType.SCENESCRIPT:
            # 创建稀疏张量
            pc_sparse_tensor = torchsparse.SparseTensor(coords=coords, feats=feats)
            # 转换为批处理形式(批大小为1)
            pc_sparse_tensor = sparse_collate([pc_sparse_tensor])
            pc_sparse_tensor = pc_sparse_tensor.to(device)
            # 通过点云编码器获取特征
            encoded_features = self.point_backbone(pc_sparse_tensor)
            # 投影到模型隐藏层维度并转换为指定数据类型
            return self.point_proj(encoded_features["context"].to(dtype))
        else:
            raise ValueError(f"Unknown point backbone type: {self.point_backbone_type}")

    def set_point_backbone_dtype(self, dtype: torch.dtype):
        # 设置点云骨干网络的数据类型
        for param in self.point_backbone.parameters():
            param.data = param.data.to(dtype)

    def get_model(self):
        # 获取基础语言模型
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token ID序列
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，标识哪些位置需要被关注
        position_ids: Optional[torch.LongTensor] = None,  # 位置编码ID
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,  # 缓存的键值对，用于加速生成
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 直接输入的嵌入表示，替代input_ids
        labels: Optional[torch.LongTensor] = None,  # 用于计算损失的标签
        use_cache: Optional[bool] = None,  # 是否使用缓存加速生成
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出所有隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
        cache_position: Optional[torch.LongTensor] = None,  # 缓存位置信息
        num_logits_to_keep: int = 0,  # 需要保留的logits数量(0表示全部保留)
        point_clouds: Optional[torch.Tensor] = None,  # 输入的点云数据
        **loss_kwargs,  # 其他损失函数参数
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        参数说明：

        labels (torch.LongTensor，形状为(batch_size, sequence_length)，可选):
        用于计算掩码语言建模损失的标签。索引值应在[0, ..., config.vocab_size]范围内或为-100（参见input_ids说明文档）。索引值为-100的标记将被忽略（掩码处理），损失仅计算标签值在[0, ..., config.vocab_size]范围内的标记。

        point_clouds (torch.Tensor，形状为(batch_size, n_points, n_features)，可选):
        输入点云数据，用于点云编码器处理。

        num_logits_to_keep (int，可选):
        指定需要计算logits的末尾标记数量。设为0时将为所有input_ids计算logits（特殊情况）。生成任务通常只需要最后一个标记的logits，仅计算该标记的logits可以显著节省内存，对于长序列或大词汇表尤为重要。

        返回值:

        使用示例:

        python
        复制
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> # 加载预训练模型和分词器
        >>> model = AutoModelForCausalLM.from_pretrained("manycore-research/SpatialLM-Llama-1B")
        >>> tokenizer = AutoTokenizer.from_pretrained("manycore-research/SpatialLM-Llama-1B")

        >>> # 构建包含点云标记的提示词
        >>> prompt = "<|point_start|><|point_pad|><|point_end|>检测墙壁、门、窗户、箱子。参考代码如下：{code_template}"
        >>> conversation = [{"role": "user", "content": prompt}]
        >>> input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")

        >>> # 生成文本（传入点云数据）
        >>> generate_ids = model.generate(input_ids, point_clouds=point_clouds, max_length=4096)
        >>> # 解码生成结果
        >>> tokenizer.batch_decode(generate_ids, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        # 设置输出注意力权重的配置，优先使用传入参数，其次使用模型配置
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # 设置输出隐藏状态的配置，优先使用传入参数，其次使用模型配置
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # 设置返回形式的配置，优先使用传入参数，其次使用模型配置
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # 计算点云嵌入特征
        if inputs_embeds is None:
            # 如果没有直接提供输入嵌入，则通过token嵌入层获取
            inputs_embeds = self.model.embed_tokens(input_ids)

        # 处理点云数据的条件判断：
        # 1. 点云骨干网络存在
        # 2. 不是单token生成模式(序列长度!=1)或处于训练模式
        # 3. 点云数据不为空
        if (
            self.point_backbone is not None
            and (input_ids.shape[1] != 1 or self.training)
            and point_clouds is not None
        ):
            # 获取批处理中的点云数量
            n_point_clouds = point_clouds.shape[0]
            point_features = []
            # 遍历批处理中的每个点云
            for i in range(n_point_clouds):  # * 遍历批次
                point_cloud = point_clouds[i]
                # 前向传播处理单个点云
                point_feature = self.forward_point_cloud(
                    point_cloud, inputs_embeds.device, inputs_embeds.dtype
                )
                point_features.append(point_feature)

            # 将点云特征插入到输入序列中
            point_start_end_token_pos = []  # 记录起止token位置
            new_input_embeds = []  # 存储新的输入嵌入
            new_attention_mask = []  # 存储新的注意力掩码
            cur_point_idx = 0  # 当前处理的点云索引
            max_num_tokens = 0  # 记录最大token数用于填充
            
            # 遍历批处理中的每个样本
            for cur_input_ids, cur_input_embeds, cur_attention_mask in zip(
                input_ids, inputs_embeds, attention_mask
            ):  # * input_ids: B, L; input_embeds: B, L, C
                # 获取当前点云特征并调整设备
                cur_point_features = (
                    point_features[cur_point_idx]
                    .to(device=cur_input_embeds.device)
                    .squeeze(0)
                )
                num_patches = cur_point_features.shape[0]  # * 点云token数量
                
                # 统计起止token数量
                num_point_start_tokens = (
                    (cur_input_ids == self.config.point_start_token_id).sum().item()
                )
                num_point_end_tokens = (
                    (cur_input_ids == self.config.point_end_token_id).sum().item()
                )
                # 目前仅支持单个起止token
                assert num_point_start_tokens == num_point_end_tokens == 1, (
                    "The number of point start tokens and point end tokens should be 1, "
                    f"but got {num_point_start_tokens} and {num_point_end_tokens}."
                )
                
                # 定位起止token位置
                point_start_token_pos = torch.where(
                    cur_input_ids == self.config.point_start_token_id
                )[0][0]
                point_end_token_pos = torch.where(
                    cur_input_ids == self.config.point_end_token_id
                )[0][0]
                
                # 构建新的输入嵌入序列：
                # 1. 起始token前的原始嵌入
                # 2. 点云特征嵌入
                # 3. 结束token后的原始嵌入
                cur_new_input_embeds = torch.cat(
                    (
                        cur_input_embeds[: point_start_token_pos + 1],
                        cur_point_features,
                        cur_input_embeds[point_end_token_pos:],
                    ),
                    dim=0,
                )
                
                # 构建新的注意力掩码：
                # 1. 起始token前的原始掩码
                # 2. 点云token的掩码(全1)
                # 3. 结束token后的原始掩码
                cur_new_attention_mask = torch.cat(
                    (
                        cur_attention_mask[: point_start_token_pos + 1],
                        torch.ones(num_patches, device=cur_attention_mask.device),
                        cur_attention_mask[point_end_token_pos:],
                    ),
                    dim=0,
                )

                cur_point_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
                new_attention_mask.append(cur_new_attention_mask)
                # 记录起止位置信息
                point_start_end_token_pos.append(
                    (point_start_token_pos, num_patches, point_end_token_pos)
                )
                # 更新最大token数
                if cur_new_input_embeds.shape[0] > max_num_tokens:
                    max_num_tokens = cur_new_input_embeds.shape[0]
            
            # 将所有样本填充到统一长度
            for i in range(len(new_input_embeds)):
                cur_input_embeds = new_input_embeds[i]
                # 使用最后一行进行填充
                last_row = cur_input_embeds[-1]
                padding = last_row.repeat(max_num_tokens - cur_input_embeds.shape[0], 1)
                new_input_embeds[i] = torch.cat([cur_input_embeds, padding], dim=0)

                cur_attention_mask = new_attention_mask[i]
                # 对注意力掩码进行填充(补0)
                new_attention_mask[i] = F.pad(
                    cur_attention_mask,
                    (0, max_num_tokens - cur_attention_mask.shape[0]),
                    value=0,
                )
            
            # 堆叠处理后的嵌入和掩码
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)

            # 验证嵌入和掩码维度一致
            assert (
                attention_mask.shape[1] == inputs_embeds.shape[1]
            ), "The length of attention mask and inputs embeds should be the same"

        # 解码器输出包含(解码特征, 层状态, 解码隐藏状态, 解码注意力)
        outputs = self.model(
            input_ids=None,  # 使用inputs_embeds替代input_ids
            attention_mask=attention_mask,  # 处理后的注意力掩码
            position_ids=position_ids,  # 位置编码
            past_key_values=past_key_values,  # 缓存的键值对
            inputs_embeds=inputs_embeds,  # 处理后的输入嵌入
            use_cache=use_cache,  # 是否使用缓存
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典格式
            cache_position=cache_position,  # 缓存位置
        )

        # 获取最后一层隐藏状态
        hidden_states = outputs[0]

        # 处理模型并行情况下的logits计算
        if self.config.pretraining_tp > 1:
            # 将语言模型头部分片处理
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            # 并行计算各分片的logits
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            # 合并分片结果
            logits = torch.cat(logits, dim=-1)
        else:
            # 仅计算必要的logits(节省内存)
            # 如果不计算损失则不转换为float类型
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        # 初始化损失为None
        loss = None

        # 如果提供了标签则计算损失
        if labels is not None:
            # 准备新的标签(考虑点云token)
            new_labels = []
            max_num_tokens = logits.shape[1]  # 获取最大token数
            
            # 处理每个样本的标签
            for i in range(len(point_start_end_token_pos)):
                cur_labels = labels[i]
                # 获取点云token位置信息
                (cur_point_start_token_pos,
                num_patches,
                cur_point_end_token_pos) = point_start_end_token_pos[i]
                
                # 构建新标签:
                # 1. 起始token前的原始标签
                # 2. 点云token部分使用IGNORE_INDEX(-100)
                # 3. 结束token后的原始标签
                cur_new_labels = torch.cat(
                    (
                        cur_labels[: cur_point_start_token_pos + 1],
                        torch.full(
                            (num_patches,),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                        ),
                        cur_labels[cur_point_end_token_pos:],
                    ),
                    dim=0,
                )
                # 填充标签到统一长度
                cur_new_labels = F.pad(
                    cur_new_labels,
                    (0, max_num_tokens - cur_new_labels.shape[0]),
                    value=IGNORE_INDEX,
                )
                new_labels.append(cur_new_labels)
            
            # 堆叠处理后的标签
            labels = torch.stack(new_labels, dim=0)

            # 验证标签和logits维度一致
            assert (
                labels.shape[1] == logits.shape[1]
            ), "The length of labels and logits should be the same"

            # 计算损失
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **loss_kwargs,
            )

        # 根据return_dict决定返回格式
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 返回标准化的输出结构
        return CausalLMOutputWithPast(
            loss=loss,  # 计算出的损失
            logits=logits,  # 预测logits
            past_key_values=outputs.past_key_values,  # 缓存的键值对
            hidden_states=outputs.hidden_states,  # 所有隐藏状态
            attentions=outputs.attentions,  # 注意力权重
        )

def prepare_inputs_for_generation(
        self,
        input_ids,  # 输入的token ID序列
        past_key_values=None,  # 缓存的键值对，用于加速生成
        attention_mask=None,  # 注意力掩码
        inputs_embeds=None,  # 直接输入的嵌入表示
        **kwargs,  # 其他可选参数
    ):
        # 如果存在缓存的键值对，则只保留最后一个token的ID
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # 处理输入嵌入的特殊情况：
        # 只有在第一代步且提供了inputs_embeds时才使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新模型输入字典，包含以下内容：
        # 1. 缓存的键值对
        # 2. 是否使用缓存的配置
        # 3. 注意力掩码
        # 4. 点云数据（如果有）
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
            }
        )
        return model_inputs


# 向AutoConfig注册自定义配置类
AutoConfig.register("spatiallm_llama", SpatialLMLlamaConfig)
# 向AutoModelForCausalLM注册自定义模型类
AutoModelForCausalLM.register(SpatialLMLlamaConfig, SpatialLMLlamaForCausalLM)

