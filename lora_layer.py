# lora_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    LoRA implementation for Linear layers.
    y = W x + alpha / r * BA x
    """

    def __init__(self,
                 in_features,
                 out_features,
                 r=4,
                 lora_alpha=1.0,
                 lora_dropout=0.0,
                 bias=True,
                 merge_weights=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r  # LoRA rank
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights

        # 原始线性层（冻结参数，默认不训练）
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # LoRA A, B
        if r > 0:
            # A: 下采样（in_features -> r）
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            # B: 上采样（r -> out_features）
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            # 按LoRA论文初始化
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = lora_alpha / r
            self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0. else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = None
            self.lora_dropout = nn.Identity()

        # LoRA训练开关（方便冻结LoRA或合并权重推理）
        self.merged = False

    def forward(self, x):
        # 原始Linear
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0 and not self.merged:
            lora_part = self.lora_dropout(x) @ self.lora_A.t()  # (batch, r)
            lora_part = lora_part @ self.lora_B.t()            # (batch, out_features)
            result = result + lora_part * self.scaling
        return result

    def merge_lora(self):
        """用于推理，将LoRA增量合并到原权重"""
        if self.r > 0 and not self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling  # (out_features, in_features)
            self.weight.data += delta.data
            self.merged = True

    def unmerge_lora(self):
        """反向还原（可选）"""
        if self.r > 0 and self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= delta.data
            self.merged = False

    def train(self, mode=True):
        super().train(mode)
        if self.r > 0 and self.merged:
            self.unmerge_lora()

    def eval(self):
        super().eval()
        if self.r > 0 and not self.merged:
            self.merge_lora()

import math  # 注意别忘了引入math

# 兼容外部import
__all__ = ["LoRALinear"]
