import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RelativePositionBias(nn.Module):

    def __init__(self,
                 bidirectional=True,
                 num_buckets=64,
                 max_distance=50,
                 n_heads=2):
        super(RelativePositionBias, self).__init__()
        # 初始化参数
        self.bidirectional = bidirectional  # 是否双向
        self.num_buckets = num_buckets  # 桶的数量
        self.max_distance = max_distance  # 最大距离
        self.n_heads = n_heads  # 头的数量
        # 创建一个嵌入层，用于学习相对位置偏置
        self.relative_attention_bias = nn.Embedding(self.num_buckets,
                                                    self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        # 将相对位置转换为一个桶编号
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(
                torch.long) * num_buckets  # 如果是负数，桶编号加上num_buckets/2
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) /
                                    math.log(max_distance / max_exact) *
                                    (num_buckets - max_exact)).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        # 计算相对位置偏置
        context_position = torch.arange(
            qlen,
            dtype=torch.long,
            device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(
            klen,
            dtype=torch.long,
            device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # 计算相对位置
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # 将相对位置转换为桶编号
            self.bidirectional,
            self.num_buckets,
            self.max_distance)
        return self.relative_attention_bias(
            relative_position_bucket)  # 返回相对位置偏置
