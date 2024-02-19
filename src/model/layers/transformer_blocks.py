from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch import Tensor
from src.model.layers.relative_position_bias import RelativePositionBias


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
        attn_type="norm",
    ):
        super().__init__()
        self.post_norm = post_norm

        self.norm1 = norm_layer(dim)
        if attn_type == "norm":
            self.attn = torch.nn.MultiheadAttention(
                dim,
                num_heads=num_heads,
                add_bias_kv=qkv_bias,
                dropout=attn_drop,
                batch_first=True,
            )
        elif attn_type == "T5":
            self.attn = T5Attention(
                d_model=dim,
                d_kv=64,
                num_heads=num_heads,
                dropout_rate=attn_drop,
            )
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=False,
        )
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        position_bias: Optional[bool] = None,
    ):
        # pre norm
        src2 = self.norm1(src)
        if position_bias is None:
            src2 = self.attn(
                query=src2,
                key=src2,
                value=src2,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
            )[0]
        else:
            src2 = self.attn(
                input=src2,
                mask=key_padding_mask,
                position_bias=position_bias,
            )[0]  # [B, T, D]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward_post(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        position_bias: Optional[bool] = None,
    ):
        if position_bias is None:
            src2 = self.attn(
                query=src,
                key=src,
                value=src,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
            )[0]
        else:
            src2 = self.attn(
                input=src,
                mask=key_padding_mask,
                position_bias=position_bias,
            )[0]
        src = src + self.drop_path1(self.norm1(src2))
        src = src + self.drop_path2(self.norm2(self.mlp(src)))
        return src

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        position_bias: Optional[bool] = None,
    ):
        if self.post_norm:
            return self.forward_post(src=src,
                                     mask=mask,
                                     key_padding_mask=key_padding_mask,
                                     position_bias=position_bias)

        return self.forward_pre(src=src,
                                mask=mask,
                                key_padding_mask=key_padding_mask,
                                position_bias=position_bias)


class CrossAttenderBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=True,
        kdim=None,
        vdim=None,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
            kdim=kdim,
            vdim=vdim,
        )
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        k,
        v,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        src2 = self.attn(
            query=src2,
            key=k,
            value=v,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward_post(
        self,
        src,
        k,
        v,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.attn(
            query=src,
            key=k,
            value=v,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(self.norm1(src2))
        src = src + self.drop_path2(self.norm2(self.mlp(src)))
        return src

    def forward(
        self,
        src,
        key,
        value,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        if self.post_norm:
            return self.forward_post(src=src,
                                     k=key,
                                     v=value,
                                     mask=mask,
                                     key_padding_mask=key_padding_mask)

        return self.forward_pre(src=src,
                                k=key,
                                v=value,
                                mask=mask,
                                key_padding_mask=key_padding_mask)


class T5Attention(nn.Module):

    def __init__(self,
                 d_model,
                 d_kv,
                 num_heads,
                 dropout_rate,
                 hist_step: float = 50):
        super().__init__()

        self.d_model = d_model
        self.d_kv = d_kv
        self.n_heads = num_heads
        self.dropout = dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.pos_embed = RelativePositionBias(bidirectional=True,
                                              num_buckets=64,
                                              max_distance=hist_step,
                                              n_heads=self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self,
                input,
                mask=None,
                kv=None,
                position_bias=None,
                cache=None,
                head_mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k],
                                  dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v],
                                  dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        # q = q / math.sqrt(dim_per_head)                                     # No scaling in T5
        scores = torch.einsum("bnqd,bnkd->bnqk", q,
                              k)  # (bs, n_heads, qlen, klen)
        # convert mask to float
        mask = mask.view(bs, 1, 1, qlen).   \
            expand(-1, self.n_heads, -1, -1)
        if mask is not None and mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(mask, float("-inf"))
            mask = new_attn_mask

        if position_bias:
            position_bias = self.pos_embed.compute_bias(qlen, klen)
            position_bias = position_bias.permute([2, 0, 1]).unsqueeze(
                0)  # shape (1, num_heads, qlen, klen)

            if mask is not None:
                mask = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += mask
        weights = F.softmax(scores.float(), dim=-1).type_as(
            scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(
            weights, p=self.dropout,
            )  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context, )
        outputs = outputs + (weights, )
        outputs = outputs + (position_bias, )
        return outputs
    