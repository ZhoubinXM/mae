from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch import Tensor
from src.model.layers.relative_position_bias import RelativePositionBias
from src.utils.weight_init import weight_init
import math


class RMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


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


class MLPLayer(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm) -> None:
        super(MLPLayer, self).__init__()
        if norm_layer is not None:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                norm_layer(hidden_dim),
                act_layer(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act_layer(),
                nn.Linear(hidden_dim, output_dim),
            )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


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
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
        attn_type="norm",
        max_t_len=50,
        attn_bias=True,
        ffn_bias=True,
    ):
        super().__init__()
        self.post_norm = post_norm
        self.freqs_cis = None

        self.norm1 = norm_layer(dim)
        if attn_type == "norm":
            self.attn = myMultiheadAttention(
                dim,
                num_heads=num_heads,
                bias=attn_bias,
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
        elif attn_type == "Ro":
            self.attn = RoEmbedAttention(num_heads=8,
                                         dim=dim,
                                         qkv_bias=qkv_bias,
                                         dropout=attn_drop)
            self.freqs_cis = precompute_freqs_cis(dim // num_heads,
                                                  end=max_t_len)

        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
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
        if self.freqs_cis is not None:
            position_bias = self.freqs_cis
        # pre norm: before atten and ffn
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
            return False
            # return self.forward_post(src=src,
            #                          mask=mask,
            #                          key_padding_mask=key_padding_mask,
            #                          position_bias=position_bias)

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
        norm_layer=RMSNorm,
        post_norm=True,
        kdim=None,
        vdim=None,
        attn_bias=True,
        ffn_bias=True,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.norm1 = norm_layer(dim)
        self.normk = norm_layer(dim)
        self.normv = norm_layer(dim)
        self.attn = myMultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
            kdim=kdim,
            vdim=vdim,
            bias=attn_bias,
        )
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop,
                       bias=ffn_bias)
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
        k = self.normk(k)
        v = self.normv(v)
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
                 hist_step: float = 50,
                 bias: bool = True):
        super().__init__()

        self.d_model = d_model
        self.d_kv = d_kv
        self.n_heads = num_heads
        self.dropout = dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=bias)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=bias)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=bias)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=bias)

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
            weights,
            p=self.dropout,
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


class RoEmbedAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, num_heads=8, dim=128, qkv_bias=False, dropout=0.):
        """
        Initialize the Attention module.

        Args:
            num_heads
            dim

        Attributes:
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.

        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=qkv_bias)

    def forward(
        self,
        input: torch.Tensor,
        position_bias: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        freqs_cis = position_bias.to(input.device)
        x = input
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim**0.5)
        mask = mask.view(bsz, 1, 1, seqlen).   \
            expand(-1, self.num_heads, -1, -1)
        if mask is not None and mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(mask, dtype=xq.dtype)
            new_attn_mask.masked_fill_(mask, float("-inf"))
            mask = new_attn_mask
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = F.dropout(
            scores,
            p=self.dropout,
        )  # (bs, n_heads, qlen, klen)
        output = torch.matmul(scores,
                              xv)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return (self.wo(output), )


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # import pdb
    # pdb.set_trace()
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout=0.0):
    # q           (B * nhead, tgt_len, head_dim)
    # kv          (B * nhead, src_len, head_dim)
    # attn_mask   (B * nhead, 1 or tgt_len, src_len)
    # out         (B * nhead, tgt_len, head_dim)

    B, Nt, E = q.shape
    q = q / math.sqrt(E)

    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))

    # attn mask will set -inf to attn positions that must be masked
    # mask is 0 by default so no masking takes place
    if attn_mask is not None:
        attn += attn_mask

    attn = F.softmax(attn, dim=-1)

    if dropout > 0.0:
        attn = F.dropout(attn, p=dropout)

    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)

    return output, attn


def _in_projection_packed(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          w: torch.Tensor,
                          b: Optional[torch.Tensor] = None):
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
            # q:        (B, *, in_features)         -> (..., E)
            # w:        (out_features, in_features) -> (E * 3, E)
            # b:        (out_features)              -> (E * 3)
            # lin_out:  (B, *, out_features)        -> (..., E * 3)
            # chunk_out:                            -> 3 * (..., E)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            # will concat q_out with k_out v_out
            #                            |
            #                            V
            return (F.linear(q, w_q, b_q), ) + F.linear(k, w_kv, b_kv).chunk(
                2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k,
                                               b_k), F.linear(v, w_v, b_v)


class myMultiheadAttention(nn.Module):

    def __init__(self,
                 d_model,
                 num_heads,
                 dropout=0.0,
                 batch_first=False,
                 bias=True,
                 add_bias_kv=False,
                 kdim=None,
                 vdim=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.head_dim = d_model // num_heads
        assert (self.head_dim * num_heads == d_model), "d_model % nhead != 0"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * d_model, d_model)))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None):
        #                     Enc             Dec tgt         Dec mem
        # query, key, value:  [672, 2, 256]   [100, 2, 256]   [100, 2, 256], [672, 2, 256], [672, 2, 256]
        # attn_mask:          None            None            None
        # key_padding_mask:   [2, 672]        None            [2, 672]
        # output:             [672, 2, 256]   [100, 2, 256]   [100, 2, 256]

        # key_padding_mask: used to mask out padding positions after the end
        #                   of the input sequence. It depends on the longest
        #                   sequence in the batch. Shape (B, src seq length)

        # attn_mask:        used in decoders to prevent attention to future
        #                   positions using a triangle mask.
        #                   2D shape: (tgt seq length, src seq length)
        #                   3D shape: (B*nhead, tgt seq length, src seq length)

        # q:                (tgt seq length, B, C)
        # kv:               (src seq length, B, C)
        # out:
        #   - attn_output           (tgt seq length, B, C)
        #   - attn_output_weights   (B, tgt seq length, C)

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [
                x.transpose(1, 0) for x in (query, key, value)
            ]

        tgt_len, batch_size, embed_dim = query.shape
        src_len, _, _ = key.shape

        assert (embed_dim == self.d_model
                ), f"expected hidden dim = {self.d_model}, but got {embed_dim}"
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

        # compute in-projection
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight,
                                        self.in_proj_bias)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            assert attn_mask.is_floating_point(
            ) or attn_mask.dtype == torch.bool, "wrong attn_mask type"

            if attn_mask.dim() == 2:
                assert (tgt_len,
                        src_len) == attn_mask.shape, "wrong attn_mask shape"
                attn_mask = attn_mask.unsqueeze(0)
                # add artificial batch_size=1
            elif attn_mask.dim() == 3:
                assert (batch_size * self.nhead, tgt_len,
                        src_len) == attn_mask.shape, "wrong attn_mask shape"
            else:
                assert False, "wrong attn_mask shape"

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            key_padding_mask = key_padding_mask.to(torch.bool)

        # reshape q, k, v for multihead attention and make em batch first
        # q:    (tgt_len, B, C)->(tgt_len, B, nhead * head_dim)->
        #       (tgt_len, B * nhead, head_dim)->(B * nhead, tgt_len, head_dim)
        q = q.contiguous().view(tgt_len, batch_size * self.nhead,
                                self.head_dim).transpose(0, 1)

        # kv:   (src_len, B, C)->(src_len, B, nhead * head_dim)->
        #       (src_len, B * nhead, head_dim)->(B * nhead, src_len, head_dim)
        # .view(-1, ...) lets python compute the first dim based on the other dims specified
        k = k.contiguous().view(-1, batch_size * self.nhead,
                                self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * self.nhead,
                                self.head_dim).transpose(0, 1)

        # update source sequence length after adjustments
        src_len = k.shape[1]

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                batch_size, src_len
            ), f"expecting key_padding_mask shape of {(batch_size, src_len)}, but got {key_padding_mask.shape}"

            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len)
            key_padding_mask = key_padding_mask.expand(-1, self.nhead, -1, -1)
            # -1 means not changing the size of that dimension
            key_padding_mask = key_padding_mask.reshape(
                batch_size * self.nhead, 1, src_len)

            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask,
                                                  float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        if not self.training:
            self.dropout = 0.0

        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = _scaled_dot_product_attention(
            q, k, v, attn_mask, self.dropout)
        #attn_output            [16, 100, 32]
        #attn_output_weights    [16, 100, 672]

        attn_output = attn_output.transpose(0, 1).contiguous().view(
            tgt_len, batch_size, embed_dim)
        #attn_output [16, 100, 32]->[100, 16, 32]->[100, 2, 256]

        #attn_output            [100, 2, 256]
        #self.out_proj.weight   [256, 256]
        #self.out_proj.bias     [256]
        attn_output = F.linear(attn_output, self.out_proj.weight,
                               self.out_proj.bias)
        #attn_output [100, 2, 256]
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
