from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch import Tensor
from src.model.layers.relative_position_bias import RelativePositionBias
import math


def _scaled_dot_product_attention(q,
                                  k,
                                  v,
                                  attn_mask=None,
                                  dropout=0.0,
                                  position_bias=None):
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

    if position_bias is not None:
        attn += position_bias

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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        position_bias: Optional[bool] = None,
    ):
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

        if position_bias is not None:
            position_bias = position_bias.unsqueeze(1).repeat(
                1, self.nhead, 1, 1).reshape(batch_size * self.nhead, tgt_len,
                                             src_len)

        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = _scaled_dot_product_attention(
            q, k, v, attn_mask, self.dropout, position_bias)
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
