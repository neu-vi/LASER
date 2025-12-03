# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    print('xFormers is available.')
    XFORMERS_AVAILABLE = True
except ImportError:
    print('xFormers is not available.')
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            norm_layer: nn.Module = nn.LayerNorm,
            qk_norm: bool = False,
            fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
            rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, return_attn=False, S=None) -> Tensor:
        B, N, C = x.shape
        # B, Lseq, 3, NH, C -> 3, B, NH, Lseq, C
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if return_attn:
            with torch.no_grad():
                attn_logit = (self.scale * q) @ k.transpose(-2, -1)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn_logit.detach().cpu().float()
        return x


class EfficientAttention(Attention):
    def forward(self, x: Tensor, pos=None, return_attn=False, S=None):
        if not XFORMERS_AVAILABLE:
            return super().forward(x, pos, return_attn=return_attn)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3)
        # B, NH, Lseq, C
        q, k, v = unbind(qkv, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if return_attn:
            with torch.no_grad():
                attn_logit = (self.scale * q) @ k.transpose(-2, -1)

        q = q.transpose(1, 2).to(v.dtype)
        k = k.transpose(1, 2).to(v.dtype)
        v = v.transpose(1, 2)
        x = memory_efficient_attention(q, k, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn_logit.detach().cpu().float()
        return x



class LocalAttention(Attention):
    def forward(self, x: Tensor, pos=None, return_attn=False, S=None):
        if not XFORMERS_AVAILABLE:
            print("xFormers is not available, using super().forward")
            return super().forward(x, pos, return_attn=return_attn)

        B, N, C = x.shape

        P = N // S
        window_left = P * 5
        window_right = P * 5
        attn_bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=window_left, window_right=window_right)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3)
        # B, NH, Lseq, C
        q, k, v = unbind(qkv, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if return_attn:
            with torch.no_grad():
                attn_logit = (self.scale * q) @ k.transpose(-2, -1)

        q = q.transpose(1, 2).to(v.dtype)
        k = k.transpose(1, 2).to(v.dtype)
        v = v.transpose(1, 2)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn_logit.detach().cpu().float()
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, S=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
