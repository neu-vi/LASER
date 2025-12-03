# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
import os
from typing import Callable, List, Any, Union, Tuple, Dict
import warnings

import torch
from torch import nn, Tensor

from .attention import CrossAttentionRope, MemEffCrossAttentionRope, FlashAttentionRope
from ..dinov2.layers.drop_path import DropPath
from ..dinov2.layers.layer_scale import LayerScale
from ..dinov2.layers.mlp import Mlp


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Block)")
    else:
        # warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Block)")


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs

class BlockRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = FlashAttentionRope,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool=False,
        rope=None
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(
            self,
            x: Tensor,
            xpos=None,
            use_cache=False,
            kv_cache=None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple]]:
        def attn_residual_func(x: Tensor) -> Union[Tensor, Tuple[Tensor, Tuple]]:
            if use_cache:
                x, new_cache = self.attn(self.norm1(x), xpos=xpos, use_cache=True, kv_cache=kv_cache)
            else:
                x = self.attn(self.norm1(x), xpos=xpos)

            if use_cache:
                return self.ls1(x), new_cache
            return self.ls1(x)

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if use_cache:
            attn_out, new_cache = attn_residual_func(x)
            x = x + attn_out
            x = x + ffn_residual_func(x)
            return x, new_cache

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


class CrossBlockRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = FlashAttentionRope,
        cross_attn_class: Callable[..., nn.Module] = CrossAttentionRope,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        init_values=None,
        qk_norm: bool=False,
        rope=None
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            rope=rope,
            qk_norm=qk_norm
        )

        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls_y = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm_y = norm_layer(dim)
        self.cross_attn = cross_attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            rope=rope,
            qk_norm=qk_norm
        )

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )

    def forward(
            self,
            x: Tensor,
            y: Tensor,
            xpos=None,
            ypos=None,
            use_cache=False,
            kv_cache_dict=None
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        def attn_residual_func(x: Tensor, kv_cache=None) -> Union[Tensor, Tuple[Tensor, Tuple]]:
            if use_cache:
                x, new_cache = self.attn(
                    self.norm1(x),
                    xpos=xpos,
                    use_cache=True,
                    kv_cache=kv_cache
                )
            else:
                x = self.attn(self.norm1(x), xpos=xpos)

            if use_cache:
                return self.ls1(x), new_cache
            return self.ls1(x)

        def cross_attn_residual_func(x: Tensor, y: Tensor, kv_cache=None) -> Union[Tensor, Tuple[Tensor, Tuple]]:
            if use_cache:
                x, new_cache = self.cross_attn(
                    self.norm2(x),
                    y,
                    y,
                    qpos=xpos,
                    kpos=ypos,
                    use_cache=True,
                    kv_cache=kv_cache
                )
            else:
                x = self.cross_attn(self.norm2(x), y, y, qpos=xpos, kpos=ypos)

            if use_cache:
                return self.ls_y(x), new_cache
            return self.ls_y(self.cross_attn(self.norm2(x), y, y, qpos=xpos, kpos=ypos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm3(x)))

        if use_cache:
            attn_cache = None
            cross_attn_cache = None
            if kv_cache_dict is not None:
                attn_cache, cross_attn_cache = kv_cache_dict['attn_cache'], kv_cache_dict['cross_attn_cache']
            attn_out, new_attn_cache = attn_residual_func(x, kv_cache=attn_cache)
            x = x + attn_out
            y_ = self.norm_y(y)
            cross_attn_out, new_cross_attn_cache = cross_attn_residual_func(x, y_, kv_cache=cross_attn_cache)
            x = x + cross_attn_out
            x = x + ffn_residual_func(x)
            return x, {'attn_cache': new_attn_cache, 'cross_attn_cache': new_cross_attn_cache}

        x = x + attn_residual_func(x)
        y_ = self.norm_y(y)
        x = x + cross_attn_residual_func(x, y_)
        x = x + ffn_residual_func(x)
        return x