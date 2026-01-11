# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MetaFormer baselines including IdentityFormer, RandFormer, PoolFormerV2,
ConvFormer and CAFormer.   ------sobel best------sobel best------sobel best------sobel best------sobel best------sobel best
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple
from models.DWEncoder import DWEncoder
from models.auxy import ImageProcessor
from models.detect_head import Default_head, Default_head2
from models.unetp import UnetDecoder
from models.TransitionConv import TEncoder
from models.SGCSA3 import *
from train_util import *
__all__ = [
    'dual1'
]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'identityformer_s12': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s12.pth'),
    'identityformer_s24': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s24.pth'),
    'identityformer_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s36.pth'),
    'identityformer_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m36.pth'),
    'identityformer_m48': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m48.pth'),

    'randformer_s12': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s12.pth'),
    'randformer_s24': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s24.pth'),
    'randformer_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s36.pth'),
    'randformer_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m36.pth'),
    'randformer_m48': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m48.pth'),

    'poolformerv2_s12': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s12.pth'),
    'poolformerv2_s24': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s24.pth'),
    'poolformerv2_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s36.pth'),
    'poolformerv2_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m36.pth'),
    'poolformerv2_m48': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m48.pth'),

    'convformer_s18': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18.pth'),
    'convformer_s18_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384.pth',
        input_size=(3, 384, 384)),
    'convformer_s18_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21ft1k.pth'),
    'convformer_s18_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'convformer_s18_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21k.pth',
        num_classes=21841),

    'convformer_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36.pth'),
    'convformer_s36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384.pth',
        input_size=(3, 384, 384)),
    'convformer_s36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21ft1k.pth'),
    'convformer_s36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'convformer_s36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21k.pth',
        num_classes=21841),

    'convformer_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36.pth'),
    'convformer_m36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384.pth',
        input_size=(3, 384, 384)),
    'convformer_m36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21ft1k.pth'),
    'convformer_m36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'convformer_m36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21k.pth',
        num_classes=21841),

    'convformer_b36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36.pth'),
    'convformer_b36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384.pth',
        input_size=(3, 384, 384)),
    'convformer_b36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21ft1k.pth'),
    'convformer_b36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'convformer_b36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21k.pth',
        num_classes=21841),

    'caformer_s18': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth'),
    'caformer_s18_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384.pth',
        input_size=(3, 384, 384)),
    'caformer_s18_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21ft1k.pth'),
    'caformer_s18_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'caformer_s18_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21k.pth',
        num_classes=21841),

    'caformer_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36.pth'),
    'caformer_s36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384.pth',
        input_size=(3, 384, 384)),
    'caformer_s36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21ft1k.pth'),
    'caformer_s36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'caformer_s36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21k.pth',
        num_classes=21841),

    'caformer_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36.pth'),
    'caformer_m36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384.pth',
        input_size=(3, 384, 384)),
    'caformer_m36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21ft1k.pth'),
    'caformer_m36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'caformer_m36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21k.pth',
        num_classes=21841),

    'caformer_b36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36.pth'),
    'caformer_b36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384.pth',
        input_size=(3, 384, 384)),
    'caformer_b36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21ft1k.pth'),
    'caformer_b36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'caformer_b36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21k.pth',
        num_classes=21841),
}


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """

    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
            requires_grad=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default.
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance.
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """

    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True,
                 bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster,
    because it directly utilizes otpimized F.layer_norm
    """

    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, kernel_size=7, padding=3,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """

    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


r"""
downsampling (stem) for the first stage is a layer of conv with k7, s4 and p2
downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
use `partial` to specify some arguments
"""
DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
                                         kernel_size=7, stride=4, padding=2,
                                         post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6)
                                         )] + \
                                [partial(Downsampling,
                                         kernel_size=3, stride=2, padding=1,
                                         pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True
                                         )] * 3


class MetaFormer(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (int): Feature dimension at each stage. Default: [64, 128, 320, 512].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage. Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
            None means not use the layer scale. From: https://arxiv.org/abs/2110.09456.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 mlps=Mlp,
                 norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
                 # partial(LayerNormGeneral, eps=1e-6, bias=False),
                 drop_path_rate=0.,
                 head_dropout=0.0,
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 output_norm=partial(nn.LayerNorm, eps=1e-6),
                 head_fn=nn.Linear,
                 get_feat=False,
                 dulbrn=0,
                 **kwargs,
                 ):
        super().__init__()
        self.dulbrn = dulbrn
        if self.dulbrn:
            print("add local conv layer")
            # self.local_conv = nn.Sequential(
            #     nn.Conv2d(3, self.dulbrn, 3, stride=2, padding=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(self.dulbrn, self.dulbrn, 3, padding=1),
            #     nn.ReLU(True),
            # )
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, self.dulbrn, (3, 3), stride=1, padding=1),
                nn.ReLU(inplace=True)
            )

            self.conv2 = nn.Sequential(
                # nn.MaxPool2d(2, stride=2, ceil_mode=True),
                nn.Conv2d(self.dulbrn, self.dulbrn * 2, (3, 3), stride=2, padding=1),
                nn.ReLU(inplace=True)
            )

            self.out_channels = [self.dulbrn, self.dulbrn * 2]
            # self.out_channels = [self.dulbrn]
            self.out_channels.extend(dims[:-1])
        else:
            print("not add local conv layer")
            self.out_channels = dims

        self.out_channels = [24, 48]

        self.out_channels.extend(dims)

        fusion_channels = [(192, 96), (384, 192), (768, 384), (1152, 576)]
        self.fusion_convs = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 1, 1) for in_ch, out_ch in fusion_channels])

        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        ## add by Archer, remove the last SA layer
        # num_stage = len(depths) - 1
        num_stage = len(depths)
        self.num_stage = num_stage - 1

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i + 1]) for i in range(num_stage)]
        )

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList()  # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i],
                                  token_mixer=token_mixers[i],
                                  mlp=mlps[i],
                                  norm_layer=norm_layers[i],
                                  drop_path=dp_rates[cur + j],
                                  layer_scale_init_value=layer_scale_init_values[i],
                                  res_scale_init_value=res_scale_init_values[i],
                                  ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.dw_encoder1 = DWEncoder()
        self.transition_encoder11 = TEncoder()
        self.transition_encoder2 = TEncoder()

        self.inity_down1 = nn.Conv2d(48, 24, kernel_size=1)
        self.inity_down2 = nn.Conv2d(96, 48, kernel_size=1)

        # self.mhcsa1 = MHCSA(96)
        # self.mhcsa2 = MHCSA(192)
        # self.mhcsa3 = MHCSA(384)
        # self.mhcsas = [self.mhcsa1, self.mhcsa2, self.mhcsa3]
        self.sgcsa1 = GCSA(96, bias=True)
        self.sgcsa2 = GCSA(192, bias=True)
        self.sgcsa3 = GCSA(384, bias=True)
        self.sgcsas = [self.sgcsa1, self.sgcsa2, self.sgcsa3]



    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward(self, x):

        image_processor = ImageProcessor()
        sobel_x = image_processor.process_image(x, mode='sobel')

        gradient = self.transition_encoder11(sobel_x)#-------------------------------------------------------------------------
        rgb_x = self.transition_encoder2(x)

        f1 = torch.cat((gradient[0], rgb_x[0]), dim=1)
        f1 = self.inity_down1(f1)
        f2 = torch.cat((gradient[1], rgb_x[1]), dim=1)
        f2 = self.inity_down2(f2)


        if self.dulbrn:
            f1 = self.conv1(x)
            f2 = self.conv2(f1)
            features = [f1, f2]
            # features = [self.local_conv(x)]
        else:
            features = []

        features = [f1, f2]

        cnn_encoder_out = x
        cnn_encoder_out = self.dw_encoder1(cnn_encoder_out)  # ----------------------------
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature = self.sgcsas[i](x.permute(0, 3, 1, 2), cnn_encoder_out[i])
            # feature = self.trans_rates[i] * x.permute(0, 3, 1, 2) + self.cnn_rates[i] * cnn_encoder_out[i]
            # softmax_att = self.fusion_atts[i](feature)
            # feature =torch.cat([x.permute(0, 3, 1, 2).unsqueeze(1), cnn_encoder_out[i].unsqueeze(1)], dim=1)
            # feature = (feature * softmax_att).sum(dim=1)

            # feature = torch.cat((x.permute(0, 3, 1, 2), cnn_encoder_out[i]), dim=1)
            # feature = self.fusion_convs[i](feature)
            # x = feature.permute(0, 2, 3, 1)
            features.append(feature)

        return features


@register_model
def identityformer_s12(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_s12']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def identityformer_s24(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_s24']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def identityformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def identityformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def identityformer_m48(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_m48']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_s12(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_s12']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_s24(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_s24']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_m48(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_m48']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def poolformerv2_s12(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_s12']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def poolformerv2_s24(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_s24']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def poolformerv2_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def poolformerv2_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def poolformerv2_m48(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_m48']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)

    if pretrained:
        state_dict = torch.load("model/convformer_s36_384_in21ft1k.pth",
                                map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k.replace('module.', '')  # 去除 "module." 的 prefix
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict)
    return model


@register_model
def convformer_s36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_m36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_m36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_m36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_m36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


def convformer_m36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        get_feat=True,
        **kwargs)
    if pretrained:
        state_dict = torch.load("model/convformer_m36_384_in21ft1k.pth",
                                map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k.replace('module.', '')  # 去除 "module." 的 prefix
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict)
    return model


@register_model
def convformer_m36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_m36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s18(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s18']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s18_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s18_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s18_384_in21ft1k(pretrained=False, Dulbrn=0, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        dulbrn=Dulbrn,
        **kwargs)
    if pretrained:
        state_dict = torch.load("model/caformer_s18_384_in21ft1k.pth",
                                map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def caformer_s18_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    if pretrained:
        state_dict = torch.load("model/caformer_s36_384_in21ft1k.pth",
                                map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def caformer_s36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_m36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_m36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_m36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_m36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_m36_384_in21ft1k(pretrained=False, Dulbrn=0, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        dulbrn=Dulbrn,
        **kwargs)
    if pretrained:
        state_dict = torch.load("F:\lyh\Multiattention-UNet\models\caformer_m36_384_in21ft1k.pth",
                                map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


@register_model
def caformer_m364_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_m364_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


class MyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MyLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels,
                                padding=kernel_size // 2)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=1)
        self.dwconv2 = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, groups=out_channels * 2,
                                 padding=kernel_size // 2)
        self.activation2 = nn.GELU()
        self.conv4 = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.dwconv(x1)
        x1 = self.activation(x1)
        x1 = self.conv2(x1)

        x2 = self.pool(x1)
        y = self.conv3(x2)
        y = self.dwconv2(y)
        y = self.activation2(y)
        y = self.conv4(y)

        return x1, y


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)


class sam_residual(nn.Module):
    def __init__(self, inplanes, planes):
        super(sam_residual, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Sequential(conv3x3(planes, planes), simam_module(planes))
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.silu(out)

        return out

class fusionatt(nn.Module):
    def __init__(self, channels, ratio=16, spatial_kernel=7):
        super(fusionatt, self).__init__()

        # channel attention
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.sigmoid1 = nn.Sigmoid()
        # self.sigmoid2 = nn.Sigmoid()

        self.conv1 = nn.Conv1d(2, 1, kernel_size=5, padding=2, bias=False)

        self.spatial_conv1 = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, feature):
        b1, c1, h1, w1 = feature.shape
        max_out1 = self.max_pool(feature).squeeze(-1).squeeze(-1)
        avg_out1 = self.avg_pool(feature).squeeze(-1).squeeze(-1)
        stacked1 = torch.stack([avg_out1, max_out1], dim=1)
        out1 = self.conv1(stacked1).squeeze(1)
        out1 = out1.unsqueeze(2).unsqueeze(3)
        channel_out1 = out1.repeat(1, 1, h1, w1)
        # channel_out1 = self.sigmoid1(channel_out1)

        spatial_mean_out1 = torch.mean(feature, dim=1, keepdim=True)
        spatial_max_out1, _ = torch.max(feature, dim=1, keepdim=True)
        spatial_out1 = torch.cat([spatial_mean_out1, spatial_max_out1], dim=1)
        spatial_out1 = self.spatial_conv1(spatial_out1)
        spatial_out1 = spatial_out1.repeat(1, c1, 1, 1)
        # spatial_out1 = self.sigmoid2(spatial_out1)

        out = channel_out1 + spatial_out1

        # out = feature * out

        return out

class MHCSA(nn.Module):
    def __init__(self, n_dims):
        super(MHCSA, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.cnn_att_key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.cnn_att = fusionatt(n_dims)
        self.conv1 = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.conv2 = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x, x2):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)
        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = self.cnn_att_key(self.cnn_att(x2)).view(n_batch, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)
# r wise
        energy = content_content + content_position
        attention = self.softmax(energy)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        attn1 = self.conv1(out)
        attn2 = self.conv2(out)
        att = torch.stack([attn1, attn2], dim=1)
        softmax_att = self.softmax1(att)
        feature = torch.cat([x.unsqueeze(1), x2.unsqueeze(1)], dim=1)
        out = (feature * softmax_att).sum(dim=1)

        return out

class DUAL(nn.Module):
    def __init__(self):
        super(DUAL, self).__init__()
        self.encoder1 = caformer_m36_384_in21ft1k(pretrained=True, Dulbrn=0)

        # incs = (24, 48, 96, 192, 384, 576)
        # oucs = (36, 72, 144, 288, 480)
        incs = (24, 48, 96, 192, 384)
        oucs = (36, 56, 112, 224)

        self.decoder = UnetDecoder(incs, oucs[-len(incs):])

        self.head = Default_head(oucs)
        self.head2 = Default_head2(oucs)

        # self.parallel_att = Parallel_att_skip()

    def forward(self, x):
        features = self.encoder1(x)

        # t1, t2, t3, t4, t5 = features[0], features[1], features[2], features[3], features[4]
        # t1, t2, t3, t4, t5 = self.parallel_att(t1, t2, t3, t4, t5)
        # features = [t1, t2, t3, t4, t5]

        features = self.decoder(features)
        edge = self.head(features)
        edge2 = self.head2(features)
        return edge, edge2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

def dual1(data=None):
# Model without  batch normalization
    model = DUAL()
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model