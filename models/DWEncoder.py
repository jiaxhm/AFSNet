import math
import torch
import torch.nn as nn
from models.KSE import SEBlock
from mmcv.cnn import Conv2d
from mmcv.cnn.bricks import build_activation_layer
from mmcv.runner import BaseModule, Sequential
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)


class DepthWiseConvModule(BaseModule):
    """An implementation of one Depth-wise Conv Module of LEFormer.

    Args:
        embed_dims (int): The feature dimension.
        feedforward_channels (int): The hidden dimension for FFNs.
        output_channels (int): The output channles of each cnn encoder layer.
        kernel_size (int): The kernel size of Conv2d. Default: 3.
        stride (int): The stride of Conv2d. Default: 2.
        padding (int): The padding of Conv2d. Default: 1.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default: 0.0.
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(DepthWiseConvModule, self).__init__(init_cfg)
        self.activate = build_activation_layer(act_cfg)
        fc1 = Conv2d(
            in_channels=embed_dims,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelAttentionModule(BaseModule):
    """An implementation of one Channel Attention Module of LEFormer.

        Args:
            embed_dims (int): The embedding dimension.
    """

    def __init__(self, embed_dims):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            Conv2d(embed_dims, embed_dims // 4, 1, bias=False),
            nn.ReLU(),
            Conv2d(embed_dims // 4, embed_dims, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(BaseModule):
    """An implementation of one Spatial Attention Module of LEFormer.

        Args:
            kernel_size (int): The kernel size of Conv2d. Default: 3.
    """

    def __init__(self, kernel_size=3):
        super(SpatialAttentionModule, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiscaleCBAMLayer(BaseModule):
    """An implementation of Multiscale CBAM layer of LEFormer.

        Args:
            embed_dims (int): The feature dimension.
            kernel_size (int): The kernel size of Conv2d. Default: 7.
        """

    def __init__(self, embed_dims, kernel_size=7):
        super(MultiscaleCBAMLayer, self).__init__()
        self.channel_attention1 = ChannelAttentionModule(embed_dims // 4)
        self.spatial_attention1 = SpatialAttentionModule(kernel_size)
        self.channel_attention22 = ChannelAttentionModule(embed_dims // 2)
        self.spatial_attention2 = SpatialAttentionModule(kernel_size)
        self.multiscale_conv = nn.ModuleList()
        self.kse = SEBlock(embed_dims)
        for i in range(1, 5):
            self.multiscale_conv.append(
                Conv2d(
                    in_channels=embed_dims // 4,
                    out_channels=embed_dims // 4,
                    kernel_size=3,
                    stride=1,
                    padding=(2 * i + 1) // 2,
                    bias=True,
                    dilation=(2 * i + 1) // 2)
            )

    def forward(self, x):
        outs = torch.split(x, x.shape[1] // 4, dim=1)
        out_list = []
        for (i, out) in enumerate(outs):
            out = self.multiscale_conv[i](out)
            # out = self.spatial_attention2(out) * out
            out_list.append(out)
        cam_1 = self.channel_attention1(out_list[0]) * out_list[0]
        sam_2 = self.spatial_attention1(out_list[1]) * out_list[1]
        cam_3 = self.channel_attention1(out_list[2]) * out_list[2]
        sam_4 = self.spatial_attention1(out_list[3]) * out_list[3]

        out13 = torch.cat([cam_1, cam_3], dim=1)
        out24 = torch.cat([sam_2, sam_4], dim=1)

        sam_13 = self.spatial_attention2(out13) * out13
        cam_24 = self.channel_attention22(out24) * out24

        out1234 = torch.cat([sam_13, cam_24], dim=1)
        x_kse = self.kse(x)
        outt = out1234 + x_kse

        return outt

class CnnEncoderLayer(BaseModule):
    """Implements one cnn encoder layer in LEFormer.

        Args:
            embed_dims (int): The feature dimension.
            feedforward_channels (int): The hidden dimension for FFNs.
            output_channels (int): The output channles of each cnn encoder layer.
            kernel_size (int): The kernel size of Conv2d. Default: 3.
            stride (int): The stride of Conv2d. Default: 2.
            padding (int): The padding of Conv2d. Default: 0.
            act_cfg (dict): The activation config for FFNs.
                Default: dict(type='GELU').
            ffn_drop (float, optional): Probability of an element to be
                zeroed in FFN. Default 0.0.
            init_cfg (dict, optional): Initialization config dict.
                Default: None.
        """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=2,
                 padding=0,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 init_cfg=None):
        super(CnnEncoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.output_channels = output_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        self.layers = DepthWiseConvModule(embed_dims=embed_dims,
                                          feedforward_channels=feedforward_channels // 2,
                                          output_channels=output_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          act_cfg=dict(type='GELU'),
                                          ffn_drop=ffn_drop)

        self.multiscale_cbam = MultiscaleCBAMLayer(output_channels, kernel_size)

    def forward(self, x):
        out = self.layers(x)
        out = self.multiscale_cbam(out)
        return out







class DWEncoder(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_
    """

    def __init__(self,
                 in_channels=3,   #   输入
                 embed_dims=32,  # 32*(1 2 5 6)/(2 4 8 16)
                 embed_dims_list=[96, 192, 384, 576], # [64, 128, 256, 512]
                 feedforward_channels_list=[192, 384, 576, 576], #[128, 256, 512, 1024]
                 num_stages=4,
                 patch_sizes=(7, 3, 3, 3), # (3, 3, 3, 3, 3)
                 strides=(4, 2, 2, 2), # (1, 2, 2, 2, 2)
                 padding=[2, 1, 1, 1],
                 drop_rate=0.0,
                 **kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_stages = num_stages

        self.patch_sizes = patch_sizes
        self.strides = strides

        assert num_stages == len(strides)

        # embed_dims_list = []
        # feedforward_channels_list = []

        self.cnn_encoder_layers = nn.ModuleList()
        self.fusion_conv_layers = nn.ModuleList()
        for i in range(num_stages):
            self.cnn_encoder_layers.append(
                CnnEncoderLayer(
                    embed_dims=self.in_channels if i == 0 else embed_dims_list[i - 1],
                    # 3, 32*(1 2 5 6)/(2 4 8 16)  inchannel
                    feedforward_channels=feedforward_channels_list[i],  # 32*(1 2 5 6)/(2 4 8 16)*2  midchannel
                    output_channels=embed_dims_list[i],  # outchannel = inchannel
                    kernel_size=patch_sizes[i],  # (7, 3, 3, 3)
                    stride=strides[i],  # (4, 2, 2, 2),
                    padding=padding[i], # patch_sizes[i] // 2
                    ffn_drop=drop_rate
                )
            )
            self.fusion_conv_layers.append(
                Conv2d(
                    in_channels=embed_dims_list[i] * 2,
                    out_channels=embed_dims_list[i],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
            )
    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(DWEncoder, self).init_weights()

    def forward(self, x):
        outs = []
        cnnin = x

        for i, cnn_encoder_layer in enumerate(self.cnn_encoder_layers):
            cnnin = cnn_encoder_layer(cnnin)
            outs.append(cnnin)

            # x = self.fusion_conv_layers[i](x)
        return outs

if __name__ == '__main__':
    input = torch.Tensor(5, 3, 481, 321)
    net = DWEncoder()
    out=net(input)