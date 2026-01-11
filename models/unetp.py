from torch import nn
import torch
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)

    def forward(self, x):
        return self.attention(x)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.InstanceNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(True)
        )

    def forward(self, x, skip=None):

        if skip is not None:
            x = self.deconv(x)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            self.deconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class DDblock(nn.Module):
    def __init__(
            self,
            skip1_channels,
            in_channels,
            skip2_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip1_channels + skip2_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip1_channels + skip2_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(skip2_channels, skip2_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(skip2_channels),
            nn.SiLU(True)
        )
        self.upconv = nn.Sequential(
            nn.Conv2d(skip1_channels, skip1_channels, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(skip1_channels),
            nn.SiLU(True)
        )

    def forward(self, skip1, x, skip2):
        skip1 = self.upconv(skip1)
        skip2 = self.deconv(skip2)
        x = torch.cat([skip1, x, skip2], dim=1)
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()


        self.width = len(encoder_channels)
        convs = dict()
        for w in range(self.width - 1):
            for d in range(self.width - w - 1):
                if w == 0 and d == 0:
                    convs["conv{}_{}".format(d, w)] = DecoderBlock(encoder_channels[d + 1],
                                                                   encoder_channels[d],
                                                                   decoder_channels[d])
                elif w == 0 and d > 0:
                    convs["conv{}_{}".format(d, w)] = DDblock(encoder_channels[d - 1], encoder_channels[d], encoder_channels[d+1], decoder_channels[d])
                elif w > 0 and d == 0:
                    convs["conv{}_{}".format(d, w)] = DecoderBlock(decoder_channels[d + 1],
                                                                   decoder_channels[d],
                                                                   decoder_channels[d])
                else:
                    convs["conv{}_{}".format(d, w)] = DDblock(decoder_channels[d - 1], decoder_channels[d], decoder_channels[d+1], decoder_channels[d])

        self.convs = nn.ModuleDict(convs)


    def forward(self, features):

        new_features = features.copy()

        for w in range(self.width - 1):
            for d in range(self.width - w - 1):
                if d == 0:
                    new_features[d] = self.convs["conv{}_{}".format(d, w)](features[d + 1], features[d])
                else:
                    new_features[d] = self.convs["conv{}_{}".format(d, w)](features[d - 1], features[d], features[d + 1])

            features = new_features
        return features