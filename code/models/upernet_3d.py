import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain
from typing import Sequence


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class PSPModule3D(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule3D, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(
                in_channels + (out_channels * len(bin_sizes)),
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool3d(output_size=bin_sz)
        conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm3d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        d, h, w = features.size()[2], features.size()[3], features.size()[4]
        pyramids = [features]
        pyramids.extend(
            [
                F.interpolate(
                    stage(features),
                    size=(d, h, w),
                    mode="trilinear",
                    align_corners=True,
                )
                for stage in self.stages
            ]
        )
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


def up_and_add(x, y):
    return (
        F.interpolate(
            x,
            size=(y.size(2), y.size(3), y.size(4)),
            mode="trilinear",
            align_corners=True,
        )
        + y
    )


class FPN_fuse3D(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse3D, self).__init__()
        self.conv1x1 = nn.ModuleList(
            [
                nn.Conv3d(ft_size, fpn_out, kernel_size=1)
                for ft_size in feature_channels[1:]
            ]
        )
        self.smooth_conv = nn.ModuleList(
            [nn.Conv3d(fpn_out, fpn_out, kernel_size=3, padding=1)]
            * (len(feature_channels) - 1)
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv3d(
                len(feature_channels) * fpn_out,
                fpn_out,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(fpn_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):

        features[1:] = [
            conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)
        ]
        P = [
            up_and_add(features[i], features[i - 1])
            for i in reversed(range(1, len(features)))
        ]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        D, H, W = P[0].size(2), P[0].size(3), P[0].size(4)
        P[1:] = [
            F.interpolate(feature, size=(D, H, W), mode="trilinear", align_corners=True)
            for feature in P[1:]
        ]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class UperNet3D(nn.Module):
    # Implementing only the object path
    def __init__(
        self,
        image_size: Sequence[int],
        num_classes: int,
        feature_channels: int = [64, 128, 256, 512],
        fpn_out: int = 64,
        freeze_bn: bool = False,
        **_
    ):
        super(UperNet3D, self).__init__()
        self.image_size = image_size

        # if backbone == "resnet34" or backbone == "resnet18":
        #     feature_channels = [64, 128, 256, 512]
        # else:
        #     feature_channels = [256, 512, 1024, 2048]
        assert feature_channels[0] == fpn_out

        # self.backbone = ResNet(
        #     in_channels=in_channels,
        #     output_stride=output_stride,
        #     backbone=backbone,
        #     pretrained=pretrained,
        # )
        self.PPN = PSPModule3D(feature_channels[-1])
        self.FPN = FPN_fuse3D(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv3d(fpn_out, num_classes, kernel_size=3, padding=1)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, features):

        # features = self.backbone(x)
        # for feat in features:
        #     print(feat.size())
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=self.image_size, mode="trilinear", align_corners=True)
        return x

    # def get_backbone_params(self):
    #     return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(
            self.PPN.parameters(), self.FPN.parameters(), self.head.parameters()
        )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()


if __name__ == "__main__":
    # x = [
    #     torch.randn(2, 192, 24, 12, 12),
    #     torch.randn(2, 384, 24, 6, 6),
    #     torch.randn(2, 768, 32, 3, 3),
    #     torch.randn(2, 768, 32, 3, 3),
    # ]
    x = [
        torch.randn(2, 256, 24, 12, 12),
        torch.randn(2, 512, 24, 6, 6),
        torch.randn(2, 1024, 24, 3, 3),
        torch.randn(2, 1024, 24, 3, 3),
    ]

    model = UperNet3D(
        image_size=(96, 96, 96),
        num_classes=14,
        # feature_channels=[192, 384, 768, 768],
        feature_channels=[256, 512, 1024, 1024],
        # fpn_out=192,
        fpn_out=256,
        freeze_bn=False,
    )

    y = model(x)
    print(y.size())
