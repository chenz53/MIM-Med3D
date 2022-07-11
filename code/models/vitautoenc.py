from typing import Sequence, Union
import math

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep
from timm.models.layers import trunc_normal_

__all__ = ["ViTAutoEnc"]


class ViTAutoEnc(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Modified to also give same dimension outputs as the input size of the image
    """

    def __init__(
        self,
        pretrained: Union[str, None],
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        out_channels: int = 1,
        deconv_chns: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        revise_keys=[("model.", "")],
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            out_channels: number of output channels.
            deconv_chns: number of channels for the deconvolution layers.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            # It will provide an output of same size as that of the input
            >>> net = ViTAutoEnc(in_channels=1, patch_size=(16,16,16), img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), output will be same size as of input
            >>> net = ViTAutoEnc(in_channels=3, patch_size=(16,16,16), img_size=(128,128,128), pos_embed='conv')

        """

        super().__init__()

        self.pretrained = pretrained
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.spatial_dims = spatial_dims

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

        new_patch_size = [4] * self.spatial_dims
        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        # self.conv3d_transpose* is to be compatible with existing 3d model weights.
        self.conv3d_transpose = conv_trans(
            hidden_size, deconv_chns, kernel_size=new_patch_size, stride=new_patch_size
        )
        self.conv3d_transpose_1 = conv_trans(
            in_channels=deconv_chns,
            out_channels=out_channels,
            kernel_size=new_patch_size,
            stride=new_patch_size,
        )

        self.init_weights(revise_keys=revise_keys)

    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        # spatial_size = x.shape[2:]
        # x = self.patch_embedding(x)
        # hidden_states_out = []
        # for blk in self.blocks:
        #     x = blk(x)
        #     hidden_states_out.append(x)
        # x = self.norm(x)
        # x = x.transpose(1, 2)
        # d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        # x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)
        # return x, hidden_states_out
        return x

    def init_weights(self, pretrained=None, revise_keys=[]):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv3d):
                fan_out = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            load_checkpoint(
                self,
                filename=self.pretrained,
                map_location=torch.device("cpu"),
                strict=False,
                revise_keys=revise_keys,
            )
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")


if __name__ == "__main__":
    x = torch.randn(1, 768, 24, 3, 3)

    model = ViTAutoEnc(
        pretrained=None,
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        out_channels=1,
        deconv_chns=16,
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        pos_embed="conv",
        dropout_rate=0.0,
        spatial_dims=3,
    )

    y = model(x)
    print(y.size())
