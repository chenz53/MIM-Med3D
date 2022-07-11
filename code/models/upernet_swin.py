from typing import Sequence, Tuple, Union

import torch

from .swin_3d import SwinTransformer3D
from .upernet_3d import UperNet3D
from mmcv.runner import load_checkpoint


class UperNetSwin(torch.nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        pretrained: Union[str, None],
        backbone_dict: dict,
        decode_head_dict: dict,
        revise_keys=[],
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()
        self.pretrained = pretrained

        self.backbone = SwinTransformer3D(**backbone_dict)
        self.decode_head = UperNet3D(**decode_head_dict)

        self.init_weights(revise_keys=revise_keys)

    def init_weights(self, pretrained=None, revise_keys=[]):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            print("load checkpoints from {}".format(self.pretrained))
            load_checkpoint(
                self,
                filename=self.pretrained,
                # map_location="cpu",
                strict=False,
                revise_keys=revise_keys,
            )
        elif self.pretrained is None:
            pass
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x_in):
        # x_in [B, C, H, W, D]
        x_in = x_in.permute(0, 1, 4, 2, 3)  # b, c, d, h, w
        x, stage_outputs = self.backbone.forward_features(x_in)
        y = self.decode_head(stage_outputs)  # b, c, d, h, w

        return y.permute(0, 1, 3, 4, 2)  # b, c, h, w, d


if __name__ == "__main__":
    import torch

    x = torch.randn(1, 1, 96, 96, 96)

    backbone_dict = {
        "pretrained": None,
        "pretrained2d": True,
        "img_size": (96, 96, 96),
        "patch_size": (4, 4, 4),
        "in_chans": 1,
        "num_classes": 0,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": (7, 7, 7),
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "norm_layer": nn.LayerNorm,
        "ape": False,
        "patch_norm": True,
        "frozen_stages": -1,
        "use_checkpoint": False,
    }
    decode_head_dict = {
        "image_size": (96, 96, 96),
        "num_classes": 14,
        "feature_channels": [192, 384, 768, 768],
        "fpn_out": 192,
        "freeze_bn": False,
    }
    model = UperNetSwin(backbone_dict=backbone_dict, decode_head_dict=decode_head_dict)

    y = model(x)
    print(y.shape)
