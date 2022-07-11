from typing import Union, Sequence
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from .swin_3d import SwinTransformer3D
from monai.networks.layers import Conv
from monai.networks.nets import ViT
from mmcv.runner import load_checkpoint
from timm.models.layers import DropPath, trunc_normal_


class ViTSimMIM(nn.Module):
    def __init__(
        self,
        pretrained: Union[str, None],
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        masking_ratio: float = 0.5,
        revise_keys=[("model.", "")],
        **kwargs,
    ):
        super().__init__()

        self.pretrained = pretrained
        self.spatial_dims = spatial_dims

        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        self.encoder = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        # patch embedding block
        self.to_patch, self.patch_to_emb = self.encoder.patch_embedding.patch_embeddings
        n_patches = self.encoder.patch_embedding.n_patches
        patch_dim = self.encoder.patch_embedding.patch_dim

        # simple linear head
        self.mask_token = nn.Parameter(torch.randn(hidden_size))
        self.to_pixels = nn.Linear(hidden_size, patch_dim)

        self.init_weights(revise_keys=revise_keys)

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
            raise TypeError("pretrained must be a path(str) or None")

    def forward(self, img):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.patch_embedding.position_embeddings

        # for indexing purposes
        batch_range = torch.arange(batch, device=device)[:, None]

        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_patches)
        mask_tokens = mask_tokens + self.encoder.patch_embedding.position_embeddings

        # calculate of patches needed to be masked, and get positions (indices) to be masked
        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = (
            torch.rand(batch, num_patches, device=device)
            .topk(k=num_masked, dim=-1)
            .indices
        )
        masked_bool_mask = (
            torch.zeros((batch, num_patches), device=device)
            .scatter_(-1, masked_indices, 1)
            .bool()
        )

        # mask tokens
        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        encoded = tokens

        # get the masked tokens
        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values
        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # # get the masked patches for the final reconstruction loss
        # masked_patches = patches[batch_range, masked_indices]

        # # calculate reconstruction loss
        # recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked

        return pred_pixel_values, patches, batch_range, masked_indices


class SwinSimMIM(nn.Module):
    def __init__(
        self,
        pretrained: Union[None, str],
        patch_size: Sequence[int] = (4, 4, 4),
        in_chans: int = 1,
        embed_dim: int = 96,
        depths: Sequence[int] = [2, 2, 6, 2],
        num_heads: Sequence[int] = [3, 6, 12, 24],
        window_size: Sequence[int] = (2, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Union[None, bool] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer=nn.LayerNorm,
        patch_norm: bool = False,
        frozen_stages: int = -1,
        use_checkpoint: bool = False,
        masking_ratio: float = 0.5,
        revise_keys=[("model.", "")],
        **kwargs,
    ):
        super().__init__()

        self.pretrained = pretrained
        self.patch_size = patch_size

        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        self.encoder = SwinTransformer3D(
            pretrained=pretrained,
            pretrained2d=False,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint,
        )

        # patch embedding block
        num_features = self.encoder.num_features
        num_layers = self.encoder.num_layers
        final_resolution = self.encoder.final_resolution
        self.num_features = num_features
        self.num_layers = num_layers
        self.final_resolution = self.encoder.final_resolution

        # masked tokens
        self.mask_token = nn.Parameter(torch.randn(embed_dim))

        # simple linear head
        conv_trans = Conv[Conv.CONVTRANS, 3]
        self.conv3d_transpose = conv_trans(
            num_features,
            16,
            kernel_size=(
                self.patch_size[0],
                2 ** (num_layers - 1),
                2 ** (num_layers - 1),
            ),
            stride=(self.patch_size[0], 2 ** (num_layers - 1), 2 ** (num_layers - 1),),
        )
        self.conv3d_transpose_1 = conv_trans(
            in_channels=16,
            out_channels=in_chans,
            kernel_size=(1, self.patch_size[1], self.patch_size[2]),
            stride=(1, self.patch_size[1], self.patch_size[2]),
        )  # B C D H W

        self.init_weights(revise_keys=revise_keys)

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
            raise TypeError("pretrained must be a path(str) or None")

    def forward(self, img):
        # B, C, D, H, W = img.shape
        device = img.device

        # get patches
        patches = rearrange(
            img,
            "b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)",
            p1=self.patch_size[0],
            p2=self.patch_size[0],
            p3=self.patch_size[0],
        )
        tokens = self.encoder.patch_embed(img)
        batch, num_patches, *_ = tokens.shape
        assert num_patches == patches.shape[1]

        # for indexing purposes
        batch_range = torch.arange(batch, device=device)[:, None]

        # calculate of patches needed to be masked, and get positions (indices) to be masked
        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = (
            torch.rand(batch, num_patches, device=device)
            .topk(k=num_masked, dim=-1)
            .indices
        )
        masked_bool_mask = (
            torch.zeros((batch, num_patches), device=device)
            .scatter_(-1, masked_indices, 1)
            .bool()
        )

        # prepare mask tokens
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_patches)
        # mask_tokens = mask_tokens + self.encoder.patch_embedding.position_embeddings

        # mask tokens
        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        if self.encoder.ape:
            tokens = tokens + self.encoder.absolute_pos_embed
        tokens = self.encoder.pos_drop(tokens)

        # attend with vision transformer
        for layer in self.encoder.layers:
            tokens = layer(tokens)
        tokens = self.encoder.norm(tokens)

        # small linear projection for predicted pixel values
        tokens = tokens.transpose(1, 2).view(
            -1, self.num_features, *self.final_resolution
        )
        tokens = self.conv3d_transpose(tokens)
        pred_pixel_values = self.conv3d_transpose_1(tokens)

        pred_pixel_values = rearrange(
            pred_pixel_values,
            "b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)",
            p1=self.patch_size[0],
            p2=self.patch_size[0],
            p3=self.patch_size[0],
        )

        return pred_pixel_values, patches, batch_range, masked_indices


if __name__ == "__main__":
    model = SwinSimMIM(
        pretrained=None,
        pretrained2d=True,
        img_size=(96, 96, 96),
        patch_size=(4, 4, 4),
        in_chans=1,
        num_classes=0,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(7, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        frozen_stages=-1,
        use_checkpoint=False,
        masking_ratio=0.75,
        revise_keys=[],
    )

    x = torch.randn(1, 1, 96, 96, 96)
    y = model(x)

    print(y[0].shape)
