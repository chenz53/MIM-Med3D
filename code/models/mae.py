import math
import logging
from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.nets import ViT

from einops import repeat
from mmcv.runner import load_checkpoint
from timm.models.layers import DropPath, trunc_normal_

__all__ = ["MAE"]


class MAE(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Modified to also give same dimension outputs as the input size of the image
    """

    def __init__(
        self,
        pretrained: str,
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
        decoder_dim: int = 768,
        decoder_depth: int = 1,
        decoder_heads: int = 8,
        masking_ratio: float = 0.75,
        revise_keys=[("model.", "")],
        **kwargs,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            decoder_dim: dimension of decoder hidden layer.
            decoder_depth: number of decoder transformer layer.
            decoder_heads: number of decoder heads.
            masking_ratio: ratio of masking patches.

        Examples::

            # for single channel input with image size of (96,96,96), patch position embedding and segmentation backbone
            # It will provide an output of same size as that of the input
            >>> net = MAE(in_channels=1, patch_size=(16,16,16), img_size=(96,96,96), pos_embed='perceptron')

        """

        super().__init__()

        self.pretrained = pretrained
        self.spatial_dims = spatial_dims

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
        patch_embedding = self.encoder.patch_embedding
        self.to_patch, self.patch_to_emb = patch_embedding.patch_embeddings
        n_patches = patch_embedding.n_patches
        patch_dim = patch_embedding.patch_dim

        # connect encoder and decoder if mismatch dimension
        self.enc_to_dec = (
            nn.Linear(hidden_size, decoder_dim)
            if hidden_size != decoder_dim
            else nn.Identity()
        )

        # build up decoder transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_dim, decoder_dim * 4, decoder_heads, dropout_rate
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.masking_ratio = masking_ratio
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder_pos_emb = nn.Embedding(n_patches, decoder_dim)

        # embeddings to pixels
        self.to_pixels = nn.Linear(decoder_dim, patch_dim)

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
            logging.info(f"load model from: {self.pretrained}")

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

    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        device = x.device

        # get patches
        patches = self.to_patch(x)
        batch, n_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.patch_embedding.position_embeddings

        # calculate of patches needed to be masked, and get random indices
        num_masked = int(self.masking_ratio * n_patches)
        rand_indices = torch.rand(batch, n_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        # masked_patches = patches[batch_range, masked_indices]

        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        encoded_tokens = tokens

        decoder_tokens = self.enc_to_dec(encoded_tokens)
        decoder_tokens += self.decoder_pos_emb(unmasked_indices)

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        for blk in self.decoder_blocks:
            decoder_tokens = blk(decoder_tokens)
        decoded_tokens = self.decoder_norm(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)

        return pred_pixel_values, patches, batch_range, masked_indices
