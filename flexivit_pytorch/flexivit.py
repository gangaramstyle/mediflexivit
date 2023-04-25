# from .patch_embed import FlexiPatchEmbed
# from .utils import resize_abs_pos_embed, to_2tuple

from functools import partial
from typing import Optional, Sequence, Tuple, Union

import torch
from timm.models.vision_transformer import Block, VisionTransformer
from torch import Tensor, nn


class MediFlexiVisionTransformer(VisionTransformer):
    def __init__(
        self,
        img_size: int = 240,
        base_patch_size: Union[int, Tuple[int, int]] = (32, 32, 32),
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = True,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
        weight_init: str = "",
        embed_layer: nn.Module = FlexiPatchEmbed,  # type:ignore
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
        block_fn: nn.Module = Block,  # type:ignore
        patch_size_seq: Sequence[int] = ((8, 8, 8), (10, 10, 10)),
        base_pos_embed_size: Union[int, Tuple[int, int, int]] = (256, 256, 128),
        patch_size_probs: Optional[Sequence[float]] = None,
        interpolation: str = "trilinear",
        antialias: bool = False,
    ) -> None:
        """Vision transformer w/ flexible patch sizes

        From: https://arxiv.org/abs/2212.08013

        Args:
            img_size: input image size
            patch_size: patch size
            in_chans: number of input channels
            num_classes: number of classes for classification head
            global_pool: type of global pooling for final sequence (default: 'token')
            embed_dim: embedding dimension
            depth: depth of transformer
            num_heads: number of attention heads
            mlp_ratio: ratio of mlp hidden dim to embedding dim
            qkv_bias: enable bias for qkv if True
            init_values: layer-scale init values
            class_token: use class token
            fc_norm: pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate: dropout rate
            attn_drop_rate: attention dropout rate
            drop_path_rate: stochastic depth rate
            weight_init: weight init scheme
            embed_layer: patch embedding layer
            norm_layer: normalization layer
            act_layer: MLP activation layer
            patch_size_seq: List of patch sizes to randomly sample from
            base_pos_embed_size: Base position embedding size. i.e. the size of the parameter buffer
            patch_size_probs: Optional list of probabilities of sample corresponding
                patch_size_seq element. If None, then uniform distribution is used
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """

        assert embed_layer == FlexiPatchEmbed, "embed_layer should be a FlexiPatchEmbed"

        # Pre-initialize the flexi specific patch embed arguments
        embed_layer_fn = partial(
            FlexiPatchEmbed,
            patch_size_seq=patch_size_seq,
            patch_size_probs=patch_size_probs,
            grid_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
        )

        # Position embedding resizing function
        self.resize_pos_embed = partial(
            resize_abs_pos_embed,
            old_size=base_pos_embed_size,
            interpolation=interpolation,
            antialias=antialias,
            num_prefix_tokens=1 if class_token and not no_embed_class else 0,
        )

        self.img_size = to_3tuple(img_size)

        self.base_pos_embed_size = base_pos_embed_size
        X, Y, Z = base_pos_embed_size

        arr_4d = np.zeros((X, Y, Z, 3), dtype=int)
        arr_4d[..., 0] = np.arange(-X//2, X//2)[:, None, None]
        arr_4d[..., 1] = np.arange(-Y//2, Y//2)[None, :, None]
        arr_4d[..., 2] = np.arange(-Z//2, Z//2)[None, None, :]

        p = np.zeros((X, Y, Z, embed_dim), dtype=np.float32)
        p = np.apply_along_axis(lambda pos: self.get_sincos_pos_embed_for_ijk(embed_dim=embed_dim, pos=pos), -1, arr_4d)

        self.pos_embeddings_arr = p

        super().__init__(
            img_size,
            base_patch_size,  # type:ignore
            in_chans,
            num_classes,
            global_pool,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            init_values,
            class_token,
            no_embed_class,
            pre_norm,
            fc_norm,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            weight_init,
            embed_layer_fn,  # type:ignore
            norm_layer,
            act_layer,
            block_fn,  # type:ignore
        )

    def get_sincos_pos_embed_for_ijk(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: i, j
        """
        assert embed_dim % 6 == 0
        omega = np.arange(embed_dim // 6, dtype=float)
        omega /= embed_dim / 6.
        omega = 1. / 10000**omega  # (D/2,)
        
        emb_sin_i = np.sin(pos[0] * omega).reshape(1, -1) # (1, D/6)
        emb_cos_i = np.cos(pos[0] * omega).reshape(1, -1) # (1, D/6)
        emb_sin_j = np.sin(pos[1] * omega).reshape(1, -1) # (1, D/6)
        emb_cos_j = np.cos(pos[1] * omega).reshape(1, -1) # (1, D/6)
        emb_sin_k = np.sin(pos[2] * omega).reshape(1, -1) # (1, D/6)
        emb_cos_k = np.cos(pos[2] * omega).reshape(1, -1) # (1, D/6)

        emb = np.empty((1, embed_dim), dtype=float)
        emb[:, 0::6] = emb_sin_i
        emb[:, 1::6] = emb_cos_i
        emb[:, 2::6] = emb_sin_j
        emb[:, 3::6] = emb_cos_j
        emb[:, 4::6] = emb_sin_k
        emb[:, 5::6] = emb_cos_k
        return emb

    def _pos_embed(self, x: Tensor, image_size: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> Tensor:
        # Resize position embedding based on current patch size
        X, Y, Z = self.base_pos_embed_size

        p = self.pos_embeddings_arr[
            (X//2)-(image_size[0]//2):(X//2)+(image_size[0]//2):patch_size[0],
            (Y//2)-(image_size[1]//2):(Y//2)+(image_size[1]//2):patch_size[1],
            (Z//2)-(image_size[2]//2):(Z//2)+(image_size[2]//2):patch_size[2]]

        p = p.squeeze(3)
        p = p.transpose(3, 0, 1, 2)

        p = p[np.newaxis, ...]
        f_x, f_y, f_z = np.floor_divide(image_size, patch_size)
        p = p[:, :, :f_x, :f_y, :f_z]
        p = p.reshape(1, x.shape[1], -1)

        # convert p to tensor
        p = torch.from_numpy(p).to(x.device).float()
        pos_embed = p

        if self.no_embed_class:
            # Position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # Position embedding has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward_features(
        self, x: Tensor, patch_size: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Tensor:
        image_size = (x.shape[-3], x.shape[-2], x.shape[-1])
        x, ps = self.patch_embed(x, patch_size, return_patch_size=True)
        x = self._pos_embed(x, image_size, ps)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(
        self, x: Tensor, patch_size: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Tensor:
        x = self.forward_features(x, patch_size)
        x = self.forward_head(x)
        return x


def flexivit_tiny(**kwargs) -> MediFlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=192, depth=12, num_heads=3, **kwargs)


def flexivit_small(**kwargs) -> MediFlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=384, depth=12, num_heads=6, **kwargs)


def flexivit_base(**kwargs) -> MediFlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=768, depth=12, num_heads=12, **kwargs)


def flexivit_large(**kwargs) -> MediFlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=1024, depth=24, num_heads=16, **kwargs)


def flexivit_huge(**kwargs) -> MediFlexiVisionTransformer:
    return FlexiVisionTransformer(embed_dim=1280, depth=32, num_heads=16, **kwargs)

# test code
x = torch.randn(1, 3, 224, 224, 112)
model = MediFlexiVisionTransformer(img_size=(224, 224, 112))
out = model.forward_features(x)
print(out.shape)