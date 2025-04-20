# TODO: Remove pos_embed and cls_token from weight decay
# TODO: Add DropPath
import torch
import torch.nn as nn
from collections.abc import Iterable
from typing import Union, Optional, Type, Tuple


def _int_tuple(x: Union[Iterable[int, int], int]) -> Tuple[int, int]:
    if isinstance(x, Iterable):
        x = tuple(x)
        if all(isinstance(i, int) for i in x):
            return x
        raise ValueError(f"Iterable contains non-int values: {x}")
    elif isinstance(x, int):
        return (x, x)
    else:
        raise ValueError(f"Invalid input type: {type(x)}")


class PatchEmbed(nn.Module):
    """
    Extracts a set of patches from an image and embeds them.
    """

    def __init__(
        self,
        img_size: Union[Iterable[int, int], int] = 224,
        patch_size: Union[Iterable[int, int], int] = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()

        self.img_size = _int_tuple(img_size)
        self.patch_size = _int_tuple(patch_size)
        self.in_channels = in_channels

        w, h = self.img_size
        pw, ph = self.patch_size
        assert (
            w % pw == 0 and h % ph == 0
        ), "Image size must be divisible by patch size."
        self.num_patches = (w // pw) * (h // ph)

        self.embed_dim = embed_dim
        self.projection = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """
    Simple MultiLayerPerceptron with 2 linear layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.ffwd = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor):
        return self.ffwd(x)


class Attention(nn.Module):
    """
    Multi-head self-attention.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, head_dim: int, drop: float = 0.0
    ):
        super().__init__()
        self.inner_dim = num_heads * head_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(embed_dim, self.inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop)
        self.proj = nn.Linear(self.inner_dim, self.embed_dim)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        x = self.qkv(x)
        qkv = x.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * (self.head_dim**-0.5)
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and MLP. Uses pre-norm.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 8,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_ratio: int = 4,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, num_heads, embed_dim // num_heads, attn_drop)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(embed_dim, embed_dim * mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Minimal implementation of a Vision Transformer.
    """

    def __init__(
        self,
        img_size: Union[Iterable[int, int], int],
        patch_size: Union[Iterable[int, int], int],
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_ratio: int = 4,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, embed_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, drop, attn_drop, mlp_ratio, norm_layer
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = norm_layer(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        return self.mlp_head(x[:, 0])


if __name__ == "__main__":
    vit = ViT(
        img_size=224,
        patch_size=8,
        in_channels=3,
        num_classes=200,
        embed_dim=384,
        depth=6,
        num_heads=8,
        drop=0.2,
        attn_drop=0.2,
    )
    x = torch.ones(size=(2, 3, 224, 224))
    y = vit(x)
    print(y.shape)