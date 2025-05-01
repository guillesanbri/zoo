from typing import Optional, Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: Union[int, Tuple[int, int, int]],
        pool_kernel_size: Union[int, Tuple[int, int, int]],
        downsample: bool = True,
    ):
        super().__init__()
        self.downsample = downsample
        self.pool = (
            nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
            if downsample
            else nn.Identity()
        )

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, conv_kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, conv_kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(x)
        x = self.double_conv(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: Union[int, Tuple[int, int, int]],
    ):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, conv_kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, conv_kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        size = skip.size()[2:]
        x = F.interpolate(x, size=size, mode="nearest")
        x = torch.cat((skip, x), dim=1)
        x = self.double_conv(x)
        return x


class UNet3D(nn.Module):
    """
    UNet with 3D convolutions adapted to seq-to-seq image generation.
    Input/output shape: [Batch, Channels, Time/depth, Height, Width]
    """

    # TODO: Compare both bilinear interpolation and transpose convolutions to upscale
    # TODO: Compare batchnorm to group norm
    # TODO: Compare with and without downsampling the time dimension
    # n_classes should be 1 if doing regression + MSE
    # n_classes should be 256 if each pixel is a classification problem (CE)
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        depth: int = 4,
        first_layer_features: int = 32,
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        final_activation: Optional[Callable] = None,
    ):
        super().__init__()

        _encoder_blocks = []
        for i in range(depth):
            if i == 0:
                in_c = in_channels
                downsample = False
            else:
                in_c = first_layer_features * (2 ** (i - 1))
                downsample = True
            out_c = first_layer_features * (2**i)
            _encoder_blocks.append(
                EncoderBlock(
                    in_c, out_c, conv_kernel_size, pool_kernel_size, downsample
                )
            )
        self.encoder_blocks = nn.ModuleList(_encoder_blocks)

        _decoder_blocks = []
        for i in range(depth - 1, 0, -1):
            in_c = int(
                (first_layer_features * (2**i))
                + (first_layer_features * (2 ** (i - 1)))
            )
            out_c = int(first_layer_features * (2 ** (i - 1)))
            _decoder_blocks.append(DecoderBlock(in_c, out_c, conv_kernel_size))
        self.decoder_blocks = nn.ModuleList(_decoder_blocks)

        self.final_conv = nn.Conv3d(first_layer_features, num_classes, 1)
        self.final_activation = final_activation

    def forward(self, x):
        encoder_outputs = []
        for eb in self.encoder_blocks:
            x = eb(x)
            encoder_outputs.insert(0, x)
        # remove the output from the bottleneck level
        encoder_outputs = encoder_outputs[1:]
        for db, eo in zip(self.decoder_blocks, encoder_outputs):
            x = db(x, eo)
        # project to output channels
        x = self.final_conv(x)
        if self.final_activation:
            x = self.final_activation(x)
        return x


if __name__ == "__main__":
    device = "cuda"
    x = torch.rand(1, 1, 10, 128, 128).to(device)
    unet = UNet3D(1, 1, pool_kernel_size=(2, 2, 2)).to(device)
    y = unet(x)
    print(y.shape)
