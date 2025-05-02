from typing import Optional, Callable, Tuple, Union, List

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
        pool_kernel_size: Union[int, Tuple[int, int, int]],
        up: str = "interpolate",
    ):
        super().__init__()
        assert up in ["interpolate", "conv"], "up must be 'interpolate' or 'conv'"
        self.up = up
        if self.up == "conv":
            if isinstance(pool_kernel_size, int):
                double_pks = pool_kernel_size * 2
            else:
                double_pks = [pks * 2 for pks in pool_kernel_size]
            self.transpose_conv = nn.ConvTranspose3d(
                in_channels, out_channels, double_pks, pool_kernel_size
            )
            in_channels = out_channels * 2
        elif self.up == "interpolate":
            in_channels = in_channels + out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, conv_kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, conv_kernel_size, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.up == "conv":
            x = self.transpose_conv(x)
        # fixes mismatching in temporal dimension if up == conv
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

    # TODO: Compare batchnorm to group norm
    # n_classes should be 1 if doing regression + MSE
    # n_classes should be 256 if each pixel is a classification problem (CE)
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        features: List[int],
        conv_kernel_size: int = 3,
        pool_kernel_size: int = 2,
        final_activation: Optional[Callable] = None,
        up: str = "interpolate",
    ):
        super().__init__()
        self.features = [in_channels, *features]
        self.up = up

        _encoder_blocks = []
        for i in range(len(self.features) - 1):
            downsample = i != 0
            in_c = self.features[i]
            out_c = self.features[i + 1]
            _encoder_blocks.append(
                EncoderBlock(
                    in_c, out_c, conv_kernel_size, pool_kernel_size, downsample
                )
            )
        self.encoder_blocks = nn.ModuleList(_encoder_blocks)

        _decoder_blocks = []
        for i in range(len(self.features) - 1, 1, -1):
            in_c = self.features[i]
            out_c = self.features[i - 1]
            _decoder_blocks.append(
                DecoderBlock(in_c, out_c, conv_kernel_size, pool_kernel_size, up)
            )
        self.decoder_blocks = nn.ModuleList(_decoder_blocks)

        self.final_conv = nn.Conv3d(self.features[1], num_classes, 1)
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
    unet = UNet3D(
        1,
        1,
        features=[32, 64, 128, 256],
        pool_kernel_size=(2, 2, 2),
        up="conv",
        final_activation=nn.Sigmoid(),
    ).to(device)
    y = unet(x)
    print(y.shape)
