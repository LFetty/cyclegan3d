import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock


def _is_stride_one(stride) -> bool:
    return bool(np.all(np.atleast_1d(stride) == 1))


class ImagePool:
    """Replay buffer of previously generated images to stabilise discriminator training."""

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, dim=0)


class UnetUpBlockInterp(nn.Module):
    """
    DynUNet-style up block with trilinear interpolation + conv instead of
    transpose convolution — eliminates checkerboard artifacts.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        lightweight: bool = False,
    ):
        super().__init__()
        sf = stride if isinstance(stride, (list, tuple)) else (stride,) * spatial_dims
        self.scale_factor: list[float] = [float(s) for s in sf]
        self.upsample_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        if lightweight:
            # single conv at full resolution to minimise activation storage
            self.conv_block = nn.Sequential(
                nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.01, inplace=True),
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=out_channels * 2,  # after cat with skip
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
            )

    def forward(self, inp, skip):
        out = F.interpolate(inp, scale_factor=self.scale_factor, mode="trilinear", align_corners=False)
        out = self.upsample_conv(out)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class ResUNetGenerator(nn.Module):
    """
    nnU-Net style encoder-decoder generator with:
    - UnetResBlock encoder (LeakyReLU + InstanceNorm + residual connections)
    - Trilinear upsample + conv decoder (no checkerboard artifacts)
    - Anisotropic kernel/stride support
    - Tanh output
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: Sequence[int] | int = 3,
        strides: Sequence[Sequence[int] | int] = (1, 2, 2, 2),
        filters: Sequence[int] = (24, 48, 96, 192),
        norm_name: tuple | str = "instance",
    ):
        super().__init__()
        assert len(strides) == len(filters), "strides and filters must have the same length"

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for f, s in zip(filters, strides):
            if _is_stride_one(s):
                # lightweight single-conv input projection at full resolution
                block = nn.Sequential(
                    nn.Conv3d(in_ch, f, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(f),
                    nn.LeakyReLU(0.01, inplace=True),
                )
            else:
                block = UnetResBlock(
                    spatial_dims=3,
                    in_channels=in_ch,
                    out_channels=f,
                    kernel_size=kernel_size,
                    stride=s,
                    norm_name=norm_name,
                )
            self.encoders.append(block)
            in_ch = f

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        for i in range(len(filters) - 1, 0, -1):
            is_last = (i == 1)  # last decoder block outputs at full resolution
            self.decoders.append(
                UnetUpBlockInterp(
                    spatial_dims=3,
                    in_channels=filters[i],
                    out_channels=filters[i - 1],
                    kernel_size=kernel_size,
                    stride=strides[i],
                    norm_name=norm_name,
                    lightweight=is_last,
                )
            )

        # ── Output ───────────────────────────────────────────────────────────
        self.out_conv = nn.Sequential(
            nn.Conv3d(filters[0], out_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        x = skips.pop()  # bottleneck
        for decoder in self.decoders:
            skip = skips.pop()
            x = decoder(x, skip)

        return self.out_conv(x)


def smoothing_loss_3d(flow):
    """Bending-energy smoothness loss on a 3-channel 3D displacement field."""
    dy = flow[:, :, 1:] - flow[:, :, :-1]
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)


class SpatialTransformer3D(nn.Module):
    """Differentiable 3D spatial transformer.

    Warps a source volume using a displacement field in voxel units.
    The identity grid is pre-computed at construction time for the given
    spatial size; the module is therefore tied to a fixed patch size.
    """

    def __init__(self, size):
        super().__init__()
        D, H, W = size
        vecs = [torch.arange(s, dtype=torch.float32) for s in (D, H, W)]
        grids = torch.meshgrid(vecs, indexing="ij")  # each (D, H, W)
        # Normalise to [-1, 1] and stack to (D, H, W, 3)
        grid = torch.stack(
            [
                grids[0] / (D - 1) * 2 - 1,
                grids[1] / (H - 1) * 2 - 1,
                grids[2] / (W - 1) * 2 - 1,
            ],
            dim=-1,
        )
        self.register_buffer("grid", grid.unsqueeze(0))  # (1, D, H, W, 3)
        self.D, self.H, self.W = D, H, W

    def forward(self, src, flow):
        """
        Args:
            src:  (B, C, D, H, W)
            flow: (B, 3, D, H, W) — displacement in voxel units
        Returns:
            warped (B, C, D, H, W)
        """
        # Normalise displacement to [-1, 1] coordinate space
        flow_norm = torch.stack(
            [
                flow[:, 0] / (self.D - 1) * 2,
                flow[:, 1] / (self.H - 1) * 2,
                flow[:, 2] / (self.W - 1) * 2,
            ],
            dim=-1,
        )  # (B, D, H, W, 3)

        new_locs = self.grid + flow_norm  # (B, D, H, W, 3)
        # grid_sample expects (x=W, y=H, z=D) ordering
        new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(
            src, new_locs, mode="bilinear", padding_mode="border", align_corners=True
        )


class RegistrationNet(nn.Module):
    """3D registration network (ResUNet) that predicts a deformation field.

    Input:  fake_B and real_B concatenated along the channel dimension.
    Output: (B, 3, D, H, W) displacement field in voxel units,
            zero-initialised so the network starts as the identity transform.
    """

    def __init__(
        self,
        strides: Sequence[Sequence[int] | int] = (1, 2, 2, 2),
        filters: Sequence[int] = (16, 32, 64, 128),
        norm_name: tuple | str = "instance",
    ):
        super().__init__()
        in_channels = 2  # fake_B concatenated with real_B

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for f, s in zip(filters, strides):
            if _is_stride_one(s):
                block = nn.Sequential(
                    nn.Conv3d(in_ch, f, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(f),
                    nn.LeakyReLU(0.01, inplace=True),
                )
            else:
                block = UnetResBlock(
                    spatial_dims=3,
                    in_channels=in_ch,
                    out_channels=f,
                    kernel_size=3,
                    stride=s,
                    norm_name=norm_name,
                )
            self.encoders.append(block)
            in_ch = f

        self.decoders = nn.ModuleList()
        for i in range(len(filters) - 1, 0, -1):
            self.decoders.append(
                UnetUpBlockInterp(
                    spatial_dims=3,
                    in_channels=filters[i],
                    out_channels=filters[i - 1],
                    kernel_size=3,
                    stride=strides[i],
                    norm_name=norm_name,
                    lightweight=(i == 1),
                )
            )

        # 3-channel output (dx, dy, dz), zero-initialised → identity transform
        self.out_conv = nn.Conv3d(filters[0], 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, fake_b, real_b):
        x = torch.cat([fake_b, real_b], dim=1)
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        x = skips.pop()
        for decoder in self.decoders:
            skip = skips.pop()
            x = decoder(x, skip)
        return self.out_conv(x)


class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator with spectral normalisation."""

    def __init__(self, in_channels=1, ndf=24):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.model = nn.Sequential(
            # Layer 1: no norm
            sn(nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            sn(nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            sn(nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4 — stride 1, matching original 70×70 PatchGAN design
            sn(nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: no activation (LSGAN)
            sn(nn.Conv3d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.model(x)
