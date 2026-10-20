# Copyright 2026 <Your Name>
# SPDX-License-Identifier: Apache-2.0
"""Compact 2D Fourier Neural Operator (Li et al. 2021) in plain PyTorch.

The spectral multiply is written with real tensors (view_as_real) so it also runs on Apple MPS,
where complex einsum support is limited. `build_model("physicsnemo", ...)` swaps in NVIDIA
PhysicsNeMo's FNO with the same in/out contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (c_in * c_out)
        # two blocks: low positive and low negative frequencies along dim -2; last dim is (real, imag)
        self.w1 = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2, 2))

    @staticmethod
    def _cmul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """(B, Ci, m1, m2, 2) x (Ci, Co, m1, m2, 2) -> (B, Co, m1, m2, 2), complex product via reals."""
        ar, ai = a[..., 0], a[..., 1]
        wr, wi = w[..., 0], w[..., 1]
        real = torch.einsum("bixy,ioxy->boxy", ar, wr) - torch.einsum(
            "bixy,ioxy->boxy", ai, wi
        )
        imag = torch.einsum("bixy,ioxy->boxy", ar, wi) + torch.einsum(
            "bixy,ioxy->boxy", ai, wr
        )
        return torch.stack([real, imag], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        m1, m2 = self.modes1, self.modes2
        x_ft = torch.view_as_real(
            torch.fft.rfft2(x, norm="ortho")
        )  # (B, Ci, h, w//2+1, 2)
        out = torch.zeros(
            b, self.w1.shape[1], h, w // 2 + 1, 2, device=x.device, dtype=x.dtype
        )
        out[:, :, :m1, :m2] = self._cmul(x_ft[:, :, :m1, :m2], self.w1)
        out[:, :, -m1:, :m2] = self._cmul(x_ft[:, :, -m1:, :m2], self.w2)
        return torch.fft.irfft2(
            torch.view_as_complex(out.contiguous()), s=(h, w), norm="ortho"
        )


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        width: int = 32,
        modes1: int = 16,
        modes2: int = 24,
        n_layers: int = 4,
        padding: int = 8,
    ):
        super().__init__()
        self.padding = padding
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
        )
        self.pointwise = nn.ModuleList(
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        p = self.padding
        x = F.pad(x, (0, p, 0, p))  # domain is not periodic
        for k, (sc, pw) in enumerate(zip(self.spectral, self.pointwise)):
            x = sc(x) + pw(x)
            if k < len(self.spectral) - 1:
                x = F.gelu(x)
        x = x[..., : x.shape[-2] - p, : x.shape[-1] - p]
        return self.proj(x)


def build_model(arch: str, in_channels: int, out_channels: int = 3, **kw) -> nn.Module:
    if arch == "fno":
        return FNO2d(in_channels, out_channels, **kw)
    if arch == "physicsnemo":
        from physicsnemo.models.fno import FNO  # pip install nvidia-physicsnemo

        return FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            dimension=2,
            latent_channels=kw.get("width", 32),
            num_fno_layers=kw.get("n_layers", 4),
            num_fno_modes=kw.get("modes1", 16),
            padding=kw.get("padding", 8),
        )
    raise ValueError(f"unknown arch {arch!r}")
