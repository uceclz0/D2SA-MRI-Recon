"""Differential / detail-enhancing convolution blocks (DEConv, DEBlock)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Conv2dCD(nn.Module):
    """Central-difference convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def get_weight(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange("c_in c_out k1 k2 -> c_in c_out (k1 k2)")(conv_weight)
        conv_weight_cd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 9)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange("c_in c_out (k1 k2) -> c_in c_out k1 k2", k1=3, k2=3)(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2dAD(nn.Module):
    """Angular-difference convolution."""

    def __init__(self, in_channels: int, out_channels: int, theta: float = 1.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.theta = theta

    def get_weight(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange("c_in c_out k1 k2 -> c_in c_out (k1 k2)")(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange("c_in c_out (k1 k2) -> c_in c_out k1 k2", k1=3, k2=3)(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2dHD(nn.Module):
    """Horizontal-difference convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def get_weight(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 9)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange("c_in c_out (k1 k2) -> c_in c_out k1 k2", k1=3, k2=3)(conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2dVD(nn.Module):
    """Vertical-difference convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def get_weight(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 9)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange("c_in c_out (k1 k2) -> c_in c_out k1 k2", k1=3, k2=3)(conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class DEConv(nn.Module):
    """Detail-enhancing convolution that fuses several difference kernels."""

    def __init__(self, dim: int, ratio: int = 4, mode: str = "all") -> None:
        super().__init__()
        self.mode = mode
        if mode in {"cd_ad", "all"}:
            self.conv_cd = Conv2dCD(dim, dim // ratio)
            self.conv_ad = Conv2dAD(dim, dim // ratio)
        if mode in {"hd_vd", "all"}:
            self.conv_hd = Conv2dHD(dim, dim // ratio)
            self.conv_vd = Conv2dVD(dim, dim // ratio)
        if mode in {"basic", "all"}:
            self.conv_basic = nn.Conv2d(dim, dim // ratio, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_list: list[torch.Tensor] = []
        bias_list: list[torch.Tensor] = []

        if self.mode in {"cd_ad", "all"}:
            w_cd, b_cd = self.conv_cd.get_weight()
            w_ad, b_ad = self.conv_ad.get_weight()
            weight_list.extend([w_cd, w_ad])
            if b_cd is not None:
                bias_list.append(b_cd)
            if b_ad is not None:
                bias_list.append(b_ad)

        if self.mode in {"hd_vd", "all"}:
            w_hd, b_hd = self.conv_hd.get_weight()
            w_vd, b_vd = self.conv_vd.get_weight()
            weight_list.extend([w_hd, w_vd])
            if b_hd is not None:
                bias_list.append(b_hd)
            if b_vd is not None:
                bias_list.append(b_vd)

        if self.mode in {"basic", "all"}:
            weight_list.append(self.conv_basic.weight)
            if self.conv_basic.bias is not None:
                bias_list.append(self.conv_basic.bias)

        weight = sum(weight_list)
        bias = sum(bias_list) if bias_list else None
        return F.conv2d(x, weight=weight, bias=bias, stride=1, padding=1)


class DEBlock(nn.Module):
    """Diffusion-guided detail-enhancing residual block."""

    def __init__(self, conv: type[DEConv], dim: int, dim_out: int, ratio: int = 4, mode: str = "all") -> None:
        super().__init__()
        self.conv1 = conv(dim, ratio=ratio, mode=mode)
        self.conv2 = nn.Conv2d(dim // ratio, dim_out, kernel_size=1, padding=0, bias=True)
        self.K = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("laplacian_kernel", laplacian_kernel)

    def diffusion_coefficient(self, signal: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + (signal / self.K) ** 2)

    def compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, _, _ = x.shape
        kernel = self.laplacian_kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(x, kernel.to(x.device), padding=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad = self.conv1(x)
        diffused = self.diffusion_coefficient(grad) * grad
        _ = self.compute_laplacian(diffused)
        return self.conv2(diffused)
