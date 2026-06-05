"""Implicit neural representation blocks used to build the FunCTA prompt."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def create_grid(height: int, width: int, device: torch.device | None = None) -> torch.Tensor:
    """Return a normalized (H, W, 2) coordinate grid in [0, 1]."""

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, steps=height, device=device),
        torch.linspace(0.0, 1.0, steps=width, device=device),
    )
    return torch.stack([grid_y, grid_x], dim=-1)


class SirenLayer(nn.Module):
    """A single sine-activated linear layer (SIREN)."""

    def __init__(self, in_features: int, out_features: int, w0: float = 30.0, is_first: bool = False, is_last: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        bound = 1 / self.in_features if self.is_first else np.sqrt(6 / self.in_features) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class PositionalEncoder(nn.Module):
    """Gaussian Fourier feature positional encoding."""

    def __init__(self, embedding_size: int = 128, coordinates_size: int = 2, scale: float = 8.0) -> None:
        super().__init__()
        basis = torch.randn((embedding_size, coordinates_size), dtype=torch.float32) * scale
        self.register_buffer("basis", basis)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x_embedding = (2.0 * np.pi * x) @ self.basis.t()
        return torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)


class SIRENPrompt(nn.Module):
    """SIREN that outputs a base image plus FiLM-style (gamma, beta) modulation."""

    def __init__(self, network_depth: int = 4, network_width: int = 256, network_input_size: int = 384, network_output_size: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [SirenLayer(network_input_size, network_width, is_first=True)]
        for _ in range(1, network_depth - 1):
            layers.append(SirenLayer(network_width, network_width))
        self.model = nn.Sequential(*layers)
        self.last_layer1 = SirenLayer(network_width, network_output_size * 64, is_last=True)
        self.last_layer2 = SirenLayer(network_width, network_output_size, is_last=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        gamma_beta = self.last_layer1(out)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        siren_output = self.last_layer2(out)
        return siren_output, gamma, beta
