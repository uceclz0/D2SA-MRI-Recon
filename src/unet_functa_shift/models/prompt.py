"""The prompt-conditioned U-Net and helpers to make it test-time adaptable."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..external import ConvBlock
from .diff_conv import DEBlock, DEConv
from .inr import PositionalEncoder, SIRENPrompt


class UNetPrompt(nn.Module):
    """Wraps a pretrained U-Net and modulates its last decoder block with a SIREN prompt."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.siren = SIRENPrompt(network_depth=4, network_input_size=384)
        self.position_encoder = PositionalEncoder(embedding_size=128)
        self.unet_ttt = model

    def freeze_unet_weights(self) -> None:
        for param in self.unet_ttt.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, coord: torch.Tensor, latent_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_coord = self.position_encoder.embedding(coord)
        latent_vec_expanded = latent_vec.unsqueeze(1).unsqueeze(2).expand(-1, x_coord.shape[1], x_coord.shape[2], -1)
        latent_coord = torch.cat([x_coord, latent_vec_expanded], dim=-1)

        siren_output, prompt_gamma, prompt_beta = self.siren(latent_coord)
        prompt_gamma = torch.moveaxis(prompt_gamma, -1, 1)
        prompt_beta = torch.moveaxis(prompt_beta, -1, 1)
        siren_output = torch.moveaxis(siren_output, -1, 1)

        stack = []
        output = x
        for layer in self.unet_ttt.unet.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.unet_ttt.unet.conv(output)

        for index, (transpose_conv, conv) in enumerate(zip(self.unet_ttt.unet.up_transpose_conv, self.unet_ttt.unet.up_conv)):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            if index == len(self.unet_ttt.unet.up_conv) - 1:
                feature_map = conv[:-1](output)
                feature_map = prompt_gamma * feature_map + prompt_beta
                output = conv[-1](feature_map)
            else:
                output = conv(output)

        return output, siren_output, prompt_gamma, prompt_beta


def configure_model_with_deblock(model: nn.Module) -> nn.Module:
    """Insert a DEBlock branch in parallel to every ConvBlock in the model."""

    for name, module in list(model.named_modules()):
        if not isinstance(module, ConvBlock):
            continue

        conv_layer = module.layers[0]
        out_chans = conv_layer.out_channels
        deblock = DEBlock(DEConv, out_chans, out_chans).to(conv_layer.weight.device)

        class CombinedBlock(nn.Module):
            def __init__(self, conv_block: nn.Module, de_block: nn.Module) -> None:
                super().__init__()
                self.conv_block = conv_block
                self.de_block = de_block

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                conv_out = self.conv_block(x)
                return conv_out + self.de_block(conv_out)

        combined_block = CombinedBlock(module, deblock)
        parent_name, child_name = name.rsplit(".", 1)
        parent_module = dict(model.named_modules())[parent_name]
        setattr(parent_module, child_name, combined_block)

    return model


def freeze_conv_blocks(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, ConvBlock):
            for param in module.parameters():
                param.requires_grad = False
    return model


def collect_trainable_parameters(model: nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    params: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
            names.append(name)
    return params, names


def clone_model(model: nn.Module) -> nn.Module:
    return deepcopy(model)
