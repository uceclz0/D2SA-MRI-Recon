"""Reconstruction + metric computation for plain U-Net and prompt U-Net."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..external import center_crop, complex_abs, ifft2, rss_torch
from ..metrics import compute_lpips_score, compute_psnr_score, compute_ssim_score


def _score(reconstruction: np.ndarray, orig: torch.Tensor, loss_fn: nn.Module, device: torch.device) -> tuple[np.ndarray, float, float, float]:
    target = orig.detach().cpu().numpy()
    ssim_score = compute_ssim_score(target, reconstruction)
    psnr_score = compute_psnr_score(target, reconstruction)
    lpips_score = compute_lpips_score(loss_fn, target, reconstruction, device)
    return reconstruction, ssim_score, psnr_score, lpips_score


def evaluate_backbone(model: nn.Module, slice_tt: torch.Tensor, orig: torch.Tensor, loss_fn: nn.Module, device: torch.device) -> tuple[np.ndarray, float, float, float]:
    """Evaluate a plain U-Net (no test-time training)."""

    model_input = torch.moveaxis(rss_torch(ifft2(slice_tt))[None, :], -1, 1)
    with torch.no_grad():
        output = torch.moveaxis(model(model_input.to(device)), 1, -1)
    reconstruction = center_crop(complex_abs(output[0]).detach().cpu().numpy(), tuple(orig.shape))
    return _score(reconstruction, orig, loss_fn, device)


def evaluate_prompt(model: nn.Module, slice_tt: torch.Tensor, orig: torch.Tensor, latent_vec: torch.Tensor, coord: torch.Tensor, loss_fn: nn.Module, device: torch.device) -> tuple[np.ndarray, float, float, float]:
    """Evaluate a prompt-conditioned U-Net for one slice."""

    model_input = torch.moveaxis(rss_torch(ifft2(slice_tt))[None, :], -1, 1)
    with torch.no_grad():
        output, _, _, _ = model(model_input.to(device), coord.to(device), latent_vec.to(device))
        output = torch.moveaxis(output, 1, -1)
    reconstruction = center_crop(complex_abs(output[0]).detach().cpu().numpy(), tuple(orig.shape))
    return _score(reconstruction, orig, loss_fn, device)
