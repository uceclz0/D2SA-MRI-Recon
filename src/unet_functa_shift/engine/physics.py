"""Forward MRI physics (coil sensitivity maps) and the data-consistency loss."""

from __future__ import annotations

import numpy as np
import torch

from ..external import fft2, transform


def relative_l1_loss(reference: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Scale-invariant L1 loss used for the k-space data-consistency term."""

    return torch.norm(prediction - reference, p=1) / (torch.norm(reference.detach(), p=1) + 1e-8)


def apply_sensitivity_maps_batch(sens_map: np.ndarray, output: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Expand a batched single-channel image to multi-coil images (image domain)."""

    sensitivity = transform.to_tensor(sens_map).to(device)
    batch_size = output.shape[0]
    images = torch.zeros(batch_size, sensitivity.shape[0], sensitivity.shape[1], sensitivity.shape[2], 2, device=device)

    for batch_index in range(batch_size):
        for coil_index, single_map in enumerate(sensitivity):
            safe_map = single_map.clone()
            zeros = torch.abs(safe_map) == 0.0
            if zeros.any():
                safe_map[zeros] = torch.abs(safe_map).max()
            images[batch_index, coil_index, :, :, 0] = output[batch_index, :, :, 0] * safe_map[:, :, 0] - output[batch_index, :, :, 1] * safe_map[:, :, 1]
            images[batch_index, coil_index, :, :, 1] = output[batch_index, :, :, 0] * safe_map[:, :, 1] + output[batch_index, :, :, 1] * safe_map[:, :, 0]
    return images


def apply_sensitivity_maps_single(sens_map: np.ndarray, output: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Expand one image to multi-coil k-space (returns FFT of coil images)."""

    sensitivity = transform.to_tensor(sens_map).to(device)
    images = torch.zeros_like(sensitivity)
    for coil_index, single_map in enumerate(sensitivity):
        safe_map = single_map.clone()
        zeros = torch.abs(safe_map) == 0.0
        if zeros.any():
            safe_map[zeros] = torch.abs(safe_map).max()
        images[coil_index, :, :, 0] = output[0, :, :, 0] * safe_map[:, :, 0] - output[0, :, :, 1] * safe_map[:, :, 1]
        images[coil_index, :, :, 1] = output[0, :, :, 0] * safe_map[:, :, 1] + output[0, :, :, 1] * safe_map[:, :, 0]
    return fft2(images)
