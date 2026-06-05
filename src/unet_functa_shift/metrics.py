from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    value_range = float(image.max() - image.min())
    if value_range < 1e-8:
        return np.zeros_like(image)
    return (image - image.min()) / value_range


def compute_ssim_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    data_range = float(target.max() - target.min())
    if data_range < 1e-8:
        data_range = 1.0
    return float(structural_similarity(target, prediction, data_range=data_range))


def compute_psnr_score(target: np.ndarray, prediction: np.ndarray) -> float:
    target_norm = normalize_image(target)
    prediction_norm = normalize_image(prediction)
    return float(
        peak_signal_noise_ratio(
            target_norm,
            prediction_norm,
            data_range=max(float(target_norm.max() - target_norm.min()), 1e-8),
        )
    )


def compute_lpips_score(loss_fn: torch.nn.Module, target: np.ndarray, prediction: np.ndarray, device: torch.device) -> float:
    target_tensor = torch.from_numpy(normalize_image(target)).float()[None, None, :, :].to(device)
    prediction_tensor = torch.from_numpy(normalize_image(prediction)).float()[None, None, :, :].to(device)

    # LPIPS expects 3-channel input. Repeat grayscale MRI slices across channels.
    target_tensor = target_tensor.repeat(1, 3, 1, 1)
    prediction_tensor = prediction_tensor.repeat(1, 3, 1, 1)
    return float(loss_fn(prediction_tensor, target_tensor).item())


def save_triplet_figure(original: np.ndarray, reconstructed: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_norm = normalize_image(original)
    reconstructed_norm = normalize_image(reconstructed)
    residual = np.abs(original_norm - reconstructed_norm)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Original")

    axes[1].imshow(reconstructed, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Reconstruction")

    heatmap = axes[2].imshow(residual, cmap="jet")
    axes[2].axis("off")
    axes[2].set_title("Residual")
    fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
