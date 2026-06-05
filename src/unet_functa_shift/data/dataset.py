"""Per-patient k-space dataset, intensity scaling, and patient/sens-map discovery."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from ..config import ExperimentConfig
from ..external import ifft2, rss_torch, transform
from ..models import create_grid
from .masking import SSDUMasks, build_base_mask, build_n2n_train_mask


def compute_scale_factor(masked_slice: np.ndarray, model: nn.Module, device: torch.device) -> float:
    """Estimate the intensity scale that matches model input/output norms."""

    with torch.no_grad():
        slice_tt = transform.to_tensor(masked_slice).to(device)
        model_input = torch.moveaxis(rss_torch(ifft2(slice_tt))[None, :], -1, 1)
        scale = torch.tensor(1.0, device=device)
        for _ in range(5):
            output = torch.moveaxis(model(model_input), 1, -1)
            scale = scale * (torch.norm(output.detach()) / (torch.norm(model_input.detach()) + 1e-8))
            model_input = torch.moveaxis(rss_torch(ifft2(slice_tt * scale))[None, :], -1, 1)
        return float(scale.item())


def compute_slice_scales(kspace: np.ndarray, mask2d: np.ndarray, model: nn.Module, device: torch.device) -> list[float]:
    scales: list[float] = []
    for slice_index in tqdm(range(kspace.shape[0]), desc="  computing scale factors", leave=False):
        slice_kspace = kspace[slice_index]
        masked_slice = slice_kspace * np.tile(mask2d[None, ...], (slice_kspace.shape[0], 1, 1))
        scales.append(compute_scale_factor(masked_slice, model, device))
    return scales


class PatientSliceDataset(Dataset):
    """Loads one patient `.h5` volume and produces per-slice training/eval tensors.

    The ``method`` controls which mask feeds the network input and which mask
    defines the self-supervised loss:

    - ``datafidelity``: input and loss both use the acquired mask.
    - ``n2n``: input uses a further sub-sampled mask, loss uses the acquired mask.
    - ``ssdu``: the acquired mask is split into disjoint train / loss masks.
    """

    def __init__(self, filename: Path, method: str, base_model: nn.Module, config: ExperimentConfig, device: torch.device) -> None:
        with h5py.File(filename, "r") as handle:
            kspace = handle["kspace"][:]
            orig = handle["reconstruction_rss"][:]

        num_slices = max(1, kspace.shape[0] - config.slice_trim)
        self.kspace = kspace[:num_slices]
        self.orig = orig[:num_slices]
        self.method = method
        self.device = device

        _, height, width = self.kspace[0].shape
        self.base_mask = build_base_mask(height, width, config.mask_center_fraction, config.mask_acceleration, config.seed)
        self.coord = create_grid(height, width).cpu()
        self.scales = compute_slice_scales(self.kspace, self.base_mask, base_model, device)

        if method == "n2n":
            self.train_mask = build_n2n_train_mask(self.base_mask, config.n2n_sub_acceleration, config.seed + 1)
            self.loss_mask = self.base_mask
        elif method == "ssdu":
            split = SSDUMasks(rho=config.ssdu_rho).uniform_selection(self.base_mask, seed=config.seed + 1)
            self.train_mask = split.train_mask
            self.loss_mask = split.loss_mask
        else:
            self.train_mask = self.base_mask
            self.loss_mask = self.base_mask

    def __len__(self) -> int:
        return self.kspace.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        slice_kspace = self.kspace[index]
        orig = self.orig[index].astype(np.float32)
        scale = self.scales[index]

        full_masked_kspace = slice_kspace * np.tile(self.base_mask[None, ...], (slice_kspace.shape[0], 1, 1))
        train_masked_kspace = slice_kspace * np.tile(self.train_mask[None, ...], (slice_kspace.shape[0], 1, 1))
        loss_masked_kspace = slice_kspace * np.tile(self.loss_mask[None, ...], (slice_kspace.shape[0], 1, 1))

        kspace_scaled = transform.to_tensor(full_masked_kspace) * scale
        train_kspace_scaled = transform.to_tensor(train_masked_kspace) * scale
        loss_kspace_scaled = transform.to_tensor(loss_masked_kspace) * scale

        eval_input = torch.moveaxis(rss_torch(ifft2(kspace_scaled)), -1, 0)
        train_input = torch.moveaxis(rss_torch(ifft2(train_kspace_scaled)), -1, 0)

        return {
            "idx": torch.tensor(index, dtype=torch.long),
            "coord": self.coord.float(),
            "orig": torch.from_numpy(orig),
            "k_space_scaled": kspace_scaled.float(),
            "train_input": train_input.float(),
            "eval_input": eval_input.float(),
            "loss_kspace": loss_kspace_scaled.float(),
            "loss_mask": torch.from_numpy(self.loss_mask).float(),
            "base_mask": torch.from_numpy(self.base_mask).float(),
        }

    def get_slice_for_refinement(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.kspace[index], self.orig[index]


def get_patient_code_map(sensitivity_maps_folder: Path) -> dict[str, Path]:
    """Map each patient code to its sensitivity-map `.npy` file."""

    mapping: dict[str, Path] = {}
    for path in sorted(sensitivity_maps_folder.glob("*.npy")):
        patient_code = path.stem.replace("sens_map_", "").replace("_middle", "").replace("_all_slices", "")
        mapping[patient_code] = path
    return mapping


def find_matching_patient_file(patient_code: str, original_files_folder: Path) -> Path | None:
    for path in sorted(original_files_folder.glob("*.h5")):
        if patient_code in path.name:
            return path
    return None
