from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MethodName = Literal["datafidelity", "n2n", "ssdu"]
RunMode = Literal["batch", "single"]


@dataclass
class ExperimentConfig:
    image_dir_base: Path
    sensitivity_maps_folder: Path
    original_files_folder: Path
    modality: str
    method: MethodName
    mode: RunMode
    checkpoint_root: Path
    device: str = "cuda"
    seed: int = 2024
    num_epochs: int = 25
    batch_size: int = 2
    num_workers: int = 4
    latent_dim: int = 128
    latent_mean: float = 0.0
    latent_std: float = 0.01
    lr_latent: float = 1e-3
    lr_model: float = 1e-4
    latent_weight: float = 1e-4
    eval_interval: int = 7
    single_slice_iters: int = 1000
    single_slice_window: int = 30
    mask_center_fraction: float = 0.08
    mask_acceleration: int = 4
    n2n_sub_acceleration: int = 2
    ssdu_rho: float = 0.4
    slice_trim: int = 5
    save_figures: bool = True
    reuse_checkpoints: bool = True

    @property
    def output_root(self) -> Path:
        return self.image_dir_base / f"{self.method}_{self.mode}_{self.modality}"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_root / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.output_root / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    def ensure_dirs(self) -> None:
        for path in (self.output_root, self.checkpoint_dir, self.metrics_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)
