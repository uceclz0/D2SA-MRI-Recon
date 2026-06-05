"""Top-level orchestration: build models, loop over patients, write metrics."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ExperimentConfig
from ..data import (
    PatientSliceDataset,
    find_matching_patient_file,
    get_patient_code_map,
)
from ..external import UnetMRIModelDemo
from ..metrics import save_triplet_figure, seed_everything
from ..models import UNetPrompt, clone_model
from .adapt import adapt_batch, refine_single_slice
from .evaluate import evaluate_backbone, evaluate_prompt

# (display name, run test-time training?, checkpoint file or None for random init)
MODEL_ORDER = [
    ("knee_no_ttt", False, "knee_with_self_supervision.pt"),
    ("knee_ttt", True, "knee_with_self_supervision.pt"),
    ("brain_no_ttt", False, "brain_with_self_supervision.pt"),
    ("brain_ttt", True, "brain_with_self_supervision.pt"),
    ("train_from_scratch", True, None),
]


def build_backbone(device: torch.device) -> nn.Module:
    hparams = SimpleNamespace(num_pools=4, drop_prob=0.0, num_chans=64, in_chans=2)
    return UnetMRIModelDemo(hparams).to(device)


def load_model_variants(config: ExperimentConfig, device: torch.device) -> list[tuple[str, nn.Module, bool]]:
    base_model = build_backbone(device)
    variants: list[tuple[str, nn.Module, bool]] = []
    for model_name, use_ttt, checkpoint_name in MODEL_ORDER:
        model = clone_model(base_model)
        if checkpoint_name is not None:
            checkpoint = torch.load(config.checkpoint_root / checkpoint_name, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        variants.append((model_name, model, use_ttt))
    return variants


def _make_loaders(dataset: PatientSliceDataset, config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    common = dict(
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, **common)
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, **common)
    return train_loader, eval_loader


def _prepare_prompt_model(
    base_model: nn.Module,
    dataset: PatientSliceDataset,
    train_loader: DataLoader,
    sens_map: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    checkpoint_path: Path,
    model_name: str,
) -> tuple[nn.Module, torch.Tensor]:
    if config.reuse_checkpoints and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        prompt_model = UNetPrompt(clone_model(base_model)).to(device)
        prompt_model.load_state_dict(checkpoint["model_state_dict"])
        return prompt_model, checkpoint["latent_vectors"].to(device)

    if config.num_epochs <= 0:
        raise FileNotFoundError(
            f"Checkpoint not found for {model_name}: {checkpoint_path}. "
            "Set `--num-epochs` > 0 to create it or point to the correct output directory."
        )

    prompt_model, latent_vectors = adapt_batch(base_model, dataset, train_loader, sens_map, config, device)
    torch.save(
        {
            "model_state_dict": prompt_model.state_dict(),
            "latent_vectors": latent_vectors.cpu(),
            "config": asdict(config),
        },
        checkpoint_path,
    )
    return prompt_model, latent_vectors


def _evaluate_slice(
    batch: dict,
    base_model: nn.Module,
    prompt_model: nn.Module | None,
    latent_vectors: torch.Tensor | None,
    dataset: PatientSliceDataset,
    sens_map: np.ndarray,
    use_ttt: bool,
    config: ExperimentConfig,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[np.ndarray, float, float, float]:
    slice_index = int(batch["idx"].item())
    slice_tt = batch["k_space_scaled"][0]
    orig = batch["orig"][0]

    if not use_ttt:
        return evaluate_backbone(base_model, slice_tt, orig, loss_fn, device)

    assert prompt_model is not None and latent_vectors is not None
    if config.mode == "single":
        slice_kspace_np, orig_np = dataset.get_slice_for_refinement(slice_index)
        return refine_single_slice(
            prompt_model=prompt_model,
            base_model=base_model,
            latent_vector=latent_vectors[slice_index].detach().cpu(),
            coord=batch["coord"][0].detach().cpu(),
            slice_kspace=slice_kspace_np,
            orig=orig_np,
            sens_map=sens_map,
            base_mask=dataset.base_mask,
            config=config,
            device=device,
            loss_fn=loss_fn,
        )

    return evaluate_prompt(
        prompt_model,
        slice_tt,
        orig,
        latent_vectors[slice_index][None, :].detach().cpu(),
        batch["coord"][0][None, :, :, :].detach().cpu(),
        loss_fn,
        device,
    )


def run_patient_model(
    model_name: str,
    base_model: nn.Module,
    use_ttt: bool,
    patient_file: Path,
    sens_map_path: Path,
    config: ExperimentConfig,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[dict[str, float], list[dict[str, float | str | int]]]:
    print(f"  [{model_name}] building dataset (ttt={use_ttt}) ...", flush=True)
    sens_map = np.load(sens_map_path, allow_pickle=True)
    dataset = PatientSliceDataset(patient_file, config.method, base_model, config, device)
    train_loader, eval_loader = _make_loaders(dataset, config)

    prompt_model: nn.Module | None = None
    latent_vectors: torch.Tensor | None = None
    if use_ttt:
        checkpoint_path = config.checkpoint_dir / f"{patient_file.stem}_{model_name}_{config.method}_{config.mode}.pt"
        print(f"  [{model_name}] test-time adaptation ...", flush=True)
        prompt_model, latent_vectors = _prepare_prompt_model(
            base_model, dataset, train_loader, sens_map, config, device, checkpoint_path, model_name
        )

    summary_rows: list[dict[str, float | str | int]] = []
    ssim_scores: list[float] = []
    psnr_scores: list[float] = []
    lpips_scores: list[float] = []

    for batch in tqdm(eval_loader, desc=f"  [{model_name}] evaluating", leave=False):
        slice_index = int(batch["idx"].item())
        reconstruction, ssim_score, psnr_score, lpips_score = _evaluate_slice(
            batch, base_model, prompt_model, latent_vectors, dataset, sens_map, use_ttt, config, device, loss_fn
        )

        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
        lpips_scores.append(lpips_score)

        if config.save_figures:
            figure_path = config.figures_dir / patient_file.stem / f"slice_{slice_index}" / f"{model_name}.png"
            save_triplet_figure(batch["orig"][0].detach().cpu().numpy(), reconstruction, figure_path)

        summary_rows.append(
            {
                "patient": patient_file.name,
                "model": model_name,
                "method": config.method,
                "mode": config.mode,
                "slice": slice_index,
                "ssim": ssim_score,
                "psnr": psnr_score,
                "lpips": lpips_score,
                "ttt": use_ttt,
            }
        )

    patient_metrics = {
        f"{model_name}_mean_ssim": float(np.mean(ssim_scores)),
        f"{model_name}_std_ssim": float(np.std(ssim_scores)),
        f"{model_name}_mean_psnr": float(np.mean(psnr_scores)),
        f"{model_name}_std_psnr": float(np.std(psnr_scores)),
        f"{model_name}_mean_lpips": float(np.mean(lpips_scores)),
        f"{model_name}_std_lpips": float(np.std(lpips_scores)),
    }
    return patient_metrics, summary_rows


def run_experiment(config: ExperimentConfig) -> None:
    seed_everything(config.seed)
    config.ensure_dirs()

    device = torch.device(config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `lpips`. Install project dependencies before running experiments."
        ) from exc

    loss_fn = lpips.LPIPS(net="vgg").to(device)
    model_variants = load_model_variants(config, device)
    sens_map_lookup = get_patient_code_map(config.sensitivity_maps_folder)
    print(f"Found {len(sens_map_lookup)} patient(s). Device: {device}.", flush=True)

    all_rows: list[dict[str, float | str | int]] = []
    all_patient_metrics: list[dict[str, float | str]] = []

    for patient_code, sens_map_path in sens_map_lookup.items():
        patient_file = find_matching_patient_file(patient_code, config.original_files_folder)
        if patient_file is None:
            continue

        patient_start = time.time()
        print(f"\n=== Patient: {patient_file.name} ===", flush=True)
        patient_metrics: dict[str, float | str] = {"patient": patient_file.name}

        for model_name, model, use_ttt in model_variants:
            model_start = time.time()
            metrics, rows = run_patient_model(
                model_name=model_name,
                base_model=model,
                use_ttt=use_ttt,
                patient_file=patient_file,
                sens_map_path=sens_map_path,
                config=config,
                device=device,
                loss_fn=loss_fn,
            )
            patient_metrics.update(metrics)
            patient_metrics[f"{model_name}_time_taken"] = float(time.time() - model_start)
            all_rows.extend(rows)

        patient_metrics["total_processing_time"] = float(time.time() - patient_start)
        all_patient_metrics.append(patient_metrics)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            config.metrics_dir / f"slice_metrics_{config.method}_{config.mode}_{config.modality}.csv",
            index=False,
        )
    if all_patient_metrics:
        pd.DataFrame(all_patient_metrics).to_csv(
            config.metrics_dir / f"patient_metrics_{config.method}_{config.mode}_{config.modality}.csv",
            index=False,
        )
