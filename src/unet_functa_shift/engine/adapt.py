"""Test-time adaptation: batch (whole-volume) and single-slice refinement."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config import ExperimentConfig
from ..data import compute_scale_factor
from ..external import fft2, ifft2, rss_torch, transform
from ..models import (
    UNetPrompt,
    clone_model,
    collect_trainable_parameters,
    configure_model_with_deblock,
    freeze_conv_blocks,
)
from .evaluate import evaluate_prompt
from .physics import (
    apply_sensitivity_maps_batch,
    apply_sensitivity_maps_single,
    relative_l1_loss,
)


def adapt_batch(
    base_model: nn.Module,
    dataset: Dataset,
    dataloader: DataLoader,
    sens_map: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor]:
    """Adapt the prompt model and per-slice latent codes over a whole volume."""

    prompt_model = UNetPrompt(clone_model(base_model)).to(device)
    trainable_params, _ = collect_trainable_parameters(prompt_model)
    latent_vectors = torch.nn.Parameter(
        torch.randn(len(dataset), config.latent_dim, device=device) * config.latent_std + config.latent_mean,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [latent_vectors], "lr": config.lr_latent},
            {"params": trainable_params, "lr": config.lr_model},
        ]
    )

    for epoch in range(config.num_epochs):
        prompt_model.train()
        running_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            batch_indices = batch["idx"].to(device)
            optimizer.zero_grad()

            train_input = batch["train_input"].to(device)
            coord = batch["coord"].to(device)
            loss_kspace = batch["loss_kspace"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            full_kspace = batch["k_space_scaled"].to(device)
            latent_batch = latent_vectors[batch_indices]

            output, siren_output, _, _ = prompt_model(train_input, coord, latent_batch)
            output = torch.moveaxis(output, 1, -1)
            siren_output = torch.moveaxis(siren_output, 1, -1)

            predicted = fft2(apply_sensitivity_maps_batch(sens_map, output, device))
            regularized = fft2(apply_sensitivity_maps_batch(sens_map, siren_output, device))

            predicted = predicted * loss_mask.unsqueeze(1).unsqueeze(-1)
            regularized = regularized * batch["base_mask"].to(device).unsqueeze(1).unsqueeze(-1)

            loss_unsup = relative_l1_loss(loss_kspace, predicted) / config.batch_size
            loss_reg = relative_l1_loss(full_kspace, regularized) / config.batch_size
            latent_norm = (
                config.latent_weight
                * torch.pow(latent_batch, 2).sum(dim=-1).mean()
                * (1 / (config.latent_std ** 2))
            )
            loss = loss_unsup + loss_reg + latent_norm
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        print(f"    [adapt] epoch {epoch + 1}/{config.num_epochs}  loss={avg_loss:.4f}", flush=True)

    return prompt_model, latent_vectors.detach()


def refine_single_slice(
    prompt_model: nn.Module,
    base_model: nn.Module,
    latent_vector: torch.Tensor,
    coord: torch.Tensor,
    slice_kspace: np.ndarray,
    orig: np.ndarray,
    sens_map: np.ndarray,
    base_mask: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[np.ndarray, float, float, float]:
    """Further refine a single slice with a self-validated DEBlock optimization loop."""

    masked_kspace = slice_kspace * np.tile(base_mask[None, ...], (slice_kspace.shape[0], 1, 1))
    scale = compute_scale_factor(masked_kspace, base_model, device)
    slice_tt = transform.to_tensor(masked_kspace).to(device) * scale
    orig_tensor = torch.from_numpy(orig.astype(np.float32))

    where_ones = np.array(np.where(base_mask == 1))
    num_points = len(where_ones[0])
    held_out = where_ones[:, np.random.randint(0, num_points, max(1, num_points // 20))]

    model_ttt = copy.deepcopy(prompt_model)
    model_ttt = configure_model_with_deblock(model_ttt)
    model_ttt = freeze_conv_blocks(model_ttt)
    params, _ = collect_trainable_parameters(model_ttt)
    optimizer = torch.optim.Adam(params, lr=config.lr_model)

    model_input = torch.moveaxis(rss_torch(ifft2(slice_tt))[None, :], -1, 1)
    mask = torch.from_numpy(np.tile(base_mask[None, :, :, None], (slice_tt.shape[0], 1, 1, 2))).float().to(device)
    validation_errors: list[float] = []

    for iteration in range(config.single_slice_iters):
        optimizer.zero_grad()

        output, siren_output, _, _ = model_ttt(model_input.to(device), coord.to(device), latent_vector[None, :].to(device))
        output = torch.moveaxis(output, 1, -1)
        predicted_kspace = apply_sensitivity_maps_single(sens_map, output, device) * mask

        train_prediction = predicted_kspace.clone()
        train_prediction[:, held_out[0], held_out[1], :] = 0
        train_target = slice_tt.clone()
        train_target[:, held_out[0], held_out[1], :] = 0

        loss = relative_l1_loss(train_target, train_prediction)

        if config.method != "ssdu":
            siren_output = torch.moveaxis(siren_output, 1, -1)
            regularized_kspace = apply_sensitivity_maps_single(sens_map, siren_output, device) * mask
            loss = loss + relative_l1_loss(slice_tt, regularized_kspace)

        validation_loss = relative_l1_loss(
            slice_tt[:, held_out[0], held_out[1], :],
            predicted_kspace[:, held_out[0], held_out[1], :],
        )
        validation_errors.append(float(validation_loss.item()))

        loss.backward()
        optimizer.step()

        if iteration > 3 * config.single_slice_window:
            recent = np.mean(validation_errors[-config.single_slice_window :])
            previous = np.mean(validation_errors[-2 * config.single_slice_window : -config.single_slice_window])
            if recent > previous:
                break

    return evaluate_prompt(
        model_ttt,
        slice_tt.cpu(),
        orig_tensor,
        latent_vector[None, :].cpu(),
        coord[None, :, :, :].cpu(),
        loss_fn,
        device,
    )
