"""Under-sampling masks and self-supervised mask splits (N2N / SSDU)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MaskFunc:
    """Cartesian under-sampling mask generator."""

    def __init__(self, center_fractions: list[float], accelerations: list[int]) -> None:
        if len(center_fractions) != len(accelerations):
            raise ValueError("center_fractions and accelerations must have equal length")
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape: list[int], seed: int | None = None) -> np.ndarray:
        if len(shape) < 3:
            raise ValueError("shape must have at least 3 dimensions")

        self.rng.seed(seed)
        _, nrow, ncol, _ = shape
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        num_low_freqs = int(round(ncol * center_fraction))
        prob = (ncol / acceleration - num_low_freqs) / (ncol - num_low_freqs)
        mask = self.rng.uniform(size=ncol) < prob
        pad = (ncol - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        shape[-1] = 1
        mask_out = np.ones(shape, dtype=np.float32)
        mask_prod = np.zeros((1, 1, ncol, 1), dtype=np.float32)
        mask_prod[0, 0, :, 0] = mask.astype(np.float32)
        return mask_out * mask_prod


@dataclass
class MaskSplit:
    train_mask: np.ndarray
    loss_mask: np.ndarray


def _index_flatten_to_2d(indices: np.ndarray, shape: tuple[int, int]) -> tuple[list[int], list[int]]:
    array = np.zeros(np.prod(shape), dtype=np.float32)
    array[indices] = 1.0
    ind_nd = np.nonzero(np.reshape(array, shape))
    return list(ind_nd[0]), list(ind_nd[1])


class SSDUMasks:
    """Split an acquired mask into disjoint training / loss masks (SSDU)."""

    def __init__(self, rho: float = 0.4, small_acs_block: tuple[int, int] = (4, 4)) -> None:
        self.rho = rho
        self.small_acs_block = small_acs_block

    def uniform_selection(self, input_mask: np.ndarray, seed: int = 2025) -> MaskSplit:
        np.random.seed(seed)
        nrow, ncol = input_mask.shape
        center_kx = nrow // 2
        center_ky = ncol // 2

        temp_mask = np.copy(input_mask)
        temp_mask[
            center_kx - self.small_acs_block[0] // 2 : center_kx + self.small_acs_block[0] // 2,
            center_ky - self.small_acs_block[1] // 2 : center_ky + self.small_acs_block[1] // 2,
        ] = 0

        probs = np.ndarray.flatten(temp_mask)
        chosen = np.random.choice(
            np.arange(nrow * ncol),
            size=int(np.count_nonzero(probs) * self.rho),
            replace=False,
            p=probs / np.sum(probs),
        )
        idx_x, idx_y = _index_flatten_to_2d(chosen, (nrow, ncol))

        loss_mask = np.zeros_like(input_mask, dtype=np.float32)
        loss_mask[idx_x, idx_y] = 1.0
        train_mask = input_mask.astype(np.float32) - loss_mask
        return MaskSplit(train_mask=train_mask, loss_mask=loss_mask)


def build_base_mask(height: int, width: int, center_fraction: float, acceleration: int, seed: int) -> np.ndarray:
    mask_fn = MaskFunc([center_fraction], [acceleration])
    return mask_fn([1, height, width, 2], seed=seed)[0, :, :, 0].astype(np.float32)


def build_n2n_train_mask(base_mask: np.ndarray, sub_acceleration: int, seed: int) -> np.ndarray:
    sub_mask_fn = MaskFunc([0.08], [sub_acceleration])
    sub_mask = sub_mask_fn([1, base_mask.shape[0], base_mask.shape[1], 2], seed=seed)[0, :, :, 0]
    return (base_mask * sub_mask).astype(np.float32)
