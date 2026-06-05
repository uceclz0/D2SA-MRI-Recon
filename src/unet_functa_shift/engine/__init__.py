"""Training/evaluation engine: physics, adaptation, and the experiment runner."""

from .adapt import adapt_batch, refine_single_slice
from .evaluate import evaluate_backbone, evaluate_prompt
from .physics import (
    apply_sensitivity_maps_batch,
    apply_sensitivity_maps_single,
    relative_l1_loss,
)
from .runner import run_experiment, run_patient_model

__all__ = [
    "adapt_batch",
    "apply_sensitivity_maps_batch",
    "apply_sensitivity_maps_single",
    "evaluate_backbone",
    "evaluate_prompt",
    "refine_single_slice",
    "relative_l1_loss",
    "run_experiment",
    "run_patient_model",
]
