from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig


def build_parser(default_method: str | None = None, default_mode: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cleaned FunCTA U-Net anatomy-shift experiments.")
    parser.add_argument("--image-dir-base", type=Path, required=True, help="Base output directory.")
    parser.add_argument("--sensitivity-maps-folder", type=Path, required=True, help="Folder containing `.npy` sensitivity maps.")
    parser.add_argument("--original-files-folder", type=Path, required=True, help="Folder containing patient `.h5` files.")
    parser.add_argument("--modality", type=str, required=True, help="Experiment tag such as `t2`.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["datafidelity", "n2n", "ssdu"],
        default=default_method,
        required=default_method is None,
        help="Training/adaptation objective.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default=default_mode,
        required=default_mode is None,
        help="Adaptation granularity.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("./unet_model_checkpoints"),
        help="Folder containing pretrained U-Net checkpoints.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="`cuda` or `cpu`.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--latent-mean", type=float, default=0.0)
    parser.add_argument("--latent-std", type=float, default=0.01)
    parser.add_argument("--lr-latent", type=float, default=1e-3)
    parser.add_argument("--lr-model", type=float, default=1e-4)
    parser.add_argument("--latent-weight", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=7)
    parser.add_argument("--single-slice-iters", type=int, default=1000)
    parser.add_argument("--single-slice-window", type=int, default=30)
    parser.add_argument("--mask-center-fraction", type=float, default=0.08)
    parser.add_argument("--mask-acceleration", type=int, default=4)
    parser.add_argument("--n2n-sub-acceleration", type=int, default=2)
    parser.add_argument("--ssdu-rho", type=float, default=0.4)
    parser.add_argument("--slice-trim", type=int, default=5)
    parser.add_argument("--no-save-figures", action="store_true", help="Skip `.png` reconstruction figures.")
    parser.add_argument("--no-reuse-checkpoints", action="store_true", help="Always train a fresh prompt model.")
    return parser


def namespace_to_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        image_dir_base=args.image_dir_base,
        sensitivity_maps_folder=args.sensitivity_maps_folder,
        original_files_folder=args.original_files_folder,
        modality=args.modality,
        method=args.method,
        mode=args.mode,
        checkpoint_root=args.checkpoint_root,
        device=args.device,
        seed=args.seed,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        latent_dim=args.latent_dim,
        latent_mean=args.latent_mean,
        latent_std=args.latent_std,
        lr_latent=args.lr_latent,
        lr_model=args.lr_model,
        latent_weight=args.latent_weight,
        eval_interval=args.eval_interval,
        single_slice_iters=args.single_slice_iters,
        single_slice_window=args.single_slice_window,
        mask_center_fraction=args.mask_center_fraction,
        mask_acceleration=args.mask_acceleration,
        n2n_sub_acceleration=args.n2n_sub_acceleration,
        ssdu_rho=args.ssdu_rho,
        slice_trim=args.slice_trim,
        save_figures=not args.no_save_figures,
        reuse_checkpoints=not args.no_reuse_checkpoints,
    )


def main(default_method: str | None = None, default_mode: str | None = None) -> None:
    parser = build_parser(default_method=default_method, default_mode=default_mode)
    args = parser.parse_args()
    from .engine import run_experiment

    run_experiment(namespace_to_config(args))


if __name__ == "__main__":
    main()
