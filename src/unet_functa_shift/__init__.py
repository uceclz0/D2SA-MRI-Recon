"""Clean entry points for FunCTA U-Net anatomy-shift experiments.

Package layout
--------------
- ``config``   : the :class:`ExperimentConfig` dataclass (all run options).
- ``external`` : bridge to the upstream ``ttt_for_deep_learning_cs`` repo.
- ``metrics``  : SSIM / PSNR / LPIPS scoring, figure saving, seeding.
- ``data``     : masks, the per-patient k-space dataset, file discovery.
- ``models``   : INR prompt, differential conv blocks, prompt-conditioned U-Net.
- ``engine``   : MRI physics, evaluation, adaptation, and the experiment runner.
- ``cli``      : command-line interface used by the wrapper scripts.
"""

from .config import ExperimentConfig

__all__ = ["ExperimentConfig"]
