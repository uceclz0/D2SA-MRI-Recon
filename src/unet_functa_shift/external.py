from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_external_repo_to_path() -> None:
    current_file = Path(__file__).resolve()
    candidates = []

    env_root = os.getenv("MRI_TTT_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    # Current project lives in `.../all_code/unet_functa_anatomy_shift_project/src/...`
    candidates.append(current_file.parents[3])
    candidates.append(current_file.parents[2])

    for root in candidates:
        package_dir = root / "ttt_for_deep_learning_cs"
        if package_dir.exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return

    raise ImportError(
        "Cannot locate `ttt_for_deep_learning_cs`. "
        "Set `MRI_TTT_ROOT` to the directory that contains it."
    )


_add_external_repo_to_path()

try:
    from ttt_for_deep_learning_cs.unet.functions.helpers import rss_torch  # noqa: E402
    from ttt_for_deep_learning_cs.unet.functions.include import transforms as transform  # noqa: E402
    from ttt_for_deep_learning_cs.unet.functions.include.transforms import (  # noqa: E402
        center_crop,
        complex_abs,
        fft2,
        ifft2,
    )
    from ttt_for_deep_learning_cs.unet.functions.unet_model import (  # noqa: E402
        ConvBlock,
        TransposeConvBlock,
        UnetMRIModelDemo,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import `ttt_for_deep_learning_cs` runtime modules. "
        "Make sure the external repository is reachable via `MRI_TTT_ROOT` "
        "and its dependencies (for example `pytorch_lightning`) are installed."
    ) from exc

__all__ = [
    "ConvBlock",
    "TransposeConvBlock",
    "UnetMRIModelDemo",
    "center_crop",
    "complex_abs",
    "fft2",
    "ifft2",
    "rss_torch",
    "transform",
]
