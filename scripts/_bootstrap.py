"""Shared path setup for the wrapper scripts.

Makes the scripts runnable without installing the package, by adding ``src/``
to ``sys.path`` and re-exporting the CLI entry point.
"""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unet_functa_shift.cli import main  # noqa: E402

__all__ = ["main"]
