"""Triton TLE MegaMoE operator PR payload."""

from pathlib import Path

_PRODUCTION_ROOT = Path(__file__).resolve().parent / "production"
PRODUCTION_V25_RUNNER = _PRODUCTION_ROOT / "v25" / "run.py"
PRODUCTION_V234_RUNNER = _PRODUCTION_ROOT / "v234" / "run.py"

__all__ = [
    "PRODUCTION_V25_RUNNER",
    "PRODUCTION_V234_RUNNER",
]
