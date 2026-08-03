"""Triton TLE MegaMoE operator PR payload."""

from pathlib import Path

_PRODUCTION_ROOT = Path(__file__).resolve().parent / "production"
PRODUCTION_V25_RUNNER = _PRODUCTION_ROOT / "v25" / "run.py"
PRODUCTION_V33_RUNNER = _PRODUCTION_ROOT / "v33" / "run.py"

__all__ = [
    "PRODUCTION_V25_RUNNER",
    "PRODUCTION_V33_RUNNER",
]
