# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Install the tutorial's reviewed kernel layouts before compiling artifacts."""
from importlib import import_module


def activate():
    """Install once per process; distribution and resource gates stay active."""
    import_module(f"{__name__}.norm")
