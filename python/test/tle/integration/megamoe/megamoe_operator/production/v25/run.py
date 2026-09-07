#!/usr/bin/env python3
"""V25 launcher: live v23 plus descriptorless raw TMA1D for D8 tokens only."""

import os

os.environ.setdefault("W_D8_TMA1D", "1")

if __package__:
    from .kernel import main
else:
    from kernel import main


if __name__ == "__main__":
    raise SystemExit(main())
