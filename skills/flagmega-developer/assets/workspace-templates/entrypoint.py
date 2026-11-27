#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bind this workspace entry point to the owning agent's real model/tool function."""
import argparse
import importlib
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, metavar="MODULE:FUNCTION")
    args, forwarded = parser.parse_known_args()
    module, separator, function = args.binding.partition(":")
    if not separator or not module or not function:
        parser.error("--binding must be MODULE:FUNCTION")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    target = getattr(importlib.import_module(module), function)
    result = target(forwarded)
    if result is not None and type(result) is not int:
        raise TypeError("bound entry point must return an integer exit code or None")
    return result or 0


if __name__ == "__main__":
    raise SystemExit(main())
