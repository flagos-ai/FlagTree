# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Render and build one C++ package inside an artifact directory.

This is the C++ counterpart of ``render_triton_package``: the single entry
``artifacts/module.py`` calls when the artifact's codegen backend is ``cpp``.
It emits the source, compiles it, and returns the manifest's codegen
descriptor -- including the library's own hash, so a stale or hand-edited
library is rejected at load time the same way a stale generated source is.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from triton.flagmega.codegen.cpp.build import build_cpp_package
from triton.flagmega.codegen.cpp.renderer import render_cpp_source_package

if TYPE_CHECKING:
    from triton.flagmega.diagnostics import Dumper
    from triton.flagmega.ir import IRModule


LIBRARY_NAME = "libflagmega_generated.so"


def render_cpp_package_artifact(
    module: IRModule,
    output_dir: str | Path,
    *,
    target: str,
    dumper: Dumper | None = None,
) -> dict[str, object]:
    destination = Path(output_dir)
    package = render_cpp_source_package(module, destination, dumper=dumper)
    library = build_cpp_package(
        destination / str(package["source"]),
        destination / LIBRARY_NAME,
        target=target,
    )
    return {
        **package,
        "library": library.name,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }


__all__ = ["LIBRARY_NAME", "render_cpp_package_artifact"]
