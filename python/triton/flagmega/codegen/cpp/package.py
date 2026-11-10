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
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from triton.flagmega.codegen.cpp.build import build_cpp_package, validate_cpp_max_registers
from triton.flagmega.codegen.cpp.renderer import render_cpp_source_package

if TYPE_CHECKING:
    from triton.flagmega.diagnostics import Dumper
    from triton.flagmega.ir import IRModule


LIBRARY_NAME = "libflagmega_generated.so"


def cpp_runtime_metadata(package: Mapping[str, object]) -> dict[str, object]:
    """Keep runtime/provenance facts without duplicating source render contexts.

    ``programs`` and ``native_program_contexts`` are consumed only during C++
    source rendering and diagnostic annotation. The typed ``cpp_programs``
    manifest carries the runtime entries, invocations, signatures and resources;
    it is independently verified against the artifact's authoritative TIR.
    Full contexts remain available from ``render_cpp_source_package`` and its
    CODEGEN dump, and can be reconstructed from that TIR by the normal renderer.
    """
    return {key: value for key, value in package.items()
            if key not in {"programs", "native_program_contexts"}}


def render_cpp_package_artifact(
    module: IRModule,
    output_dir: str | Path,
    *,
    target: str,
    cpp_max_registers: int | None = None,
    cpp_operator_profile: str | None = None,
    dumper: Dumper | None = None,
) -> dict[str, object]:
    validate_cpp_max_registers(cpp_max_registers, target=target)
    if cpp_operator_profile is not None:
        from .operator_profile import validate_option
        validate_option(cpp_operator_profile, target=target)
    destination = Path(output_dir)
    package = (render_cpp_source_package(module, destination, dumper=dumper)
               if cpp_operator_profile is None else render_cpp_source_package(
                   module, destination, dumper=dumper, cpp_operator_profile=cpp_operator_profile))
    provenance: dict[str, object] = {}
    library = build_cpp_package(
        destination / str(package["source"]),
        destination / LIBRARY_NAME,
        target=target,
        max_registers_per_thread=cpp_max_registers,
        build_provenance=provenance,
    )
    return {
        **cpp_runtime_metadata(package),
        "library": library.name,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "build_options": {"max_registers_per_thread": cpp_max_registers},
        "build_provenance": provenance,
    }


__all__ = ["LIBRARY_NAME", "cpp_runtime_metadata", "render_cpp_package_artifact"]
