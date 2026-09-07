# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reader-only renderer for selected and bufferized Triton IR."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from triton.flagmega.codegen.triton.diagnostics import (
    dump_codegen,
    dump_schedules,
)
from triton.flagmega.codegen.triton.registry import (
    PackageRendererRegistry,
    PackageRendererSpec,
)
from triton.flagmega.codegen.triton.tir_package import (
    describe_tir_package,
    render_tir_package,
)
from triton.flagmega.ir import (
    IRModule,
    verify_buffer_plan,
    verify_module,
)

if TYPE_CHECKING:
    from triton.flagmega.diagnostics import Dumper


RENDERER_VERSION = "flagmega-triton-renderer/v20"


renderer_registry = PackageRendererRegistry()
renderer_registry.register(PackageRendererSpec(
    "bufferized-tir",
    "tir_call_graph/v1",
    lambda module: (
        module.stage == "bufferized_tir"
        and module.dialect == "bufferized_tir"
    ),
    describe_tir_package,
    render_tir_package,
))


def render_triton_package(
    module: IRModule,
    output_dir: str | Path,
    *,
    dumper: Dumper | None = None,
) -> dict[str, object]:
    verify_module(
        module,
        expected_stage="bufferized_tir",
        expected_dialect="bufferized_tir",
    )
    verify_buffer_plan(module)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    spec = renderer_registry.resolve(module)
    descriptor = spec.describe(module)
    source = spec.render(descriptor, RENDERER_VERSION)
    source_path = destination / "generated_kernels.py"
    source_path.write_text(source, encoding="utf-8")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    package = {
        "renderer": RENDERER_VERSION,
        "renderer_spec": spec.name,
        "kind": spec.kind,
        "source": source_path.name,
        "source_sha256": source_hash,
        "symbol": descriptor["symbol"],
        "grid": descriptor["grid"],
        "num_warps": descriptor.get("num_warps", 1),
        "dynamic_argument_indices": descriptor.get(
            "dynamic_argument_indices", ()
        ),
        **{
            key: value
            for key, value in descriptor.items()
            if key not in {"symbol", "kernel_templates", "render_calls"}
        },
    }
    dump_schedules(module, dumper)
    dump_codegen(module, package, source, dumper)
    return package


__all__ = [
    "RENDERER_VERSION",
    "render_triton_package",
    "renderer_registry",
]
