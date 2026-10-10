"""NVIDIA backend hook: materialize deferred tle_raw DSL regions at make_llir.

This module exists mainly to validate the deferred materialization pipeline end-to-end:
trace registers pending sources, make_llir compiles them and runs the C++ pass to fill
stub dsl_region bodies before dsl_region_inline.

For now CUDA and MLIR support live together here and both reuse the existing eager
compile paths at make_llir time (CUDAJITFunction.make_llvm / MLIRJITFunction.make_llvm)
rather than dedicated deferred-only compilers. Can be replaced with separate compile hooks.
"""

from __future__ import annotations

from typing import Any

from triton._C.libtriton import nvidia
from triton.experimental.tle.raw.source_store import (
    clear_pending_sources,
    list_pending_sources,
)


def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compile pending sources to LLVM IR, dispatching by region_dialect.

    CUDA and MLIR compile helpers live in their own runtime modules; this function
    only routes to them. Both paths temporarily call the same make_llvm() used by eager
    mode.
    """
    from triton._C.libtriton import tle

    context = mod.context
    tle.load_dialects(context)

    compiled: dict[str, dict[str, Any]] = {}
    for source_id, entry in pending.items():
        payload = dict(entry)
        region_dialect = payload.get("region_dialect")
        if region_dialect == "cuda":
            from triton.experimental.tle.raw.cuda.runtime import compile_deferred_pending_source
            payload["llvm_ir"] = compile_deferred_pending_source(payload, context=context)
        elif region_dialect == "mlir":
            from triton.experimental.tle.raw.mlir.runtime import compile_deferred_pending_source
            payload["llvm_ir"] = compile_deferred_pending_source(payload, context=context)
        else:
            raise RuntimeError(f"deferred raw materialize does not support region_dialect={region_dialect!r}")
        compiled[source_id] = payload
    return compiled


def deferred_raw_materialize(pm: Any, mod: Any) -> None:
    pending = list_pending_sources()
    if not pending:
        return
    compiled = _compile_pending_raw_sources(mod, pending)
    nvidia.passes.tle_raw.deferred_raw_materialize(compiled, pm)


def finish_deferred_raw_materialize() -> None:
    if not list_pending_sources():
        return
    clear_pending_sources()
