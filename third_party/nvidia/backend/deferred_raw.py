"""NVIDIA backend hook: materialize deferred tle_raw DSL regions at make_llir.

For now CUDA and MLIR support live together here and both reuse the existing eager
compile paths at make_llir time (CUDAJITFunction.make_llvm / MLIRJITFunction.make_llvm)
rather than dedicated deferred-only compilers.
"""

from __future__ import annotations

from typing import Any

from triton._C.libtriton import nvidia
from triton.experimental.tle.raw.source_store import clear_pending_sources, list_pending_sources


# Compile pending sources to LLVM IR, dispatching by region_dialect.
def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from triton._C.libtriton import tle

    context = mod.context
    tle.load_dialects(context)

    compiled: dict[str, dict[str, Any]] = {}
    wrappers_to_gen = {}
    for wrapper_metadata in nvidia.passes.tle_raw.register_raw_wrappers(mod):
        wrappers_to_gen.setdefault(wrapper_metadata["source_id"], []).append(wrapper_metadata)
    for source_id, entry in pending.items():
        payload = dict(entry)
        region_dialect = payload.get("region_dialect")
        if region_dialect == "cuda":
            from triton.experimental.tle.raw.cuda.runtime import compile_deferred_pending_source
            payload["required_wrappers"] = wrappers_to_gen.get(source_id)
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
