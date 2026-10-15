"""Iluvatar backend hook: materialize deferred tle_raw DSL regions at make_llir.

Mirrors third_party/nvidia/backend/deferred_raw.py: trace registers pending
sources, make_llir compiles them and runs the C++ pass to fill stub dsl_region
bodies before dsl_region_inline.

corex only ships its own region dialect, "corex": CUDA-like sources compiled by
the corex clang, not by the NVIDIA CUDA toolchain. The MLIR dialect needs the
MLIR python bindings, which the corex LLVM distribution does not build, so it is
rejected here rather than silently mis-compiled.
"""

from __future__ import annotations

from typing import Any

from triton._C.libtriton import iluvatar
from triton.experimental.tle.raw.source_store import (
    clear_pending_sources,
    list_pending_sources,
)


def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compile pending sources to LLVM IR, dispatching by region_dialect.

    The corex compile helper lives in its own runtime module; this function only
    routes to it. The path temporarily calls the same make_llvm() used by eager
    mode.
    """
    context = mod.context
    iluvatar.load_dialects(context)

    compiled: dict[str, dict[str, Any]] = {}
    for source_id, entry in pending.items():
        payload = dict(entry)
        region_dialect = payload.get("region_dialect")
        if region_dialect == "corex":
            from triton.experimental.tle.raw.iluvatar.runtime import compile_deferred_pending_source
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
    iluvatar.passes.tle_raw.deferred_raw_materialize(compiled, pm)


def finish_deferred_raw_materialize() -> None:
    if not list_pending_sources():
        return
    clear_pending_sources()
