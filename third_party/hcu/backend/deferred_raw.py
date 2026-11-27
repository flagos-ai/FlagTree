from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from triton._C.libtriton import hcu
from triton.experimental.tle.raw.source_store import (
    clear_pending_sources,
    list_pending_sources,
)
from triton.backends.hcu.compiler import HIPBackend


def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compile pending sources to LLVM IR, dispatching by region_dialect.

    HCU/HIP and MLIR compile helpers live in their own runtime modules; this function
    only routes to them. Both paths use the same make_llvm() as eager mode. Device libraries stay
    unresolved; the final HCU HSACO stage owns rocSHMEM linking.
    """
    from triton._C.libtriton import tle

    context = mod.context
    tle.load_dialects(context)

    compiled: dict[str, dict[str, Any]] = {}
    for source_id, entry in pending.items():
        payload = dict(entry)
        region_dialect = payload.get("region_dialect")
        if region_dialect in ["hcu", "hip"]:
            from triton.experimental.tle.raw.hcu.runtime import compile_deferred_pending_source
            payload["llvm_ir"] = compile_deferred_pending_source(payload, context=context)
        elif region_dialect == "mlir":
            from triton.experimental.tle.raw.mlir.runtime import compile_deferred_pending_source
            payload["llvm_ir"] = compile_deferred_pending_source(payload, context=context)
        else:
            raise RuntimeError(f"deferred raw materialize does not support region_dialect={region_dialect!r}")
        compiled[source_id] = payload
    return compiled


def deferred_raw_materialize(mod: Any, pm: Any) -> tuple[str, ...]:
    pending = list_pending_sources()
    if not pending:
        return
    libraries = tuple(
        sorted({
            entry["library"]
            for entry in pending.values()
            if entry.get("region_dialect") in {"hcu", "hip"} and entry.get("library")
        }))
    try:
        compiled = _compile_pending_raw_sources(mod, pending)
        hcu.passes.tle_raw.deferred_raw_materialize(compiled, pm)
    except Exception:
        clear_pending_sources()
        raise
    return libraries


def finish_deferred_raw_materialize() -> None:
    if not list_pending_sources():
        return
    clear_pending_sources()
