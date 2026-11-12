"""Compile the current module's deferred HCU raw sources and materialize them."""

from __future__ import annotations

from typing import Any

from triton._C.libtriton import hcu, tle


def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tle.load_dialects(mod.context)
    from triton.experimental.tle.raw.hcu.runtime import compile_deferred_pending_source

    compiled = {}
    for source_id, entry in pending.items():
        region_dialect = entry.get("region_dialect")
        if region_dialect not in ("hcu", "hip"):
            raise RuntimeError(f"deferred raw source {source_id}: unsupported region_dialect={region_dialect!r}")
        try:
            # The compiler replaces the source-level name with its lowered LLVM
            # symbol. Keep that update local to this compilation.
            payload = dict(entry)
            llvm_ir = compile_deferred_pending_source(payload, context=mod.context)
            compiled[source_id] = {
                "region_dialect": payload["region_dialect"],
                "extern_func_name": payload["extern_func_name"],
                "llvm_ir": llvm_ir,
            }
        except Exception as exc:
            raise RuntimeError(f"failed to compile deferred raw source {source_id} "
                               f"({entry.get('dsl_file_name', '<missing file>')}): {exc}") from exc
    return compiled


def deferred_raw_materialize(pm: Any, mod: Any) -> None:
    pending = tle.raw_passes.collect_pending_sources(mod)
    if not pending:
        return
    compiled = _compile_pending_raw_sources(mod, pending)
    hcu.passes.tle_raw.deferred_raw_materialize(compiled, pm)
