"""Compile the current module's deferred raw sources and materialize them."""

from __future__ import annotations

from typing import Any

from triton._C.libtriton import iluvatar


def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    iluvatar.load_dialects(mod.context)
    from triton.experimental.tle.raw.iluvatar.runtime import compile_deferred_pending_source
    compiled = {}
    for source_id, entry in pending.items():
        region_dialect = entry.get("region_dialect")
        if region_dialect != "corex":
            detail = f"unsupported region_dialect={region_dialect!r}"
            raise RuntimeError(f"deferred raw source {source_id}: {detail}")
        try:
            llvm_ir = compile_deferred_pending_source(entry, context=mod.context)
            compiled[source_id] = {
                "region_dialect": entry["region_dialect"],
                "extern_func_name": entry["extern_func_name"],
                "llvm_ir": llvm_ir,
            }
        except Exception as exc:
            raise RuntimeError(f"failed to compile deferred raw source {source_id} "
                               f"({entry.get('dsl_file_name', '<missing file>')}): {exc}") from exc
    return compiled


def deferred_raw_materialize(pm: Any, mod: Any) -> None:
    pending = iluvatar.tle.raw_passes.collect_pending_sources(mod)
    if not pending:
        return
    compiled = _compile_pending_raw_sources(mod, pending)
    iluvatar.passes.tle_raw.deferred_raw_materialize(compiled, pm)
