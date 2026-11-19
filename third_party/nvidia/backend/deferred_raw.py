"""Compile the current module's deferred raw sources and materialize them."""

from __future__ import annotations

from typing import Any

from triton._C.libtriton import nvidia, tle


def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tle.load_dialects(mod.context)
    from triton.experimental.tle.raw.cuda.runtime import compile_deferred_pending_source

    wrappers = {}
    for metadata in nvidia.passes.tle_raw.register_raw_wrappers(mod):
        wrappers.setdefault(metadata["source_id"], []).append(metadata)

    compiled = {}
    for source_id, entry in pending.items():
        region_dialect = entry.get("region_dialect")
        if region_dialect != "cuda":
            detail = f"unsupported region_dialect={region_dialect!r}"
            raise RuntimeError(f"deferred raw source {source_id}: {detail}")
        try:
            entry = dict(entry, required_wrappers=wrappers.get(source_id, []))
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
    pending = tle.raw_passes.collect_pending_sources(mod)
    if not pending:
        return
    compiled = _compile_pending_raw_sources(mod, pending)
    nvidia.passes.tle_raw.deferred_raw_materialize(compiled, pm)
