"""Backend hook: compile deferred `tle.raw` payloads and hand them to MLIR.

Called from a backend's `make_llir`, right before
`convert-tritonxpu-to-llvm`. By then the target architecture is known, so every
payload registered during tracing can be compiled for it and injected into the
matching `triton_xpu.raw` ops (matched by source id).

Nothing is compiled when the kernel comes from the Triton cache: tracing never
runs, so no payload is pending.
"""

from __future__ import annotations

from typing import Any, Callable

from .source_store import list_pending_sources

__all__ = ["materialize_deferred_raw"]


def materialize_deferred_raw(pm: Any, add_pass: Callable[[Any, dict], None], *, arch, dialect=None) -> dict:
    """Compile pending payloads for `arch` and add the materialization pass.

    Args:
        pm: the PassManager being assembled.
        add_pass: `xpu.passes.ttxpuir.add_tritonxpu_materialize_deferred_raw_pass`.
        arch: numeric XPU arch the backend is building for (3).
        dialect: only materialize payloads of this dialect ("xpu"), or all.

    Returns the {source_id: llvm_ir} map that was handed over (mostly for tests).
    """
    pending = list_pending_sources()
    if not pending:
        return {}

    compiled = {}
    for source_id, entry in pending.items():
        if dialect is not None and entry.get("dialect") != dialect:
            continue
        handle = entry.get("handle")
        if handle is None:
            raise RuntimeError(f"tle.raw: pending payload {source_id} has no compiler handle")
        compiled[source_id] = handle.make_llvm(arch=arch)

    if not compiled:
        return {}
    add_pass(pm, compiled)
    return compiled
