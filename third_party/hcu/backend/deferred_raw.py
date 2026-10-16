"""HCU backend hook: materialize deferred tle.raw regions at make_llir.

Mirrors third_party/nvidia/backend/deferred_raw.py. The HIP body itself is not
imported here. ``compile_deferred_pending_source`` records the original source
so ``make_amdgcn`` can ``llvm-link`` the DTK clang bitcode, keeping the
``target-features`` that ``translateLLVMIRToModule`` would drop.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from triton._C.libtriton import hcu
from triton.experimental.tle.raw.hcu.runtime import take_link_sources
from triton.experimental.tle.raw.source_store import (
    clear_pending_sources,
    list_pending_sources,
)


def _compile_pending_raw_sources(mod: Any, pending: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from triton._C.libtriton import tle

    context = mod.context
    tle.load_dialects(context)
    hcu.load_dialects(context)

    compiled: dict[str, dict[str, Any]] = {}
    for source_id, entry in pending.items():
        payload = dict(entry)
        region_dialect = payload.get("region_dialect")
        if region_dialect != "hcu":
            raise RuntimeError(f"deferred raw materialize does not support region_dialect={region_dialect!r}")
        from triton.experimental.tle.raw.hcu.runtime import compile_deferred_pending_source
        payload["llvm_ir"] = compile_deferred_pending_source(payload, context=context)
        compiled[source_id] = payload
    return compiled


def deferred_raw_materialize(pm: Any, mod: Any) -> None:
    pending = list_pending_sources()
    if not pending:
        return
    compiled = _compile_pending_raw_sources(mod, pending)
    hcu.passes.tle_raw.deferred_raw_materialize(compiled, pm)


def finish_deferred_raw_materialize() -> None:
    if not list_pending_sources():
        return
    clear_pending_sources()


def consume_link_sources() -> list[dict[str, str]]:
    return take_link_sources()


def link_raw_bitcode(llir: str, sources: list[dict[str, str]], arch: str) -> str:
    """Compile each HIP source with DTK clang and llvm-link it into ``llir``."""
    from triton.experimental.tle.raw.hcu.runtime import _clang_command, _resolve_clang

    clang = _resolve_clang()
    llvm_link = str(Path(clang).with_name("llvm-link"))
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel = Path(tmpdir) / "kernel.ll"
        kernel.write_text(llir)
        bitcodes: list[str] = []
        for index, source in enumerate(sources):
            hip = Path(tmpdir) / f"raw_{index}.hip"
            bc = Path(tmpdir) / f"raw_{index}.bc"
            hip.write_text(source["source"])
            command = _clang_command(
                clang,
                source.get("arch") or arch,
                str(hip),
                str(bc),
                bitcode=True,
                dushmem=source.get("library") == "dushmem",
            )
            build = subprocess.run(command, capture_output=True, text=True)
            if build.returncode != 0:
                raise RuntimeError(f"HCU clang failed to emit bitcode for {source.get('file')}:\n"
                                   f"{build.stderr}")
            bitcodes.append(str(bc))
            if source.get("library") == "dushmem":
                from triton.experimental.tle.raw.hcu.dushmem import resolve_dushmem_device_bitcode
                bitcodes.append(str(resolve_dushmem_device_bitcode(source.get("arch") or arch)))
        # hipcc links these device libraries before codegen. The copies under
        # Triton's backend lib do not define __ockl_get_local_id.
        device_lib = Path("/opt/dtk/amdgcn/bitcode")
        isa = "".join(ch for ch in (sources[0].get("arch") or arch) if ch.isdigit())
        for name in (
                "ockl.bc",
                "ocml.bc",
                "oclc_wavefrontsize64_on.bc",
                f"oclc_isa_version_{isa}.bc",
                "oclc_abi_version_500.bc",
        ):
            path = device_lib / name
            if path.is_file():
                bitcodes.append(str(path))
        linked = Path(tmpdir) / "linked.ll"
        command = [llvm_link, str(kernel), *bitcodes, "-S", "-o", str(linked)]
        build = subprocess.run(command, capture_output=True, text=True)
        if build.returncode != 0:
            raise RuntimeError(f"llvm-link failed:\ncommand: {' '.join(command)}\n{build.stderr}")
        return linked.read_text()
