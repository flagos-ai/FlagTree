# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Render independently compiled program namespaces into separate files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from triton.flagmega.errors import CodegenError


LEGACY_PROGRAM_SOURCE_LAYOUT = "flagmega.triton-program-sources/v1"
PROGRAM_SOURCE_LAYOUT = "flagmega.triton-program-sources/v2"


def render_program_source_files(descriptor, renderer_version, render, destination, source):
    programs = descriptor.get("programs") or ()
    aliases = {program["symbol"]: program["entry_alias"] for program in programs
               if program.get("entry_alias")}
    records = []
    if not programs:
        programs = ({"symbol": descriptor["symbol"]},)
    for program in programs:
        symbol = program["symbol"]
        if not isinstance(symbol, str) or not symbol.isidentifier():
            raise CodegenError("Program source requires a valid entry symbol.")
        if program.get("entry_alias"):
            continue
        invocations = tuple(invocation for invocation in descriptor.get("kernels", ())
                            if invocation.get("entry_symbol", invocation["symbol"]) == symbol)
        if descriptor.get("programs"):
            # The main source and manifest retain complete metadata. Each
            # independent namespace carries only its own invocation records.
            context = {**descriptor, "programs": (program,), "kernels": list(invocations),
                       "source_invocations": invocations}
            text = render(context, renderer_version)
        else:
            text = source
        # Repeated calls may also name a program alias. Its definition belongs
        # to the alias table, rather than being exported a second time.
        exports = tuple(dict.fromkeys((symbol, *(invocation["symbol"] for invocation in invocations
                                                if invocation["symbol"] not in aliases))))
        relative = f"program_sources/{symbol}.py"
        path = Path(destination) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        records.append({"symbol": symbol, "symbols": exports, "source": relative,
                        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    return {"program_source_layout": PROGRAM_SOURCE_LAYOUT,
            "program_sources": records, "program_entry_aliases": aliases}
