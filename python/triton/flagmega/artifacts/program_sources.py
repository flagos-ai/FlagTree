# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Validate the source files and exported entries of a Triton package."""

from __future__ import annotations

from collections.abc import Mapping
import re

from triton.flagmega.codegen.triton.program_sources import (
    LEGACY_PROGRAM_SOURCE_LAYOUT, PROGRAM_SOURCE_LAYOUT,
)
from triton.flagmega.errors import ArtifactError


PROGRAM_SOURCE_KIND = "codegen.triton-program/v1"


def program_source_records(codegen):
    if "program_source_layout" not in codegen:
        if "program_sources" in codegen or "program_entry_aliases" in codegen:
            raise ArtifactError("Program sources require a declared source layout.")
        return ()
    if codegen["program_source_layout"] not in {LEGACY_PROGRAM_SOURCE_LAYOUT, PROGRAM_SOURCE_LAYOUT}:
        raise ArtifactError("Unsupported Triton program source layout.")
    records = codegen.get("program_sources")
    aliases = codegen.get("program_entry_aliases", {})
    if not isinstance(records, (tuple, list)) or not records or not isinstance(aliases, Mapping):
        raise ArtifactError("Program source layout requires source records and entry aliases.")
    entries, exports, paths = set(), set(), set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactError("Program source record must be an object.")
        entry, names, path = record.get("symbol"), record.get("symbols"), record.get("source")
        if (not isinstance(entry, str) or not entry.isidentifier()
                or not isinstance(names, (tuple, list)) or entry not in names
                or any(not isinstance(name, str) or not name.isidentifier() for name in names)
                or len(set(names)) != len(names) or entries.intersection({entry})
                or exports.intersection(names) or not isinstance(path, str) or not path or path in paths
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("source_sha256", "")))):
            raise ArtifactError("Invalid or duplicated program source exports or identity.")
        entries.add(entry)
        exports.update(names)
        paths.add(path)
    if any(not isinstance(name, str) or not name.isidentifier()
           or not isinstance(target, str) or not target.isidentifier()
           or name in exports for name, target in aliases.items()):
        raise ArtifactError("Invalid program source entry alias.")
    for name in aliases:
        active = set()
        target = name
        while target in aliases:
            if target in active:
                raise ArtifactError("Cyclic program source entry aliases.")
            active.add(target)
            target = aliases[target]
        if target not in exports:
            raise ArtifactError("Program source entry alias has no definition.")
    kernels = codegen.get("kernels")
    required = ({spec["symbol"] for spec in kernels} if kernels is not None else {codegen["symbol"]})
    if not required <= exports | set(aliases):
        raise ArtifactError("Program source files do not cover the declared entries.")
    return tuple(records)


def program_source_section_name(record):
    return f"codegen.program.{record['symbol']}"
