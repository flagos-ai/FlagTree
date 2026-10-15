# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Immutable compiler options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from triton.flagmega.diagnostics.dump import DumpFlags
from triton.flagmega.ir import Placement


def parse_placement(value: str) -> Placement:
    """Parse the stable CLI placement spelling ``name=AxB/levels``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("placement must be a non-empty string")
    text = value.strip()
    try:
        name, rest = text.split("=", 1)
        dimensions, levels = rest.split("/", 1)
        hierarchy = tuple(int(item) for item in dimensions.lower().split("x"))
    except (ValueError, TypeError) as error:
        raise ValueError(
            "placement must use the form name=AxB[/...] / levels, "
            "for example cyx=4x8x16/cbb"
        ) from error
    if not name or not hierarchy or not levels:
        raise ValueError(
            "placement must use the form name=AxB[/...] / levels, "
            "for example cyx=4x8x16/cbb"
        )
    try:
        return Placement(hierarchy, name, levels)
    except Exception as error:
        raise ValueError(f"invalid placement {value!r}: {error}") from error


@dataclass(frozen=True)
class CompileOptions:
    target: str = "nvidia-sm90"
    emit_every_stage: bool = False
    require_review: bool = False
    work_dir: Path | None = None
    policy: str | None = None
    dump_flags: DumpFlags = DumpFlags.NONE
    # None preserves the target default, or the allocator in an existing plan.
    bufferize_opt_level: str | None = None
    placement: Placement | None = None

    def __post_init__(self) -> None:
        if self.bufferize_opt_level not in (None, "fast", "optimized"):
            raise ValueError("bufferize_opt_level must be 'fast' or 'optimized'.")

    @property
    def effective_dump_flags(self) -> DumpFlags:
        """Keep ``emit_every_stage`` as a compatibility alias for Compile."""

        flags = DumpFlags(self.dump_flags)
        if self.emit_every_stage:
            flags |= DumpFlags.COMPILE
        return flags


@dataclass(frozen=True)
class RunOptions:
    device: str = "cuda:0"
