# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""MemSpan-based alias analysis for bufferization."""

from __future__ import annotations

from dataclasses import dataclass
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.bufferization import AliasKind, MemSpan, PhysicalBuffer


@dataclass(frozen=True)
class BufferView:
    logical_id: str
    mem_span: MemSpan
    kind: AliasKind
    source: str | None = None

    @property
    def physical_id(self) -> str:
        return self.mem_span.buffer.id

    @property
    def byte_offset(self) -> int:
        return self.mem_span.byte_offset

    @property
    def nbytes(self) -> int:
        return self.mem_span.nbytes


class AliasAnalysis:
    """Track aliases by physical object and half-open byte range.

    This is intentionally independent of liveness.  Lifetime analysis decides
    whether an in-place relation is legal; this analysis answers whether two
    resulting logical values can or must refer to intersecting physical bytes.
    """

    def __init__(self) -> None:
        self._views: dict[str, BufferView] = {}

    def define(self, logical_id: str, physical_id: str, nbytes: int) -> BufferView:
        """Compatibility helper for fixed-size tests and external callers."""

        physical = PhysicalBuffer(physical_id, "analysis", nbytes, 1)
        return self.define_span(logical_id, MemSpan(physical))

    def define_span(self, logical_id: str, mem_span: MemSpan) -> BufferView:
        return self._insert(BufferView(logical_id, mem_span, AliasKind.IDENTITY))

    def add_alias(
        self,
        logical_id: str,
        source: str,
        *,
        byte_offset: int = 0,
        nbytes: int | None = None,
        kind: AliasKind = AliasKind.INPLACE,
    ) -> BufferView:
        try:
            source_view = self._views[source]
        except KeyError as error:
            raise IRVerificationError(
                f"Alias {logical_id!r} references unknown source {source!r}."
            ) from error
        size = source_view.nbytes - byte_offset if nbytes is None else int(nbytes)
        if byte_offset < 0 or size < 0 or byte_offset + size > source_view.nbytes:
            raise IRVerificationError(
                f"Alias {logical_id!r} exceeds source {source!r} byte range."
            )
        return self._insert(BufferView(
            logical_id,
            source_view.mem_span.subspan(byte_offset, size),
            AliasKind(kind),
            source,
        ))

    def view(self, logical_id: str) -> BufferView:
        try:
            return self._views[logical_id]
        except KeyError as error:
            raise IRVerificationError(f"Unknown alias-analysis value {logical_id!r}.") from error

    def replace_span(self, logical_id: str, mem_span: MemSpan) -> BufferView:
        """Rebind a known logical view after authoritative storage promotion."""

        current = self.view(logical_id)
        replacement = BufferView(
            current.logical_id,
            mem_span,
            current.kind,
            current.source,
        )
        self._views[logical_id] = replacement
        return replacement

    def may_alias(self, lhs: str, rhs: str) -> bool:
        return self.view(lhs).mem_span.may_alias(self.view(rhs).mem_span)

    def must_alias(self, lhs: str, rhs: str) -> bool:
        return self.view(lhs).mem_span.must_alias(self.view(rhs).mem_span)

    def groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        values: dict[str, list[str]] = {}
        for view in self._views.values():
            values.setdefault(view.physical_id, []).append(view.logical_id)
        return tuple(
            (physical_id, tuple(logical_ids))
            for physical_id, logical_ids in sorted(values.items())
        )

    def _insert(self, view: BufferView) -> BufferView:
        if not view.logical_id or not view.physical_id or view.nbytes < 0:
            raise IRVerificationError("Alias-analysis values require valid ids and byte bounds.")
        if view.logical_id in self._views:
            raise IRVerificationError(f"Duplicate alias-analysis value {view.logical_id!r}.")
        self._views[view.logical_id] = view
        return view


PhysicalView = BufferView


__all__ = ["AliasAnalysis", "AliasKind", "BufferView", "PhysicalView"]
