# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Serializable importer source locations carried through editable IR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from triton.flagmega.errors import ImporterError
from triton.flagmega.ir import IRModule, Node


SOURCE_LOCATION_SCHEMA = "flagmega.import-source/v1"


@dataclass(frozen=True)
class ImportSourceLocation:
    uri: str
    symbol: str
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.uri or not self.symbol:
            raise ImporterError("Import source locations require non-empty uri and symbol.")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SOURCE_LOCATION_SCHEMA,
            "uri": self.uri,
            "symbol": self.symbol,
            "revision": self.revision,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> ImportSourceLocation:
        if data.get("schema") != SOURCE_LOCATION_SCHEMA:
            raise ImporterError(
                f"Unsupported import source schema {data.get('schema')!r}."
            )
        revision = data.get("revision")
        return cls(
            str(data.get("uri", "")),
            str(data.get("symbol", "")),
            None if revision is None else str(revision),
        )


def attach_import_source_locations(
    module: IRModule,
    *,
    architecture: str,
    revision: str | None,
) -> IRModule:
    """Attach one canonical source location to every imported node."""

    if not architecture:
        raise ImporterError("Imported modules require a non-empty architecture source.")
    nodes = []
    for node in module.nodes:
        existing = node.metadata.get("source_location")
        if existing is not None:
            if not isinstance(existing, Mapping):
                raise ImporterError(
                    f"Node {node.id!r} has a malformed import source location."
                )
            location = ImportSourceLocation.from_data(existing)
        elif node.op == "builtin.weight":
            location = ImportSourceLocation(
                f"safetensors://{node.attrs['source']}",
                str(node.attrs["key"]),
                revision,
            )
        else:
            location = ImportSourceLocation(
                f"huggingface://{architecture}",
                node.id,
                revision,
            )
        nodes.append(replace(
            node,
            metadata={**dict(node.metadata), "source_location": location.to_data()},
        ))
    return replace(module, nodes=tuple(nodes))


def source_location_of(node: Node) -> ImportSourceLocation | None:
    value = node.metadata.get("source_location")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ImporterError(f"Node {node.id!r} has a malformed import source location.")
    return ImportSourceLocation.from_data(value)


__all__ = [
    "ImportSourceLocation",
    "SOURCE_LOCATION_SCHEMA",
    "attach_import_source_locations",
    "source_location_of",
]
