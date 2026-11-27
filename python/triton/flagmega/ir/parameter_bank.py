# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lazy, schema-checked parameter banks for structurally repeated layers."""

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import IRType, RefType, TupleType, type_from_data


@dataclass(frozen=True)
class BankFieldProgram:
    """One pure offline transform applied per row at the bank boundary.

    ``source`` names the placeholder standing for the row's raw stored value,
    so the same closure applies to every row. Keeping the program in the
    schema is what lets a selected physical layout stay offline work: a loop
    body indexes the bank by its induction value, so an equivalent transform
    copied into the body would become per-iteration runtime work instead.
    """

    sources: tuple
    nodes: tuple
    output: str

    @property
    def node_map(self):
        return {node.id: node for node in self.nodes}

    @property
    def source(self):
        """The single stored source, for the common one-input transform."""
        if len(self.sources) != 1:
            raise IRSchemaError("This BankFieldProgram consumes more than one stored source.")
        return self.sources[0]

    def __post_init__(self):
        if isinstance(self.sources, str):
            object.__setattr__(self, "sources", (self.sources,))
        object.__setattr__(self, "sources", tuple(str(value) for value in self.sources))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        ids = {node.id for node in self.nodes}
        if not self.sources or len(set(self.sources)) != len(self.sources) or not self.nodes \
                or self.output not in ids or ids.intersection(self.sources):
            raise IRSchemaError("BankFieldProgram requires free sources and a produced output.")
        if any(not node.effect.is_pure for node in self.nodes):
            raise IRSchemaError("BankFieldProgram transforms must be pure.")
        produced: set[str] = set(self.sources)
        for node in self.nodes:
            if any(value not in produced for value in node.inputs):
                raise IRSchemaError("BankFieldProgram nodes must be topological and closed.")
            produced.add(node.id)

    def to_data(self):
        return {"sources": list(self.sources), "nodes": [node.to_data() for node in self.nodes],
                "output": self.output}

    @classmethod
    def from_data(cls, data):
        from triton.flagmega.ir.model import Node

        # Accept the single-source spelling so earlier checkpoints still load.
        sources = data["sources"] if "sources" in data else (data["source"],)
        return cls(tuple(str(value) for value in sources),
                   tuple(Node.from_data(value) for value in data["nodes"]), str(data["output"]))


@dataclass(frozen=True)
class BankField:
    name: str
    type: IRType
    transform: str = "identity"
    program: BankFieldProgram | None = None
    source_type: IRType | None = None
    source_types: tuple | None = None

    def __post_init__(self):
        if self.source_type is not None and self.source_types is None:
            object.__setattr__(self, "source_types", (self.source_type,))
        if self.source_types is not None:
            object.__setattr__(self, "source_types", tuple(self.source_types))
            object.__setattr__(self, "source_type", self.source_types[0])
        if not self.name or not isinstance(self.type, IRType) or not self.transform:
            raise IRSchemaError("BankField requires a name, type and transform.")
        if self.program is not None:
            if not isinstance(self.program, BankFieldProgram) or self.source_types is None \
                    or any(not isinstance(value, IRType) for value in self.source_types) \
                    or len(self.source_types) != len(self.program.sources):
                raise IRSchemaError("A transformed BankField requires a program and one type per source.")
            if self.program.node_map[self.program.output].type != self.type:
                raise IRSchemaError("BankField type must match its program output type.")
        elif self.source_types is not None:
            raise IRSchemaError("BankField source types require a program.")

    @property
    def stored_type(self):
        """The type held in readonly storage, for a single-source field."""
        if self.source_types is None:
            return self.type
        if len(self.source_types) != 1:
            raise IRSchemaError(f"Bank field {self.name!r} stores more than one source tensor.")
        return self.source_types[0]

    @property
    def stored_types(self):
        """Every tensor held in readonly storage for one row of this field."""
        return (self.type,) if self.source_types is None else self.source_types

    def to_data(self):
        data = {"name": self.name, "type": self.type.to_data(), "transform": self.transform}
        if self.program is not None:
            data["program"] = self.program.to_data()
            data["source_types"] = [value.to_data() for value in self.source_types]
        return data

    @classmethod
    def from_data(cls, data):
        program = data.get("program")
        if program is None:
            return cls(str(data["name"]), type_from_data(data["type"]),
                       str(data.get("transform", "identity")))
        # Accept the single-type spelling so earlier checkpoints still load.
        raw = data["source_types"] if "source_types" in data else (data["source_type"],)
        return cls(str(data["name"]), type_from_data(data["type"]), str(data.get("transform", "identity")),
                   BankFieldProgram.from_data(program), None,
                   tuple(type_from_data(value) for value in raw))


@dataclass(frozen=True)
class ParameterBankSchema:
    name: str
    fields: tuple[BankField, ...]

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.name or not self.fields or any(not isinstance(field, BankField) for field in self.fields):
            raise IRSchemaError("ParameterBankSchema requires BankField entries.")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise IRSchemaError("ParameterBankSchema requires distinct nonempty fields.")

    @property
    def element_type(self):
        return TupleType(tuple(field.type for field in self.fields))

    @property
    def type(self):
        return RefType("parameter_bank", (("element", self.element_type),))

    def to_data(self):
        return {"name": self.name, "fields": [field.to_data() for field in self.fields]}

    @classmethod
    def from_data(cls, data):
        return cls(str(data["name"]), tuple(BankField.from_data(value) for value in data["fields"]))


__all__ = ["BankField", "BankFieldProgram", "ParameterBankSchema"]
