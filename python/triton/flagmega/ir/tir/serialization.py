# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Deterministic data encoding for TIR checkpoints and semantic hashes."""

from __future__ import annotations

from dataclasses import fields
from enum import Enum
from typing import Mapping

from triton.flagmega.ir.bufferization import MemSpan, PhysicalBuffer
from triton.flagmega.ir.dim_expr import Dimension
from triton.flagmega.ir.model import IRType, type_from_data
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.tir.base import TIRNode, node_type
from triton.flagmega.ir.tir.buffer import DistributedBufferStorageKind
from triton.flagmega.ir.tir.barrier import BarrierScope
from triton.flagmega.ir.tir.for_loop import LoopMode, LoopPartition
from triton.flagmega.ir.tir.memory_pool_frame import MemoryPoolFrame
from triton.flagmega.ir.tir.prim_function import PrimParameterRole
from triton.flagmega.ir.tir.workspace_requirement import WorkspaceLifetime
from triton.flagmega.ir.types import (
    DType,
    MaskVectorType,
    PointerType,
    VectorType,
    data_type_from_data,
    data_type_to_data,
)


_ENUMS = {
    "LoopMode": LoopMode,
    "LoopPartition": LoopPartition,
    "PrimParameterRole": PrimParameterRole,
    "DistributedBufferStorageKind": DistributedBufferStorageKind,
    "WorkspaceLifetime": WorkspaceLifetime,
    "BarrierScope": BarrierScope,
}


def tir_to_data(node: TIRNode) -> dict[str, object]:
    result = {
        "kind": node.kind,
        **{field.name: _encode(getattr(node, field.name)) for field in fields(node)},
    }
    if (
        node.kind == "kernel_dispatch"
        and getattr(node, "inplace_alias_candidates", None) is None
    ):
        # Preserve semantic hashes of editable checkpoints written before the
        # named TIR alias contract existed. Explicit ``()`` remains encoded
        # and means the producer proved that no alias opportunity exists.
        result.pop("inplace_alias_candidates", None)
    if (
        node.kind == "buffer"
        and getattr(node, "distributed_backing_type", None) is None
    ):
        result.pop("distributed_backing_type", None)
    return result


def tir_from_data(data: Mapping[str, object]) -> TIRNode:
    try:
        cls = node_type(str(data["kind"]))
    except (KeyError, ValueError) as error:
        raise ValueError(f"Unknown TIR node kind {data.get('kind')!r}.") from error
    values = {
        field.name: _decode(data[field.name])
        for field in fields(cls)
        if field.name in data
    }
    if str(data.get("kind")) == "kernel_dispatch" and "semantic_candidate" not in data:
        for legacy_name in ("candidate", "parameters", "facts"):
            if legacy_name in data:
                values[legacy_name] = _decode(data[legacy_name])
    if str(data.get("kind")) == "prim_function_call" and "memory_pools" not in data:
        workspace_offset = int(data.get("workspace_offset", 0))
        workspace_nbytes = int(data.get("workspace_nbytes", 0))
        values["memory_pools"] = (
            MemoryPoolFrame(
                "workspace", None, workspace_offset, workspace_nbytes
            ),
        ) if workspace_offset or workspace_nbytes else ()
    return cls(**values)


def _encode(value: object) -> object:
    if isinstance(value, TIRNode):
        return tir_to_data(value)
    if isinstance(value, IRType):
        return {"$ir_type": value.to_data()}
    if isinstance(value, (DType, VectorType, PointerType, MaskVectorType)):
        return {"$data_type": data_type_to_data(value)}
    if isinstance(value, Dimension):
        return {"$dimension": value.to_data()}
    if isinstance(value, MemSpan):
        return {
            "$mem_span": {
                "buffer": value.buffer.to_data(),
                "start": value.start.to_data(),
                "size": value.size.to_data(),
            }
        }
    if isinstance(value, PhysicalBuffer):
        return {"$physical_buffer": value.to_data()}
    if isinstance(value, MemoryEffect):
        return {"$memory_effect": value.to_data()}
    if isinstance(value, Enum):
        return {"$enum": type(value).__name__, "value": value.value}
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if isinstance(value, list):
        return {"$list": [_encode(item) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported TIR serialization value {type(value).__name__}.")


def _decode(value: object) -> object:
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if not isinstance(value, Mapping):
        return value
    if "$ir_type" in value:
        return type_from_data(value["$ir_type"])
    if "$data_type" in value:
        return data_type_from_data(value["$data_type"])
    if "$dimension" in value:
        return Dimension.from_data(value["$dimension"])
    if "$physical_buffer" in value:
        return PhysicalBuffer.from_data(value["$physical_buffer"])
    if "$memory_effect" in value:
        return MemoryEffect.from_data(value["$memory_effect"])
    if "$mem_span" in value:
        encoded = value["$mem_span"]
        buffer = PhysicalBuffer.from_data(encoded["buffer"])
        return MemSpan(
            buffer,
            Dimension.from_data(encoded["start"]),
            Dimension.from_data(encoded["size"]),
        )
    if "$enum" in value:
        return _ENUMS[str(value["$enum"])](value["value"])
    if "$tuple" in value:
        return tuple(_decode(item) for item in value["$tuple"])
    if "$list" in value:
        return [_decode(item) for item in value["$list"]]
    if "kind" in value:
        return tir_from_data(value)
    return {str(key): _decode(item) for key, item in value.items()}


__all__ = ["tir_from_data", "tir_to_data"]
