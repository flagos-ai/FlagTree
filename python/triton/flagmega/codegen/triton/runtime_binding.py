# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bind bufferized function storage to stable Triton pointer arguments."""

from __future__ import annotations

from copy import deepcopy
import keyword
import re

from triton.flagmega.codegen.triton.call_abi import describe_function_call_abi
from triton.flagmega.codegen.triton.dimension_expression import emit_dimension
from triton.flagmega.ir.dim_expr import Dimension
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import IRModule, verify_buffer_plan
from triton.flagmega.codegen.triton.pool_abi import (
    is_runtime_pool_space,
    memory_scope_count,
)


FUNCTION_RUNTIME_BINDING_SCHEMA = "flagmega.function-runtime-binding/v1"


def describe_function_runtime_binding(
    module: IRModule,
    *,
    function_name: str | None = None,
    call_abi: dict[str, object] | None = None,
) -> dict[str, object]:
    """Attach every call buffer to an external or pooled pointer root.

    The result is independent of model roles.  External function values become
    explicit pointer arguments, readonly data uses its module pool, and
    SAT-allocated temporaries use target-declared function-relative pools.
    Views and aliases resolve by physical-buffer identity, not SSA spelling.
    """

    plan = verify_buffer_plan(module)
    name = module.entry if function_name is None else str(function_name)
    call_abi = deepcopy(
        describe_function_call_abi(module, function_name=name)
        if call_abi is None
        else call_abi
    )
    try:
        function_plan = plan.function_map[name]
    except KeyError as error:
        raise CodegenError(
            f"Cannot bind runtime arguments for unknown function {name!r}."
        ) from error

    argument_by_physical: dict[str, str] = {}
    arguments: list[dict[str, object]] = []
    value_bindings: list[dict[str, object]] = []
    used_names: set[str] = set()

    def add_external(
        value: str,
        buffer_ids: tuple[str, ...],
        role: str,
    ) -> None:
        for index, buffer_id in enumerate(buffer_ids):
            buffer = plan.buffer_map[buffer_id]
            if _is_runtime_pool_storage(plan, buffer.storage):
                value_bindings.append({
                    "role": role,
                    "value": value,
                    "buffer": buffer_id,
                    "storage": buffer.storage,
                    "runtime_argument": None,
                })
                continue
            existing = argument_by_physical.get(buffer.physical_id)
            if existing is not None:
                value_bindings.append({
                    "role": role,
                    "value": value,
                    "buffer": buffer_id,
                    "storage": buffer.storage,
                    "runtime_argument": existing,
                    "alias": True,
                })
                continue
            suffix = buffer.field or (str(index) if len(buffer_ids) > 1 else "")
            argument = _unique_argument_name(
                "_".join(part for part in (value, suffix) if part),
                used_names,
            )
            argument_by_physical[buffer.physical_id] = argument
            arguments.append({
                "name":
                argument,
                "role":
                role,
                "value":
                value,
                "buffer":
                buffer_id,
                **({"runtime_value_kind": "scalar",
                    "scalar_dtype": buffer.dtype.value} if buffer.storage == "scalar" else {}),
            })
            value_bindings.append({
                "role": role,
                "value": value,
                "buffer": buffer_id,
                "storage": buffer.storage,
                "runtime_argument": argument,
                "alias": False,
            })

    for value, buffer_ids in function_plan.parameters:
        add_external(value, buffer_ids, "parameter")
    for value, buffer_ids in function_plan.outputs:
        add_external(value, buffer_ids, "result")

    used_storages = {
        str(value["storage"])
        for value in value_bindings
        if _is_runtime_pool_storage(plan, str(value["storage"]))
    }
    for call in call_abi["kernel_calls"]:
        for parameter in (
            *call["inputs"],
            *call["outputs"],
            *call["workspaces"],
        ):
            for binding in parameter["buffers"]:
                used_storages.add(str(binding["abi"]["storage"]))
    for event in call_abi["events"]:
        if event["kind"] != "function_call":
            continue
        for edge in (*event["arguments"], *event["results"]):
            used_storages.add(str(edge["actual_abi"]["storage"]))
        used_storages.update(
            str(frame["memory_space"])
            for frame in event["memory_pools"]
        )

    pools: list[dict[str, object]] = []
    ordered_spaces = sorted(
        enumerate(plan.memory_spaces),
        key=lambda value: (
            0 if value[1].allocation_scope.value == "module" else 1,
            value[0],
        ),
    )
    for _, space in ordered_spaces:
        storage = space.name
        if storage not in used_storages or not _is_runtime_pool_storage(plan, storage):
            continue
        scope_nbytes = (
            plan.rdata_bytes
            if storage == "rdata"
            else plan.function_memory_space_bytes(name, storage)
        )
        physical_scope_count = memory_scope_count(module, space.sharing_scope)
        scope_count = physical_scope_count if name == module.entry else 1
        pool_name = _unique_argument_name(storage, used_names)
        pool = {
            "name": pool_name,
            "storage": storage,
            "nbytes": scope_nbytes * scope_count,
        }
        if physical_scope_count != 1:
            pool.update({
                "scope": space.sharing_scope.value,
                "scope_nbytes": scope_nbytes,
                "scope_count": scope_count,
                # This is an ABI strategy, not a generated-source variable.
                # Codegen lowers it through a noinline scope-base helper so the
                # physical program id cannot become a common SSA root for all
                # logical shard coordinates in a large inlined call graph.
                "scope_index": "program_id_x",
                "scope_local": name != module.entry,
            })
        pools.append(pool)
    pool_by_storage = {
        str(value["storage"]): str(value["name"]) for value in pools
    }
    for value in value_bindings:
        if value["runtime_argument"] is None:
            value["runtime_argument"] = pool_by_storage[str(value["storage"])]
            value["alias"] = True

    for call in call_abi["kernel_calls"]:
        for parameter in (
            *call["inputs"],
            *call["outputs"],
            *call["workspaces"],
        ):
            for binding in parameter["buffers"]:
                if name != module.entry:
                    _make_pool_abi_scope_local(plan, binding["abi"])
                abi = binding["abi"]
                argument, value_kind = _resolve_runtime_root(
                    module,
                    plan,
                    abi,
                    str(binding["actual"]),
                    pool_by_storage,
                    argument_by_physical,
                )
                if argument is None:
                    raise CodegenError(
                        f"Function @{name} call {call['call']!r} buffer "
                        f"{binding['actual']!r} has no runtime storage root."
                    )
                binding["runtime_argument"] = argument
                binding["runtime_value_kind"] = value_kind
                address_arguments = _bind_view_offset(module, plan, abi, pool_by_storage, argument_by_physical)
                if address_arguments:
                    binding["address_arguments"] = address_arguments

    for event in call_abi["events"]:
        if event["kind"] != "function_call":
            continue
        for edge in (*event["arguments"], *event["results"]):
            if name != module.entry:
                _make_pool_abi_scope_local(plan, edge["actual_abi"])
                _make_pool_abi_scope_local(plan, edge["formal_abi"])
            actual_abi = edge["actual_abi"]
            argument, value_kind = _resolve_runtime_root(
                module,
                plan,
                actual_abi,
                str(edge["actual"]),
                pool_by_storage,
                argument_by_physical,
            )
            if argument is None:
                raise CodegenError(
                    f"Function @{name} nested call {event['call']!r} actual "
                    f"buffer {edge['actual']!r} has no runtime storage root."
                )
            edge["actual_runtime_argument"] = argument
            edge["actual_runtime_value_kind"] = value_kind
            _bind_view_offset(module, plan, actual_abi, pool_by_storage, argument_by_physical)
        for frame in event["memory_pools"]:
            frame["runtime_argument"] = pool_by_storage[
                str(frame["memory_space"])
            ]

    return {
        "schema": FUNCTION_RUNTIME_BINDING_SCHEMA,
        "function": name,
        "arguments": arguments,
        "values": value_bindings,
        "pools": pools,
        "signature": [
            value["name"]
            for value in (*arguments, *pools)
        ],
        "call_abi": call_abi,
    }


def _make_pool_abi_scope_local(plan, abi: dict[str, object]) -> None:
    space = plan.memory_space_map.get(str(abi.get("memory_space", "")))
    if space is None or space.sharing_scope.value != "block":
        return
    abi.pop("pool_scope_stride_bytes", None)
    abi.pop("pool_scope_index", None)
    abi["pool_scope_count"] = 1
    abi["pool_scope_local"] = True


def _bind_view_offset(module, plan, abi, pools, arguments):
    encoded = abi.get("view_byte_offset")
    if encoded is None:
        return ()
    expressions = {}
    dependencies = []
    for symbol, buffer_id in abi.get("offset_bindings", {}).items():
        buffer = plan.buffer_map[buffer_id]
        if buffer.storage != "scalar":
            raise CodegenError("A view offset must reference scalar SSA storage.")
        argument, kind = _resolve_runtime_root(module, plan,
                                               {"storage": buffer.storage, "physical_buffer": buffer.physical_id},
                                               buffer_id, pools, arguments)
        if argument is None:
            raise CodegenError(f"View offset {symbol!r} has no runtime scalar binding.")
        # Promote before multiplication: a valid byte span can exceed int32
        # even when its runtime index and shape extents fit in int32.
        expressions[symbol] = (f"tl.full((), {argument}, tl.int64)"
                               if kind == "immediate" else f"({argument}).to(tl.int64)")
        if kind != "immediate" and argument not in dependencies:
            dependencies.append(argument)
    abi["view_byte_offset_expression"] = emit_dimension(Dimension.from_data(encoded), symbols=expressions)
    return tuple(dependencies)


def _is_runtime_pool_storage(plan, storage: str) -> bool:
    space = plan.memory_space_map.get(storage)
    return bool(
        space is not None
        and is_runtime_pool_space(space)
    )


def _resolve_runtime_root(
    module: IRModule,
    plan,
    abi: dict[str, object],
    buffer_id: str,
    pool_by_storage: dict[str, str],
    argument_by_physical: dict[str, str],
) -> tuple[str | None, str]:
    storage = str(abi["storage"])
    if storage in pool_by_storage:
        return pool_by_storage[storage], "pointer"
    argument = argument_by_physical.get(str(abi["physical_buffer"]))
    if argument is not None:
        return argument, "pointer" if storage != "scalar" else "scalar"
    if storage == "scalar":
        return _scalar_immediate(module, plan, buffer_id), "immediate"
    return None, "pointer"


def _scalar_immediate(module: IRModule, plan, buffer_id: str) -> str:
    seen: set[str] = set()
    current = buffer_id
    while current not in seen:
        seen.add(current)
        try:
            buffer = plan.buffer_map[current]
        except KeyError as error:
            raise CodegenError(
                f"Scalar runtime binding references unknown buffer {current!r}."
            ) from error
        node = module.node_map.get(str(buffer.source_node or current))
        if node is not None and node.op in {
            "builtin.scalar_const",
            "tir.scalar_const",
        }:
            return repr(node.attrs["value"])
        if buffer.alias_of is None:
            break
        current = buffer.alias_of
    raise CodegenError(
        f"Scalar buffer {buffer_id!r} is neither a function argument nor a "
        "compile-time scalar constant."
    )


def _unique_argument_name(value: str, used: set[str]) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not stem:
        stem = "argument"
    if stem[0].isdigit() or keyword.iskeyword(stem):
        stem = f"argument_{stem}"
    result = stem
    ordinal = 1
    while result in used:
        result = f"{stem}_{ordinal}"
        ordinal += 1
    used.add(result)
    return result


__all__ = [
    "FUNCTION_RUNTIME_BINDING_SCHEMA",
    "describe_function_runtime_binding",
]
