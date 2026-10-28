# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Preserve verified pool origins across reusable kernel-function boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from math import gcd

from triton.flagmega.codegen.triton.physical_access import emit_triton_scalar_type
from triton.flagmega.errors import CodegenError

OFFSET_TABLE = "_flagmega_function_offsets"
CALL_RECORD = "_flagmega_call_offsets"
ROW_INDEX = "_flagmega_bank_row_index"
_POOLS = {"rdata", "workspace"}


def bank_row_strides(module, plan):
    """Dynamic bank rows have a stride; static grouped weights have none.

    ``rdata_group`` also describes ordinary unrolled checkpoint tensors.
    Their call records already hold absolute row addresses, so grouping
    alone must not introduce another runtime row displacement.
    """
    groups = {
        f"{node.id}.{field.name}"
        for node in module.nodes if node.op in {"builtin.parameter_bank", "tir.parameter_bank"}
        for field in node.attrs["schema"].fields
    }
    result = {}
    for buffer in plan.buffers:
        if getattr(buffer, "rdata_group", None) not in groups:
            continue
        span = buffer.mem_span
        size = getattr(getattr(span, "size", None), "fixed_value", None)
        if size is None:
            size = getattr(getattr(span, "size", None), "value", None)
        if isinstance(size, int) and size > 0:
            result[str(getattr(span.buffer, "id", span.buffer))] = size
    return result


def plan_function_pool_abis(bindings: Mapping[str, Mapping],
                            row_strides: Mapping[str, int] | None = None,
                            entry: str | None = None) -> tuple[dict, list]:
    """Prove each physical argument's origin at every closed direct call.

    A function-relative pool frame is an address within its caller's pool,
    not a new allocation. Its frame displacement belongs in the same record
    as pooled tensor views. External allocations and dynamic views remain
    explicit arguments because these contracts do not prove their origins.

    The entry is the one caller whose pool parameters are the kernel's own
    arguments rather than addresses within an enclosing pool, so a callee it
    invokes directly cannot resolve a frame against them: folding the formal
    away there would emit pool arithmetic over a plain kernel argument.
    """
    calls: dict[str, list[tuple[str, Mapping]]] = {}
    for caller, binding in bindings.items():
        for event in binding["call_abi"]["events"]:
            if event["kind"] == "function_call":
                calls.setdefault(str(event["callee"]), []).append((caller, event))
    plans = {}
    values = []
    for name, rows in calls.items():
        if name not in bindings:
            continue
        callee = bindings[name]
        candidates = []
        for argument in (*callee["arguments"], *callee["pools"]):
            origins = [_origin(bindings[caller], argument, event, row_strides)
                       for caller, event in rows]
            if any(origin is None for origin in origins):
                continue
            if len({(origin[0], origin[2]) for origin in origins}) != 1:
                continue
            candidates.append((str(argument["name"]), origins[0][0], origins[0][2],
                               [origin[3] for origin in origins],
                               origins[0][4], gcd(*(origin[5] for origin in origins)),
                               argument.get("runtime_value_kind", "pointer")))
        roots = sorted({candidate[1] for candidate in candidates})
        # A dynamic bank-row displacement is required address semantics,
        # including when pooling does not reduce the signature's size.
        dynamic_rows = any(candidate[4] for candidate in candidates)
        if len(candidates) < 2 and not dynamic_rows:
            continue
        fields = []
        groups = []
        for formal, storage, dtype, offsets, row_stride, alignment, value_kind in candidates:
            field = {"formal": formal, "storage": storage, "scalar_dtype": dtype,
                     "alignment_bytes": gcd(alignment, row_stride) if row_stride else alignment}
            if value_kind == "scalar":
                field["value_kind"] = "scalar"
            if row_stride:
                # A parameter-bank row: the callee is invoked once per
                # loop trip against one binding, so its address is the
                # group base plus the trip index times this stride.
                field["row_stride_bytes"] = int(row_stride)
            if len(set(offsets)) == 1:
                field["byte_offset"] = offsets[0]
            else:
                match = None
                for column, (root, anchor) in enumerate(groups):
                    deltas = {a - b for a, b in zip(offsets, anchor, strict=True)}
                    if root == storage and len(deltas) == 1:
                        delta = deltas.pop()
                        if -(2**63) <= delta < 2**63:
                            match = column, delta
                            break
                if match is None:
                    match = len(groups), 0
                    groups.append((storage, offsets))
                field.update(record_column=match[0], byte_offset=match[1])
            fields.append(field)
        if len(candidates) <= len(roots) + bool(groups) and not dynamic_rows:
            continue
        start = len(values)
        width = len(groups)
        for field in fields:
            if "record_column" not in field:
                continue
            offsets = groups[field["record_column"]][1]
            stride = offsets[1] - offsets[0]
            if all(value == offsets[0] + index * stride
                   for index, value in enumerate(offsets)):
                field["record_affine"] = {
                    "record_start": start, "record_width": width,
                    "base": offsets[0], "stride": stride,
                    "records": len(offsets),
                }
        for row in range(len(rows)):
            values.extend(anchor[row] for _, anchor in groups)
        plans[name] = {
            "schema": "flagmega.function-pool-abi/v2",
            "bindings": fields,
            "pools": [{"storage": storage, "parameter": pool_parameter(storage)} for storage in roots],
            "record_width": width,
            "calls": [{"caller": caller, "call": str(event["call"]),
                       "record_offset": start + row * width}
                      for row, (caller, event) in enumerate(rows)],
        }
    tables = [{"name": OFFSET_TABLE, "dtype": "int64", "values": values}] if values else []
    return plans, tables


def _origin(caller: Mapping, argument: Mapping, event: Mapping,
            row_strides: Mapping[str, int] | None = None):
    pools = {str(pool["storage"]): pool for pool in caller["pools"]}
    if "buffer" in argument:
        edge = next((edge for edge in (*event["arguments"], *event["results"])
                     if str(edge["formal"]) == str(argument["buffer"])), None)
        if edge is None or edge.get("actual_runtime_value_kind") != "pointer":
            return None
        abi = edge["actual_abi"]
        storage = str(abi.get("storage"))
        pool = pools.get(storage)
        if (storage not in _POOLS or pool is None or not abi.get("pooled", False)
                or str(edge["actual_runtime_argument"]) != str(pool["name"])
                or int(abi.get("pool_scope_stride_bytes", 0))
                or "view_byte_offset" in abi):
            return None
        dtype = str(abi["scalar_dtype"])
        offset = int(abi["pool_byte_offset"])
        itemsize = int(abi["scalar_itemsize"])
        if itemsize <= 0 or offset % itemsize:
            raise CodegenError("Function pool ABI requires scalar-aligned byte offsets.")
        stride = int((row_strides or {}).get(str(abi.get("physical_buffer")), 0))
        alignment = abi.get("alignment_bytes", 1)
    else:
        storage = str(argument["storage"])
        pool = pools.get(storage)
        if storage not in _POOLS or pool is None or int(pool.get("scope_count", 1)) != 1:
            return None
        frame = next((frame for frame in event["memory_pools"]
                      if str(frame["memory_space"]) == storage), None)
        dtype = "uint8"
        offset = int(frame["offset"]) if frame is not None else 0
        # A pool frame is not a bank row, so it has no per-row stride.
        stride = 0
        # Old records contain no frame-alignment proof. Do not infer one
        # from an offset or borrow it from a neighboring tensor.
        alignment = frame.get("alignment_bytes", 1) if frame is not None else 1
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise CodegenError("Function pool ABI alignment must be a positive power of two.")
    if not 0 <= offset < 2**63:
        raise CodegenError("Function pool ABI displacement does not fit signed i64.")
    return storage, str(pool["name"]), dtype, offset, stride, alignment


def pool_parameter(storage: str) -> str:
    return f"_flagmega_call_pool_{storage}"


def function_pool_aliases(plan: Mapping | None) -> dict[str, str]:
    if plan is None:
        return {}
    aliases = {}
    for field in plan["bindings"]:
        offset = str(field["byte_offset"])
        if "record_column" in field:
            # The module owns an immutable offset table. Ordinary loads allow
            # reuse within an invocation while retaining its dynamic record index.
            affine = field.get("record_affine")
            if affine is None:
                offset = (f"tl.load({OFFSET_TABLE}.to(tl.pointer_type(tl.int64)) + "
                          f"{CALL_RECORD} + {field['record_column']})")
            else:
                if any(type(affine[name]) is not int for name in
                       ("record_start", "record_width", "base", "stride", "records")):
                    raise CodegenError("Affine record proof requires integer parameters")
                if affine["record_width"] != plan["record_width"] or affine["records"] != len(plan["calls"]):
                    raise CodegenError("Affine record proof must match its complete call domain")
                if any(call["record_offset"] != affine["record_start"] + index * affine["record_width"]
                       for index, call in enumerate(plan["calls"])):
                    raise CodegenError("Affine record proof requires contiguous scalar records")
                index = (f"((tl.cast({CALL_RECORD}, tl.int64) - "
                         f"{affine['record_start']}) // {affine['record_width']})")
                offset = f"({affine['base']} + {index} * {affine['stride']})"
            if field["byte_offset"]:
                offset = f"({offset} + {field['byte_offset']})"
        if field.get("row_stride_bytes"):
            # The planner binds a bank row to its group's index-0 member and
            # records the group identity so codegen can reach the other rows;
            # this is where the trip's row index becomes an address.
            stride = int(field["row_stride_bytes"])
            # Static calls may specialize the index to a Python integer.
            # tl.cast normalizes constants and runtime scalars alike before
            # multiplying the byte stride in signed 64-bit arithmetic.
            displacement = f"tl.cast({ROW_INDEX}, tl.int64) * {stride}"
            offset = (displacement if offset == "0"
                      else f"({offset} + {displacement})")
        base = f"{pool_parameter(field['storage'])}.to(tl.pointer_type(tl.uint8))"
        pointer = base if offset == "0" else f"({base} + {offset})"
        alignment = int(field.get("alignment_bytes", 1))
        if alignment > 1:
            pointer = f"tl.multiple_of({pointer}, {alignment})"
        alias = (
            f"{pointer}.to(tl.pointer_type({emit_triton_scalar_type(str(field['scalar_dtype']))}))"
        )
        # A pooled address can back a by-value scalar formal. Preserve that
        # formal's value ABI when folding its physical allocation into a pool.
        aliases[str(field["formal"])] = f"tl.load({alias})" if field.get("value_kind") == "scalar" else alias
    return aliases


def function_pool_signature(signature, plan: Mapping | None, *, has_tables: bool):
    removed = set(function_pool_aliases(plan))
    retained = [value for value in signature if value not in removed]
    if plan is not None and any(
            field.get("row_stride_bytes") for field in plan["bindings"]):
        retained.append(ROW_INDEX)
    if has_tables:
        retained.append(OFFSET_TABLE)
    if plan is not None:
        retained.extend(pool["parameter"] for pool in plan["pools"])
        if plan["record_width"]:
            retained.append(CALL_RECORD)
    return retained


def function_pool_actuals(caller, call, plan: Mapping | None, actuals: Mapping[str, str], *,
                          has_tables: bool, row_index: str | None = None):
    result = []
    if plan is not None and any(
            field.get("row_stride_bytes") for field in plan["bindings"]):
        # One call site serves every trip, so the row this invocation reads is
        # only known from the loop's induction variable.
        result.append(row_index if row_index is not None else "0")
    if has_tables:
        result.append(actuals.get(OFFSET_TABLE, OFFSET_TABLE))
    if plan is None:
        return result
    pools = {str(pool["storage"]): str(pool["name"]) for pool in caller["pools"]}
    result.extend(actuals[pools[str(pool["storage"])]] for pool in plan["pools"])
    if plan["record_width"]:
        row = next(row for row in plan["calls"]
                   if row["caller"] == caller["function"] and row["call"] == call.call_id)
        displacement = str(row["record_offset"])
        if row_index is not None:
            # A loop body has one call site but one record per iteration, so
            # stride the scalar record displacement by the runtime row index.
            displacement = f"{displacement} + ({row_index}) * {plan['record_width']}"
        # Pass a tensor scalar to avoid constexpr specialization for each
        # static call. A pointer here makes LLVM hoist every row's address
        # into the entry and keep it live across the whole persistent graph.
        result.append(f"tl.cast({displacement}, tl.int64)")
    return result
