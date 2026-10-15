# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Preserve verified pool origins across reusable kernel-function boundaries."""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.codegen.triton.physical_access import emit_triton_scalar_type
from triton.flagmega.errors import CodegenError

OFFSET_TABLE = "_flagmega_function_offsets"
CALL_RECORD = "_flagmega_call_offsets"
ROW_INDEX = "_flagmega_bank_row_index"
_POOLS = {"rdata", "workspace"}


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
                               origins[0][4] if len(origins[0]) > 4 else 0))
        roots = sorted({candidate[1] for candidate in candidates})
        if len(candidates) < 2:
            continue
        fields = []
        groups = []
        for formal, storage, dtype, offsets, row_stride in candidates:
            field = {"formal": formal, "storage": storage, "scalar_dtype": dtype}
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
        if len(candidates) <= len(roots) + bool(groups):
            continue
        start = len(values)
        width = len(groups)
        for row in range(len(rows)):
            values.extend(anchor[row] for _, anchor in groups)
        plans[name] = {
            "schema": "flagmega.function-pool-abi/v1",
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
    if not 0 <= offset < 2**63:
        raise CodegenError("Function pool ABI displacement does not fit signed i64.")
    return storage, str(pool["name"]), dtype, offset, stride


def pool_parameter(storage: str) -> str:
    return f"_flagmega_call_pool_{storage}"


def function_pool_aliases(plan: Mapping | None) -> dict[str, str]:
    if plan is None:
        return {}
    aliases = {}
    for field in plan["bindings"]:
        offset = str(field["byte_offset"])
        if "record_column" in field:
            offset = f"tl.load({CALL_RECORD} + {field['record_column']}, volatile=True)"
            if field["byte_offset"]:
                offset = f"({offset} + {field['byte_offset']})"
        elif field.get("row_stride_bytes"):
            # The planner binds a bank row to its group's index-0 member and
            # records the group identity so codegen can reach the other rows;
            # this is where the trip's row index becomes an address.
            stride = int(field["row_stride_bytes"])
            displacement = f"({ROW_INDEX}).to(tl.int64) * {stride}"
            offset = (displacement if offset == "0"
                      else f"({offset} + {displacement})")
        base = f"{pool_parameter(field['storage'])}.to(tl.pointer_type(tl.uint8))"
        pointer = base if offset == "0" else f"({base} + {offset})"
        aliases[str(field["formal"])] = (
            f"{pointer}.to(tl.pointer_type({emit_triton_scalar_type(str(field['scalar_dtype']))}))"
        )
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
        table = f"{actuals.get(OFFSET_TABLE, OFFSET_TABLE)}.to(tl.pointer_type(tl.int64))"
        displacement = str(row["record_offset"])
        if row_index is not None:
            # A loop body has one call site but one record per iteration, so
            # stride the record pointer by the runtime row index.
            displacement = f"{displacement} + ({row_index}) * {plan['record_width']}"
        result.append(f"{table} + {displacement}")
    return result
