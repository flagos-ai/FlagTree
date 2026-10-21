# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""C++ device layouts over verified canonical readonly artifact bytes.

The artifact and source-layout programs retain their IR-defined canonical
representation. Only the C++ runtime's device pool uses this bijective byte
permutation, matching the explicit group strides in its generated kernels.
"""
from __future__ import annotations

from math import prod
from collections.abc import Mapping

from triton.flagmega.errors import ArtifactError, CodegenError
from triton.flagmega.ir import (
    ContiguousSplit, DistributedBufferStorageKind, block_owner_axes,
    block_owner_count, verify_buffer_plan,
)

READONLY_LAYOUT_SCHEMA = "flagmega.cpp-readonly-layout/v1"


def _readonly_roots(plan, buffer, seen=()):
    if buffer.storage == "rdata":
        return (buffer,)
    if buffer.storage != "parameter" or buffer.function in seen:
        return ()
    callers = tuple(c for c in plan.call_map.values() if c.callee == buffer.function)
    if not callers:
        return ()
    roots = []
    for call in callers:
        actual = plan.resolve_call_buffer(call.call, buffer.id)
        resolved = _readonly_roots(plan, actual, (*seen, buffer.function))
        if not resolved:
            return ()
        roots.extend(resolved)
    return tuple(roots)


def select_readonly_layouts(module, descriptor):
    """Prove shard-major K walks from the actual buffer/call ABI.

    A split K axis, noncanonical storage or externally supplied weight keeps
    its original address contract. Selection depends on geometry and storage,
    never a model/layer/parameter name.
    """
    plan = verify_buffer_plan(module)
    layouts = {}
    for call in descriptor.get("render_calls", ()):
        if call.get("family") == "dense_matmul_glu":
            formals = ("gate_weight", "up_weight")
        elif call.get("family") == "dense_matmul" and call.get("variant") == "packed_k_major_gemv_norm_stats":
            formals = ("rhs",)
        else:
            continue
        lane = int(call.get("weight_k_lane") or 0)
        group = int(call.get("local_n_capacity") or 0) * lane
        groups = int(call.get("local_k_capacity") or 0) // lane if lane else 0
        stride = int(call.get("weight_group_stride") or 0)
        if not groups or not group or group % 512 or stride % group:
            continue
        owners = stride // group
        if owners <= 1:
            continue
        # A permuted buffer cannot enter the scalar canonical-address walk.
        # Require the exact SME geometry before changing its device layout.
        warps = descriptor["block_threads"] // descriptor["warp_size"]
        slots = descriptor["sme_slots"]
        batch = slots // 2
        tiles = group // 512
        if (slots < 2 or not batch or descriptor["sme_pad"] != 0 or lane != 16
                or not call.get("weight_block_n") or call.get("weight_block_k") != lane):
            continue
        if call["family"] == "dense_matmul_glu":
            if (tiles > warps or group * slots * 4 > 114688
                    or groups <= slots or groups % batch):
                continue
        else:
            if (tiles >= warps or warps % tiles or warps * 512 * slots * 2 > 114688
                    or groups < 2 * (warps // tiles) * batch
                    or groups % ((warps // tiles) * batch)):
                continue
        selected = []
        for formal in formals:
            argument = next((item for item in call["inputs"] if item["formal"] == formal), None)
            if argument is None or len(argument.get("buffers", ())) != 1:
                break
            binding = argument["buffers"][0]
            abi = binding["abi"]
            if (abi.get("storage_kind") != "canonical_global"
                    or abi.get("scalar_dtype") != "bfloat16"
                    or abi["logical_shape"][0] != groups
                    or abi["local_capacity_shape"][0] != groups
                    or abi.get("owner_count") != owners
                    or prod(abi["logical_shape"]) * abi["scalar_lane_count"] != groups * stride):
                break
            # Owners must select consecutive N slices in row-major placement
            # order; a block-cyclic or permuted owner map is a different layout.
            distributed = plan.buffer_map[binding["actual"]].distributed_type
            splits = [(axis, stage) for axis, policy in enumerate(distributed.axis_policies)
                      for stage in getattr(policy, "stages", ())]
            if (len(splits) != 1 or splits[0][0] != 1
                    or not isinstance(splits[0][1].distribution, ContiguousSplit)
                    or tuple(splits[0][1].hierarchy_axes) != tuple(block_owner_axes(distributed))):
                break
            roots = _readonly_roots(plan, plan.buffer_map[binding["actual"]])
            if not roots:
                break
            selected.append(roots)
        if len(selected) != len(formals):
            continue
        for roots in selected:
            for buffer in roots:
                if (buffer.distributed_storage_kind is not DistributedBufferStorageKind.CANONICAL_GLOBAL
                        or buffer.nbytes != groups * stride * 2):
                    raise CodegenError("C++ shard-major readonly layout disagrees with its concrete call buffer.")
                record = {"buffer": buffer.id, "offset": buffer.offset,
                          "nbytes": buffer.nbytes, "shape": [groups, owners, group, 2],
                          "axes": [1, 0, 2, 3]}
                prior = layouts.setdefault(buffer.offset, record)
                if prior != record:
                    raise CodegenError("Conflicting C++ readonly layouts share one byte range.")
        call["cpp_shard_major_weight"] = True
        call["canonical_weight_group_stride"] = stride
        call["weight_group_stride"] = group
        call["weight_layout_k_groups"] = groups
    # Repacking one readonly root affects all of its users. Every weight
    # consumer must have the same explicit device-address contract; a second
    # canonical reader cannot continue to use the original K-group stride.
    for call in descriptor.get("render_calls", ()):
        weight_formals = (("gate_weight", "up_weight")
                          if call.get("family") == "dense_matmul_glu" else ("rhs",))
        for argument in call.get("inputs", ()):
            for binding in argument.get("buffers", ()):
                roots = _readonly_roots(plan, plan.buffer_map[binding["actual"]])
                if any(root.offset in layouts for root in roots) and (
                        not call.get("cpp_shard_major_weight")
                        or argument["formal"] not in weight_formals):
                    raise CodegenError("A C++ readonly layout has a consumer without its device-address contract.")
    return {"schema": READONLY_LAYOUT_SCHEMA,
            "buffers": sorted(layouts.values(), key=lambda item: item["offset"])}


def apply_readonly_layout(pool, contract, module):
    """Permute exact byte views once, before graph capture or any kernel use."""
    if not isinstance(contract, Mapping) or contract.get("schema") != READONLY_LAYOUT_SCHEMA:
        raise ArtifactError("Unknown C++ readonly device layout schema.")
    buffers = verify_buffer_plan(module).buffer_map
    records = contract.get("buffers")
    if not isinstance(records, (list, tuple)):
        raise ArtifactError("C++ readonly device layouts must be a list.")
    end = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactError("C++ readonly layouts must contain buffer records.")
        buffer = buffers.get(record.get("buffer"))
        shape, axes = record.get("shape"), record.get("axes")
        if (buffer is None or buffer.storage != "rdata"
                or buffer.distributed_storage_kind is not DistributedBufferStorageKind.CANONICAL_GLOBAL
                or record.get("offset") != buffer.offset or record.get("nbytes") != buffer.nbytes
                or not isinstance(shape, (list, tuple)) or len(shape) != 4
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or shape[0] != buffer.shape[0]
                or buffer.distributed_type is None
                or shape[1] != block_owner_count(buffer.distributed_type)
                or shape[3] != 2 or axes != [1, 0, 2, 3]
                or prod(shape) != buffer.nbytes or record["offset"] < end
                or record["offset"] + buffer.nbytes > pool.numel()):
            raise ArtifactError("C++ readonly layout does not match its canonical buffer byte range.")
        end = record["offset"] + buffer.nbytes
    for record in records:
        buffer = buffers[record["buffer"]]
        shape, axes = record["shape"], record["axes"]
        view = pool.narrow(0, buffer.offset, buffer.nbytes).view(*shape)
        # clone is necessary even when an owner dimension has extent one:
        # contiguous() may alias the input in that case.
        import torch
        reordered = view.permute(*axes).clone(memory_format=torch.contiguous_format)
        view.reshape(-1).copy_(reordered.reshape(-1))
    return pool
