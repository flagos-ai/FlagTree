# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prove packed-row ABIs for warp-owned COREX SME transport."""

from itertools import product
from math import prod

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.physical_access import (
    emit_buffer_pointer,
    emit_local_scalar_offset,
)
from triton.flagmega.ir import DistributedType, local_shard_descriptor, type_from_data


def attach_flat_sme_calls(encoded_calls, source_schedule):
    """Specialize only complete contiguous BF16 rows with proven origins.

    Reusable formals may declare alignment one even when all closed calls
    resolve to aligned pool views. Use the function-pool proof in that case;
    an unknown external pointer cannot borrow a neighboring tensor's proof.
    """
    definitions = {} if source_schedule is None else {
        item["function"]: item.get("function_pool_abi")
        for item in source_schedule.get("device_functions", ())
    }
    for name, calls in encoded_calls.items():
        plan = definitions.get(name)
        alignments = {} if plan is None else {
            field["formal"]: field["alignment_bytes"] for field in plan["bindings"]
        }
        for call in calls:
            if (call["family"] == "qkv_parallel_linear"
                    and call["variant"] == "packed_fused_gemv"):
                _attach_qkv(call, alignments)
                continue
            if (call["family"] == "dense_matmul"
                    and call["variant"] == "packed_k_major_gemv"):
                _attach_gemv(call, alignments)
                continue
            if (call["family"] == "dense_matmul"
                    and call["variant"] == "split_k_n_packed_k_major_gemv"):
                _attach_gemv(call, alignments, split_k=True)
                continue
            if (call["family"] == "dense_matmul"
                    and call["variant"] == "packed_k_major_gemv_norm_stats"):
                _attach_down(call, alignments)
                continue
            if (call["family"] != "dense_matmul_glu"
                    or call["variant"] != "packed_k_major_gemv"
                    or call.get("packed_layout") != "k_major_n8_k16"
                    or call["block_k"] != 32 or call["tile_n"] != 128):
                continue
            operands = {
                item["formal"]: item
                for group in ("inputs", "outputs")
                for parameter in call[group]
                for item in parameter["buffers"]
            }
            if not {"value", "gate_weight", "up_weight", "result"} <= operands.keys():
                continue
            source, gate, up, result = (operands[key] for key in
                                        ("value", "gate_weight", "up_weight", "result"))
            if any(item["abi"].get("scalar_dtype") != "bfloat16"
                   for item in (source, gate, up, result)):
                continue
            n, k = call["local_n_capacity"], call["local_k_capacity"]
            if not (32 <= n <= 512 and n % 32 == 0 and k >= 64 and k % 32 == 0):
                continue
            if not all(_full_contiguous(item["abi"]) for item in (source, gate, up, result)):
                continue
            if not all(_matching_origins(source, item, result) for item in (gate, up)):
                continue
            if any(prod(item["abi"]["local_capacity_shape"][:-1]) != 1
                   or item["abi"].get("scalar_lane_count", 1) != 1
                   or item["abi"]["scalar_storage_strides"][-1] != 1
                   for item in (source, result)):
                continue
            if (source["abi"]["local_capacity_shape"][-1] != k
                    or result["abi"]["local_capacity_shape"][-1] != n):
                continue
            weights = []
            for item in (gate, up):
                if not _packed_weight(item, n, k, alignments):
                    break
                weights.append(_owner_pointer(item))
            if len(weights) != 2 or gate["abi"]["scalar_storage_strides"] != up["abi"]["scalar_storage_strides"]:
                continue
            call["corex_flat_sme_glu"] = {
                "source": _owner_pointer(source), "gate_weight": weights[0],
                "up_weight": weights[1], "result": _owner_pointer(result),
                "weight_k_group_stride": gate["abi"]["scalar_storage_strides"][0],
            }


def _attach_qkv(call, alignments):
    """A fused RHS stores complete owner/K16/N rows in natural order.

    Source activity and output ownership still use the original call's
    expressions. Only the weight transport changes; every copied byte must
    belong to a complete, aligned RHS row, including pipeline lookahead.
    """
    if (call["block_k"] != 64 or call["tile_n"] < 128
            or call["n_lane"] not in (4, 8) or call["k_lane"] != 16
            or len(call["outputs"]) != 3):
        return
    outputs = call["outputs"]
    if (tuple(item["local_n_capacity"] for item in outputs) != (128, 64, 64)
            or tuple(item["weight_group_start"] * call["n_lane"]
                     for item in outputs) != (0, 128, 192)):
        return
    operands = {item["formal"]: item for parameter in call["inputs"]
                for item in parameter["buffers"]}
    if set(operands) != {"input", "q_weight_qkv_fused"}:
        return
    source, weight = (operands[key] for key in ("input", "q_weight_qkv_fused"))
    x, w = source["abi"], weight["abi"]
    k = call["local_k_capacity"]
    shape = tuple(w["local_capacity_shape"])
    if (k < 128 or k % 64 or x.get("scalar_dtype") != "bfloat16"
            or x.get("scalar_lane_count", 1) != 1
            or prod(x["local_capacity_shape"][:-1]) != 1
            or x["local_capacity_shape"][-1] != k
            or x["scalar_storage_strides"][-1] != 1):
        return
    if (w.get("scalar_dtype") != "bfloat16" or len(shape) != 3
            or shape[1:] != (k // 16, 256 // call["n_lane"])
            or tuple(w.get("scalar_lane_shape", ())) != (call["n_lane"], 2, 8)
            or tuple(w["logical_shape"]) != shape
            or tuple(w["active_shape_expressions"]) != tuple(map(str, shape))
            or w.get("distributed_type") is not None
            or w.get("coordinate_space") != "canonical_global"
            or w.get("storage_kind") != "compact_local"
            or tuple(w["scalar_storage_strides"])
            != (call["weight_owner_stride"], 4096, call["n_lane"] * 16)
            or call["weight_k_group_stride"] != 4096
            or call["weight_n_group_stride"] != call["n_lane"] * 16
            or call["weight_owner_stride"] < k * 256):
        return
    mesh = call.get("parameters", {}).get("distribution", {}).get("placement", {})
    if shape[0] != prod(mesh.get("hierarchy", ())):
        return
    alignment = max(int(w.get("alignment_bytes", 1)),
                    int(alignments.get(weight["runtime_argument"], 1)))
    if (alignment < 64 or "view_byte_offset" in w
            or int(w.get("pool_byte_offset", 0)) % 64
            or call["weight_owner_stride"] * 2 % 64
            or int(w.get("pool_scope_stride_bytes", 0)) % 64):
        return
    call["corex_flat_sme_qkv"] = True


def _attach_gemv(call, alignments, *, split_k=False):
    if (call.get("packed_layout") != "k_major_n8_k16"
            or call["block_k"] != 64 or call["tile_n"] != 128
            or call.get("local_m_capacity") != 1):
        return
    operands = {item["formal"]: item for group in ("inputs", "outputs")
                for parameter in call[group] for item in parameter["buffers"]}
    if set(operands) != {"lhs", "rhs", "result"}:
        return
    source, weight, result = (operands[key] for key in ("lhs", "rhs", "result"))
    n, k = call["local_n_capacity"], call["local_k_capacity"]
    # Packed N32 transport tiles fit the narrow and wide warp layouts.
    # Partial tiles use masked scalar loads within the same packed ABI.
    if not (n >= 256 and n % 8 == 0 and k >= 128 and k % 64 == 0):
        return
    if not all(_full_contiguous(item["abi"]) for item in (source, weight)):
        return
    if not _full_contiguous(result["abi"], allow_sum_partial=split_k):
        return
    if split_k and not _matching_partial_axes(source, weight, result):
        return
    if split_k and (result["abi"].get("storage_kind") != "compact_per_owner"
                    or int(result["abi"].get("owner_count", 0))
                    != prod(type_from_data(result["abi"]["distributed_type"]).placement.hierarchy)
                    or int(result["abi"].get("component_stride_scalar_elements", 0)) < n):
        return
    if any(item["abi"].get("scalar_dtype") != "bfloat16" for item in (source, weight)):
        return
    if (source["abi"].get("scalar_lane_count", 1) != 1
            or result["abi"].get("scalar_dtype") not in {"bfloat16", "float32"}
            or not _matching_origins(source, weight, result)):
        return
    for item, capacity in ((source, k), (result, n)):
        abi = item["abi"]
        lanes = int(abi.get("scalar_lane_count", 1))
        if (prod(abi["local_capacity_shape"][:-1]) != 1
                or abi["local_capacity_shape"][-1] * lanes != capacity
                or abi["scalar_storage_strides"][-1] != lanes):
            return
    if not _packed_weight(weight, n, k, alignments):
        return
    call["corex_flat_sme_gemv"] = {
        "source": _owner_pointer(source), "weight": _owner_pointer(weight),
        "result": _owner_pointer(result),
        "weight_k_group_stride": weight["abi"]["scalar_storage_strides"][0],
    }
    if split_k and not weight["abi"].get("scalar_lane_shape"):
        # Scalar packed coordinates coalesce along N in the original 32-warp
        # kernel. Its four K registers are 0/16/32/48, before the K16 warp
        # reduction. Vector-lane packed coordinates retain adjacent K4 folds.
        call["corex_flat_sme_gemv"]["strided_k16_fold"] = True


def _attach_down(call, alignments):
    if call.get("packed_layout") != "k_major_n8_k16" or call["block_k"] != 128 or call["tile_n"] != 64:
        return
    operands = {item["formal"]: item for group in ("inputs", "outputs")
                for parameter in call[group] for item in parameter["buffers"]}
    if not {"lhs", "rhs", "addend", "result_0", "result_1"} <= operands.keys():
        return
    source, weight, residual, result, stats = (operands[key] for key in
                                               ("lhs", "rhs", "addend", "result_0", "result_1"))
    n, k = call["local_n_capacity"], call["local_k_capacity"]
    if not (32 <= n <= 128 and n % 32 == 0 and k >= 256 and k % 128 == 0):
        return
    if not all(_full_contiguous(item["abi"]) for item in (source, weight, residual, result)):
        return
    if any(item["abi"].get("scalar_dtype") != "bfloat16" for item in (source, weight)):
        return
    if (source["abi"].get("scalar_lane_count", 1) != 1
            or residual["abi"].get("scalar_dtype") not in {"bfloat16", "float32"}
            or result["abi"].get("scalar_dtype") != residual["abi"].get("scalar_dtype")):
        return
    if not _matching_origins(source, weight, result) or not _matching_origins(source, weight, residual):
        return
    for item, capacity in ((source, k), (residual, n), (result, n)):
        abi = item["abi"]
        lanes = int(abi.get("scalar_lane_count", 1))
        if (prod(abi["local_capacity_shape"][:-1]) != 1
                or abi["local_capacity_shape"][-1] * lanes != capacity
                or abi["scalar_storage_strides"][-1] != lanes):
            return
    if (stats["abi"].get("scalar_dtype") != "float32"
            or prod(stats["abi"]["local_capacity_shape"]) != 1
            or stats["abi"].get("scalar_lane_count", 1) != 1):
        return
    if not _packed_weight(weight, n, k, alignments):
        return
    call["corex_flat_sme_down"] = {
        "source": _owner_pointer(source), "weight": _owner_pointer(weight),
        "weight_k_group_stride": weight["abi"]["scalar_storage_strides"][0],
    }


def _packed_weight(item, n, k, alignments):
    abi = item["abi"]
    shape = tuple(abi["local_capacity_shape"])
    strides = tuple(abi["scalar_storage_strides"])
    lanes = tuple(abi.get("scalar_lane_shape", ()))
    if not ((shape == (k // 16, n // 8, 2, 64) and not lanes
             and strides[1:] == (128, 64, 1))
            or (shape == (k // 16, n // 8) and lanes == (8, 2, 8) and strides[1] == 128)):
        return False
    alignment = max(int(abi.get("alignment_bytes", 1)),
                    int(alignments.get(item["runtime_argument"], 1)))
    return (alignment >= 64 and not int(abi.get("pool_byte_offset", 0)) % 64
            and "view_byte_offset" not in abi and not strides[0] * 2 % 64
            and not int(abi.get("pool_scope_stride_bytes", 0)) % 64
            and not int(abi.get("component_stride_scalar_elements", 0)) * 2 % 64)


def _full_contiguous(abi, *, allow_sum_partial=False):
    raw = abi.get("distributed_type")
    if not isinstance(raw, dict):
        return False
    value = type_from_data(raw)
    if not isinstance(value, DistributedType) or value.exclusive is not None:
        return False
    if value.partial is not None and not (
            allow_sum_partial and value.partial.reduce_op == fm.ReduceOp.SUM):
        return False
    shape = tuple(abi["local_capacity_shape"])
    for owner in product(*(range(size) for size in value.placement.hierarchy)):
        descriptor = local_shard_descriptor(value, owner)
        if any(not axis.active_extent.is_fixed or axis.active_extent.fixed_value != extent
               or axis.affine_stride != 1
               for axis, extent in zip(descriptor.axes, shape, strict=True)):
            return False
    return abi.get("coordinate_space") in {"canonical_global", "local"}


def _matching_partial_axes(source, weight, result):
    """A local projection publishes one sum partial per K owner.

    Contiguous local rows alone do not prove partial ownership. Both operands
    must split K over exactly the result's sum axes; N origins are checked
    independently by _matching_origins.
    """
    x, w, y = [type_from_data(item["abi"]["distributed_type"])
               for item in (source, weight, result)]
    if y.partial is None or y.partial.reduce_op != fm.ReduceOp.SUM:
        return False
    x_k, w_k = x.axis_policies[-1], w.axis_policies[0]
    return (isinstance(x_k, fm.SBPSplit) and isinstance(w_k, fm.SBPSplit)
            and set(x_k.hierarchy_axes) == set(w_k.hierarchy_axes)
            == set(y.partial.axes))


def _owner_pointer(binding):
    abi = binding["abi"]
    pointer = emit_buffer_pointer(abi, binding["runtime_argument"])
    offset = emit_local_scalar_offset(
        abi, ("0",) * len(abi["local_capacity_shape"]),
        lane_coordinate="0" if abi.get("scalar_lane_count", 1) != 1 else None,
    )
    return f"({pointer} + ({offset}))"


def _matching_origins(source, weight, result):
    values = [type_from_data(item["abi"]["distributed_type"])
              for item in (source, weight, result)]
    if any(value.placement != values[0].placement for value in values):
        return False
    for owner in product(*(range(size) for size in values[0].placement.hierarchy)):
        x, w, y = [local_shard_descriptor(value, owner) for value in values]
        if (w.axes[0].map_local_to_global(0).fixed_value * 16
                != x.axes[-1].map_local_to_global(0).fixed_value * source["abi"].get("scalar_lane_count", 1)
                or w.axes[1].map_local_to_global(0).fixed_value * 8
                != y.axes[-1].map_local_to_global(0).fixed_value * result["abi"].get("scalar_lane_count", 1)):
            return False
    return True
