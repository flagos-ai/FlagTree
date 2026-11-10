# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit ABI for warp-state GDN; only projection/norm statistics stay C1.

Adapted from the core scalar head validator. No old identity or fake context is
used: every check below receives the actual new32B descriptor and block128.
"""
import ast
from itertools import product

from triton.flagmega.codegen.triton.candidates.gdn_head_gates import (
    dense_head_geometry, dense_value_geometry,
)
IMPLEMENTATION = "tir.gdn_recurrent.persistent_warp_state_v1"

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import type_from_data


def _binding(raw, group, formal, index=0):
    return next(p for p in raw[group] if p["formal"] == formal)["buffers"][index]


def _reject_alias(a, b):
    aa, ba = a["abi"], b["abi"]
    if a["runtime_argument"] != b["runtime_argument"]:
        return
    if aa.get("physical_buffer") == ba.get("physical_buffer"):
        raise CodegenError("Warp-state GDN require disjoint mutable and readonly buffer spans.")
    # Runtime parameter banks may carry symbolic layer offsets. If two operands
    # share their pointer, reject unless their complete static intervals prove
    # disjoint. Normal pooled value/workspace buffers have fixed planned offsets.
    if aa.get("view_byte_offset") or ba.get("view_byte_offset"):
        raise CodegenError("Warp-state GDN cannot prove disjoint symbolic aliases.")
    def interval(abi):
        shape = tuple(abi["local_capacity_shape"])
        strides = tuple(abi["scalar_storage_strides"])
        lanes = int(abi.get("scalar_lane_count", 1))
        count = lanes + sum((n - 1) * s for n, s in zip(shape, strides))
        start = int(abi.get("pool_byte_offset", 0))
        return start, start + count * int(abi["scalar_itemsize"])
    a0, a1 = interval(aa); b0, b1 = interval(ba)
    if max(a0, b0) < min(a1, b1):
        raise CodegenError("Warp-state GDN require disjoint mutable and readonly buffer spans.")


def prepare_fields(raw, call):
    if raw.get("implementation") != IMPLEMENTATION:
        return {}
    if not raw.get("parameters", {}).get("head_statistics_cache"):
        raise CodegenError("Warp-state GDN identity requires its cache parameters.")
    workspaces = raw.get("shared_workspaces", ())
    if len(workspaces) != 1:
        raise CodegenError("Warp-state GDN require one planned Shared descriptor.")
    ws = workspaces[0]
    offset = ws.get("offset_bytes")
    if (ws.get("name") != "head_statistics" or ws.get("dtype") != "float32"
            or tuple(ws.get("shape", ())) != (2, 4) or tuple(ws.get("strides", ())) != (4, 1)
            or ws.get("nbytes") != 32 or ws.get("alignment_bytes") != 16
            or type(offset) is not int or offset < 0 or offset % 16
            or not ws.get("physical_buffer") or ws.get("matrix_compatible", False)):
        raise CodegenError("Warp-state GDN require planned F32[2,4], strides(4,1), 32B/alignment16.")
    writes = [_binding(raw, "inputs", "state", 1), _binding(raw, "outputs", "result_0"),
              _binding(raw, "workspaces", "core_scratch")]
    reads = [_binding(raw, "inputs", name) for name in
             ("qkv", "z", "projection_input", "b_weight", "a_weight", "a_log", "dt_bias", "norm_weight")]
    for write in writes:
        for read in reads:
            _reject_alias(write, read)
    for i, write in enumerate(writes):
        for other in writes[i+1:]:
            _reject_alias(write, other)
    # Canonical projection/state pointers in this kernel must retain their
    # complete rows. The ABI may have a bank/subspan offset but no row padding.
    for name, shape, strides, dtype, lanes in (
            ("projection_input", (1, 5120), (5120, 1), "bfloat16", 1),
            ("b_weight", (48, 2720), (2720, 1), "uint8", 1),
            ("a_weight", (48, 2720), (2720, 1), "uint8", 1),
            ("qkv", (1, 10240), (10240, 1), "bfloat16", 1)):
        abi = _binding(raw, "inputs", name)["abi"]
        if (tuple(abi.get("local_capacity_shape", ())) != shape
                or tuple(abi.get("scalar_storage_strides", ())) != strides
                or abi.get("scalar_dtype") != dtype or abi.get("scalar_lane_count", 1) != lanes):
            raise CodegenError("Warp-state GDN require canonical contiguous projection/QKV rows.")
    state = writes[0]["abi"]
    if (tuple(state.get("local_capacity_shape", ())) != (1, 48, 128, 32)
            or tuple(state.get("scalar_storage_strides", ())) != (786432, 16384, 128, 4)
            or state.get("scalar_dtype") != "float32" or tuple(state.get("scalar_lane_shape", ())) != (4,)):
        raise CodegenError("Warp-state GDN require contiguous F32<4> key state rows.")
    return {"head_statistics_cache": True, "head_statistics_offset_bytes": offset,
            "head_statistics_arena_bytes": offset + 32, "head_statistics_physical_buffer": ws["physical_buffer"]}


def validate_call(call, context):
    if call.get("implementation") != IMPLEMENTATION:
        return
    if (call.get("implementation") != IMPLEMENTATION
            or context.get("platform") != "nvidia" or context.get("target_architecture") != "sm120"
            or context.get("block_threads") != 128 or context.get("warp_size") != 32):
        raise CodegenError("Warp-state GDN require explicit CPP SM120/block128/warp32.")
    expected = {"b_weight_encoding": "IQ4_XS", "a_weight_encoding": "IQ4_XS",
                "value_heads": 48, "key_heads": 16,
                "value_dim": 128, "key_dim": 128, "projection_dim": 5120,
                "tiled_value_heads": True, "round_beta": True, "round_core": False,
                "round_normalized_qk": False, "qk_norm_add": False,
                "gate_math_mode": "accurate", "accurate_norm_silu": False}
    hierarchy = tuple(context.get("mesh_hierarchy", ()))
    geometry = dense_head_geometry(hierarchy, call.get("value_heads"), call.get("value_dim"))
    if geometry is None:
        raise CodegenError("Warp-state GDN require complete dense CTA intervals touching one or two heads.")
    capacity, bounds = geometry
    if hierarchy not in ((4, 8), (4, 24)) or capacity % 4:
        raise CodegenError("Warp-state GDN supports only complete4x8 or4x24 block128 ownership")
    if (any(call.get(k) != v for k, v in expected.items())
            or call.get("local_capacity") != capacity
            or float(call.get("qk_norm_epsilon", "nan")) != 1e-12
            or str(call.get("active_values")) != str(capacity) or str(call.get("writer_active")) != "True"
            or str(call.get("participation_active")) != "True"):
        raise CodegenError("Warp-state GDN do not advertise altered/tail/exclusive/numeric modes.")
    parameters, facts = call.get("parameters", {}), call.get("facts", {})
    if (parameters.get("head_cache_slots") != 2
            or parameters.get("projection_order") != "ascending_scalar_C1"
            or tuple(parameters.get("cache_contents", ())) != ("beta", "decay", "query_norm", "key_norm")
            or parameters.get("cache_epoch") != "per_GDN_invocation"
            or parameters.get("qk_norm_order") != "ascending_scalar_C1"
            or parameters.get("state_reduction_order") != "warp32_xor_16_8_4_2_1"
            or parameters.get("state_rows_per_warp") != 1
            or parameters.get("state_values_per_lane") != 4
            or parameters.get("normalized_qk_math") != "f32_division"
            or facts.get("state_reassociation") is not True
            or facts.get("state_bitwise_equivalent") is not False
            or facts.get("internal_cta_barriers") != 1 or call.get("internal_grid_barriers") != 1
            or facts.get("required_codegen_backend") != "cpp"
            or call.get("head_statistics_arena_bytes", 0) > context["launch_resource_limits"]["max_shared_memory_bytes"]):
        raise CodegenError("Warp-state GDN disagree with the declared publication/resource contract.")
    if call.get("outputs"):
        abi = _binding(call, "outputs", "result_0")["abi"]
        distributed = type_from_data(abi["distributed_type"])
        if (distributed.placement.hierarchy != hierarchy
                or dense_value_geometry(distributed, call["value_heads"], call["value_dim"]) != geometry
                or tuple(abi.get("local_capacity_shape", ())) != (1, capacity)
                or abi.get("scalar_dtype") != "bfloat16" or abi.get("scalar_lane_count", 1) != 1):
            raise CodegenError("Warp-state GDN result ABI differs from its dense proposal geometry.")
    allowed = (ast.Expression, ast.BinOp, ast.Add, ast.Mult, ast.Name, ast.Load, ast.Constant)
    def affine(expression):
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, TypeError) as error:
            raise CodegenError("Warp-state GDN cannot prove the value map.") from error
        if any(not isinstance(n, allowed) for n in ast.walk(tree)):
            raise CodegenError("Warp-state GDN require a proven dense affine value map.")
        return compile(tree, "<scalar-head-gates-map>", "eval")
    code = affine(call["global_value"])
    first_code = affine(call.get("head_statistics_first_value"))
    for shard, coordinates in enumerate(product(*(range(n) for n in hierarchy))):
        env = {"shard_index": shard, **{f"shard_coord{i}": v for i, v in enumerate(coordinates)}}
        if len(hierarchy) == 2:
            env.update(shard_y=coordinates[0], shard_x=coordinates[1])
        try:
            # First value must be independent of the per-thread induction variable.
            if eval(first_code, {"__builtins__": {}}, env) != shard * capacity:
                raise CodegenError("Warp-state GDN first value differs from its proposal proof.")
            for value in range(capacity):
                actual = eval(code, {"__builtins__": {}}, {**env, "local_values": value})
                if actual != value + shard * capacity:
                    raise CodegenError("Warp-state GDN value map differs from its proposal proof.")
        except (NameError, TypeError) as error:
            raise CodegenError("Warp-state GDN cannot prove the value map.") from error
