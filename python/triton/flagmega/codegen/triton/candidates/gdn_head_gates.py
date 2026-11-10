# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Admission for a nondefault scalar-ordered, per-invocation CTA gate cache."""
import os
from math import prod

from triton.flagmega.ir import (
    DType, DistributedType, RefType, SBPBroadCast, SBPSplit, VectorType,
)
from triton.flagmega.ir.execution_mesh import effective_execution_mesh, ExecutionMeshKind

IMPLEMENTATION = "tir.gdn_recurrent.persistent_scalar_head_gates_v1"


def dense_head_geometry(hierarchy, heads, dim):
    """Prove complete dense CTA intervals fit the two-cell head cache."""
    hierarchy = tuple(hierarchy)
    if (not hierarchy or any(type(n) is not int or n <= 0 for n in hierarchy)
            or type(heads) is not int or heads <= 0 or type(dim) is not int or dim <= 0):
        return None
    size = prod(hierarchy)
    if heads * dim % size:
        return None
    capacity = heads * dim // size
    if capacity <= 0:
        return None
    bounds = []
    for shard in range(size):
        first = shard * capacity // dim
        last = (shard * capacity + capacity - 1) // dim
        if not 1 <= last - first + 1 <= 2 or last >= heads:
            return None
        bounds.append((first, last - first + 1))
    return capacity, tuple(bounds)


def dense_value_geometry(out, heads, dim):
    """One scalar contiguous split across all physical block axes."""
    if (not isinstance(out, DistributedType) or out.partial is not None or out.exclusive is not None
            or out.placement.hierarchy_levels != "b" * len(out.placement.hierarchy)
            or out.tensor.dtype != DType.BFLOAT16 or _shape(out.tensor) != (1, heads * dim)
            or not _dense(out.tensor) or not isinstance(out.axis_policies[0], SBPBroadCast)):
        return None
    split = out.axis_policies[1]
    if (not isinstance(split, SBPSplit) or not split.is_contiguous or len(split.stages) != 1
            or split.hierarchy_axes != tuple(range(len(out.placement.hierarchy)))
            or split.stages[0].distribution.granularity is not None):
        return None
    return dense_head_geometry(out.placement.hierarchy, heads, dim)


def _shape(tensor):
    return tuple(d.fixed_value for d in tensor.shape) if all(d.is_fixed for d in tensor.shape) else None


def _dense(tensor):
    layout = tensor.layout
    return not layout.order and not layout.strides and not layout.vector_lanes and layout.tag == "dense"


def _broadcast(value, placement, shape, dtype):
    return (isinstance(value, DistributedType) and value.placement == placement
            and value.partial is None and value.exclusive is None
            and value.tensor.dtype == dtype and _shape(value.tensor) == shape and _dense(value.tensor)
            and all(isinstance(p, SBPBroadCast) for p in value.axis_policies))


def _ref_derived(value, module):
    node = module.node_map[value]
    if isinstance(node.type, RefType) or node.op.startswith("refs."):
        return True
    # Follow zero-copy views, but stop at a producer that owns a fresh tensor.
    if node.op in {"tensors.reshape", "tensors.view", "tensors.reinterpret", "ntt.view"}:
        return any(_ref_derived(i, module) for i in node.inputs)
    return False


def applicable(node, context):
    """First certified dense SM120 map; the normal candidate remains available.

    Tensor operands have value semantics, while the mutable state is a Ref.
    Ref-derived aliases are deliberately excluded at proposal; materialized
    byte spans are checked again at CPP emission, including planner scratch.
    """
    if (node.op != "nn.gdn_recurrent_core" or context.codegen_backend != "cpp"
            or not context.cooperative_grid
            or getattr(context.target, "codegen_platform", None) != "nvidia"
            or getattr(context.target, "codegen_architecture", None) != "sm120"
            or os.environ.get("FLAGMEGA_CPP_BLOCK_THREADS", "128") != "128"):
        return False
    attrs = node.attrs
    flags = {"round_beta": True, "round_core": False, "round_normalized_qk": False,
             "qk_norm_mode": "clamp", "qk_norm_epsilon": 1e-12,
             "gate_math_mode": "accurate", "norm_math_mode": "fast"}
    defaults = dict(flags)
    if (attrs.get("b_weight_encoding") != "IQ4_XS" or attrs.get("a_weight_encoding") != "IQ4_XS"
            or any(attrs.get(k, defaults[k]) != v for k, v in flags.items())
            or attrs.get("key_head_dim") != 128 or attrs.get("value_head_dim") != 128
            or attrs.get("num_key_heads") != 16 or attrs.get("num_value_heads") != 48
            or attrs.get("value_head_order", "grouped") != "tiled"):
        return False
    out = node.type.fields[0]
    if dense_value_geometry(out, attrs["num_value_heads"], attrs["value_head_dim"]) is None:
        return False
    mesh = effective_execution_mesh(context.module, node)
    if mesh is not None and (mesh.kind != ExecutionMeshKind.FIXED
                             or mesh.shape != out.placement.hierarchy or mesh.mapping != "row_major"):
        return False
    operands = [context.module.node_map[i].type for i in node.inputs]
    if operands[2] != out:
        return False
    for index, shape, dtype in (
            (1, (1, 2 * 16 * 128 + 48 * 128), DType.BFLOAT16),
            (3, (1, 5120), DType.BFLOAT16),
            (4, (48, 5120 // 256 * 136), DType.UINT8),
            (5, (48, 5120 // 256 * 136), DType.UINT8),
            (6, (48,), DType.FLOAT32), (7, (48,), DType.FLOAT32), (8, (128,), DType.FLOAT32)):
        if not _broadcast(operands[index], out.placement, shape, dtype):
            return False
    if any(_ref_derived(i, context.module) for i in node.inputs[1:]):
        return False
    state = operands[0]
    if not isinstance(state, RefType):
        return False
    recurrent = dict(state.fields).get("recurrent")
    return (recurrent is not None and recurrent.dtype == VectorType(DType.FLOAT32, (4,))
            and _shape(recurrent) == (1, 48, 128, 32) and _dense(recurrent))
