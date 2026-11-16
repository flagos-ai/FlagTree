# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit candidates for vectorized residual/local16 statistics."""
from dataclasses import replace
from math import prod

from triton.flagmega.ir import DistributedType, DType, TupleType, VectorType
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.op_fusion import has_partial_type
from triton.flagmega.ir.ops.ntt.vectorized_add_norm_stats import LOCAL16_ORDER, VectorizedAddNormStats, binary_attrs
from triton.flagmega.codegen.triton.fusion import require_fusion
from triton.flagmega.codegen.triton.vectorization import configured_vector_schedule, vectorization_contract
from .core import TritonCandidateProposal


class VectorizedAddNormStatsCandidateProvider:
    op_names = frozenset({VectorizedAddNormStats.op_name})

    def propose(self, node, context):
        inputs = tuple(context.module.node_map[k] for k in node.inputs)
        if (not isinstance(node.type, TupleType) or len(node.type.fields) != 2
                or node.attrs.get("post_ops") or node.attrs.get("local_reduction_order") != LOCAL16_ORDER
                or any(has_partial_type(n.type) for n in inputs)):
            return None
        # Exact inference validates effective PreOps, packing and both owners.
        if VectorizedAddNormStats.infer_call_type(inputs, node.attrs) != node.type:
            return None
        require_fusion(node, inputs, family="vectorized_add_norm_stats")
        raw, stats = node.type.fields
        if not context.cooperative_grid or not isinstance(raw, DistributedType) or not has_partial_type(stats):
            return None
        local = lambda ty: local_tensor_type(ty) if isinstance(ty, DistributedType) else tensor_of(ty)
        r, p = local(raw), local(stats)
        if any(not d.is_fixed for t in (r, p) for d in t.shape):
            return None
        elements = prod(d.fixed_value for d in r.shape) * (r.dtype.lane_count if isinstance(r.dtype, VectorType) else 1)
        if elements != 16 or prod(d.fixed_value for d in p.shape) != 1 or p.dtype is not DType.FLOAT32:
            return None
        raw_view = replace(node, op="math.vectorized_binary", type=raw,
                           attrs={**binary_attrs(node.attrs), **({"pre_ops": node.attrs["pre_ops"]} if node.attrs.get("pre_ops") else {})})
        vector = vectorization_contract(raw_view)
        choices = tuple(context.configure_implementation(implementation,
            semantic_parameters={"vector_schedule": configured_vector_schedule(vector,
                lowering="packed_axes", elements_per_program=16)},
            facts={"explicit_local16_canonical_stats": True})
            for implementation in context.implementations("vectorized_add_norm_stats", local_reduction_order=LOCAL16_ORDER)
            if implementation.variant == "local16_xor_fma" and implementation.parameters.get("tile") == 16
            and not implementation.shared_workspaces and implementation.transfer_pipeline is None)
        return None if not choices else TritonCandidateProposal(choices, context.choose_default("vectorized_add_norm_stats", choices))


__all__ = ["VectorizedAddNormStatsCandidateProvider"]
