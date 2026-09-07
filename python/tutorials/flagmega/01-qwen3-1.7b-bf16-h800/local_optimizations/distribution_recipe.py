"""Workload-owned layout choices; derive all coupled choices with CP-SAT."""

from triton.flagmega.ir import DistributedType, SBP, TupleType
from triton.flagmega.ir.distributed_inference import all_broadcast

CHOICES = (
    ("distribution.decode_layer_qkv_projection.packed_projection",
     'distribution.decode_layer_qkv_projection.packed_projection.packed_qkv_output_k_sbp_partial.mesh_yx_8x16_bb.in_d_b_s_bc_h0_b64__d_s_bc_h0_b4_s_bc_h1_b8__d_s_bc_h0_b4_s_bc_h1_b8__d_s_bc_h0_b4_s_bc_h1_b8__nonetype__nonetype__nonetype__nonetype__nonetype__nonetype__nonetype__nonetype__nonetype.out_tuple_d_b_s_bc_h1_b8_partial_p_sum_h0__d_b_s_bc_h1_b8_partial_p_sum_h0__d_b_s_bc_h1_b8_partial_p_sum_h0'),
)


def _full_output_shard(value):
    if (not isinstance(value, DistributedType) or value.tensor.rank != 2
            or not value.tensor.shape[-1].is_fixed or value.partial is not None):
        return False
    extent = value.tensor.shape[-1].fixed_value
    if extent % value.placement.size:
        return False
    return value.axis_policies == (SBP.broadcast(), SBP.split_contiguous(
        tuple(range(value.placement.rank)), extent // value.placement.size))


def sharded_residual_norm_choices(graph, module, *, shard_casts=False):
    """Keep one output-N shard and its statistics per owner for local GEMV fusion.

    This is an explicit experiment choice, not a compiler fallback or cost-model
    correction. The solver must prove the caller/callee and later boxing edges.
    """
    choices = {}
    for bucket in graph.buckets:
        if module.node_map[bucket.node_id].op != "ntt.matmul_norm_stats_combine":
            continue
        matches = []
        for candidate in bucket.candidates:
            result = candidate.return_type
            if not isinstance(result, TupleType):
                continue
            value = result.fields[0]
            if _full_output_shard(value) and candidate.input_types == (value, value):
                matches.append(candidate)
        if len(matches) != 1:
            raise ValueError(f"Expected one full-mesh contiguous residual shard choice for {bucket.node_id}, got {len(matches)}")
        choices[bucket.node_id] = matches[0].id
        if shard_casts:
            value = module.node_map[module.node_map[bucket.node_id].inputs[1]]
            buckets = {item.node_id: item for item in graph.buckets}
            while value.op in {"tensors.cast", "ntt.vectorized_cast"}:
                matches = [candidate for candidate in buckets[value.id].candidates
                           if _full_output_shard(candidate.return_type)
                           and len(candidate.input_types) == 1
                           and _full_output_shard(candidate.input_types[0])]
                if len(matches) != 1:
                    raise ValueError(f"Expected one shard-local residual Cast choice for {value.id}, got {len(matches)}")
                choices[value.id] = matches[0].id
                value = module.node_map[value.inputs[0]]
    if not choices:
        raise ValueError("The requested residual experiment has no residual/stats combines")
    return choices


def replicated_norm_choices(graph, module, *, residual_only=False):
    """Trade redundant row normalization for owner-private GEMV inputs.

    The shared statistics are still reduced; only the materialized value,
    scale/bias and result use B. The solver must retain the required input
    publication, while the local output no longer needs a grid publication.
    """
    choices = {}
    for bucket in graph.buckets:
        node = module.node_map[bucket.node_id]
        if node.op != "nn.norm_apply":
            continue
        if residual_only:
            value = module.node_map[node.inputs[0]]
            if (value.op != "builtin.get_item"
                    or module.node_map[value.inputs[0]].op != "ntt.matmul_norm_stats_combine"):
                continue
        candidates = [c for c in bucket.candidates
                      if isinstance(c.return_type, DistributedType) and c.return_type.tensor.rank == 2
                      and all_broadcast(c.return_type) and all(
                          isinstance(t, DistributedType) and all_broadcast(t) for t in c.input_types)]
        if len(candidates) != 1:
            raise ValueError(f"Expected one replicated NormApply candidate for {bucket.node_id}, got {len(candidates)}")
        choices[bucket.node_id] = candidates[0].id
    if not choices:
        raise ValueError("The requested normalization experiment has no NormApply nodes")
    return choices
