# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SM90 selected-kernel workspace ABI policy."""

from dataclasses import replace
from math import prod

from triton.flagmega.ir import (
    Candidate,
    DType,
    IRModule,
    Node,
    TupleType,
    local_tensor_type,
    DistributedType,
    tensor_type,
)
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.nn.greedy_sample import GreedySample


def attach_workspace_requirements(
    node: Node,
    candidates: tuple[Candidate, ...],
    module: IRModule,
    *,
    mesh_hierarchy: tuple[int, ...],
) -> tuple[Candidate, ...]:
    """Make internal scratch explicit before candidate selection is frozen."""

    result = []
    for candidate in candidates:
        requirements = _requirements(node, candidate, module, mesh_hierarchy)
        result.append(
            replace(candidate, parameters={
                **dict(candidate.parameters),
                "workspaces": requirements,
            })
            if requirements else candidate
        )
    return tuple(result)


def _requirements(node, candidate, module, mesh_hierarchy):
    family = str(candidate.parameters.get("family", ""))
    variant = str(candidate.parameters.get("variant", ""))
    mesh_size = prod(mesh_hierarchy)
    values = []

    def workspace(
        name, dtype, shape, alignment=256, *, lifetime="invocation"
    ):
        values.append({
            "name": name,
            "type": tensor_type(dtype, shape),
            "memory_space": "workspace",
            "alignment": alignment,
            "lifetime": lifetime,
        })

    if family == "gdn_recurrent":
        # Each owner computes its dense local value slice, while RMS
        # normalization is over a complete value head. nncase materializes
        # this as the recurrent core scratch buffer; making it a caller-owned
        # PrimFunction workspace gives the inter-owner barrier a real ABI.
        workspace(
            "core_scratch",
            DType.FLOAT32,
            (
                int(node.attrs["num_value_heads"]),
                int(node.attrs["value_head_dim"]),
            ),
            128,
        )

    if family == "greedy_sample" and variant == "distributed_argmax":
        logits = module.node_map[GreedySample.logits.read(node.inputs)].type
        local = local_tensor_type(logits) if isinstance(logits, DistributedType) else logits
        workspace("partial_max", DType.FLOAT32, (local.shape[0], mesh_size), 128)
        workspace("partial_index", DType.INT32, (local.shape[0], mesh_size), 128)

    if family == "gather_reduce_add_norm_stats":
        if not isinstance(node.type, TupleType) or len(node.type.fields) != 2:
            return ()
        value = tensor_of(node.type.fields[0])
        workspace("collective", value.dtype, value.shape)
        workspace("norm_stats_partials", DType.FLOAT32, (mesh_size,), 128)

    if family == "gather_reduce_add_norm_apply":
        components = 2 if bool(node.attrs.get("use_mean", False)) else 1
        collective_owner_count = int(
            candidate.parameters.get("owner_count", mesh_size)
        )
        workspace(
            "norm_stats_partials",
            DType.FLOAT32,
            (components, collective_owner_count),
            128,
        )

    if family == "add_norm_stats" and variant == "rms":
        workspace("norm_stats_partials", DType.FLOAT32, (mesh_size,), 128)

    if (
        node.op == "ntt.matmul_norm_stats"
        and family == "dense_matmul"
        and candidate.parameters.get("distribution_schedule", {}).get("kind")
        == "reduction_split"
    ):
        if not isinstance(node.type, TupleType) or len(node.type.fields) != 2:
            return ()
        value = tensor_of(node.type.fields[0])
        workspace(
            "matmul_partial",
            value.dtype,
            (mesh_size, *value.shape),
        )
        workspace("norm_stats_partials", DType.FLOAT32, (mesh_size,), 128)

    return tuple(values)


__all__ = ["attach_workspace_requirements"]
