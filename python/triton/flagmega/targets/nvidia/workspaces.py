# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SM90 selected-kernel workspace ABI policy."""

from dataclasses import replace
from math import prod

from triton.flagmega.ir import (
    Candidate,
    DistributedType,
    DType,
    IRModule,
    Node,
    TupleType,
    local_tensor_type,
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

    if family == "mhc_pre" and variant in {"cooperative_stream_fma", "cooperative_contiguous_tf32"}:
        streams = tensor_of(module.node_map[node.inputs[0]].type)
        hc = streams.shape[1].fixed_value
        splits = int(node.attrs.get("projection_splits", 1))
        # Coefficient and square partials are indexed by logical split. The
        # last hc entries publish pre coefficients for the collapse phase.
        workspace("projection_partials", DType.FLOAT32,
                  (streams.shape[0], (hc * (hc + 2) + 1) * splits + hc))

    if family == "sparse_experts_gate_up" and variant == "cooperative_wgmma":
        from triton.flagmega.codegen.triton.sparse_experts.cooperative_geometry import cooperative_gate_geometry
        from triton.flagmega.ir.ops.core import get_definition

        geometry = cooperative_gate_geometry({
            parameter.name: module.node_map[node.inputs[index]].type
            for index, parameter in enumerate(get_definition(node.op).input_parameters)
        }, node.type)
        if geometry is None:
            raise ValueError("Cooperative FP8 expert workspace requires a proven chip owner domain.")
        workspace("projection_partials", DType.FLOAT32, geometry.partial_shape)
        workspace("activation_scales", DType.FLOAT32, geometry.scale_shape)

    if family == "sparse_experts_gate_up" and variant == "complete_k_wgmma":
        from triton.flagmega.codegen.triton.sparse_experts.cooperative_geometry import cooperative_gate_geometry
        from triton.flagmega.ir.ops.core import get_definition

        geometry = cooperative_gate_geometry({
            parameter.name: module.node_map[node.inputs[index]].type
            for index, parameter in enumerate(get_definition(node.op).input_parameters)
        }, node.type)
        if geometry is None:
            raise ValueError("Complete-K experts require a proven chip owner domain.")
        workspace("quantized_activations", DType.FLOAT8_E4M3FN,
                  (geometry.tokens, geometry.chip_routes, geometry.reduction))
        workspace("activation_scales", DType.FLOAT32, geometry.scale_shape)

    if family == "matmul_glu" and variant == "cooperative_wgmma":
        from triton.flagmega.codegen.triton.matmul_glu_geometry import cooperative_glu_geometry
        from triton.flagmega.ir.ops.core import get_definition

        geometry = cooperative_glu_geometry({
            parameter.name: module.node_map[node.inputs[index]].type
            for index, parameter in enumerate(get_definition(node.op).input_parameters)
        }, node.type, node.attrs)
        if geometry is None:
            raise ValueError("Cooperative FP8 GLU requires matching complete chip owner domains.")
        workspace("gate_partials", DType.FLOAT32, geometry.partial_shape)
        workspace("up_partials", DType.FLOAT32, geometry.partial_shape)
        workspace("activation_scales", DType.FLOAT32, geometry.scale_shape)

    if family == "block_fp8" and variant == "cooperative_wgmma":
        from triton.flagmega.codegen.triton.block_fp8_geometry import cooperative_block_geometry

        geometry = cooperative_block_geometry(dict(zip(
            ("value", "weight", "weight_scale"),
            (module.node_map[value].type for value in node.inputs), strict=True,
        )), node.type, node.attrs)
        if geometry is None:
            raise ValueError("Cooperative FP8 projection requires a proven complete chip owner domain.")
        workspace("projection_partials", DType.FLOAT32, geometry.partial_shape)
        workspace("activation_scales", DType.FLOAT32, geometry.scale_shape)

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
        rows = prod(dimension.fixed_value for dimension in value.shape[:-1])
        workspace("norm_stats_partials", DType.FLOAT32, (mesh_size * rows,), 128)

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
