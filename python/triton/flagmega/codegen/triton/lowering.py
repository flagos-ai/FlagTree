# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Selected high-level IR to Triton-oriented first-class TIR realization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from triton.flagmega.ir.tir.kernel_definition import replace_kernel_dispatch, replace_kernel_callables
from math import prod

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, Node, TIRMicroKernelSelection, kernel_dispatch_of
from triton.flagmega.codegen.triton.kernel_dispatch import selected_kernel_nodes
from triton.flagmega.codegen.triton.distribution import requires_distributed_grid
from triton.flagmega.passes.tir import materialize_kernel_definitions
from triton.flagmega.codegen.triton.microkernels.materialization import (
    materialize_shared_workspace_buffers,
    validate_transfer_pipeline,
)
from triton.flagmega.codegen.triton.vectorization import require_vector_schedule
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import normalize_axes
from triton.flagmega.ir.ops.tensors.unpack import Unpack
from triton.flagmega.ir.types import VectorType


_VECTORIZABLE_SEMANTIC_OPS = frozenset({
    "math.add",
    "math.matmul",
    "math.mul",
    "math.div",
    "math.sigmoid",
    "math.packed_dense_matmul",
    "math.silu",
    "math.vectorized_binary",
    "math.vectorized_unary",
    "nn.rms_norm",
    "nn.norm_apply",
    "ntt.matmul_norm_stats",
    "ntt.matmul_norm_stats_combine",
    "ntt.packed_matmul",
    "ntt.vectorized_cast",
    "tensors.cast",
})


class TritonTirLoweringPolicy:
    def lower(self, module: IRModule, target) -> IRModule:
        selection_points = {point.id: point for point in module.selection_points}
        selections = module.selection_map
        nodes: list[Node] = []
        aliases: dict[str, str] = {}
        distribution_metadata = module.metadata.get("auto_distribution")
        placement_data = (
            dict(distribution_metadata["placement"])
            if isinstance(distribution_metadata, Mapping) and "placement" in distribution_metadata
            else None
        )
        distribution_adapters: list[dict[str, object]] = []

        def resolve(node_id: str) -> str:
            while node_id in aliases:
                node_id = aliases[node_id]
            return node_id

        for node in module.nodes:
            inputs = tuple(resolve(input_id) for input_id in node.inputs)
            if node.op in {"nn.gdn_state_slice", "tir.ref_slice"}:
                # A reference view is address computation, never a compute
                # kernel or a fresh state allocation. Bufferize owns its spans.
                nodes.append(
                    replace(node, op="tir.ref_slice", inputs=inputs, attrs={"length": int(node.attrs.get("length", 1))},
                            metadata={**node.metadata, "lowered_from": node.op}))
                continue
            if node.op in {"distributed.sharded_view", "distributed.boxing"}:
                if len(inputs) != 1:
                    raise IRVerificationError(
                        f"Distributed adapter {node.id!r} must have exactly one input.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                source_type = module.node_map[node.inputs[0]].type
                distribution_adapters.append({
                    "id": node.id,
                    "op": node.op,
                    "input": node.inputs[0],
                    "source_type": source_type,
                    "target_type": node.type,
                    "consumer": node.metadata.get("consumer"),
                    "realization": node.metadata.get("realization"),
                })
                if node.op == "distributed.sharded_view":
                    # A sharded view changes only the logical coordinate map.
                    # Keep it in graph TIR so bufferization can attach a typed
                    # alias to the source MemSpan; executable boxing continues
                    # through ordinary candidate lowering below.
                    nodes.append(replace(node, inputs=inputs))
                    continue
            if node.op == "builtin.call":
                nodes.append(replace(
                    node,
                    op="tir.call",
                    inputs=inputs,
                    metadata={**dict(node.metadata), "lowered_from": "builtin.call"},
                ))
                continue
            if node.op == "builtin.scalar_const":
                nodes.append(replace(
                    node,
                    op="tir.scalar_const",
                    inputs=inputs,
                    metadata={
                        **dict(node.metadata),
                        "lowered_from": "builtin.scalar_const",
                    },
                ))
                continue
            if node.op == "tensors.bitcast":
                # Bitcast is defined as a byte-preserving storage view.  It
                # must share the source MemSpan; proposing a compute kernel
                # here would contradict both its IR semantics and nncase's
                # UnpackToBitcast canonicalization.
                nodes.append(replace(
                    node,
                    op="tir.buffer_view",
                    inputs=inputs,
                    attrs={"alias_kind": "vector_reinterpret"},
                    metadata={
                        **dict(node.metadata),
                        "lowered_from": "tensors.bitcast",
                    },
                ))
                continue
            if node.op == "tensors.unpack" and _is_zero_copy_unpack(node, module):
                nodes.append(replace(
                    node,
                    op="tir.buffer_view",
                    inputs=inputs,
                    attrs={"alias_kind": "vector_reinterpret"},
                    metadata={
                        **dict(node.metadata),
                        "lowered_from": "tensors.unpack",
                    },
                ))
                continue
            if node.op == "tensors.pack" and _is_zero_copy_pack(node, module):
                nodes.append(replace(
                    node,
                    op="tir.buffer_view",
                    inputs=inputs,
                    attrs={"alias_kind": "vector_reinterpret"},
                    metadata={
                        **dict(node.metadata),
                        "lowered_from": "tensors.pack",
                    },
                ))
                continue
            if node.op == "tensors.reshape" and _is_zero_copy_reshape(node, module):
                nodes.append(replace(
                    node,
                    op="tir.buffer_view",
                    inputs=inputs,
                    attrs={"alias_kind": "reshape"},
                    metadata={
                        **dict(node.metadata),
                        "lowered_from": "tensors.reshape",
                    },
                ))
                continue
            if node.op.startswith("builtin."):
                nodes.append(replace(node, inputs=inputs))
                continue
            semantic_op = node.op
            semantic_inputs = inputs
            semantic_attrs = dict(node.attrs)
            point_id = f"tir.{node.id}"
            parameters: dict[str, object]
            facts: dict[str, object]
            semantic_tir_candidate = False
            if point_id in selection_points:
                try:
                    candidate_id = selections[point_id].candidate_id
                except KeyError as error:
                    raise IRVerificationError(
                        f"TIR selection point {point_id!r} has no applied selection.",
                        stage=module.stage,
                        node_id=node.id,
                    ) from error
                candidate = next(
                    candidate for candidate in selection_points[point_id].candidates if candidate.id == candidate_id)
                parameters = dict(candidate.parameters)
                facts = dict(candidate.facts)
                semantic_tir_candidate = (
                    selection_points[point_id].kind == "semantic_tir"
                )
            else:
                raise IRVerificationError(
                    f"No reviewed Triton TIR candidate was proposed for op {semantic_op!r}; "
                    "generic kernel fabrication is not permitted.",
                    stage=module.stage,
                    node_id=node.id,
                )
            if semantic_op in _VECTORIZABLE_SEMANTIC_OPS:
                require_vector_schedule(parameters, node.id)
            if placement_data is not None:
                parameters["distribution"] = {
                    "placement": placement_data,
                    "input_types": tuple(
                        module.node_map[input_id].type for input_id in node.inputs
                    ),
                    "output_type": node.type,
                }
            dispatch_attrs = {
                "semantic_op": semantic_op,
                "semantic_attrs": semantic_attrs,
            }
            if semantic_tir_candidate:
                dispatch_attrs.update({
                    "semantic_candidate": candidate_id,
                    "semantic_parameters": parameters,
                    "semantic_facts": facts,
                })
            else:
                dispatch_attrs.update({
                    "candidate": candidate_id,
                    "parameters": parameters,
                    "facts": facts,
                })
            nodes.append(replace(
                node,
                op="tir.kernel",
                inputs=semantic_inputs,
                attrs=dispatch_attrs,
                metadata={**dict(node.metadata), "lowered_from": semantic_op},
            ))
        cooperative_grid = requires_distributed_grid(module)
        launch_parameters = target.plan_launch(
            module, tuple(node for node in nodes if node.op == "tir.kernel")
        )
        metadata = {
            **dict(module.metadata),
            "target": target.name,
            "target_backend": getattr(target, "backend_name", target.name),
            "target_machine": getattr(target, "machine_name", target.name),
            "target_machine_policy": getattr(
                target, "machine_policy_version", target.policy_version
            ),
            "target_policy": target.policy_version,
            "target_capability": target.capability.to_data(),
            "target_options": target.options.to_data(),
            "target_implementation_model": dict(
                target.triton_implementation_model.snapshot()
            ),
            "codegen_template_target": {
                "platform": target.codegen_platform,
                "architecture": target.codegen_architecture,
            },
            "launch_contract": {
                "kind": "single_prepared_entry",
                "entry": f"flagmega_{module.entry}",
                "ordinary_launch_allowed": False,
                "cooperative_grid": cooperative_grid,
                "grid_mesh": placement_data,
                **launch_parameters,
            },
            "tir_distribution": {
                "schema": "flagmega.tir-distribution/v1",
                "placement": placement_data,
                "adapters": distribution_adapters,
            },
        }
        functions = tuple(
            replace(function, outputs=tuple(resolve(output_id) for output_id in function.outputs))
            for function in module.functions
        )
        selected = replace(module, nodes=tuple(nodes), functions=functions, metadata=metadata)
        selected = materialize_kernel_definitions(selected)
        selected = _materialize_direct_implementation_resources(selected, target)
        dispatches = tuple(
            dispatch
            for function in selected.kernel_callable_map.values()
            if (dispatch := kernel_dispatch_of(function)) is not None
        )
        if not all(dispatch.microkernel is not None for dispatch in dispatches):
            return selected
        package_plan = target.plan_codegen_package(selected, selected_kernel_nodes(selected))
        return replace(
            selected,
            metadata={
                **dict(selected.metadata),
                "codegen_package_plan": package_plan,
            },
        )


def _materialize_direct_implementation_resources(module: IRModule, target) -> IRModule:
    """Attach catalog-owned resources to legacy direct implementation choices.

    Most semantic TIR families select their physical microkernel in the later
    ``select-microkernels`` stage.  Existing dense-matmul families still expose
    the concrete implementation in the earlier TIR selection point.  The
    selected candidate already carries a verified catalog identity, but the
    compatibility constructor cannot serialize typed Shared workspaces or a
    transfer pipeline through ``Candidate``.  Materialize those target-owned
    resources here rather than weakening verification or reconstructing them
    in codegen.
    """

    functions = []
    changed = False
    model = target.triton_implementation_model
    for function in module.kernel_callable_map.values():
        dispatch = kernel_dispatch_of(function)
        selection = None if dispatch is None else dispatch.microkernel
        implementation = (
            None
            if selection is None
            else model.implementation(selection.implementation)
        )
        if (
            dispatch is None
            or selection is None
            or implementation is None
            or (
                not implementation.shared_workspaces
                and implementation.transfer_pipeline is None
            )
        ):
            functions.append(function)
            continue
        if selection.shared_workspaces or selection.transfer_pipeline is not None:
            if (
                selection.shared_workspaces != implementation.shared_workspaces
                or selection.transfer_pipeline != implementation.transfer_pipeline
            ):
                raise IRVerificationError(
                    f"PrimFunction @{function.name} direct microkernel resources "
                    "differ from the active implementation catalog.",
                    stage=module.stage,
                )
            functions.append(function)
            continue
        materialized = TIRMicroKernelSelection(
            implementation=selection.implementation,
            family=selection.family,
            variant=selection.variant,
            parameters=selection.parameters,
            facts=selection.facts,
            requires=selection.requires,
            shared_workspaces=implementation.shared_workspaces,
            transfer_pipeline=implementation.transfer_pipeline,
        )
        validate_transfer_pipeline(
            function,
            dispatch,
            materialized,
            stage=module.stage,
        )
        rewritten = replace(
            dispatch,
            microkernel=materialized,
            shared_workspace_buffers=materialize_shared_workspace_buffers(
                function, materialized
            ),
        )
        functions.append(replace_kernel_dispatch(function, rewritten))
        changed = True
    return module if not changed else replace_kernel_callables(module, functions)



def _is_zero_copy_unpack(node: Node, module: IRModule) -> bool:
    """Prove that Unpack is a dense trailing-lane reinterpretation.

    Moving vector lanes into an earlier tensor axis requires a real
    permutation/copy and deliberately falls through to reviewed candidate
    selection.  Only trailing-axis lane folding preserves byte order.
    """

    if len(node.inputs) != 1:
        return False
    source_node = module.node_map[node.inputs[0]]
    source = tensor_of(source_node.type)
    if not isinstance(source.dtype, VectorType):
        return False
    if "axes" in node.attrs:
        raw_axes = tuple(int(value) for value in node.attrs["axes"])
    elif "axis" in node.attrs:
        raw_axes = (int(node.attrs["axis"]),) * len(source.dtype.lanes)
    else:
        return False
    try:
        axes = normalize_axes(raw_axes, source.rank)
        inferred = Unpack.infer_type((source_node,), node.attrs)
    except Exception:
        return False
    return (
        bool(axes)
        and all(axis == source.rank - 1 for axis in axes)
        and inferred == node.type
    )


def _is_zero_copy_pack(node: Node, module: IRModule) -> bool:
    """Prove that Pack only groups dense lanes on the trailing axis."""

    if len(node.inputs) != 1:
        return False
    source = tensor_of(module.node_map[node.inputs[0]].type)
    target = tensor_of(node.type)
    if isinstance(source.dtype, VectorType) or not isinstance(target.dtype, VectorType):
        return False
    raw_axes = node.attrs.get("axes")
    if raw_axes is None:
        raw_axis = node.attrs.get("axis")
        if raw_axis is None:
            return False
        raw_axes = (int(raw_axis),) * len(target.dtype.lanes)
    try:
        axes = normalize_axes(tuple(int(value) for value in raw_axes), source.rank)
    except Exception:
        return False
    return (
        bool(axes)
        and all(axis == source.rank - 1 for axis in axes)
        and source.dtype == target.dtype.elem_type
    )


def _is_zero_copy_reshape(node: Node, module: IRModule) -> bool:
    if len(node.inputs) != 1:
        return False
    source = tensor_of(module.node_map[node.inputs[0]].type)
    target = tensor_of(node.type)
    if source.dtype != target.dtype:
        return False
    source_shape = tuple(dimension.value for dimension in source.shape)
    target_shape = tuple(dimension.value for dimension in target.shape)
    if any(value is None for value in (*source_shape, *target_shape)):
        return False
    return prod(source_shape) == prod(target_shape)


__all__ = ["TritonTirLoweringPolicy"]
