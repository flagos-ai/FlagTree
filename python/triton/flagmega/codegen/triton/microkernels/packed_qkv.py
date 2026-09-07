# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Portable selector contract for canonical fused-RHS Packed-QKV TIR."""

from __future__ import annotations

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import (
    DType,
    DistributedType,
    NoneType,
    ReduceOp,
    SBPSplit,
    TensorType,
    VectorType,
    logical_type,
)
from triton.flagmega.ir.distributed_type import local_shape
from triton.flagmega.ir.distributed_type import (
    is_fully_sharded_across_placement,
    placement_owner_count,
)

from .core import TIRMicroKernelContext, TIRMicroKernelProposal


class PackedQKVMicroKernelProvider:
    """Map canonical Packed-QKV semantics to an injected implementation family.

    Tile sizes, pipeline depths, instruction capabilities, and platform names
    belong to ``TritonImplementationModel`` entries supplied by the machine.
    """

    op_names = frozenset({"ntt.packed_qkv_parallel_linear_fused_rhs"})
    family = "qkv_parallel_linear"

    def propose(
        self, context: TIRMicroKernelContext
    ) -> TIRMicroKernelProposal | None:
        dispatch = context.dispatch
        rhs_layout = str(dispatch.semantic_attrs.get("rhs_layout", ""))
        capacities = dispatch.semantic_attrs.get("projection_n_capacities")
        if rhs_layout != "k_major":
            raise CodegenError(
                "Canonical Packed-QKV microkernel selection requires K-major RHS."
            )
        if (
            not isinstance(capacities, tuple)
            or len(capacities) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in capacities)
        ):
            raise CodegenError(
                "Canonical Packed-QKV microkernel selection requires three positive "
                "projection N capacities."
            )
        if len(dispatch.outputs) != 3:
            raise CodegenError(
                "Canonical Packed-QKV microkernel selection requires Q/K/V outputs."
            )
        implementations = tuple(
            implementation
            for implementation in context.implementations(
                self.family,
                input_kind="fused_rhs",
                rhs_layout=rhs_layout,
            )
            if _applicable(context, implementation.contract)
        )
        if not implementations:
            raise CodegenError(
                f"Implementation model {context.implementation_model.name!r} has no "
                "canonical Packed-QKV implementation."
            )
        candidates = tuple(context.candidate(value) for value in implementations)
        return TIRMicroKernelProposal(
            candidates,
            context.choose_default(self.family, candidates),
        )


def _applicable(context: TIRMicroKernelContext, contract) -> bool:
    """Match optional target-owned profile constraints to semantic local ABI."""

    dispatch = context.dispatch
    parameters = context.function.parameter_map
    try:
        source_type = parameters[dispatch.arguments[0]].type
        weight_type = logical_type(parameters[dispatch.arguments[1]].type)
        output_types = tuple(parameters[name].type for name in dispatch.outputs)
    except (IndexError, KeyError):
        return False
    source = logical_type(source_type)
    outputs = tuple(logical_type(value) for value in output_types)
    if (
        not isinstance(source, TensorType)
        or not isinstance(weight_type, TensorType)
        or any(not isinstance(value, TensorType) for value in outputs)
        or source.rank != 2
        or weight_type.rank not in {2, 3}
        or any(value.rank != 2 for value in outputs)
    ):
        return False
    source_shape = _local_fixed_shape(source_type)
    output_shapes = tuple(_local_fixed_shape(value) for value in output_types)
    if source_shape is None or any(value is None for value in output_shapes):
        return False
    assert all(value is not None for value in output_shapes)
    output_shapes = tuple(value for value in output_shapes if value is not None)
    source_lanes = _lanes(source.dtype)
    weight_lanes = _lanes(weight_type.dtype)
    output_lanes = tuple(_lanes(value.dtype) for value in outputs)
    observed = {
        "required_input_dtype": _scalar_dtype(source.dtype).value,
        "required_weight_dtype": _scalar_dtype(weight_type.dtype).value,
        "required_input_lanes": source_lanes,
        "required_weight_lanes": weight_lanes,
        "required_weight_rank": weight_type.rank,
        "required_output_lanes": output_lanes[0],
        "required_local_rows": source_shape[-2],
        "required_local_reduction_extent": source_shape[-1]
        * _lane_count(source.dtype),
        "required_local_output_extent": sum(
            shape[-1] * _lane_count(value.dtype)
            for shape, value in zip(output_shapes, outputs, strict=True)
        ),
    }
    for key, value in observed.items():
        required = contract.get(key)
        if required is not None and required != value:
            return False
    required_output_dtype = contract.get("required_output_dtype")
    if required_output_dtype is not None and any(
        _scalar_dtype(value.dtype).value != required_output_dtype
        for value in outputs
    ):
        return False
    if contract.get("requires_matching_output_lanes", False) and any(
        lanes != output_lanes[0] for lanes in output_lanes[1:]
    ):
        return False
    if contract.get("requires_full_output_placement_ownership", False) and any(
        not isinstance(value, DistributedType)
        or not is_fully_sharded_across_placement(value)
        for value in output_types
    ):
        return False
    if contract.get("requires_weight_owner_count_match", False):
        if (
            any(not isinstance(value, DistributedType) for value in output_types)
            or len({placement_owner_count(value) for value in output_types}) != 1
            or not weight_type.shape[0].is_fixed
            or weight_type.shape[0].fixed_value
            != placement_owner_count(output_types[0])
        ):
            return False
    required_partial_reduce_op = contract.get(
        "required_output_partial_reduce_op"
    )
    if required_partial_reduce_op is not None:
        if any(
            not isinstance(value, DistributedType)
            or value.partial is None
            or value.partial.reduce_op.value != required_partial_reduce_op
            for value in output_types
        ):
            return False
    if contract.get("requires_partial_axes_match_input_reduction_ownership", False):
        if not isinstance(source_type, DistributedType):
            return False
    if contract.get("requires_uniform_full_input_reduction_tiles", False):
        if not isinstance(source_type, DistributedType):
            return False
        reduction_policy = source_type.axis_policies[-1]
        if (
            not isinstance(reduction_policy, SBPSplit)
            or not source.shape[-1].is_fixed
        ):
            return False
        reduction_owner_count = 1
        for axis in reduction_policy.hierarchy_axes:
            reduction_owner_count *= source_type.placement.hierarchy[axis]
        if (
            source.shape[-1].fixed_value * _lane_count(source.dtype)
            != source_shape[-1]
            * _lane_count(source.dtype)
            * reduction_owner_count
        ):
            return False
        reduction_policy = source_type.axis_policies[-1]
        if not isinstance(reduction_policy, SBPSplit):
            return False
        reduction_axes = frozenset(reduction_policy.hierarchy_axes)
        if any(
            not isinstance(value, DistributedType)
            or value.partial is None
            or value.partial.reduce_op is not ReduceOp.SUM
            or frozenset(value.partial.axes) != reduction_axes
            for value in output_types
        ):
            return False
    if contract.get("requires_none_optional_inputs", False) and any(
        not isinstance(parameters[name].type, NoneType)
        for name in dispatch.arguments[2:]
    ):
        return False
    return True


def _local_fixed_shape(value) -> tuple[int, ...] | None:
    tensor = logical_type(value)
    if not isinstance(tensor, TensorType):
        return None
    shape = local_shape(value) if isinstance(value, DistributedType) else tensor.shape
    if any(not dimension.is_fixed for dimension in shape):
        return None
    return tuple(dimension.fixed_value for dimension in shape)


def _scalar_dtype(value) -> DType:
    return value.elem_type if isinstance(value, VectorType) else value


def _lanes(value) -> tuple[int, ...]:
    return tuple(value.lanes) if isinstance(value, VectorType) else ()


def _lane_count(value) -> int:
    result = 1
    for lane in _lanes(value):
        result *= lane
    return result


__all__ = ["PackedQKVMicroKernelProvider"]
