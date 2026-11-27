# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Row projection with explicit BF16 rounding of contiguous K partitions."""

from itertools import combinations
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import (
    SBP,
    SBPBroadCast,
    SBPPartial,
    SBPSplit,
    local_tensor_type,
    scale_split_units,
)
from triton.flagmega.ir.model import DistributedType, DType, TensorType, VectorType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpCostFactors,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_tensor


def _chip_policy(policy, placement):
    return isinstance(policy, SBPBroadCast) or (
        isinstance(policy, SBPSplit) and policy.is_contiguous and len(policy.stages) == 1
        and policy.stages[0].distribution.granularity is None
        and all(not placement.is_physical_block_axis(axis) for axis in policy.hierarchy_axes)
    )


@op_definition("nn.partitioned_linear", namespace="nn", functional_name="partitioned_linear",
               display_name="NN.PartitionedLinear")
class PartitionedLinear(OpDefinition):
    """BF16 ``[1,K]`` times BF16 ``[N,K]``, split into P contiguous K parts.

    Each dot accumulates in FP32 and is rounded independently to BF16. Optional
    ``accumulation_splits`` divides each semantic part into equal contiguous
    sub-dots; their FP32 results add in split order before that BF16 boundary.
    No sub-dot is narrowed independently. The default one preserves old dots.
    The
    rounded parts are then accumulated in FP32 and rounded once to BF16.
    As with ReduceSum, FP32 reduction trees are implementation dependent;
    the per-part and final BF16 boundaries are mandatory. Packed input lanes
    belong only to K; packed result lanes belong only to N.

    N may be chip-sharded. K may instead be chip-sharded only when there is
    exactly one complete semantic partition per K owner. Its result is a BF16
    Sum partial: materialization widens those rounded partials to FP32 before
    the final BF16 rounding. Splitting a semantic partition or merging several
    parts into a rounded chip partial would change the contract and is illegal.
    """

    const_evaluable = True
    value = input_parameter(is_tensor())
    weight = input_parameter(is_tensor())
    partitions = attribute_parameter()
    output_lanes = attribute_parameter(default=())
    accumulation_splits = attribute_parameter(default=1)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        parts, lanes = attrs["partitions"], attrs["output_lanes"]
        splits = attrs["accumulation_splits"]
        if isinstance(parts, bool) or not isinstance(parts, int) or parts <= 0:
            raise IRSchemaError("PartitionedLinear partitions must be a positive integer.")
        if isinstance(splits, bool) or not isinstance(splits, int) or splits <= 0:
            raise IRSchemaError("PartitionedLinear accumulation_splits must be a positive integer.")
        if not isinstance(lanes, (tuple, list)) or any(
            isinstance(lane, bool) or not isinstance(lane, int) or lane <= 0 for lane in lanes
        ):
            raise IRSchemaError("PartitionedLinear output_lanes must be positive integers.")
        return {"partitions": parts, "output_lanes": tuple(lanes), "accumulation_splits": splits}

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items()
                if (key != "output_lanes" or value)
                and (key != "accumulation_splits" or value != 1)}

    @classmethod
    def infer_type(cls, inputs, attrs):
        x_kind, w_kind = (parameter.type_of(inputs) for parameter in cls.input_parameters)
        x, w = tensor_of(x_kind), tensor_of(w_kind)
        lanes, parts = attrs.get("output_lanes", ()), attrs["partitions"]
        splits = attrs.get("accumulation_splits", 1)
        input_lanes = getattr(x.dtype, "lane_count", 1)
        scalar = x.dtype.elem_type if isinstance(x.dtype, VectorType) else x.dtype
        if (x.rank != 2 or w.rank != 2 or scalar != DType.BFLOAT16 or w.dtype != DType.BFLOAT16
                or any(not d.is_fixed or d.fixed_value <= 0 for t in (x, w) for d in t.shape)
                or x.shape[0].fixed_value != 1):
            raise IRSchemaError("PartitionedLinear requires one BF16 token and scalar BF16 [N,K] weights.")
        k, n = w.shape[1].fixed_value, w.shape[0].fixed_value
        if x.shape[1].fixed_value * input_lanes != k or k % (parts * splits) or n % prod(lanes):
            raise IRSchemaError("PartitionedLinear K must match and divide into complete parts and accumulation splits; N must divide by lanes.")
        output = tensor_type(VectorType(DType.BFLOAT16, lanes) if lanes else DType.BFLOAT16,
                             (1, n // prod(lanes)))
        if not isinstance(x_kind, DistributedType) and not isinstance(w_kind, DistributedType):
            return output
        if not isinstance(x_kind, DistributedType) or not isinstance(w_kind, DistributedType):
            raise IRSchemaError("PartitionedLinear operands must both be distributed.")
        placement = x_kind.placement
        n_policy, k_policy = w_kind.axis_policies
        if (w_kind.placement != placement or x_kind.partial is not None or w_kind.partial is not None
                or x_kind.exclusive is not None or w_kind.exclusive is not None
                or x_kind.axis_policies[0] != SBP.broadcast()
                or not all(_chip_policy(policy, placement) for policy in (n_policy, k_policy))):
            raise IRSchemaError("PartitionedLinear requires complete contiguous chip ownership, no input partials.")
        packed_k = (scale_split_units(k_policy, 1, input_lanes)
                    if isinstance(k_policy, SBPSplit) else k_policy)
        if packed_k is None or x_kind.axis_policies[1] != packed_k:
            raise IRSchemaError("PartitionedLinear input and weight K owners must agree in scalar units.")
        partial = None
        if isinstance(k_policy, SBPSplit):
            owners = prod(placement.hierarchy[axis] for axis in k_policy.hierarchy_axes)
            if owners != parts or (k // parts) % input_lanes:
                raise IRSchemaError("PartitionedLinear K sharding requires exactly one complete partition per owner.")
            partial = SBPPartial(k_policy.hierarchy_axes)
        packed_n = (scale_split_units(n_policy, 1, prod(lanes))
                    if isinstance(n_policy, SBPSplit) else n_policy)
        n_owners = prod(placement.hierarchy[axis] for axis in n_policy.hierarchy_axes) if isinstance(n_policy, SBPSplit) else 1
        if packed_n is None or n % (n_owners * prod(lanes)):
            raise IRSchemaError("PartitionedLinear N owners must hold complete result lane groups.")
        return DistributedType(output, (SBP.broadcast(), packed_n), placement, partial=partial)

    @classmethod
    def infer_distributed_input_types(cls, output_type, logical_input_types, attrs):
        if (not isinstance(output_type, DistributedType) or output_type.exclusive is not None
                or output_type.axis_policies[0] != SBP.broadcast()
                or not all(isinstance(t, TensorType) for t in logical_input_types)):
            return ()
        x, w = logical_input_types
        placement, partial = output_type.placement, output_type.partial
        if partial is not None and partial != SBPPartial(partial.axes):
            return ()
        k_policy = SBP.split_contiguous(partial.axes) if partial else SBP.broadcast()
        n_policy = output_type.axis_policies[1]
        if isinstance(n_policy, SBPSplit):
            n_policy = scale_split_units(n_policy, prod(attrs.get("output_lanes", ())), 1)
        if n_policy is None:
            return ()
        return ((DistributedType(x, (SBP.broadcast(), k_policy), placement),
                 DistributedType(w, (n_policy, k_policy), placement)),)

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        x_choices = {(kind.placement, kind.axis_policies[1]): kind for kind in choices[0]
                     if isinstance(kind, DistributedType) and kind.axis_policies[0] == SBP.broadcast()
                     and kind.partial is None and kind.exclusive is None}
        for w in choices[1]:
            if not isinstance(w, DistributedType) or w.partial is not None or w.exclusive is not None:
                continue
            x_sample = next(iter(choices[0]), None)
            if x_sample is None:
                continue
            k_policy = w.axis_policies[1]
            if isinstance(k_policy, SBPSplit):
                k_policy = scale_split_units(k_policy, 1, getattr(tensor_of(x_sample).dtype, "lane_count", 1))
            x = x_choices.get((w.placement, k_policy))
            if x is not None:
                yield (x, w)

    @classmethod
    def distributed_output_type_candidates(cls, choices, output_type, attrs):
        results = []
        for placement in dict.fromkeys(kind.placement for options in choices for kind in options
                                       if isinstance(kind, DistributedType)):
            axes = tuple(i for i in range(len(placement.hierarchy)) if not placement.is_physical_block_axis(i))
            results.append(DistributedType(output_type, (SBP.broadcast(),) * 2, placement))
            for count in range(1, len(axes) + 1):
                for owner_axes in combinations(axes, count):
                    results.append(DistributedType(output_type, (SBP.broadcast(), SBP.split_contiguous(owner_axes)), placement))
                    if prod(placement.hierarchy[axis] for axis in owner_axes) == attrs["partitions"]:
                        results.append(DistributedType(output_type, (SBP.broadcast(),) * 2, placement,
                                                       partial=SBPPartial(owner_axes)))
        return tuple(results)

    @classmethod
    def evaluate(cls, node, arguments, context):
        x, w = arguments[0].reshape(1, -1).float(), arguments[1].float()
        parts, span = node.attrs["partitions"], w.shape[1] // node.attrs["partitions"]
        total = context.torch.zeros((1, w.shape[0]), dtype=context.torch.float32, device=x.device)
        for part in range(parts):
            start, end = part * span, (part + 1) * span
            if node.attrs.get("accumulation_splits", 1) == 1:
                rounded = (x[:, start:end] @ w[:, start:end].T).bfloat16()
            else:
                splits = node.attrs["accumulation_splits"]
                inner_span = span // splits
                partial = context.torch.zeros_like(total)
                for split in range(splits):
                    lower = start + split * inner_span
                    upper = lower + inner_span
                    partial = partial + x[:, lower:upper] @ w[:, lower:upper].T
                rounded = partial.bfloat16()
            total = total + rounded.float()
        kind = tensor_of(node.type)
        return total.bfloat16().reshape(*(d.fixed_value for d in kind.shape), *getattr(kind.dtype, "lanes", ()))

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("fp32-dot-bf16-parts-fp32-sum-bf16",))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        x, w = tuple(local_tensor_type(value.type) if isinstance(value.type, DistributedType) else value.type
                     for value in inputs)
        output = local_tensor_type(return_type) if isinstance(return_type, DistributedType) else return_type
        weight_kind = inputs[1].type
        parts = (1 if isinstance(weight_kind, DistributedType)
                 and isinstance(weight_kind.axis_policies[1], SBPSplit) else attrs["partitions"])
        split_additions = parts * (attrs.get("accumulation_splits", 1) - 1) * w.shape[0].fixed_value
        return OpCostFactors(cpu_cycles=2 * prod(d.fixed_value for d in w.shape) + split_additions,
                             chip_aggregate_memory_load_bytes=tensor_nbytes(x) + tensor_nbytes(w),
                             chip_aggregate_memory_store_bytes=tensor_nbytes(output))


__all__ = ["PartitionedLinear"]
