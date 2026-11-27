# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Six BF16 projections of the same KDA decode input, before convolution/gates."""

from itertools import combinations
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import SBPBroadCast, SBPSplit, local_tensor_type
from triton.flagmega.ir.model import SBP, DistributedType, DType, TensorType, TupleType, VectorType, tensor_type
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

_NO_LANES = ((),) * 6


def _typed(tensor, placement, head, axis):
    policies = [SBP.broadcast()] * tensor.rank
    if axis is not None:
        policies[axis] = head
    return DistributedType(tensor, tuple(policies), placement)


def _chip_head(head, placement):
    # Blocks cooperate on complete chip tensors rather than owning head shards.
    return isinstance(head, SBPBroadCast) or (
        isinstance(head, SBPSplit) and head.is_contiguous
        and all(not placement.is_physical_block_axis(axis) for axis in head.hierarchy_axes)
    )


@op_definition("nn.kda_input_projection", namespace="nn", functional_name="kda_input_projection",
               display_name="NN.KdaInputProjection")
class KdaInputProjection(OpDefinition):
    """Return ``(q, k, v, beta_logits, forget_a, output_a)``.

    Q/K/V weights are head-major ``[H,D,K]`` views of the original row-major
    matrices; beta is ``[H,K]``, and the two independent low-rank weights are
    ``[Rf,K]``/``[Rg,K]``. Each projection accumulates in FP32 and rounds once
    to BF16. No convolution, sigmoid, scaling, bias, or state access is hidden.

    ``output_lanes`` contains six independent final-axis lane shapes. Input
    lanes also pack only the final K axis. Scalar byte order is unchanged.
    In the implemented distributed contract only head ownership varies across
    chips; K and both low-rank projections remain complete on each chip.
    """

    const_evaluable = True
    value = input_parameter(is_tensor())
    query_weight = input_parameter(is_tensor())
    key_weight = input_parameter(is_tensor())
    value_weight = input_parameter(is_tensor())
    beta_weight = input_parameter(is_tensor())
    forget_weight = input_parameter(is_tensor())
    output_weight = input_parameter(is_tensor())
    output_lanes = attribute_parameter(default=_NO_LANES)

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        lanes = attrs["output_lanes"]
        if not isinstance(lanes, (tuple, list)) or len(lanes) != 6 or any(
            not isinstance(field, (tuple, list)) or any(
                isinstance(lane, bool) or not isinstance(lane, int) or lane <= 0 for lane in field
            ) for field in lanes
        ):
            raise IRSchemaError("KdaInputProjection output_lanes requires six positive-integer lane shapes.")
        return {**attrs, "output_lanes": tuple(tuple(field) for field in lanes)}

    @classmethod
    def ir_attrs(cls, attrs):
        return {} if attrs["output_lanes"] == _NO_LANES else attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        kinds = tuple(parameter.type_of(inputs) for parameter in cls.input_parameters)
        x, q, k, v, b, f, g = tensors = tuple(tensor_of(kind) for kind in kinds)
        if any(not dim.is_fixed or dim.fixed_value <= 0 for t in tensors for dim in t.shape):
            raise IRSchemaError("KdaInputProjection requires fixed positive extents.")
        scalar = x.dtype.elem_type if isinstance(x.dtype, VectorType) else x.dtype
        if scalar != DType.BFLOAT16 or any(w.dtype != DType.BFLOAT16 for w in tensors[1:]):
            raise IRSchemaError("KdaInputProjection requires BF16 values and scalar BF16 weights.")
        if x.rank != 2 or x.shape[0].fixed_value != 1:
            raise IRSchemaError("KdaInputProjection supports one decode token [1,K].")
        reduction = x.shape[1].fixed_value * getattr(x.dtype, "lane_count", 1)
        if any(w.rank != 3 for w in (q, k, v)) or not q.shape == k.shape == v.shape:
            raise IRSchemaError("KdaInputProjection Q/K/V weights must share [heads,dim,K].")
        if any(w.rank != 2 for w in (b, f, g)) or b.shape[0] != q.shape[0]:
            raise IRSchemaError("KdaInputProjection requires beta [heads,K] and low-rank [rank,K].")
        if any(w.shape[-1].fixed_value != reduction for w in tensors[1:]):
            raise IRSchemaError("KdaInputProjection weights must match the complete input K.")
        heads, width = q.shape[0].fixed_value, q.shape[1].fixed_value
        shapes = ((1, heads, width),) * 3 + ((1, heads), (1, f.shape[0].fixed_value),
                                                          (1, g.shape[0].fixed_value))
        fields = []
        for shape, lanes in zip(shapes, attrs.get("output_lanes", _NO_LANES), strict=True):
            if shape[-1] % prod(lanes):
                raise IRSchemaError("KdaInputProjection result width must be divisible by its output_lanes.")
            fields.append(tensor_type(VectorType(DType.BFLOAT16, lanes) if lanes else DType.BFLOAT16,
                                      (*shape[:-1], shape[-1] // prod(lanes))))
        if not any(isinstance(kind, DistributedType) for kind in kinds):
            return TupleType(tuple(fields))
        if not all(isinstance(kind, DistributedType) for kind in kinds):
            raise IRSchemaError("KdaInputProjection operands must be all plain or all distributed.")
        placement, head = kinds[1].placement, kinds[1].axis_policies[0]
        expected = tuple(_typed(t, placement, head, 0 if 1 <= i <= 4 else None)
                         for i, t in enumerate(tensors))
        if kinds != expected or not _chip_head(head, placement):
            raise IRSchemaError("KdaInputProjection requires common chip head ownership and broadcast K/low ranks.")
        # Packing beta's head axis changes its ownership units, unlike packing
        # the within-head Q/K/V dimension. Keep that unsupported case explicit.
        if isinstance(head, SBPSplit) and attrs.get("output_lanes", _NO_LANES)[3]:
            raise IRSchemaError("Head-sharded KdaInputProjection beta must have scalar elements.")
        return TupleType(tuple(_typed(field, placement, head, 1 if i < 4 else None)
                               for i, field in enumerate(fields)))

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        del attrs
        keys = dict.fromkeys((kind.placement, kind.axis_policies[0])
                             for kind in choices[1] if isinstance(kind, DistributedType))
        for placement, head in keys:
            if not _chip_head(head, placement):
                continue
            values = tuple(next((kind for kind in options if isinstance(kind, DistributedType)
                                 and kind == _typed(kind.tensor, placement, head, 0 if 1 <= i <= 4 else None)), None)
                           for i, options in enumerate(choices))
            if all(kind is not None for kind in values):
                yield values

    @classmethod
    def distributed_output_type_candidates(cls, choices, output_type, attrs):
        del attrs
        results = []
        keys = dict.fromkeys((kind.placement, kind.axis_policies[0])
                             for options in choices[1:5] for kind in options if isinstance(kind, DistributedType))
        # Tuple results need explicit output seeds. A weight-side constant view
        # may not itself offer the head layout until this inverse demand exists.
        for placement in dict.fromkeys(kind.placement for options in choices for kind in options
                                       if isinstance(kind, DistributedType)):
            axes = tuple(i for i in range(len(placement.hierarchy)) if not placement.is_physical_block_axis(i))
            for count in range(1, len(axes) + 1):
                for owner_axes in combinations(axes, count):
                    keys[(placement, SBP.split_contiguous(owner_axes))] = None
        for placement, head in keys:
            if not _chip_head(head, placement):
                continue
            try:
                results.append(TupleType(tuple(_typed(tensor_of(field), placement, head, 1 if i < 4 else None)
                                               for i, field in enumerate(output_type.fields))))
            except IRSchemaError:
                continue
        return tuple(results)

    @classmethod
    def infer_distributed_input_types(cls, output_type, logical_input_types, attrs):
        del attrs
        if not isinstance(output_type, TupleType) or len(output_type.fields) != 6:
            return ()
        fields = output_type.fields
        if not all(isinstance(field, DistributedType) for field in fields):
            return ()
        placement, head = fields[0].placement, fields[0].axis_policies[1]
        if not _chip_head(head, placement) or any(
            field != _typed(field.tensor, placement, head, 1 if i < 4 else None)
            for i, field in enumerate(fields)
        ):
            return ()
        if any(not isinstance(t, TensorType) for t in logical_input_types):
            return ()
        return (tuple(_typed(t, placement, head, 0 if 1 <= i <= 4 else None)
                      for i, t in enumerate(logical_input_types)),)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments).reshape(1, -1).float()
        results = []
        for weight, field, lanes in zip(arguments[1:], node.type.fields, node.attrs.get("output_lanes", _NO_LANES), strict=True):
            # The evaluator's FP32 projection keeps the sole BF16 boundary at
            # the result. Packed types reshape that already rounded result.
            result = (value @ weight.reshape(-1, value.shape[-1]).float().T).bfloat16()
            shape = tuple(dim.fixed_value for dim in tensor_of(field).shape)
            results.append(result.reshape(*shape, *lanes))
        return tuple(results)

    @classmethod
    def cost(cls, node):
        fields = tuple(tensor_of(field) for field in node.type.fields)
        outputs = sum(prod(dim.fixed_value for dim in t.shape) * getattr(t.dtype, "lane_count", 1) for t in fields)
        # K is recoverable from input types only in cost factors; byte and
        # scalar result counts remain accurate for every lane representation.
        return OpCost(bytes_written=sum(tensor_nbytes(field) for field in node.type.fields),
                      notes=("six-independent-fp32-projections", f"scalar-results={outputs}"))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        del attrs
        tensors = tuple(local_tensor_type(value.type) if isinstance(value.type, DistributedType)
                        else value.type for value in inputs)
        fields = tuple(local_tensor_type(field) if isinstance(field, DistributedType) else field
                       for field in return_type.fields)
        reduction = tensors[0].shape[-1].fixed_value * getattr(tensors[0].dtype, "lane_count", 1)
        outputs = sum(tensor_nbytes(field) // 2 for field in fields)
        # Aggregate chip work is shared by CTAs, not repeated in every block.
        return OpCostFactors(cpu_cycles=2 * outputs * reduction,
                             chip_aggregate_memory_load_bytes=sum(tensor_nbytes(t) for t in tensors),
                             chip_aggregate_memory_store_bytes=2 * outputs)


__all__ = ["KdaInputProjection"]
