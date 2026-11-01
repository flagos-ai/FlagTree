# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block-scaled FP8 projection with an explicit, rounded K-partition axis."""

from itertools import combinations
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import (
    SBP,
    SBPSplit,
    local_tensor_type,
    scale_split_units,
)
from triton.flagmega.ir.model import DistributedType, TensorType, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpCostFactors,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.math._block_scaled import block_scaled_linear, normalize_output_lanes
from triton.flagmega.ir.ops.nn.partitioned_linear import _chip_policy
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType, VectorType


@op_definition("nn.partitioned_block_scaled_linear", namespace="nn",
               functional_name="partitioned_block_scaled_linear", display_name="NN.PartitionedBlockScaledLinear")
class PartitionedBlockScaledLinear(OpDefinition):
    """Project BF16 [T,K] and E4M3FN [N,K] into BF16 [P,T,N].

    Partition K into P equal, block-aligned contiguous intervals. Each part
    quantizes its original activation independently in weight_block_k groups,
    accumulates its block-scaled dot in FP32, and rounds to BF16. No reduction
    over P occurs: consumers may combine other rounded branches before summing.
    As with BlockScaledMatMul, FP32 dot association is implementation dependent.
    SM90 quantization uses ordered combined-scale FP32 FMA. The ordinary dynamic
    mode retains separately rounded scale multiplies and compensated addition.

    Final input lanes address K; final output lanes address N. N can be owned
    by contiguous chips. A P chip split requires matching K owners holding
    exactly one complete partition. P is a material axis, never a Partial sum.
    Scales are broadcast and indexed by global N/K quantization groups.
    """

    const_evaluable = True
    value = input_parameter(is_tensor())
    weight = input_parameter(is_tensor())
    weight_scale = input_parameter(is_tensor())
    partitions = attribute_parameter()
    weight_block_n = attribute_parameter(default=128)
    weight_block_k = attribute_parameter(default=128)
    activation_quantization = attribute_parameter(default="dynamic_fp8")
    output_lanes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        for name in ("partitions", "weight_block_n", "weight_block_k"):
            if type(attrs[name]) is not int or attrs[name] <= 0:
                raise IRSchemaError(f"PartitionedBlockScaledLinear {name} must be a positive integer.")
        if (not isinstance(attrs["activation_quantization"], str)
                or attrs["activation_quantization"] not in {"dynamic_fp8", "dynamic_fp8_sm90"}):
            raise IRSchemaError("PartitionedBlockScaledLinear requires dynamic_fp8 or dynamic_fp8_sm90 quantization.")
        attrs["output_lanes"] = normalize_output_lanes(attrs["output_lanes"], "PartitionedBlockScaledLinear")
        return attrs

    @classmethod
    def ir_attrs(cls, attrs):
        return {key: value for key, value in attrs.items() if key != "output_lanes" or value}

    @classmethod
    def infer_type(cls, inputs, attrs):
        kinds = tuple(parameter.type_of(inputs) for parameter in cls.input_parameters)
        x, w, scale = map(tensor_of, kinds)
        scalar = x.dtype.elem_type if isinstance(x.dtype, VectorType) else x.dtype
        if (x.rank != 2 or w.rank != 2 or scale.rank != 2 or scalar != DType.BFLOAT16
                or w.dtype != DType.FLOAT8_E4M3FN or scale.dtype != DType.FLOAT32
                or any(not d.is_fixed or d.fixed_value <= 0 for t in (x, w, scale) for d in t.shape)):
            raise IRSchemaError("PartitionedBlockScaledLinear requires fixed positive BF16 [T,K], scalar E4M3FN [N,K] and FP32 scales.")
        t, n, k = x.shape[0].fixed_value, w.shape[0].fixed_value, w.shape[1].fixed_value
        parts, bn, bk = (attrs[name] for name in ("partitions", "weight_block_n", "weight_block_k"))
        in_width, out_width = getattr(x.dtype, "lane_count", 1), prod(attrs.get("output_lanes", ()))
        if x.shape[1].fixed_value * in_width != k or k % (parts * bk) or n % out_width:
            raise IRSchemaError("PartitionedBlockScaledLinear K must match complete block-aligned parts and N complete output lanes.")
        if tuple(d.fixed_value for d in scale.shape) != ((n + bn - 1) // bn, k // bk):
            raise IRSchemaError("PartitionedBlockScaledLinear scales must cover the original global N/K blocks.")
        lanes = attrs.get("output_lanes", ())
        output = tensor_type(VectorType(DType.BFLOAT16, lanes) if lanes else DType.BFLOAT16,
                             (parts, t, n // out_width))
        if not any(isinstance(kind, DistributedType) for kind in kinds):
            return output
        if not all(isinstance(kind, DistributedType) for kind in kinds):
            raise IRSchemaError("PartitionedBlockScaledLinear requires every operand to name its placement.")
        x_kind, w_kind, s_kind = kinds
        placement = x_kind.placement
        n_policy, k_policy = w_kind.axis_policies
        b = SBP.broadcast()
        if (any(kind.placement != placement or kind.partial is not None or kind.exclusive is not None for kind in kinds)
                or x_kind.axis_policies[0] != b or any(policy != b for policy in s_kind.axis_policies)
                or not all(_chip_policy(policy, placement) for policy in (n_policy, k_policy))
                or set(getattr(n_policy, "hierarchy_axes", ())) & set(getattr(k_policy, "hierarchy_axes", ()))):
            raise IRSchemaError("PartitionedBlockScaledLinear needs complete chip owners, broadcast scales and no input partials.")
        packed_k = scale_split_units(k_policy, 1, in_width) if isinstance(k_policy, SBPSplit) else k_policy
        if packed_k is None or x_kind.axis_policies[1] != packed_k:
            raise IRSchemaError("PartitionedBlockScaledLinear input and weight must have matching scalar K ownership.")
        if isinstance(k_policy, SBPSplit) and (
                prod(placement.hierarchy[axis] for axis in k_policy.hierarchy_axes) != parts
                or (k // parts) % in_width):
            raise IRSchemaError("PartitionedBlockScaledLinear K owners must each hold one complete partition and lane group.")
        packed_n = scale_split_units(n_policy, 1, out_width) if isinstance(n_policy, SBPSplit) else n_policy
        owners_n = prod(placement.hierarchy[axis] for axis in getattr(n_policy, "hierarchy_axes", ()))
        if packed_n is None or n % (owners_n * out_width):
            raise IRSchemaError("PartitionedBlockScaledLinear N owners must hold complete output lane groups.")
        return DistributedType(output, (k_policy, b, packed_n), placement)

    @classmethod
    def infer_distributed_input_types(cls, output_type, logical_input_types, attrs):
        if (not isinstance(output_type, DistributedType) or output_type.partial is not None
                or output_type.exclusive is not None or output_type.axis_policies[1] != SBP.broadcast()
                or not all(isinstance(kind, TensorType) for kind in logical_input_types)):
            return ()
        x, w, scale = logical_input_types
        p_policy, _, n_policy = output_type.axis_policies
        b, placement = SBP.broadcast(), output_type.placement
        k_policy = scale_split_units(p_policy, 1, getattr(x.dtype, "lane_count", 1)) if isinstance(p_policy, SBPSplit) else p_policy
        n_policy = scale_split_units(n_policy, prod(attrs.get("output_lanes", ())), 1) if isinstance(n_policy, SBPSplit) else n_policy
        if k_policy is None or n_policy is None:
            return ()
        return ((DistributedType(x, (b, k_policy), placement),
                 DistributedType(w, (n_policy, p_policy), placement),
                 DistributedType(scale, (b, b), placement)),)

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        x_map = {(kind.placement, kind.axis_policies): kind for kind in choices[0]
                 if isinstance(kind, DistributedType) and kind.partial is None and kind.exclusive is None}
        s_map = {kind.placement: kind for kind in choices[2] if isinstance(kind, DistributedType)
                 and kind.partial is None and kind.exclusive is None
                 and all(policy == SBP.broadcast() for policy in kind.axis_policies)}
        sample = next(iter(choices[0]), None)
        if sample is None:
            return
        width = getattr(tensor_of(sample).dtype, "lane_count", 1)
        for w in choices[1]:
            if not isinstance(w, DistributedType) or w.partial is not None or w.exclusive is not None:
                continue
            k_policy = w.axis_policies[1]
            if isinstance(k_policy, SBPSplit):
                k_policy = scale_split_units(k_policy, 1, width)
            x, scale = x_map.get((w.placement, (SBP.broadcast(), k_policy))), s_map.get(w.placement)
            if x is not None and scale is not None:
                yield x, w, scale

    @classmethod
    def distributed_output_type_candidates(cls, choices, output_type, attrs):
        results = []
        b = SBP.broadcast()
        for placement in dict.fromkeys(kind.placement for options in choices for kind in options
                                       if isinstance(kind, DistributedType)):
            axes = tuple(i for i in range(len(placement.hierarchy)) if not placement.is_physical_block_axis(i))
            groups = [(), *(group for size in range(1, len(axes) + 1) for group in combinations(axes, size))]
            for p_axes in groups:
                if p_axes and prod(placement.hierarchy[i] for i in p_axes) != attrs["partitions"]:
                    continue
                for n_axes in groups:
                    if set(p_axes) & set(n_axes):
                        continue
                    results.append(DistributedType(output_type,
                        (SBP.split_contiguous(p_axes) if p_axes else b, b,
                         SBP.split_contiguous(n_axes) if n_axes else b), placement))
        return tuple(results)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value, weight, scales = arguments
        value = value.reshape(value.shape[0], -1)
        parts, bk = node.attrs["partitions"], node.attrs["weight_block_k"]
        span = weight.shape[1] // parts
        values = [block_scaled_linear(value[:, p * span:(p + 1) * span], weight[:, p * span:(p + 1) * span],
                    scales[:, p * span // bk:(p + 1) * span // bk], block_n=node.attrs["weight_block_n"], block_k=bk,
                    output_dtype=context.torch.bfloat16, activation_quantization=node.attrs["activation_quantization"])
                  for p in range(parts)]
        result = context.torch.stack(values)
        kind = tensor_of(node.type)
        return result.reshape(*(d.fixed_value for d in kind.shape), *getattr(kind.dtype, "lanes", ()))

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=tensor_nbytes(node.type), notes=("block-fp8-rounded-k-partitions",))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        del attrs
        x, weight, scale = (local_tensor_type(v.type) if isinstance(v.type, DistributedType) else v.type for v in inputs)
        output = local_tensor_type(return_type) if isinstance(return_type, DistributedType) else return_type
        return OpCostFactors(cpu_cycles=2 * x.shape[0].fixed_value * prod(d.fixed_value for d in weight.shape),
            chip_aggregate_memory_load_bytes=sum(tensor_nbytes(kind) for kind in (x, weight, scale)),
            chip_aggregate_memory_store_bytes=tensor_nbytes(output))


__all__ = ["PartitionedBlockScaledLinear"]
