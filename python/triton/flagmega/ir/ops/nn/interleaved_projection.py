# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One native row projection with two independently stored interleaved fields."""

from itertools import combinations
from math import prod
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of, all_broadcast, broadcast_type
from triton.flagmega.ir.distributed_type import SBP, SBPBroadCast, SBPSplit, scale_split_units
from triton.flagmega.ir.model import DistributedType, DType, TupleType, VectorType, tensor_type
from triton.flagmega.ir.ops.core import OpDefinition, OpCost, OpCostFactors, input_parameter, attribute_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.math._quantized import matrix_width
from triton.flagmega.ir.ops.math._block_scaled import normalize_output_lanes, block_scaled_linear
from triton.flagmega.ir.type_pattern import is_tensor, has_rank


def _output(tensor, placement, head):
    return DistributedType(tensor, (SBP.broadcast(), head, SBP.broadcast()), placement)


@op_definition("nn.interleaved_projection", namespace="nn", functional_name="interleaved_projection",
               display_name="NN.InterleavedProjection")
class InterleavedProjection(OpDefinition):
    """Project original rows ``(head*2+field)*D+d`` once, then store each field.

    Native activations and FP16 header scales widen to FP32. Each projection
    sum rounds once to BF16 at its result; no activation quantization, copied
    weight recipe, or intermediate split tensor is implied. Fields have final
    D-axis dense packet lanes and share whole-head owners. K remains complete.
    """
    const_evaluable = True
    value = input_parameter(is_tensor() & has_rank(2))
    weight = input_parameter(is_tensor() & has_rank(2))
    weight_scale = input_parameter(is_tensor() & has_rank(2))
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()
    weight_encoding = attribute_parameter()
    feature_shape = attribute_parameter()
    field_count = attribute_parameter(default=2)
    output_lanes = attribute_parameter(default=((), ()))

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        shape = attrs["feature_shape"]
        if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
            raise IRSchemaError("InterleavedProjection feature_shape requires positive (heads, width).")
        if attrs["field_count"] != 2 or type(attrs["field_count"]) is not int:
            raise IRSchemaError("InterleavedProjection currently supports exactly two fields.")
        if attrs["weight_encoding"] not in {"IQ4_XS", "Q4_K"}:
            raise IRSchemaError("InterleavedProjection requires native IQ4_XS or Q4_K storage.")
        for name in ("weight_block_n", "weight_block_k"):
            if type(attrs[name]) is not int or attrs[name] <= 0:
                raise IRSchemaError(f"InterleavedProjection {name} must be positive.")
        lanes = attrs["output_lanes"]
        if not isinstance(lanes, (tuple, list)) or len(lanes) != 2:
            raise IRSchemaError("InterleavedProjection output_lanes requires two lane shapes.")
        return {**attrs, "feature_shape": tuple(shape),
                "output_lanes": tuple(normalize_output_lanes(x, "InterleavedProjection") for x in lanes)}

    @classmethod
    def ir_attrs(cls, attrs):
        return {k: v for k, v in attrs.items() if not (k == "output_lanes" and v == ((), ())) and k != "field_count"}

    @classmethod
    def infer_type(cls, inputs, attrs):
        kinds = tuple(p.type_of(inputs) for p in cls.input_parameters)
        value, weight, scale = tuple(tensor_of(t) for t in kinds)
        heads, width = attrs["feature_shape"]
        if any(not d.is_fixed or d.fixed_value <= 0 for d in value.shape):
            raise IRSchemaError("InterleavedProjection currently requires fixed positive activation extents.")
        if value.dtype not in {DType.BFLOAT16, DType.FLOAT32}:
            raise IRSchemaError("InterleavedProjection input must use scalar BF16 or FP32.")
        if value.shape[1] != matrix_width(weight, attrs["weight_encoding"]):
            raise IRSchemaError("InterleavedProjection value/weight logical K must match.")
        if not weight.shape[0].is_fixed or weight.shape[0].fixed_value != heads * 2 * width:
            raise IRSchemaError("InterleavedProjection native rows must equal heads*fields*width.")
        bn, bk = attrs["weight_block_n"], attrs["weight_block_k"]
        k = value.shape[1]
        if (scale.dtype != DType.FLOAT32 or not k.is_fixed
                or scale.shape != tensor_type(DType.FLOAT32, ((heads * 2 * width + bn - 1)//bn, (k.fixed_value+bk-1)//bk)).shape):
            raise IRSchemaError("InterleavedProjection requires logical-N/K FP32 placeholder scales.")
        fields = []
        for lanes in attrs.get("output_lanes", ((), ())):
            if width % prod(lanes):
                raise IRSchemaError("InterleavedProjection field width must divide output_lanes.")
            fields.append(tensor_type(VectorType(DType.BFLOAT16, lanes) if lanes else DType.BFLOAT16,
                                      (value.shape[0], heads, width // prod(lanes))))
        if not any(isinstance(t, DistributedType) for t in kinds):
            return TupleType(tuple(fields))
        if not all(isinstance(t, DistributedType) for t in kinds) or len({t.placement for t in kinds}) != 1:
            raise IRSchemaError("InterleavedProjection operands must share one placement.")
        if any(t.exclusive is not None for t in kinds):
            raise IRSchemaError("InterleavedProjection does not support exclusive distributed operands.")
        if not all_broadcast(kinds[0]) or not all_broadcast(kinds[2]) or kinds[1].partial is not None:
            raise IRSchemaError("InterleavedProjection requires complete K and broadcast activation/scales.")
        rows, reduction = kinds[1].axis_policies
        if not isinstance(reduction, SBPBroadCast):
            raise IRSchemaError("InterleavedProjection byte reduction axis must remain broadcast.")
        if isinstance(rows, SBPBroadCast):
            head = rows
        elif isinstance(rows, SBPSplit):
            # Scaling all split stages rejects a unit boundary inside a head.
            head = scale_split_units(rows, 1, 2 * width)
            if head is None:
                raise IRSchemaError("InterleavedProjection row split must own whole heads.")
        else:
            raise IRSchemaError("InterleavedProjection row owners must be broadcast or whole heads.")
        return TupleType(tuple(_output(field, kinds[0].placement, head) for field in fields))

    @classmethod
    def infer_distributed_input_types(cls, output_type, logical_input_types, attrs):
        if not isinstance(output_type, TupleType) or len(output_type.fields) != 2:
            return ()
        fields = output_type.fields
        if not all(isinstance(f, DistributedType) for f in fields):
            return ()
        placement, head = fields[0].placement, fields[0].axis_policies[1]
        if any(f != _output(f.tensor, placement, head) for f in fields):
            return ()
        rows = scale_split_units(head, 2 * attrs["feature_shape"][1], 1) if isinstance(head, SBPSplit) else head
        if rows is None:
            return ()
        x, w, s = (tensor_of(t) for t in logical_input_types)
        return ((broadcast_type(x, placement), DistributedType(w, (rows, SBP.broadcast()), placement),
                 broadcast_type(s, placement)),)

    @classmethod
    def distributed_output_type_candidates(cls, choices, output_type, attrs):
        del attrs
        results = []
        for placement in dict.fromkeys(t.placement for options in choices for t in options if isinstance(t, DistributedType)):
            axes = tuple(range(placement.rank))
            for count in range(1, len(axes) + 1):
                for selected in combinations(axes, count):
                    head = SBP.split_contiguous(selected)
                    try:
                        results.append(TupleType(tuple(_output(tensor_of(f), placement, head) for f in output_type.fields)))
                    except IRSchemaError:
                        continue
        return tuple(results)

    @classmethod
    def evaluate(cls, node, arguments, context):
        projected = block_scaled_linear(*arguments, block_n=node.attrs["weight_block_n"],
            block_k=node.attrs["weight_block_k"], weight_encoding=node.attrs["weight_encoding"], output_dtype=context.torch.bfloat16)
        heads, width = node.attrs["feature_shape"]
        projected = projected.reshape(projected.shape[0], heads, 2, width)
        return tuple(projected[:, :, index, :].contiguous().reshape(projected.shape[0], heads,
                    width//prod(lanes), *lanes) for index, lanes in enumerate(node.attrs.get("output_lanes", ((), ()))))

    @classmethod
    def cost(cls, node):
        return OpCost(bytes_written=sum(tensor_nbytes(tensor_of(t)) for t in node.type.fields),
                      notes=("single-native-interleaved-row-projection",))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        from triton.flagmega.ir.distributed_type import local_tensor_type
        fields = tuple(local_tensor_type(t) if isinstance(t, DistributedType) else t for t in return_type.fields)
        tensors = tuple(local_tensor_type(n.type) if isinstance(n.type, DistributedType) else tensor_of(n.type) for n in inputs)
        k = tensor_of(inputs[0].type).shape[1]
        if not k.is_fixed or any(not d.is_fixed for t in fields for d in t.shape):
            return None
        outputs = sum(tensor_nbytes(t)//2 for t in fields)
        return OpCostFactors(cpu_cycles=2*outputs*k.fixed_value,
                             chip_aggregate_memory_load_bytes=sum(tensor_nbytes(t) for t in tensors),
                             chip_aggregate_memory_store_bytes=2*outputs)
