# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense BF16 gate/up projections with SiLU GLU."""

from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import all_broadcast, placement_of, split_policy, tensor_of
from triton.flagmega.ir.model import DistributedType, IRType, Node, SBP, tensor_type
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, attribute_parameter, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.type_pattern import has_rank, is_tensor


@op_definition(
    "nn.dense_matmul_glu",
    namespace="nn",
    functional_name="dense_matmul_glu",
    display_name="NN.DenseMatMulGlu",
)
class DenseMatMulGlu(OpDefinition):
    value = input_parameter(is_tensor() & has_rank(2))
    gate_weight = input_parameter(is_tensor() & has_rank(2))
    up_weight = input_parameter(is_tensor() & has_rank(2))
    activation = attribute_parameter(default="silu")
    round_activation = attribute_parameter(default=True)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if attrs["activation"] != "silu":
            raise IRSchemaError("DenseMatMulGlu currently supports only SiLU.")
        if not isinstance(attrs["round_activation"], bool):
            raise IRSchemaError("DenseMatMulGlu round_activation must be boolean.")
        if attrs["round_activation"]:
            attrs.pop("round_activation")  # Preserve the existing default IR encoding.
        return attrs

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value_type = cls.value.type_of(inputs)
        gate_type = cls.gate_weight.type_of(inputs)
        up_type = cls.up_weight.type_of(inputs)
        value = tensor_of(value_type)
        gate = tensor_of(gate_type)
        up = tensor_of(up_type)
        if gate != up:
            raise IRSchemaError("DenseMatMulGlu gate/up weight types must match.")
        if value.dtype != gate.dtype or value.shape[-1] != gate.shape[-1]:
            raise IRSchemaError("DenseMatMulGlu value and weights have incompatible dtype/K dimension.")
        output = tensor_type(value.dtype, [value.shape[0], gate.shape[0]])
        placement = placement_of(value_type, gate_type, up_type)
        if placement is None:
            return output
        if not all(isinstance(item, DistributedType) for item in (value_type, gate_type, up_type)):
            raise IRSchemaError("Distributed DenseMatMulGlu requires all operands to name a placement.")
        if all_broadcast(value_type) and all_broadcast(gate_type) and all_broadcast(up_type):
            return DistributedType(output, (SBP.broadcast(), SBP.broadcast()), placement)
        gate_split = split_policy(gate_type, 0)
        up_split = split_policy(up_type, 0)
        if all_broadcast(value_type) and gate_split is not None and gate_split == up_split:
            if gate_type.axis_policies[1] == SBP.broadcast() and up_type.axis_policies[1] == SBP.broadcast():
                return DistributedType(output, (SBP.broadcast(), gate_split), placement)
        raise IRSchemaError("Distributed DenseMatMulGlu requires replicated input and identical gate/up N splits.")

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        gate = context.torch.nn.functional.linear(value, cls.gate_weight.read(arguments))
        up = context.torch.nn.functional.linear(value, cls.up_weight.read(arguments))
        if not node.attrs.get("round_activation", True):
            return (context.torch.nn.functional.silu(gate.float()) * up.float()).to(value.dtype)
        return context.torch.nn.functional.silu(gate) * up

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(flops=None, bytes_read=None, bytes_written=tensor_nbytes(node.type), notes=("two-dense-matmuls+glu",))


__all__ = ["DenseMatMulGlu"]
