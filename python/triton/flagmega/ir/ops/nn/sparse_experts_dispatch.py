# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Expand tokens into explicitly owned TopK slots, including always-active slots."""

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import DistributedType, tensor_type
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.ops.core import OpDefinition, OpCost, OpCostFactors, input_parameter, op_definition, tensor_nbytes
from triton.flagmega.ir.ops.nn._sparse_experts import ROUTER_IDS, distributed_inputs, floating_tensor, role_axes


@op_definition("nn.sparse_experts_dispatch", namespace="nn", functional_name="sparse_experts_dispatch",
               display_name="NN.SparseExpertsDispatch")
class SparseExpertsDispatch(OpDefinition):
    value = input_parameter(floating_tensor(2, packed=True))
    router_expert_ids = input_parameter(ROUTER_IDS, memory_effect=MemoryEffect.NONE)
    supports_broadcast_lifting = False

    @classmethod
    def infer_type(cls, inputs, attrs):
        value, ids = (parameter.type_of(inputs) for parameter in cls.input_parameters)
        x, routes = tensor_of(value), tensor_of(ids)
        if x.shape[0] != routes.shape[0]:
            raise IRSchemaError("SparseExpertsDispatch token dimensions must match")
        output = tensor_type(x.dtype, (x.shape[0], routes.shape[1], x.shape[1]))
        placement = distributed_inputs({"value": value, "ids": ids})
        if placement is None:
            return output
        token, hidden = value.axis_policies
        if ids.axis_policies[0] != token:
            raise IRSchemaError("SparseExpertsDispatch token owners must match")
        route = ids.axis_policies[1]
        role_axes(token, route, hidden)
        return DistributedType(output, (token, route, hidden), placement)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value, ids = arguments
        return value.unsqueeze(1).expand(value.shape[0], ids.shape[1], *value.shape[1:]).contiguous()

    @classmethod
    def cost(cls, node):
        size = tensor_nbytes(node.type)
        return OpCost(bytes_read=size, bytes_written=size, notes=("token-to-route-expansion",))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        output = local_tensor_type(return_type) if isinstance(return_type, DistributedType) else return_type
        size = tensor_nbytes(output)
        if size is None:
            return None
        return OpCostFactors(block_local_memory_load_bytes=size, block_local_memory_store_bytes=size)
