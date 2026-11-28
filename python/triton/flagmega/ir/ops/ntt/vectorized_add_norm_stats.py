# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed residual addition with explicit ordered owner-local statistics."""
from math import prod

from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.model import DistributedType, Node, TupleType
from triton.flagmega.ir.ops.core import (OpDefinition, OpCost, OpCostFactors, attribute_parameter, input_parameter,
                                         op_definition, tensor_elements, tensor_nbytes)
from triton.flagmega.ir.ops.math.vectorized_binary import VectorizedBinary
from triton.flagmega.ir.ops.nn._norm import normalize_axis
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import DType, VectorType

LOCAL16_ORDER = "square_rn_xor8_fma_xor4_2_1_lane0"
_BINARY_KEYS = ("binary_op", "lhs_pack", "rhs_pack", "result_unpack")


def binary_attrs(attrs):
    return {key: attrs[key] for key in _BINARY_KEYS if key in attrs}


@op_definition("ntt.vectorized_add_norm_stats", namespace="ntt", functional_name="vectorized_add_norm_stats",
               display_name="NTT.VectorizedAddNormStats")
class VectorizedAddNormStats(OpDefinition):
    """Preserve the raw VectorizedBinary result and one local16 statistic.

    For raw lane j, compute RN(x_j*x_j), XOR8 peer square, then
    FMA_RN(x_j,x_j,peer_square), followed by RN XOR4,2,1 additions.
    Logical local lane zero names the statistic, independently of the physical
    thread chosen by code generation. Other butterfly replicas need not agree.
    No input partial is materialized and no output overwrites an operand.
    """
    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())
    binary_op = attribute_parameter(default="add")
    lhs_pack = attribute_parameter(default=None)
    rhs_pack = attribute_parameter(default=None)
    result_unpack = attribute_parameter(default=None)
    axis = attribute_parameter(default=-1)
    use_mean = attribute_parameter(default=False)
    local_reduction_order = attribute_parameter()
    supports_broadcast_lifting = False
    supports_materialized_pre_ops_partial_results = True

    @classmethod
    def normalize_attrs(cls, attributes):
        attrs = super().normalize_attrs(attributes)
        binary = VectorizedBinary.normalize_attrs(binary_attrs(attrs))
        if binary["binary_op"] != "add":
            raise IRSchemaError("VectorizedAddNormStats requires addition.")
        if type(attrs["axis"]) is not int or attrs["use_mean"] is not False:
            raise IRSchemaError("VectorizedAddNormStats requires an integer axis and use_mean=False.")
        if attrs["local_reduction_order"] != LOCAL16_ORDER:
            raise IRSchemaError("VectorizedAddNormStats requires the explicit local16 canonical-lane order.")
        return {**binary, "axis": attrs["axis"], "use_mean": False, "local_reduction_order": LOCAL16_ORDER}

    @classmethod
    def infer_type(cls, inputs, attrs):
        raw = VectorizedBinary.infer_type(inputs, binary_attrs(attrs))
        tensor = tensor_of(raw)
        scalar = tensor.dtype.elem_type if isinstance(tensor.dtype, VectorType) else tensor.dtype
        axis = normalize_axis(attrs["axis"], tensor.rank)
        if scalar is not DType.FLOAT32 or axis != tensor.rank - 1:
            raise IRSchemaError("VectorizedAddNormStats requires a last-axis FP32 raw result.")
        # NormStats owns the complete/raw-input and partial-axis contract.
        raw_node = Node("<raw>", "builtin.var", (), raw, attrs={"name": "<raw>"})
        stats = NormStats.infer_type((raw_node, ), {"axis": axis, "use_mean": False})
        local = local_tensor_type(raw) if isinstance(raw, DistributedType) else tensor
        lanes = local.dtype.lane_count if isinstance(local.dtype, VectorType) else 1
        if not local.shape[axis].is_fixed or local.shape[axis].fixed_value * lanes != 16:
            raise IRSchemaError("VectorizedAddNormStats requires exactly16 owner-local reduction scalars.")
        return TupleType((raw, stats))

    @classmethod
    def evaluate(cls, node, arguments, context):
        raise EvaluationError(
            "Exact local16 owner-partial evaluation requires the explicit owner ABI; generic dense sum is unsupported.")

    @classmethod
    def cost(cls, node):
        raw, stats = node.type.fields
        raw_tensor, stats_tensor = tensor_of(raw), tensor_of(stats)
        # tensor_elements already includes every vector lane. This logical
        # diagnostic is separate from per-owner storage in cost_factors.
        elements = tensor_elements(raw_tensor)
        size = tensor_nbytes(raw_tensor)
        writes = (tensor_nbytes(raw_tensor), tensor_nbytes(stats_tensor))
        return OpCost(flops=None if elements is None else 3 * elements, bytes_read=None if size is None else 2 * size,
                      bytes_written=None if None in writes else sum(writes),
                      notes=("vectorized-add+ordered-local16-statistics", ))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        local = lambda ty: local_tensor_type(ty) if isinstance(ty, DistributedType) else tensor_of(ty)
        tensors = tuple(local(n.type) for n in inputs)
        raw, stats = (local(ty) for ty in return_type.fields)
        if any(not d.is_fixed for t in (*tensors, raw, stats) for d in t.shape):
            return None
        elements = prod(d.fixed_value
                        for d in raw.shape) * (raw.dtype.lane_count if isinstance(raw.dtype, VectorType) else 1)
        return OpCostFactors(elementwise_operations=3 * elements,
                             block_local_memory_load_bytes=sum(tensor_nbytes(t) for t in tensors),
                             block_local_memory_store_bytes=tensor_nbytes(raw) + tensor_nbytes(stats))


__all__ = ["VectorizedAddNormStats", "LOCAL16_ORDER", "binary_attrs"]
