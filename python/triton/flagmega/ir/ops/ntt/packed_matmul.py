# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Matrix multiplication over a typed-vector packed RHS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import placement_of, tensor_of
from triton.flagmega.ir.distributed_type import SBP, SBPPartial, SBPSplit, scale_split_units
from triton.flagmega.ir.model import (
    DType,
    DistributedType,
    IRType,
    Node,
    NoneType,
    TensorType,
    tensor_type,
)
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpCostFactors,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.type_pattern import is_none, is_tensor
from triton.flagmega.ir.types import VectorType
from triton.flagmega.ir.ops.math.matmul import matmul_value
from triton.flagmega.ir.vector_layout import split_vector_lanes


_OPTIONAL_TENSOR = is_tensor() | is_none()


@op_definition(
    "ntt.packed_matmul",
    namespace="ntt",
    functional_name="packed_matmul",
    display_name="NTT.PackedMatMul",
)
class PackedMatMul(OpDefinition):
    """``lhs @ rhs`` with an nncase-compatible typed-vector RHS.

    The K-major physical form is ``[KGroup, NGroup]`` with vector lanes
    ``(NVector, KPack, KVector)``.  Its scalar interpretation is the ordinary
    logical matrix ``[K, N]`` and the result retains ``NVector`` logical values,
    split into wider-output packets when necessary. Geometry remains part of the type rather
    than becoming target or model metadata.
    """

    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())
    scale = input_parameter(_OPTIONAL_TENSOR)
    addend = input_parameter(_OPTIONAL_TENSOR)
    fused_reduce = attribute_parameter(default=False)
    output_data_type = attribute_parameter(default=DType.FLOAT32)
    rhs_layout = attribute_parameter(default="k_major")

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        if attrs["rhs_layout"] != "k_major":
            raise IRSchemaError("PackedMatMul currently requires a K-major RHS.")
        return {
            "fused_reduce": bool(attrs["fused_reduce"]),
            "output_data_type": DType(attrs["output_data_type"]),
            "rhs_layout": "k_major",
        }

    @classmethod
    def ir_attrs(cls, attrs: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "fused_reduce": bool(attrs["fused_reduce"]),
            "output_data_type": DType(attrs["output_data_type"]).value,
            "rhs_layout": attrs["rhs_layout"],
        }

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        lhs_type = cls.lhs.type_of(inputs)
        rhs_type = cls.rhs.type_of(inputs)
        lhs = tensor_of(lhs_type)
        rhs = tensor_of(rhs_type)
        if lhs.rank != 2 or rhs.rank != 2:
            raise IRSchemaError("PackedMatMul currently requires rank-2 operands.")
        if isinstance(lhs.dtype, VectorType):
            raise IRSchemaError("PackedMatMul K-major lhs must have a scalar dtype.")
        if not isinstance(rhs.dtype, VectorType) or len(rhs.dtype.lanes) != 3:
            raise IRSchemaError(
                "PackedMatMul K-major RHS requires VectorType(NVector,KPack,KVector)."
            )
        if lhs.dtype != rhs.dtype.elem_type:
            raise IRSchemaError("PackedMatMul lhs and RHS scalar dtypes must match.")
        n_vector, k_pack, k_vector = rhs.dtype.lanes
        if lhs.shape[-1] != rhs.shape[0] * (k_pack * k_vector):
            raise IRSchemaError("PackedMatMul lhs K does not match its packed RHS K.")
        output_dtype = DType(attrs["output_data_type"])
        output_lanes, _ = split_vector_lanes((n_vector,), (1,), element_bytes=output_dtype.itemsize,
                                             vector_bytes=k_vector * rhs.dtype.elem_type.itemsize)
        output = tensor_type(
            VectorType(output_dtype, output_lanes),
            (lhs.shape[0], rhs.shape[1]),
            layout=lhs.layout,
        )
        placement = placement_of(lhs_type, rhs_type)
        if placement is None:
            result: IRType = output
        else:
            if not isinstance(lhs_type, DistributedType) or not isinstance(
                rhs_type, DistributedType
            ):
                raise IRSchemaError(
                    "Distributed PackedMatMul requires both matrix operands distributed."
                )
            if lhs_type.partial is not None or rhs_type.partial is not None:
                raise IRSchemaError("PackedMatMul operands cannot be partial values.")
            lhs_m, lhs_k = lhs_type.axis_policies
            rhs_k, rhs_n = rhs_type.axis_policies
            logical_rhs_k = (
                scale_split_units(rhs_k, k_pack * k_vector, 1)
                if isinstance(rhs_k, SBPSplit)
                else rhs_k
            )
            if lhs_k != logical_rhs_k:
                raise IRSchemaError(
                    "PackedMatMul lhs and RHS reduction policies must match."
                )
            reduction_axes = (
                tuple(lhs_k.hierarchy_axes) if isinstance(lhs_k, SBPSplit) else ()
            )
            output_axes = set(lhs_m.hierarchy_axes) if isinstance(lhs_m, SBPSplit) else set()
            output_axes.update(
                rhs_n.hierarchy_axes if isinstance(rhs_n, SBPSplit) else ()
            )
            if output_axes.intersection(reduction_axes):
                raise IRSchemaError(
                    "PackedMatMul output and reduction splits must use disjoint mesh axes."
                )
            partial = None
            if reduction_axes and not bool(attrs["fused_reduce"]):
                partial = SBPPartial(reduction_axes)
            result = DistributedType(output, (lhs_m, rhs_n), placement, partial=partial)

        scale_type = cls.scale.type_of(inputs)
        if not isinstance(scale_type, NoneType):
            scale = tensor_of(scale_type)
            if scale.rank != 0:
                raise IRSchemaError("PackedMatMul scale must be None or a scalar tensor.")
        addend_type = cls.addend.type_of(inputs)
        if not isinstance(addend_type, NoneType):
            if addend_type != result:
                raise IRSchemaError(
                    "PackedMatMul addend must have exactly the packed output type."
                )
            if isinstance(result, DistributedType) and result.partial is not None:
                raise IRSchemaError("PackedMatMul cannot add to an unmaterialized partial.")
        return result

    @classmethod
    def evaluate(cls, node, arguments, context):
        lhs = cls.lhs.read(arguments)
        rhs = cls.rhs.read(arguments)
        rhs_type = tensor_of(context.types[cls.rhs.read(node.inputs)])
        assert isinstance(rhs_type.dtype, VectorType)
        n_vector, k_pack, k_vector = rhs_type.dtype.lanes
        k_groups, n_groups = rhs_type.shape
        if not k_groups.is_fixed or not n_groups.is_fixed:
            raise IRSchemaError("PackedMatMul evaluation requires fixed RHS dimensions.")
        logical_rhs = rhs.reshape(
            k_groups.fixed_value,
            n_groups.fixed_value,
            n_vector,
            k_pack,
            k_vector,
        ).permute(0, 3, 4, 1, 2).reshape(
            k_groups.fixed_value * k_pack * k_vector,
            n_groups.fixed_value * n_vector,
        )
        value = matmul_value(lhs, logical_rhs.to(dtype=lhs.dtype), node.attrs["output_data_type"], context)
        scale = cls.scale.read(arguments)
        if scale is not None:
            value = value * scale
        value = value.to(
            dtype=context.torch_dtype(DType(node.attrs["output_data_type"]))
        )
        output_lanes = tensor_of(node.type).dtype.lanes
        value = value.reshape(value.shape[0], value.shape[1] // n_vector, *output_lanes)
        addend = cls.addend.read(arguments)
        return value if addend is None else value + addend

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost(
            bytes_written=tensor_nbytes(node.type),
            notes=("typed-vector-k-major-packed-matmul",),
        )

    @classmethod
    def cost_factors(
        cls,
        inputs: Sequence[Node],
        attrs: Mapping[str, object],
        return_type: IRType,
    ) -> OpCostFactors | None:
        """Describe the K-major SIMT GEMV candidate in target-scalable units.

        This follows nncase ``PackedMatMulEvaluator.TryGetTargetCost``.  The
        packed RHS and result are interpreted as their scalar logical matrix
        shapes for arithmetic, while memory traffic retains the physical
        typed-vector representation.  The SIMT padding is the regular Triton
        GEMV execution contract (M/N/K = 1/32/256 minima), not a model shape
        or an SM90-only choice.
        """

        lhs_type = cls.lhs.type_of(inputs)
        rhs_type = cls.rhs.type_of(inputs)
        addend_type = cls.addend.type_of(inputs)
        lhs = _local_cost_tensor(lhs_type)
        rhs = _local_cost_tensor(rhs_type)
        output = _local_cost_tensor(return_type)
        if (
            lhs.rank != 2
            or rhs.rank != 2
            or output.rank != 2
            or not isinstance(rhs.dtype, VectorType)
            or len(rhs.dtype.lanes) != 3
            or not isinstance(output.dtype, VectorType)
            or any(
                not dimension.is_fixed
                for tensor in (lhs, rhs, output)
                for dimension in tensor.shape
            )
        ):
            return None
        n_vector, k_pack, k_vector = rhs.dtype.lanes
        output_vector = prod(output.dtype.lanes)
        if n_vector != output_vector:
            return None
        m = lhs.shape[0].fixed_value
        k = lhs.shape[1].fixed_value
        rhs_k = rhs.shape[0].fixed_value * k_pack * k_vector
        n = output.shape[1].fixed_value * output_vector
        rhs_n = rhs.shape[1].fixed_value * n_vector
        if rhs_k != k or rhs_n != n:
            return None
        if m <= 1:
            padded_m = max(m, 0)
            padded_n = _ceil_div(max(n, 0), 32) * 32
            padded_k = _ceil_div(max(k, 0), 256) * 256
        else:
            padded_m = _ceil_div(max(m, 0), 16) * 16
            padded_n = _ceil_div(max(n, 0), 64) * 64
            padded_k = _ceil_div(max(k, 0), 64) * 64

        output_bytes = _fixed_tensor_nbytes(output)
        addend_bytes = 0
        addend_cycles = 0
        if not isinstance(addend_type, NoneType):
            addend = _local_cost_tensor(addend_type)
            if any(not dimension.is_fixed for dimension in addend.shape):
                return None
            addend_bytes = _fixed_tensor_nbytes(addend)
            # nncase CPU-cycle factors count the typed-vector outer shape;
            # lanes are represented by memory traffic and SIMT work.
            addend_cycles = prod(
                dimension.fixed_value for dimension in output.shape
            )

        chip_load = 0
        chip_store = 0
        all_reduce_cycles = 0
        grid_synchronizations = 0
        if (
            bool(attrs["fused_reduce"])
            and isinstance(lhs_type, DistributedType)
            and isinstance(lhs_type.axis_policies[-1], SBPSplit)
        ):
            chip_load = output_bytes * 2
            chip_store = output_bytes
            all_reduce_cycles = prod(
                dimension.fixed_value for dimension in output.shape
            )
            grid_synchronizations = 3

        return OpCostFactors(
            cpu_cycles=addend_cycles + all_reduce_cycles,
            simt_fma_operations=padded_m * padded_n * padded_k,
            block_local_memory_load_bytes=(
                _fixed_tensor_nbytes(lhs)
                + _fixed_tensor_nbytes(rhs)
                + addend_bytes
            ),
            block_local_memory_store_bytes=output_bytes,
            chip_global_memory_load_bytes=chip_load,
            chip_global_memory_store_bytes=chip_store,
            grid_synchronizations=grid_synchronizations,
        )


def _local_cost_tensor(value: IRType) -> TensorType:
    from triton.flagmega.ir.distributed_type import local_tensor_type

    return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)


def _fixed_tensor_nbytes(value: TensorType) -> int:
    return prod(dimension.fixed_value for dimension in value.shape) * value.dtype.itemsize


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


__all__ = ["PackedMatMul"]
