# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense matmul with explicit residual value and additive statistics results."""

from __future__ import annotations

from typing import Mapping, Sequence
from dataclasses import replace
from math import prod

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import (
    DType,
    IRType,
    Node,
    NoneType,
    TupleType,
)
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.math.matmul import MatMul, matmul_value, normalize_output_data_type
from triton.flagmega.ir.ops.nn._norm import norm_stats_value, unpack_default_vector
from triton.flagmega.ir.ops.ntt.packed_matmul import PackedMatMul
from triton.flagmega.ir.ops.ntt.add_norm_stats import AddNormStats
from triton.flagmega.ir.ops.ntt._matmul_promotion import promoted_projection_type
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType


@op_definition(
    "ntt.matmul_norm_stats",
    namespace="ntt",
    functional_name="matmul_norm_stats",
    display_name="NTT.MatMulNormStats",
)
class MatMulNormStats(OpDefinition):
    """A directly implementable matmul/materialize/add/statistics operation."""

    const_evaluable = True
    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())
    addend = input_parameter(is_tensor())
    transpose_a = attribute_parameter(default=False)
    transpose_b = attribute_parameter(default=False)
    rhs_layout = attribute_parameter(default=None)
    axis = attribute_parameter()
    use_mean = attribute_parameter()
    addend_cast_dtypes = attribute_parameter(default=())
    output_data_type = attribute_parameter(default=None)
    inplace_output_parameters = (addend, None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        rhs_layout = attrs["rhs_layout"]
        if rhs_layout is None:
            matmul_attrs = MatMul.normalize_attrs({
                "transpose_a": attrs["transpose_a"],
                "transpose_b": attrs["transpose_b"],
            })
        elif rhs_layout == "k_major":
            if bool(attrs["transpose_a"]) or bool(attrs["transpose_b"]):
                raise IRSchemaError(
                    "Packed MatMulNormStats has fixed lhs @ unpack(rhs) semantics; "
                    "transpose flags must be false."
                )
            matmul_attrs = {"transpose_a": False, "transpose_b": False}
        else:
            raise IRSchemaError(
                "MatMulNormStats rhs_layout must be None or 'k_major'."
            )
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("MatMulNormStats axis must be an integer.")
        casts = attrs["addend_cast_dtypes"]
        if not isinstance(casts, (tuple, list)) or any(
            value not in ("bfloat16", "float32") for value in casts
        ):
            raise IRSchemaError("MatMulNormStats addend_cast_dtypes must be a BF16/FP32 conversion sequence.")
        return normalize_output_data_type({
            **({"addend_cast_dtypes": tuple(DType(value).value for value in casts)} if casts else {}),
            **matmul_attrs,
            "rhs_layout": rhs_layout,
            "axis": axis,
            "use_mean": bool(attrs["use_mean"]),
            "output_data_type": attrs.get("output_data_type"),
        })

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        lhs = cls.lhs.read(inputs)
        rhs = cls.rhs.read(inputs)
        addend = cls.addend.read(inputs)
        casts = attrs.get("addend_cast_dtypes", ())
        addend_dtype = tensor_of(addend.type).dtype
        if isinstance(addend_dtype, VectorType):
            addend_dtype = addend_dtype.elem_type
        if casts and casts[-1] != addend_dtype:
            raise IRSchemaError("MatMulNormStats addend conversion sequence must end in the addend element dtype.")
        rhs_layout = attrs.get("rhs_layout")
        if rhs_layout is None:
            matmul_type = MatMul.infer_type((lhs, rhs), attrs)
        else:
            none = Node("<none>", "builtin.none", (), NoneType())
            partial_attrs = {
                "fused_reduce": False,
                "output_data_type": attrs.get("output_data_type", DType.BFLOAT16),
                "rhs_layout": rhs_layout,
            }
            matmul_type = PackedMatMul.infer_type(
                (lhs, rhs, none, none), partial_attrs
            )
            if promoted_projection_type(matmul_type, addend.type) != addend.type:
                raise IRSchemaError(
                    "Packed MatMulNormStats requires matching projection and "
                    "addend local shards; keep layout publication explicit."
                )
        promoted_type = promoted_projection_type(matmul_type, addend.type)
        if promoted_type is None:
            raise IRSchemaError("MatMulNormStats requires matching projection/addend or a lossless BF16-to-FP32 promotion.")
        partial = Node(
            "<matmul_partial>",
            "builtin.var",
            (),
            promoted_type,
            attrs={"name": "<matmul_partial>"},
        )
        return AddNormStats.infer_type(
            (partial, addend),
            {"axis": attrs["axis"], "use_mean": attrs["use_mean"]},
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        lhs = cls.lhs.read(arguments)
        rhs = cls.rhs.read(arguments)
        if node.attrs["rhs_layout"] is None:
            if node.attrs["transpose_a"]:
                lhs = lhs.transpose(-2, -1)
            if node.attrs["transpose_b"]:
                rhs = rhs.transpose(-2, -1)
            projected = matmul_value(lhs, rhs, node.attrs.get("output_data_type"), context)
        else:
            rhs_type = tensor_of(context.types[cls.rhs.read(node.inputs)])
            if not isinstance(rhs_type.dtype, VectorType):
                raise IRSchemaError(
                    "Packed MatMulNormStats evaluation requires a typed-vector RHS."
                )
            n_vector, k_pack, k_vector = rhs_type.dtype.lanes
            k_groups, n_groups = rhs_type.shape
            if not k_groups.is_fixed or not n_groups.is_fixed:
                raise IRSchemaError(
                    "Packed MatMulNormStats evaluation requires fixed RHS dimensions."
                )
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
            projected = matmul_value(lhs, logical_rhs.to(dtype=lhs.dtype),
                                     node.attrs.get("output_data_type", DType.BFLOAT16), context)
        value_type = tensor_of(context.types[cls.addend.read(node.inputs)])
        addend = cls.addend.read(arguments)
        for dtype in node.attrs.get("addend_cast_dtypes", ()):
            addend = addend.to(dtype=context.torch_dtype(DType(dtype)))
        projected = projected.to(dtype=addend.dtype)
        if isinstance(value_type.dtype, VectorType):
            projected = projected.reshape(
                *projected.shape[:-1],
                projected.shape[-1] // prod(value_type.dtype.lanes),
                *value_type.dtype.lanes,
            )
        value = (projected + addend).to(dtype=addend.dtype)
        logical_value = unpack_default_vector(value, value_type)
        return (
            value,
            norm_stats_value(
                logical_value,
                axis=int(node.attrs["axis"]),
                use_mean=bool(node.attrs["use_mean"]),
            ),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        if not isinstance(node.type, TupleType):
            return OpCost(notes=("invalid-matmul-norm-stats-type",))
        value = tensor_of(node.type.fields[0])
        stats = tensor_of(node.type.fields[1])
        return OpCost(
            flops=None,
            bytes_read=None,
            bytes_written=_sum_optional(tensor_nbytes(value), tensor_nbytes(stats)),
            communication_bytes=None,
            synchronizations=None,
            model="flagmega.matmul-norm-stats/v1",
            notes=("matmul-materialize-residual-add-norm-stats",),
        )

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        """Price the local epilogue without an intermediate projection load/store.

        Output-N owners publish additive statistics. Their later collective
        remains a separate Boxing edge, not an implicit matmul synchronization.
        """
        if attrs.get("rhs_layout") != "k_major" or not isinstance(return_type, TupleType):
            return None
        lhs, rhs, addend = inputs
        none = Node("<none>", "builtin.none", (), NoneType())
        matmul_inputs = (lhs, rhs, none, none)
        matmul_attrs = {
            "fused_reduce": False,
            "rhs_layout": "k_major",
            "output_data_type": attrs.get("output_data_type", DType.BFLOAT16),
        }
        projection_type = PackedMatMul.infer_type(matmul_inputs, matmul_attrs)
        if getattr(projection_type, "partial", None) is not None:
            return None
        matmul = PackedMatMul.cost_factors(matmul_inputs, matmul_attrs, projection_type)
        projection = Node("<projection>", "builtin.var", (), addend.type)
        epilogue = AddNormStats.cost_factors((projection, addend), attrs, return_type)
        if matmul is None or epilogue is None:
            return None
        from triton.flagmega.ir.distributed_type import local_tensor_type
        from triton.flagmega.ir import DistributedType

        local_addend = local_tensor_type(addend.type) if isinstance(addend.type, DistributedType) else addend.type
        addend_bytes = tensor_nbytes(local_addend)
        if addend_bytes is None:
            return None
        return replace(
            matmul,
            cpu_cycles=matmul.cpu_cycles + epilogue.cpu_cycles,
            elementwise_operations=matmul.elementwise_operations + epilogue.elementwise_operations,
            block_local_memory_load_bytes=matmul.block_local_memory_load_bytes + addend_bytes,
            block_local_memory_store_bytes=epilogue.block_local_memory_store_bytes,
        )


def _sum_optional(lhs: int | None, rhs: int | None) -> int | None:
    return None if lhs is None or rhs is None else lhs + rhs


__all__ = ["MatMulNormStats"]
