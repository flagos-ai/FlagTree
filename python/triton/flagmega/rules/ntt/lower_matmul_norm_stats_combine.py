# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lower a private MatMul producer plus combine to one explicit multi-result op."""

from __future__ import annotations

from triton.flagmega.ir import DType, DistributedType, IRModule, Node, NoneType, TupleType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.types import VectorType
from triton.flagmega.ir.ops.ntt.matmul_norm_stats import MatMulNormStats
from triton.flagmega.ir.ops.ntt._matmul_promotion import promoted_projection_type, is_projection_promotion
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteRule


def lower_matmul_norm_stats_combine_rule() -> RewriteRule:
    pattern = F.ntt.is_matmul_norm_stats_combine(
        wildcard("input", type_pattern=MatMulNormStats.lhs.type_pattern),
        wildcard("addend", type_pattern=MatMulNormStats.addend.type_pattern),
        target_name="target",
        call_name="call",
    )

    def rewrite(result, module: IRModule):
        source = result["call"]
        projection = result["input"]
        addend = result["addend"]
        assert all(isinstance(value, Node) for value in (source, projection, addend))
        producer, adapters = _projection_producer(projection, module)
        # Match nncase's LowerMaterializedPackedMatMulNormStatsCombine rule.
        # This direct multi-result op is the packed-matmul microkernel form;
        # ordinary dense matmul remains paired with the target-neutral combine
        # and is selected as two independently implementable TIR roles.
        if producer.op != "ntt.packed_matmul":
            return source
        # a partial producer must remain an explicit combine so TIR selection
        # lowers it to GatherReduceAddNormStats.  Folding a partial producer
        # into a machine-flavoured MatMul kernel here both loses the portable
        # collective boundary and makes this graph rule target dependent.
        if isinstance(producer.type, DistributedType) and producer.type.partial is not None:
            return source
        if (
            not isinstance(source.type, TupleType)
            or len(source.type.fields) != 2
            # An adapter may publish Split storage as Broadcast. Folding it
            # into a local matmul epilogue would consume remote owners before
            # their publication boundary and compute the wrong statistics.
            or promoted_projection_type(producer.type, source.type.fields[0]) != source.type.fields[0]
            or projection.type != source.type.fields[0]
            or addend.type != source.type.fields[0]
        ):
            return source
        chain = (producer.id, *adapters)
        expected_users = (*adapters, source.id)
        if any(
            _users(module, node_id) != (expected,)
            for node_id, expected in zip(chain, expected_users)
        ):
            return source
        exported = any(
            node_id in function.outputs
            for function in module.functions
            for node_id in chain
        )
        if exported:
            return source
        if (
            len(producer.inputs) != 4
            or bool(producer.attrs["fused_reduce"])
            or DType(producer.attrs["output_data_type"]) not in {DType.BFLOAT16, DType.FLOAT32}
            or not isinstance(module.node_map[producer.inputs[2]].type, NoneType)
            or not isinstance(module.node_map[producer.inputs[3]].type, NoneType)
        ):
            return source
        addend, addend_casts = _private_addend_casts(addend, source.id, module)
        inputs = (
            module.node_map[producer.inputs[0]],
            module.node_map[producer.inputs[1]],
            addend,
        )
        fusion_attrs = {
            "transpose_a": False,
            "transpose_b": False,
            "rhs_layout": str(producer.attrs["rhs_layout"]),
            "output_data_type": producer.attrs["output_data_type"],
        }
        prepared = MatMulNormStats.prepare(
            inputs,
            {
                **fusion_attrs,
                "axis": int(source.attrs["axis"]),
                "use_mean": bool(source.attrs["use_mean"]),
                "addend_cast_dtypes": addend_casts,
            },
        )
        if prepared.result_type != source.type:
            return source
        return Node(
            source.id,
            MatMulNormStats.op_name,
            tuple(value.id for value in prepared.inputs),
            prepared.result_type,
            prepared.effect,
            prepared.attrs,
            {
                **dict(source.metadata),
                "lowered_by": "LowerMatMulNormStatsCombine",
                "fused_matmul": producer.id,
                "projection_adapters": tuple(adapters),
                "matmul_vectorization": {
                    key: value
                    for key, value in producer.metadata.items()
                    if key in {
                        "selected_vectorization",
                        "selected_vector_axes",
                        "selected_vector_lanes",
                        "vectorization_candidate",
                        "vector_axes",
                        "vector_lanes",
                    }
                },
            },
        )

    return RewriteRule("LowerMatMulNormStatsCombine", pattern, rewrite)


def _projection_producer(node: Node, module: IRModule) -> tuple[Node, tuple[str, ...]]:
    """Trace storage views and proven lossless promotions to the producer."""

    adapters: list[str] = []
    current = node
    seen: set[str] = set()
    while ((current.op == "distributed.sharded_view" and len(current.inputs) == 1)
           or is_projection_promotion(current, module.node_map)):
        if current.id in seen:
            return node, ()
        seen.add(current.id)
        adapters.append(current.id)
        current = module.node_map[current.inputs[0]]
    return current, tuple(reversed(adapters))


def _users(module: IRModule, node_id: str) -> tuple[str, ...]:
    return tuple(node.id for node in module.nodes if node_id in node.inputs)


def _private_addend_casts(addend: Node, consumer_id: str, module: IRModule):
    """Move private local conversions into the epilogue, never erase rounding.

    Identical endpoint types prove the same physical local shard. Cast nodes
    themselves prove lane/split-unit conversion; no view or communication edge
    is traversed, and shared/escaping intermediate values stay materialized.
    """
    current = addend
    casts = []
    outputs = {value for function in module.functions for value in function.outputs}
    while current.op in {"tensors.cast", "ntt.vectorized_cast"}:
        if (not current.effect.is_pure or current.id in outputs
                or _users(module, current.id) != (consumer_id,)
                or isinstance(current.type, DistributedType) and current.type.partial is not None):
            break
        dtype = tensor_of(current.type).dtype
        dtype = dtype.elem_type if isinstance(dtype, VectorType) else dtype
        if dtype not in (DType.BFLOAT16, DType.FLOAT32):
            break
        casts.append(dtype.value)
        consumer_id = current.id
        current = module.node_map[current.inputs[0]]
    if not casts or current.type != addend.type:
        return addend, ()
    return current, tuple(reversed(casts))


__all__ = ["lower_matmul_norm_stats_combine_rule"]
