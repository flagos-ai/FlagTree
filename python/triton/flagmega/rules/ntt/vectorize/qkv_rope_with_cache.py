# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Vectorize fused Q/K normalization, RoPE, and cache updates by cache ABI."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DType, IRModule, Node, TensorType, TupleType, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.nn._norm import normalize_axis
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    paged_attention_state_config_from_type,
)
from triton.flagmega.ir.ops.tensors.pack import Pack
from triton.flagmega.rules import RewriteResult
from triton.flagmega.rules.ntt.vectorize.base import VectorizeCandidate
from triton.flagmega.rules.ntt.vectorize.utility import (
    internal_metadata,
    root_metadata,
)


@dataclass(frozen=True)
class _PackingPlan:
    lane: int
    tensor_axis: int
    q_parameter_axis: int
    k_parameter_axis: int
    trig_axis: int


class VectorizeQKVRoPEWithCache:
    """Port nncase ``VectorizeQKVRoPEWithCache`` without model policy."""

    name = "VectorizeQKVRoPEWithCache"
    op_names = frozenset({"nn.qkv_rope_with_cache"})

    def candidates(
        self, node: Node, module: IRModule
    ) -> tuple[VectorizeCandidate, ...]:
        plan = _try_plan(node, module)
        if plan is None:
            return ()
        try:
            _, packed_inputs = _build_packed_inputs(node, module, plan)
            inferred = module_definition(node).infer_type(packed_inputs, node.attrs)
        except (IRSchemaError, AssertionError, ValueError):
            return ()
        if inferred != node.type:
            return ()
        return (
            VectorizeCandidate(
                "vectorization.qkv_rope_with_cache.cache_layout",
                self.name,
                (plan.tensor_axis,),
                (plan.lane,),
                {
                    "tensor_axis": plan.tensor_axis,
                    "q_parameter_axis": plan.q_parameter_axis,
                    "k_parameter_axis": plan.k_parameter_axis,
                    "trig_axis": plan.trig_axis,
                    "lane": plan.lane,
                },
                {
                    "cache_abi": True,
                    "egraph_equivalent": True,
                    "boundary_type_preserved": True,
                },
            ),
        )

    def rewrite(
        self,
        node: Node,
        module: IRModule,
        candidate: VectorizeCandidate,
    ) -> RewriteResult:
        plan = _PackingPlan(
            lane=int(candidate.parameters["lane"]),
            tensor_axis=int(candidate.parameters["tensor_axis"]),
            q_parameter_axis=int(candidate.parameters["q_parameter_axis"]),
            k_parameter_axis=int(candidate.parameters["k_parameter_axis"]),
            trig_axis=int(candidate.parameters["trig_axis"]),
        )
        helpers, packed_inputs = _build_packed_inputs(node, module, plan)
        result_type = module_definition(node).infer_type(packed_inputs, node.attrs)
        if result_type != node.type:
            raise IRSchemaError(
                "VectorizeQKVRoPEWithCache changed the semantic boundary type."
            )
        replacement = Node(
            node.id,
            node.op,
            tuple(value.id for value in packed_inputs),
            result_type,
            node.effect,
            dict(node.attrs),
            root_metadata(node, candidate),
        )
        return RewriteResult(replacement, helpers)


def _try_plan(node: Node, module: IRModule) -> _PackingPlan | None:
    if node.op != "nn.qkv_rope_with_cache" or len(node.inputs) != 10:
        return None
    qkv = module.node_map[node.inputs[0]]
    if qkv.op != "builtin.tuple" or len(qkv.inputs) != 3:
        return None
    fields = tuple(module.node_map[value].type for value in qkv.inputs)
    if any(
        not isinstance(value, TensorType)
        or isinstance(value.dtype, VectorType)
        or value.dtype != DType.BFLOAT16
        for value in fields
    ):
        return None
    try:
        config = paged_attention_state_config_from_type(
            module.node_map[node.inputs[7]].type
        )
        layout = tuple(node.attrs["qkv_layout"])
        tensor_axis = layout.index("dim")
        q_axis = normalize_axis(int(node.attrs["q_axis"]), fields[0].rank)
        k_axis = normalize_axis(int(node.attrs["k_axis"]), fields[1].rank)
        trig = tensor_of(module.node_map[node.inputs[5]].type)
    except (IRSchemaError, KeyError, TypeError, ValueError):
        return None
    q_parameter_axis = tensor_axis - q_axis
    k_parameter_axis = tensor_axis - k_axis
    trig_axis = tensor_axis - (fields[0].rank - trig.rank)
    if (
        q_parameter_axis < 0
        or k_parameter_axis < 0
        or trig_axis < 0
        or trig_axis >= trig.rank
    ):
        return None
    return _PackingPlan(
        config.lanes,
        tensor_axis,
        q_parameter_axis,
        k_parameter_axis,
        trig_axis,
    )


def _build_packed_inputs(
    node: Node,
    module: IRModule,
    plan: _PackingPlan,
) -> tuple[tuple[Node, ...], tuple[Node, ...]]:
    helpers: list[Node] = []
    qkv = module.node_map[node.inputs[0]]
    packed_qkv: list[Node] = []
    for role, value_id in zip(("q", "k", "v"), qkv.inputs, strict=True):
        packed = _pack(
            module.node_map[value_id],
            (plan.lane,),
            (plan.tensor_axis,),
            f"{node.id}.vectorized.{role}",
            node.id,
        )
        helpers.append(packed)
        packed_qkv.append(packed)
    tuple_node = Node(
        f"{node.id}.vectorized.qkv",
        "builtin.tuple",
        tuple(value.id for value in packed_qkv),
        TupleType(tuple(value.type for value in packed_qkv)),
        metadata={
            **internal_metadata(node.id, "qkv"),
            "vectorization_semantic_id": qkv.id,
        },
    )
    helpers.append(tuple_node)

    packed_by_index: dict[int, Node] = {}
    for index, role, axis in (
        (1, "q_scale", plan.q_parameter_axis),
        (2, "k_scale", plan.k_parameter_axis),
        (3, "q_bias", plan.q_parameter_axis),
        (4, "k_bias", plan.k_parameter_axis),
    ):
        packed = _pack(
            module.node_map[node.inputs[index]],
            (plan.lane,),
            (axis,),
            f"{node.id}.vectorized.{role}",
            node.id,
        )
        helpers.append(packed)
        packed_by_index[index] = packed

    for index, role in ((5, "cos"), (6, "sin")):
        value = module.node_map[node.inputs[index]]
        # Cache lanes constrain logical coordinates, not the table's scalar
        # dtype. Preserve its rounding/storage contract: the fused op promotes
        # loaded elements to its arithmetic dtype without a materialized Cast.
        packed = _pack(
            value,
            (2, plan.lane),
            (plan.trig_axis, plan.trig_axis),
            f"{node.id}.vectorized.{role}",
            node.id,
        )
        helpers.append(packed)
        packed_by_index[index] = packed

    inputs = (
        tuple_node,
        *(packed_by_index[index] for index in range(1, 7)),
        *(module.node_map[node.inputs[index]] for index in range(7, 10)),
    )
    return tuple(helpers), inputs


def _pack(
    value: Node,
    lanes: tuple[int, ...],
    axes: tuple[int, ...],
    node_id: str,
    root_id: str,
) -> Node:
    attrs = {"lanes": lanes, "axes": axes}
    return Node(
        node_id,
        "tensors.pack",
        (value.id,),
        Pack.infer_type((value,), attrs),
        attrs=attrs,
        metadata=internal_metadata(root_id, "pack"),
    )


def module_definition(node: Node):
    from triton.flagmega.ir import get_definition

    return get_definition(node.op)


__all__ = ["VectorizeQKVRoPEWithCache"]
