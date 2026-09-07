# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent decomposition of Gated DeltaNet semantic operations."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.model import Function, IRModule, Node
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.verify import verify_module


_GDN_OP = "nn.gated_delta_net"
_GET_ITEM_OP = "builtin.get_item"


def decompose_gated_delta_net(module: IRModule) -> IRModule:
    """Expose the independently optimizable stages of every fused GDN call.

    The decomposition follows nncase's semantic boundary: QKV and Z
    projections, stateful convolution, recurrent core, and the final output
    projection.  It deliberately contains no target, mesh, launch, or kernel
    policy.  Tuple projections are replaced by aliases so the two stateful
    operations remain explicit effect boundaries without introducing a fake
    tuple materialization.
    """

    module = verify_module(module)
    _verify_projected_gdn_uses(module)
    projection_ids = _gdn_projection_ids(module)
    aliases: dict[str, str] = {}
    expanded: dict[str, tuple[str, str]] = {}
    nodes: list[Node] = []
    node_map: dict[str, Node] = {}

    def resolve(node_id: str) -> str:
        observed: set[str] = set()
        while node_id in aliases:
            if node_id in observed:
                raise IRVerificationError(
                    "Gated DeltaNet decomposition produced an alias cycle.",
                    stage=module.stage,
                    node_id=node_id,
                )
            observed.add(node_id)
            node_id = aliases[node_id]
        return node_id

    def append(node: Node) -> Node:
        if node.id in node_map:
            raise IRVerificationError(
                f"Gated DeltaNet decomposition node id {node.id!r} already exists.",
                stage=module.stage,
                node_id=node.id,
            )
        nodes.append(node)
        node_map[node.id] = node
        return node

    def make(
        op: str,
        node_id: str,
        input_ids: Sequence[str],
        attrs: Mapping[str, object],
        source: Node,
    ) -> Node:
        inputs = tuple(node_map[resolve(value)] for value in input_ids)
        prepared = get_definition(op).prepare(inputs, attrs)
        metadata = {
            **dict(source.metadata),
            "decomposed_from": source.id,
            "decomposition_rule": "DecomposeGatedDeltaNet",
        }
        return append(Node(
            id=node_id,
            op=op,
            inputs=tuple(value.id for value in prepared.inputs),
            type=prepared.result_type,
            effect=prepared.effect,
            attrs=prepared.attrs,
            metadata=metadata,
        ))

    for original in module.nodes:
        if original.op == _GDN_OP:
            source = replace(original, inputs=tuple(resolve(value) for value in original.inputs))
            source_inputs = tuple(node_map[value] for value in source.inputs)
            _verify_gdn_source(source, source_inputs, module.stage)
            definition = get_definition(_GDN_OP)
            values = {
                parameter.name: parameter.read(source_inputs)
                for parameter in definition.input_parameters
            }
            prefix = f"{source.id}.decomposed"
            projection_attrs = {
                "weight_block_n": int(source.attrs["weight_block_n"]),
                "weight_block_k": int(source.attrs["weight_block_k"]),
            }
            qkv = make(
                "math.block_scaled_matmul",
                f"{prefix}.qkv",
                (values["value"].id, values["qkv_weight"].id, values["qkv_scale"].id),
                projection_attrs,
                source,
            )
            z = make(
                "math.block_scaled_matmul",
                f"{prefix}.z",
                (values["value"].id, values["z_weight"].id, values["z_scale"].id),
                projection_attrs,
                source,
            )
            convolution = make(
                "nn.gdn_convolution",
                f"{prefix}.convolution",
                (qkv.id, values["state"].id, values["conv_weight"].id),
                {"conv_kernel_size": int(source.attrs["conv_kernel_size"])},
                source,
            )
            convolved = make(
                _GET_ITEM_OP,
                f"{prefix}.convolved",
                (convolution.id,),
                {"index": 0},
                source,
            )
            convolution_state = make(
                _GET_ITEM_OP,
                f"{prefix}.convolution_state",
                (convolution.id,),
                {"index": 1},
                source,
            )
            recurrent = make(
                "nn.gdn_recurrent_core",
                f"{prefix}.recurrent",
                (
                    convolution_state.id,
                    convolved.id,
                    z.id,
                    values["value"].id,
                    values["b_weight"].id,
                    values["a_weight"].id,
                    values["a_log"].id,
                    values["dt_bias"].id,
                    values["norm_weight"].id,
                ),
                {
                    "num_key_heads": int(source.attrs["num_key_heads"]),
                    "num_value_heads": int(source.attrs["num_value_heads"]),
                    "key_head_dim": int(source.attrs["key_head_dim"]),
                    "value_head_dim": int(source.attrs["value_head_dim"]),
                    "epsilon": float(source.attrs["epsilon"]),
                },
                source,
            )
            gated = make(
                _GET_ITEM_OP,
                f"{prefix}.gated",
                (recurrent.id,),
                {"index": 0},
                source,
            )
            recurrent_state = make(
                _GET_ITEM_OP,
                projection_ids[source.id][1][0],
                (recurrent.id,),
                {"index": 1},
                source,
            )
            output = make(
                "math.block_scaled_matmul",
                projection_ids[source.id][0][0],
                (gated.id, values["output_weight"].id, values["output_scale"].id),
                projection_attrs,
                source,
            )
            expanded[source.id] = (output.id, recurrent_state.id)
            continue

        if original.op == _GET_ITEM_OP and len(original.inputs) == 1:
            producer_id = resolve(original.inputs[0])
            replacement = expanded.get(producer_id)
            if replacement is not None:
                replacement_id = replacement[int(original.attrs["index"])]
                if original.id != replacement_id:
                    aliases[original.id] = replacement_id
                continue

        append(replace(original, inputs=tuple(resolve(value) for value in original.inputs)))

    functions = tuple(
        replace(
            function,
            parameters=tuple(resolve(value) for value in function.parameters),
            outputs=tuple(resolve(value) for value in function.outputs),
        )
        for function in module.functions
    )
    points = tuple(
        replace(point, owner=resolve(point.owner)) if point.owner is not None else point
        for point in module.selection_points
    )
    result = replace(module, nodes=tuple(nodes), functions=functions, selection_points=points)
    return verify_module(result)


def _verify_projected_gdn_uses(module: IRModule) -> None:
    fused = {node.id for node in module.nodes if node.op == _GDN_OP}
    if not fused:
        return
    for node in module.nodes:
        for input_id in node.inputs:
            if input_id not in fused:
                continue
            if (
                node.op != _GET_ITEM_OP
                or len(node.inputs) != 1
                or int(node.attrs.get("index", -1)) not in (0, 1)
            ):
                raise IRVerificationError(
                    "DecomposeGatedDeltaNet requires fused tuple results to be consumed through "
                    "builtin.get_item indices 0 or 1.",
                    stage=module.stage,
                    node_id=node.id,
                )
    for function in module.functions:
        if any(value in fused for value in function.outputs):
            raise IRVerificationError(
                "DecomposeGatedDeltaNet cannot expose a fused tuple directly as a function output; "
                "project its output and state fields first.",
                stage=module.stage,
            )


def _gdn_projection_ids(module: IRModule) -> dict[str, dict[int, tuple[str, ...]]]:
    values: dict[str, dict[int, list[str]]] = {
        node.id: {0: [], 1: []}
        for node in module.nodes
        if node.op == _GDN_OP
    }
    for node in module.nodes:
        if node.op == _GET_ITEM_OP and len(node.inputs) == 1 and node.inputs[0] in values:
            values[node.inputs[0]][int(node.attrs["index"])].append(node.id)
    for node_id, fields in values.items():
        if not fields[0] or not fields[1]:
            raise IRVerificationError(
                "DecomposeGatedDeltaNet requires both output and state tuple projections.",
                stage=module.stage,
                node_id=node_id,
            )
    return {
        node_id: {index: tuple(ids) for index, ids in fields.items()}
        for node_id, fields in values.items()
    }


def _verify_gdn_source(source: Node, inputs: Sequence[Node], stage: str) -> None:
    definition = get_definition(_GDN_OP)
    try:
        definition.verify_parameter_types(inputs)
        attrs = definition.normalize_attrs(source.attrs)
        inferred = definition.infer_call_type(inputs, attrs)
        effect = definition.infer_effect(inputs, attrs)
    except Exception as error:
        raise IRVerificationError(
            f"Cannot decompose invalid Gated DeltaNet call: {error}",
            stage=stage,
            node_id=source.id,
        ) from error
    if inferred != source.type or effect != source.effect:
        raise IRVerificationError(
            "Cannot decompose Gated DeltaNet whose stored type/effect differs from inference.",
            stage=stage,
            node_id=source.id,
        )


__all__ = ["decompose_gated_delta_net"]
