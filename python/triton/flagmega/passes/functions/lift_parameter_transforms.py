# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lift offline parameter transforms from reusable callees to call sites.

AutoPacking reasons about the reusable body once.  If a selected physical
layout is constructed from function parameters, executing that construction
inside the device function would turn an offline weight transform into runtime
work.  This pass changes the function ABI to accept the physical value and
clones the pure transform at every call site, where constant-island freezing
can materialize one asset per actual weight set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import Function, IRModule, Node


@dataclass(frozen=True)
class _Transform:
    root: Node
    sources: tuple[str, ...]
    nodes: tuple[Node, ...]
    parameter: Node


def lift_parameter_constant_transforms(module: IRModule) -> IRModule:
    node_map = module.node_map
    transforms_by_function: dict[str, tuple[_Transform, ...]] = {}
    for function in module.functions:
        parameter_set = set(function.parameters)
        transforms: list[_Transform] = []
        claimed_sources: set[str] = set()
        for root in module.nodes:
            packed_from = root.metadata.get("packed_from")
            if isinstance(packed_from, str):
                sources = (packed_from,)
            elif isinstance(packed_from, tuple) and packed_from:
                sources = tuple(str(value) for value in packed_from)
            else:
                continue
            if not set(sources).issubset(parameter_set):
                continue
            if claimed_sources.intersection(sources):
                raise IRVerificationError(
                    f"Overlapping lifted parameter transforms in @{function.name}.",
                    stage=module.stage,
                    node_id=root.id,
                )
            closure = _transform_closure(root, set(sources), node_map, module.stage)
            parameter = Node(
                id=f"{root.id}.parameter",
                op="builtin.var",
                inputs=(),
                type=root.type,
                attrs={"name": _parameter_name(function.name, root)},
                metadata={
                    "function_parameter": function.name,
                    "lifted_parameter_transform": root.id,
                    "packed_layout": root.metadata.get("packed_layout"),
                },
            )
            transforms.append(_Transform(root, sources, closure, parameter))
            claimed_sources.update(sources)
        if transforms:
            transforms.sort(
                key=lambda value: min(function.parameters.index(source) for source in value.sources)
            )
            transforms_by_function[function.name] = tuple(transforms)
    if not transforms_by_function:
        return module

    removed: set[str] = set()
    replacement_inputs: dict[str, str] = {}
    inserted_parameters: dict[int, list[Node]] = {}
    functions: list[Function] = []
    for function in module.functions:
        transforms = transforms_by_function.get(function.name, ())
        if not transforms:
            functions.append(function)
            continue
        by_source = {
            source: transform
            for transform in transforms
            for source in transform.sources
        }
        parameters: list[str] = []
        for source in function.parameters:
            transform = by_source.get(source)
            if transform is None:
                parameters.append(source)
            elif source == transform.sources[0]:
                parameters.append(transform.parameter.id)
        root_parameters = {
            transform.root.id: transform.parameter.id for transform in transforms
        }
        functions.append(replace(
            function,
            parameters=tuple(parameters),
            outputs=tuple(root_parameters.get(value, value) for value in function.outputs),
        ))
        for transform in transforms:
            removed.update(node.id for node in transform.nodes)
            removed.update(transform.sources)
            replacement_inputs[transform.root.id] = transform.parameter.id
        last_parameter_index = max(
            index for index, node in enumerate(module.nodes) if node.id in function.parameters
        )
        inserted_parameters.setdefault(last_parameter_index, []).extend(
            transform.parameter for transform in transforms
        )

    calls_by_id: dict[str, tuple[Function, tuple[_Transform, ...]]] = {}
    function_map = {function.name: function for function in module.functions}
    for node in module.nodes:
        if node.op != "builtin.call":
            continue
        callee_name = str(node.attrs["callee"])
        transforms = transforms_by_function.get(callee_name)
        if transforms:
            calls_by_id[node.id] = (function_map[callee_name], transforms)

    result_nodes: list[Node] = []
    for index, node in enumerate(module.nodes):
        if node.id not in removed:
            if node.id in calls_by_id:
                function, transforms = calls_by_id[node.id]
                clone_roots: dict[str, Node] = {}
                actual_by_parameter = dict(zip(function.parameters, node.inputs))
                for transform in transforms:
                    clone_roots[transform.root.id] = _clone_transform(
                        node,
                        transform,
                        actual_by_parameter,
                        node_map,
                        result_nodes,
                    )
                by_source = {
                    source: transform
                    for transform in transforms
                    for source in transform.sources
                }
                inputs: list[str] = []
                for parameter_id, actual_id in zip(function.parameters, node.inputs):
                    transform = by_source.get(parameter_id)
                    if transform is None:
                        inputs.append(actual_id)
                    elif parameter_id == transform.sources[0]:
                        inputs.append(clone_roots[transform.root.id].id)
                result_nodes.append(replace(
                    node,
                    inputs=tuple(inputs),
                    metadata={
                        **dict(node.metadata),
                        "lifted_parameter_transforms": tuple(
                            transform.root.id for transform in transforms
                        ),
                    },
                ))
            else:
                inputs = tuple(replacement_inputs.get(value, value) for value in node.inputs)
                result_nodes.append(replace(node, inputs=inputs))
        result_nodes.extend(inserted_parameters.get(index, ()))

    return replace(module, nodes=tuple(result_nodes), functions=tuple(functions))


def _transform_closure(
    root: Node,
    sources: set[str],
    node_map: Mapping[str, Node],
    stage: str,
) -> tuple[Node, ...]:
    reachable: set[str] = set()
    pending = [root.id]
    while pending:
        node_id = pending.pop()
        if node_id in sources or node_id in reachable:
            continue
        node = node_map[node_id]
        if not node.effect.is_pure:
            raise IRVerificationError(
                "A lifted parameter transform must be pure.", stage=stage, node_id=node.id
            )
        reachable.add(node_id)
        for input_id in node.inputs:
            if input_id not in sources and input_id not in reachable:
                pending.append(input_id)
    leaves = {
        input_id
        for node_id in reachable
        for input_id in node_map[node_id].inputs
        if input_id not in reachable
    }
    if leaves != sources:
        raise IRVerificationError(
            f"Lifted transform {root.id!r} has non-parameter leaves {sorted(leaves - sources)}.",
            stage=stage,
            node_id=root.id,
        )
    return tuple(node for node in node_map.values() if node.id in reachable)


def _clone_transform(
    call: Node,
    transform: _Transform,
    actual_by_parameter: Mapping[str, str],
    node_map: Mapping[str, Node],
    output: list[Node],
) -> Node:
    redirects = {source: actual_by_parameter[source] for source in transform.sources}
    actual_sources = tuple(redirects[source] for source in transform.sources)
    for original in transform.nodes:
        clone_id = f"{call.id}.{original.id}"
        metadata = dict(original.metadata)
        if original.id == transform.root.id:
            metadata.update({
                "packed_from": (
                    actual_sources[0] if len(actual_sources) == 1 else actual_sources
                ),
                "packed_for": call.id,
                "lifted_from_function": str(call.attrs["callee"]),
            })
            group = _combined_rdata_group(actual_sources, node_map)
            if group is not None:
                metadata["rdata_group"] = group
        clone = replace(
            original,
            id=clone_id,
            inputs=tuple(redirects[value] for value in original.inputs),
            metadata=metadata,
        )
        output.append(clone)
        redirects[original.id] = clone.id
    return output[-1]


def _combined_rdata_group(
    actual_sources: tuple[str, ...],
    node_map: Mapping[str, Node],
) -> dict[str, object] | None:
    """Preserve layer-major grouping for a lifted multi-source transform.

    A fused physical value may be built from any number of grouped parameters;
    its group identity is the ordered composition of the source group names.
    This keeps the pass independent of model and operator roles while retaining
    the common layer index/count required by the rdata allocator.
    """

    raw_groups = tuple(
        node_map[source].metadata.get("rdata_group")
        for source in actual_sources
    )
    if not any(isinstance(value, Mapping) for value in raw_groups):
        return None
    if not all(isinstance(value, Mapping) for value in raw_groups):
        raise IRVerificationError(
            "A lifted multi-source transform cannot mix grouped and ungrouped "
            "rdata parameters."
        )
    groups = tuple(dict(value) for value in raw_groups)
    names = tuple(str(value.get("name", "")) for value in groups)
    indexes = {value.get("index") for value in groups}
    counts = {value.get("count") for value in groups}
    if (
        any(not value for value in names)
        or len(indexes) != 1
        or len(counts) != 1
        or isinstance(next(iter(indexes)), bool)
        or not isinstance(next(iter(indexes)), int)
        or isinstance(next(iter(counts)), bool)
        or not isinstance(next(iter(counts)), int)
    ):
        raise IRVerificationError(
            "Lifted grouped transform sources require non-empty names and one "
            "integer layer index/count."
        )
    if len(groups) == 1:
        return groups[0]
    return {
        "name": "composite[" + ",".join(names) + "]",
        "index": next(iter(indexes)),
        "count": next(iter(counts)),
    }


def _parameter_name(function_name: str, root: Node) -> str:
    layout = str(root.metadata.get("packed_layout", "physical"))
    packed_from = root.metadata.get("packed_from")
    role = "parameters"
    if isinstance(packed_from, str):
        role = packed_from.removeprefix(function_name + "_")
    elif isinstance(packed_from, tuple) and len(packed_from) == 3:
        role = "qkv_weight"
    return f"packed_{role}_{layout}"


__all__ = ["lift_parameter_constant_transforms"]
