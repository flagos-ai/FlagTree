# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Interprocedural propagation of declared storage alignment contracts.

Alignment constrains a physical buffer, and an aggregate value owns one
buffer per tensor leaf.  A requirement is therefore recorded against a leaf
identity, which for a plain tensor value is the value id itself, so the common
case stays keyed exactly as before.  A requirement declared against a whole
aggregate (a ``Ref`` kernel operand, say) still applies to all of its leaves;
``effective_alignment`` reads both and returns the stricter one.
"""

from __future__ import annotations

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import DistributedType, IRModule, kernel_dispatch_of, logical_type
from triton.flagmega.ir.model import TensorType
from triton.flagmega.ir.types import VectorType
from triton.flagmega.ir.bufferization.leaves import buffer_leaf_names


def storage_alignment_requirements(module: IRModule) -> dict[str, int]:
    """Read semantic TIR ABI requirements without consulting microkernels."""
    return _propagate(module, _declared_requirements(module))


def effective_alignment(
    requirements: dict[str, int], value_id: str, leaf: str | None = None
) -> int:
    """Strictest requirement constraining one physical leaf of a value."""
    result = int(requirements.get(str(value_id), 1))
    if leaf is not None:
        result = max(result, int(requirements.get(str(leaf), 1)))
    return result


def _declared_requirements(module):
    requirements = {}
    for node in module.nodes:
        if node.op != "tir.call":
            continue
        function = module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
        if function is None:
            continue
        for parameter, value in zip(function.runtime_parameters, node.inputs, strict=True):
            if parameter.role.value != "metadata" and parameter.alignment_bytes is not None:
                _add(requirements, value, parameter.alignment_bytes)
        # An output parameter constrains the buffer it writes, not its siblings.
        # A kernel returning (value, statistics) must not raise the statistics
        # buffer to the value's alignment: that pads the per-owner stride of a
        # result whose own contract never asked for it, and the callee that
        # later receives it as a formal parameter computes the unpadded stride.
        for leaf, parameter in zip(_result_leaves(module, node), _output_parameter_leaves(function), strict=True):
            if parameter.alignment_bytes is not None:
                _add(requirements, leaf, parameter.alignment_bytes)
    # A layout transform may be represented as a zero-copy alias while
    # exposing a fixed-width vector element.  Propagate that element's
    # alignment to both sides of the view so a vectorized alias cannot start
    # at an under-aligned scalar base address.
    for node in module.nodes:
        if node.op not in {
                "tensors.pack",
                "tensors.unpack",
                "tensors.permute",
                "tensors.reshape",
                "tensors.bitcast",
                "distributed.sharded_view",
                "tir.buffer_view",
                "tir.buffer_subspan",
        }:
            continue
        for value_id in (node.id, node.inputs[0]):
            value_type = logical_type(module.node_map[value_id].type)
            if isinstance(value_type, TensorType) and isinstance(value_type.dtype, VectorType):
                _add(requirements, value_id, value_type.dtype.itemsize)
    return requirements


def transfer_source_alignment_requirements(module: IRModule) -> dict[str, int]:
    """Allocation requirements, including legacy checkpoints without contracts.

    Planned ABIs are authoritative. A selected implementation may not increase
    them. Old checkpoints lacking the field retain their existing transfer ABI.
    """

    requirements = _declared_requirements(module)
    for node in module.nodes:
        if node.op != "tir.call":
            continue
        function = module.kernel_callable_map.get(str(node.attrs.get("callee", "")))
        dispatch = None if function is None else kernel_dispatch_of(function)
        selection = None if dispatch is None else dispatch.microkernel
        pipeline = None if selection is None else selection.transfer_pipeline
        if pipeline is None:
            continue
        for channel in pipeline.channels:
            for argument_index in channel.source_argument_indices:
                if argument_index >= len(node.inputs):
                    raise IRVerificationError(
                        f"TIR call {node.id!r} transfer channel {channel.name!r} "
                        f"references missing argument {argument_index}.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                parameter = function.runtime_parameters[argument_index]
                if parameter.alignment_bytes is not None and channel.source_alignment_bytes > parameter.alignment_bytes:
                    raise IRVerificationError(
                        f"Microkernel on {node.id!r} exceeds the declared alignment contract of {parameter.name!r}.",
                        stage=module.stage, node_id=node.id)
                _add(
                    requirements,
                    node.inputs[argument_index],
                    channel.source_alignment_bytes,
                )

    return _propagate(module, requirements)


def _propagate(module, requirements):
    """Close the requirement set over views, tuples and call boundaries.

    Values that name one buffer carry one requirement, so the fixpoint relates
    leaves pairwise. Collapsing the leaves of an aggregate into a single
    scalar would let an unrelated field dictate another field's stride.
    """

    changed = True
    while changed:
        changed = False
        for node in module.nodes:
            if node.op in {
                    "builtin.get_item",
                    "builtin.tuple",
                    "distributed.sharded_view",
                    "tir.buffer_view",
                    "tir.buffer_subspan",
                    # Layout transforms may be represented as zero-copy aliases
                    # by the buffer planner.  Propagate the stricter alignment
                    # across them so a vectorized view cannot alias a scalar
                    # allocation with an under-aligned base address.
                    "tensors.pack",
                    "tensors.unpack",
                    "tensors.permute",
                    "tensors.reshape",
                    "tensors.bitcast",
            }:
                for source, target in _view_leaf_pairs(module, node):
                    changed |= _unify(requirements, source, target)
            if node.op not in {"builtin.call", "tir.call"}:
                continue
            callee = module.function_map.get(str(node.attrs.get("callee", "")))
            if callee is None:
                continue
            for formal_id, actual_id in zip(callee.parameters, node.inputs):
                for formal, actual in _value_leaf_pairs(module, formal_id, actual_id):
                    changed |= _unify(requirements, formal, actual)
            # A call result is the storage the callee returned, leaf for leaf.
            returned = [leaf for value in callee.outputs for leaf in _leaves(module, value)]
            results = _result_leaves(module, node)
            if len(returned) != len(results):
                raise IRVerificationError(
                    f"Call {node.id!r} returns {len(results)} buffer leaves; "
                    f"@{callee.name} declares {len(returned)}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            for formal, actual in zip(returned, results):
                changed |= _unify(requirements, formal, actual)
    return requirements


def _view_leaf_pairs(module, node):
    """Source/result leaf pairs of a zero-copy view or aggregate.

    These ops never copy, so each result leaf is the same buffer as the
    operand leaf it exposes.
    """

    results = _leaves(module, node.id)
    if node.op == "builtin.tuple":
        sources = [leaf for value in node.inputs for leaf in _leaves(module, value)]
    elif node.op == "builtin.get_item":
        source_type = module.node_map[node.inputs[0]].type
        fields = getattr(source_type, "fields", None)
        if fields is None:
            return ()
        counts = [len(buffer_leaf_names("leaf", field)) for field in fields]
        start = sum(counts[: int(node.attrs["index"])])
        sources = _leaves(module, node.inputs[0])[
            start : start + counts[int(node.attrs["index"])]
        ]
    else:
        # A view reinterprets its first operand; any further operands are
        # index/shape metadata rather than storage.
        sources = _leaves(module, node.inputs[0])
    if len(sources) != len(results):
        return ()
    return tuple(zip(sources, results))


def _value_leaf_pairs(module, left_id, right_id):
    """Leaf pairs of two values bound to the same storage."""
    left = _leaves(module, left_id)
    right = _leaves(module, right_id)
    if len(left) != len(right):
        return ()
    return tuple(zip(left, right))


def _leaves(module, value_id):
    """Leaf identities of a graph value, in physical binding order."""
    return list(buffer_leaf_names(str(value_id), module.node_map[value_id].type))


def _result_leaves(module, node):
    return list(buffer_leaf_names(node.id, node.type))


def _output_parameter_leaves(function):
    """One entry per returned buffer leaf, naming the parameter writing it."""
    return [
        parameter
        for parameter in function.output_parameters
        for _ in buffer_leaf_names("leaf", parameter.type)
    ]


def _unify(requirements, left, right):
    """Two names for one buffer carry one requirement."""
    required = max(requirements.get(left, 1), requirements.get(right, 1))
    if required <= 1:
        return False
    return _add(requirements, left, required) | _add(requirements, right, required)


def validate_storage_alignments(module: IRModule) -> None:
    """An edited view must still satisfy its already-declared consumer ABI."""
    from triton.flagmega.ir.distributed_type import local_shape
    from triton.flagmega.ir.ops.tir.buffer_subspan import dense_subspan_offset

    requirements = storage_alignment_requirements(module)
    for node in module.nodes:
        if node.op != "tir.buffer_subspan":
            continue
        source = module.node_map[node.inputs[0]].type
        tensor = getattr(source, "tensor", source)
        shapes = [(tensor.shape, getattr(node.type, "tensor", node.type).shape)]
        if isinstance(source, DistributedType):
            shapes.append((local_shape(source), local_shape(node.type)))
        alignment = requirements.get(node.id, 1)
        for parent, result in shapes:
            offset = dense_subspan_offset(tuple(d.fixed_value for d in parent), tuple(d.fixed_value for d in result),
                                           node.attrs["offsets"], tensor.dtype.itemsize)
            if offset % alignment:
                raise IRVerificationError(
                    f"Subspan {node.id!r} offset {offset} violates its {alignment}-byte storage alignment contract.",
                    stage=module.stage, node_id=node.id)


def _add(requirements: dict[str, int], value_id: str, alignment: int) -> bool:
    previous = requirements.get(value_id, 1)
    if alignment <= previous:
        return False
    requirements[value_id] = alignment
    return True


__all__ = [
    "effective_alignment",
    "storage_alignment_requirements",
    "transfer_source_alignment_requirements",
    "validate_storage_alignments",
]
