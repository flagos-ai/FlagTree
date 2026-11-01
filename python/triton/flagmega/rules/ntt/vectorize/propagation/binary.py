# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Push Pack/Unpack through binary dataflow."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import IRModule, Node, TensorType, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import Pack
from triton.flagmega.rules import RewriteResult, RewriteRule
from triton.flagmega.rules.ntt.vectorize.utility import (
    propagation_helper_metadata,
    propagation_result_metadata,
)


def _pack_binary_matches(node: Node, module: IRModule) -> bool:
    return node.op == "tensors.pack" and module.node_map[node.inputs[0]].op in {"math.add", "math.mul"}


def _pack_binary(node: Node, module: IRModule) -> RewriteResult:
    scalar = module.node_map[node.inputs[0]]
    boundary_lanes = tuple(int(value) for value in node.attrs["lanes"])
    boundary_axes = _pack_axes(node, len(boundary_lanes))
    helpers: list[Node] = []
    packed: list[Node] = []
    for index, input_id in enumerate(scalar.inputs):
        source = module.node_map[input_id]
        attrs = dict(node.attrs)
        value_type = Pack.infer_type((source,), attrs)
        helper = Node(
            f"{node.id}.propagated.pack{index}",
            "tensors.pack",
            (source.id,),
            value_type,
            attrs=attrs,
            metadata=propagation_helper_metadata(node, node.id, "propagated-pack"),
        )
        helpers.append(helper)
        packed.append(helper)
    replacement = Node(
        node.id,
        "math.vectorized_binary",
        tuple(value.id for value in packed),
        node.type,
        attrs={"binary_op": scalar.op.removeprefix("math.")},
        metadata=propagation_result_metadata(
            scalar,
            node,
            axes=boundary_axes,
            lanes=boundary_lanes,
            rule="VectorizeBinaryPropagation",
            internal_role="propagated-compute",
        ),
    )
    return RewriteResult(replacement, tuple(helpers))


def _binary_unpack_matches(node: Node, module: IRModule, operand_index: int) -> bool:
    if node.op not in {"math.add", "math.mul"} or len(node.inputs) != 2:
        return False
    unpack = module.node_map[node.inputs[operand_index]]
    if unpack.op != "tensors.unpack":
        return False
    vector = module.node_map[unpack.inputs[0]]
    other = module.node_map[node.inputs[1 - operand_index]]
    if (
        not isinstance(vector.type, TensorType)
        or not isinstance(vector.type.dtype, VectorType)
        or not isinstance(other.type, TensorType)
        or isinstance(other.type.dtype, VectorType)
    ):
        return False
    axes = _unpack_axes(unpack, len(vector.type.dtype.lanes))
    lanes = vector.type.dtype.lanes[: len(axes)]
    try:
        packed_type = Pack.infer_type((other,), {"lanes": lanes, "axes": axes})
    except (IRSchemaError, TypeError, ValueError):
        return False
    return packed_type == vector.type


def _binary_unpack(node: Node, module: IRModule, operand_index: int) -> RewriteResult:
    unpack = module.node_map[node.inputs[operand_index]]
    vector = module.node_map[unpack.inputs[0]]
    other = module.node_map[node.inputs[1 - operand_index]]
    assert isinstance(vector.type, TensorType) and isinstance(vector.type.dtype, VectorType)
    axes = _unpack_axes(unpack, len(vector.type.dtype.lanes))
    lanes = vector.type.dtype.lanes[: len(axes)]
    pack_attrs = {"lanes": lanes, "axes": axes}
    packed_other = Node(
        f"{node.id}.propagated.pack{1 - operand_index}",
        "tensors.pack",
        (other.id,),
        Pack.infer_type((other,), pack_attrs),
        attrs=pack_attrs,
        metadata=propagation_helper_metadata(unpack, node.id, "propagated-pack"),
    )
    vector_inputs = [packed_other.id, packed_other.id]
    vector_inputs[operand_index] = vector.id
    compute = Node(
        f"{node.id}.propagated.compute",
        "math.vectorized_binary",
        tuple(vector_inputs),
        vector.type,
        attrs={"binary_op": node.op.removeprefix("math.")},
        metadata=propagation_result_metadata(
            node,
            unpack,
            axes=axes,
            lanes=lanes,
            rule="BinaryDevectorizePropagation",
            internal_role="propagated-compute",
        ),
    )
    replacement = Node(
        node.id,
        "tensors.unpack",
        (compute.id,),
        node.type,
        attrs=dict(unpack.attrs),
        metadata=propagation_result_metadata(
            node,
            unpack,
            axes=axes,
            lanes=lanes,
            rule="BinaryDevectorizePropagation",
        ),
    )
    return RewriteResult(replacement, (packed_other, compute))


def _unpack_axes(unpack: Node, lane_count: int) -> tuple[int, ...]:
    if "axes" in unpack.attrs:
        return tuple(int(value) for value in unpack.attrs["axes"])
    return (int(unpack.attrs["axis"]),) * lane_count


def _pack_axes(pack: Node, lane_count: int) -> tuple[int, ...]:
    if "axes" in pack.attrs:
        return tuple(int(value) for value in pack.attrs["axes"])
    return (int(pack.attrs["axis"]),) * lane_count


def binary_propagation_rules() -> tuple[RewriteRule, ...]:
    return (
        RewriteRule('VectorizedBinaryOperandLaneAccess', _operand_pack_matches, _operand_pack,
                    root_ops=('math.vectorized_binary',)),
        RewriteRule('VectorizedBinaryOperandTail', _operand_pad_matches, _operand_pad,
                    root_ops=('math.vectorized_binary',)),
        RewriteRule('VectorizedBinaryResultLaneAccess',
                    lambda node, _: node.op == 'tensors.slice_to_shape', _result_lanes,
                    root_ops=('tensors.slice_to_shape',)),
        RewriteRule("VectorizeBinaryPropagation", _pack_binary_matches, _pack_binary, root_ops=("tensors.pack",)),
        RewriteRule(
            "BinaryDevectorizeLhsPropagation",
            lambda node, module: _binary_unpack_matches(node, module, 0),
            lambda node, module: _binary_unpack(node, module, 0),
            root_ops=("math.add", "math.mul"),
        ),
        RewriteRule(
            "BinaryDevectorizeRhsPropagation",
            lambda node, module: _binary_unpack_matches(node, module, 1),
            lambda node, module: _binary_unpack(node, module, 1),
            root_ops=("math.add", "math.mul"),
        ),
    )


def _operand_pack_matches(node, module):
    return (node.op == 'math.vectorized_binary' and any(
        module.node_map[value].op == 'tensors.pack' and 'padding' not in (node.attrs.get(name) or {})
        for value, name in zip(node.inputs, ('lhs_pack', 'rhs_pack'))))


def _operand_pack(node, module):
    inputs = list(node.inputs)
    attrs = dict(node.attrs)
    for index, name in enumerate(('lhs_pack', 'rhs_pack')):
        boundary = module.node_map[inputs[index]]
        if boundary.op == 'tensors.pack' and 'padding' not in (attrs.get(name) or {}):
            inputs[index] = boundary.inputs[0]
            packing = dict(boundary.attrs)
            if attrs.get(name) is not None:
                from triton.flagmega.ir.ops.math.vectorized_binary import operand_pack_axes
                rank = tensor_of(boundary.type).rank
                outer = attrs[name]
                packing = {'lanes': (*outer['lanes'], *packing['lanes']),
                           'axes': (*operand_pack_axes(outer, rank), *operand_pack_axes(packing, rank))}
            attrs[name] = packing
    return RewriteResult(replace(node, inputs=tuple(inputs), attrs=attrs))


def _operand_pad_matches(node, module):
    return (node.op == 'math.vectorized_binary' and any(
        module.node_map[value].op == 'tensors.pad' and isinstance(node.attrs.get(name), Mapping)
        and 'padding' not in node.attrs[name] for value, name in zip(node.inputs, ('lhs_pack', 'rhs_pack'))))


def _operand_pad(node, module):
    from triton.flagmega.ir.ops.math.vectorized_binary import VectorizedBinary
    inputs, attrs = list(node.inputs), dict(node.attrs)
    for index, name in enumerate(('lhs_pack', 'rhs_pack')):
        boundary = module.node_map[inputs[index]]
        packing = attrs.get(name)
        if boundary.op != 'tensors.pad' or packing is None or 'padding' in packing:
            continue
        source = module.node_map[boundary.inputs[0]]
        candidate = {**dict(packing), 'padding': dict(boundary.attrs)}
        try:
            if VectorizedBinary.operand_representation(source.type, index, {name: candidate}) != (
                    VectorizedBinary.operand_representation(boundary.type, index, {name: packing})):
                continue
        except IRSchemaError:
            continue
        inputs[index] = source.id
        attrs[name] = candidate
    return RewriteResult(replace(node, inputs=tuple(inputs), attrs=attrs))


def _result_lanes(node, module):
    """Give arithmetic its exact public result representation and tail mask."""
    from triton.flagmega.ir.ops.math.vectorized_binary import VectorizedBinary, operand_pack_axes
    from triton.flagmega.ir.ops.tensors.unpack import Unpack
    view = module.node_map[node.inputs[0]]
    if view.op not in {'tensors.unpack', 'tensors.bitcast'}:
        return None
    compute = module.node_map[view.inputs[0]]
    if compute.op != 'math.vectorized_binary' or compute.attrs.get('result_unpack') is not None:
        return None
    tensor = tensor_of(compute.type)
    if not isinstance(tensor.dtype, VectorType):
        return None
    if view.op == 'tensors.unpack':
        axes = _unpack_axes(view, len(tensor.dtype.lanes))
    else:
        # Bitcast can implement Unpack only when its byte interpretation is
        # exactly the lane expansion proved below. No other cast is absorbed.
        axes = (tensor.rank - 1,) * len(tensor.dtype.lanes)
    output = {'lanes': tensor.dtype.lanes[:len(axes)], 'axes': axes, 'shape': tuple(node.attrs['shape'])}
    try:
        axes = operand_pack_axes(output, tensor.rank)
        if Unpack.infer_type((compute,), {'axes': axes}) != view.type:
            return None
        attrs = {**compute.attrs, 'result_unpack': output}
        if VectorizedBinary.infer_type(tuple(module.node_map[i] for i in compute.inputs), attrs) != node.type:
            return None
    except IRSchemaError:
        return None
    return RewriteResult(Node(node.id, compute.op, compute.inputs, node.type, compute.effect, attrs,
                              propagation_result_metadata(compute, node, axes=axes, lanes=output['lanes'],
                                  rule='VectorizedBinaryResultLaneAccess')))


__all__ = ["binary_propagation_rules"]
