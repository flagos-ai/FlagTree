# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-style vectorized binary operation."""

from __future__ import annotations

from typing import Mapping, Sequence
from itertools import product
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.model import DistributedType, IRType, Node, tensor_type
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpCostFactors,
    attribute_parameter,
    input_parameter,
    op_definition,
    tensor_elements,
    tensor_nbytes,
)
from triton.flagmega.ir.ops.pointwise import SameTypePointwiseOp
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.ir.types import VectorType
from triton.flagmega.ir.ops.tensors.pack import Pack, axis_lane_products, normalize_axes, pack_physical
from triton.flagmega.ir.ops.tensors.pad import Pad
from triton.flagmega.ir.ops.tensors.slice_to_shape import SliceToShape
from triton.flagmega.ir.ops.tensors.unpack import Unpack, unpack_physical


@op_definition(
    "math.vectorized_binary",
    namespace="math",
    functional_name="vectorized_binary",
    display_name="Math.VectorizedBinary",
)
class VectorizedBinary(SameTypePointwiseOp):
    """Arithmetic over one common vector representation.

    Optional operand packing is a load coordinate contract. It derives real
    vector elements from scalar or existing vector inputs without a separate
    layout operation, and does not introduce broadcasting or conversion.
    Optional result unpacking writes those arithmetic lanes to an explicit
    public representation; its shape can mask only the minimum lane tail.
    """
    const_evaluable = True
    lhs = input_parameter(is_tensor())
    rhs = input_parameter(is_tensor())
    binary_op = attribute_parameter()
    lhs_pack = attribute_parameter(default=None)
    rhs_pack = attribute_parameter(default=None)
    result_unpack = attribute_parameter(default=None)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        binary_op = str(attrs["binary_op"])
        if binary_op not in {"add", "mul"}:
            raise IRSchemaError(f"Unsupported vectorized binary op {binary_op!r}.")
        result = {"binary_op": binary_op}
        for name in ('lhs_pack', 'rhs_pack'):
            packing = attrs[name]
            if packing is not None:
                if not isinstance(packing, Mapping):
                    raise IRSchemaError(f"VectorizedBinary {name} must be a lane/axis mapping.")
                result[name] = Pack.normalize_attrs({k: v for k, v in packing.items() if k != 'padding'})
                if 'padding' in packing:
                    if not isinstance(packing['padding'], Mapping):
                        raise IRSchemaError('Operand padding must declare pad_end and pad_value.')
                    result[name]['padding'] = Pad.normalize_attrs(packing['padding'])
        output = attrs['result_unpack']
        if output is not None:
            if not isinstance(output, Mapping):
                raise IRSchemaError('VectorizedBinary result_unpack must declare lanes and axes.')
            result['result_unpack'] = Pack.normalize_attrs({k: v for k, v in output.items() if k != 'shape'})
            if 'shape' in output:
                result['result_unpack'].update(SliceToShape.normalize_attrs({'shape': output['shape']}))
        return result

    @classmethod
    def operand_representation(cls, value, index, attrs):
        packing = attrs.get(('lhs_pack', 'rhs_pack')[index])
        if packing is None:
            return value
        if 'padding' in packing:
            source = tensor_of(value)
            pads = tuple(packing['padding']['pad_end'])
            products = axis_lane_products(packing['lanes'], operand_pack_axes(packing, source.rank))
            if (len(pads) != source.rank or any(pad < 0 or pad >= products.get(axis, 1)
                    or (pad and (not dim.is_fixed or (dim.fixed_value + pad) % products.get(axis, 1)))
                    for axis, (pad, dim) in enumerate(zip(pads, source.shape)))):
                raise IRSchemaError('Operand padding must be the minimum tail needed to align its lane groups.')
            padded = Pad.infer_type((_type_node(source),), packing['padding'])
            if isinstance(value, DistributedType):
                # A minimum lane tail does not change the number of logical
                # lane groups. Pack below proves every split boundary scales
                # in complete groups; source and result owner origins agree.
                # Generic Pad cannot establish this representation proof.
                padded = replace(value, tensor=padded)
            value = padded
        return Pack.infer_type((_type_node(value),), packing)

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        lhs = cls.operand_representation(cls.lhs.type_of(inputs), 0, attrs)
        rhs = cls.operand_representation(cls.rhs.type_of(inputs), 1, attrs)
        lhs_tensor = tensor_of(lhs)
        if lhs != rhs or not isinstance(lhs_tensor.dtype, VectorType):
            raise IRSchemaError("F.math.vectorized_binary requires identical VectorType tensors.")
        return cls.result_representation(lhs, attrs)

    @classmethod
    def result_representation(cls, working, attrs):
        """Map real arithmetic lanes to the public result, with a minimum tail."""
        output = attrs.get('result_unpack')
        if output is None:
            return working
        tensor = tensor_of(working)
        lanes = tuple(output['lanes'])
        if not isinstance(tensor.dtype, VectorType) or tensor.dtype.lanes[:len(lanes)] != lanes:
            raise IRSchemaError('VectorizedBinary result lanes must be a leading arithmetic lane prefix.')
        axes = operand_pack_axes(output, tensor.rank)
        result = Unpack.infer_type((_type_node(working),), {'axes': axes})
        if 'shape' in output:
            source = tensor_of(result)
            shape = tuple(output['shape'])
            products = axis_lane_products(lanes, axes)
            if (len(shape) != source.rank or any(not dim.is_fixed or not 0 <= dim.fixed_value - size
                    < products.get(axis, 1) for axis, (dim, size) in enumerate(zip(source.shape, shape)))):
                raise IRSchemaError('VectorizedBinary result shape must trim only its minimum lane tail.')
            trimmed = SliceToShape.infer_type((_type_node(source),), {'shape': shape})
            # Complete lane groups keep the same owner origin and grain after
            # trimming their last partial group. Generic Slice cannot prove it.
            result = replace(result, tensor=trimmed) if isinstance(result, DistributedType) else trimmed
        return result

    @classmethod
    def working_representation(cls, result, attrs):
        """Invert the public result mapping without changing owner/grain units."""
        output = attrs.get('result_unpack')
        if output is None:
            return result
        original = result
        tensor = tensor_of(result)
        products = axis_lane_products(tuple(output['lanes']), operand_pack_axes(output, tensor.rank))
        if 'shape' in output:
            shape = tuple(((dim.fixed_value + products.get(axis, 1) - 1) // products.get(axis, 1))
                          * products.get(axis, 1) for axis, dim in enumerate(tensor.shape))
            expanded = tensor_type(tensor.dtype, shape, layout=tensor.layout)
            result = replace(result, tensor=expanded) if isinstance(result, DistributedType) else expanded
        working = Pack.infer_type((_type_node(result),), output)
        if cls.result_representation(working, attrs) != original:
            raise IRSchemaError('VectorizedBinary result mapping does not round trip.')
        return working

    @classmethod
    def distributed_input_type_tuples(cls, choices, attrs):
        represented = []
        for index, values in enumerate(choices):
            by_type = {}
            for value in values:
                if getattr(value, 'partial', None) is not None:
                    continue
                try:
                    key = cls.operand_representation(value, index, attrs)
                except IRSchemaError:
                    continue
                by_type.setdefault(key, []).append(value)
            represented.append(by_type)
        if len(represented) != 2:
            return
        for key, lhs in represented[0].items():
            if key in represented[1]:
                yield from product(lhs, represented[1][key])

    @classmethod
    def distributed_output_type_candidates(cls, choices, output_type, attrs):
        result = {}
        for index, values in enumerate(choices):
            for value in values:
                if not isinstance(value, DistributedType) or value.partial is not None:
                    continue
                try:
                    represented = cls.operand_representation(value, index, attrs)
                    represented = cls.result_representation(represented, attrs)
                except IRSchemaError:
                    continue
                if tensor_of(represented) == tensor_of(output_type):
                    result[represented] = None
        return tuple(result)

    @classmethod
    def infer_distributed_input_types(cls, output_type, logical_input_types, attrs):
        if (not isinstance(output_type, DistributedType) or output_type.partial is not None
                or len(logical_input_types) != 2):
            return ()
        values = []
        try:
            working = cls.working_representation(output_type, attrs)
        except IRSchemaError:
            return ()
        for index, logical in enumerate(logical_input_types):
            packing = attrs.get(('lhs_pack', 'rhs_pack')[index])
            value = working
            if packing is not None:
                dtype = tensor_of(working).dtype
                lanes = tuple(packing['lanes'])
                if not isinstance(dtype, VectorType) or dtype.lanes[:len(lanes)] != lanes:
                    return ()
                value = Unpack.infer_type((_type_node(working),), {'axes': operand_pack_axes(packing,
                                                                                          working.tensor.rank)})
                if 'padding' in packing:
                    value = replace(value, tensor=tensor_of(logical))
            if tensor_of(value) != tensor_of(logical):
                return ()
            try:
                if cls.operand_representation(value, index, attrs) != working:
                    return ()
            except IRSchemaError:
                return ()
            values.append(value)
        return (tuple(values),)

    @classmethod
    def evaluate(cls, node, arguments, context):
        values = list(arguments)
        for index, name in enumerate(('lhs_pack', 'rhs_pack')):
            packing = node.attrs.get(name)
            if packing is not None:
                source = tensor_of(context.types[node.inputs[index]])
                if 'padding' in packing:
                    shape = list(values[index].shape)
                    for axis, pad in enumerate(packing['padding']['pad_end']):
                        shape[axis] += pad
                    padded = values[index].new_full(shape, packing['padding']['pad_value'])
                    padded[tuple(slice(0, extent) for extent in values[index].shape)] = values[index]
                    values[index] = padded
                values[index] = pack_physical(values[index], source.rank, tuple(packing['lanes']),
                                              operand_pack_axes(packing, source.rank),
                                              source.dtype.lanes if isinstance(source.dtype, VectorType) else ())
        lhs, rhs = values
        value = lhs + rhs if node.attrs["binary_op"] == "add" else lhs * rhs
        output = node.attrs.get('result_unpack')
        if output is not None:
            working = tensor_of(cls.operand_representation(context.types[node.inputs[0]], 0, node.attrs))
            value = unpack_physical(value, working.rank, working.dtype.lanes, operand_pack_axes(output, working.rank))
            if 'shape' in output:
                value = value[tuple(slice(0, size) for size in output['shape'])].contiguous()
        return value

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        tensor = tensor_of(cls.working_representation(node.type, node.attrs))
        elements = tensor_elements(tensor)
        if elements is not None:
            elements *= tensor.dtype.lane_count
        size = tensor_nbytes(tensor)
        return OpCost(flops=elements, bytes_read=None if size is None else size * 2,
                      bytes_written=tensor_nbytes(node.type))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        from triton.flagmega.ir.distributed_type import local_tensor_type
        def local(value):
            return local_tensor_type(value) if isinstance(value, DistributedType) else tensor_of(value)
        # Invert lane mappings on the global typed operands before measuring
        # their local shards. The default synthetic local-result adapter loses
        # the shape to which the public result extent contract belongs.
        working = local(cls.operand_representation(inputs[0].type, 0, attrs))
        elements = tensor_elements(working)
        reads = tuple(tensor_nbytes(local(value.type)) for value in inputs)
        written = tensor_nbytes(local(return_type))
        if elements is None or any(size is None for size in (*reads, written)):
            return None
        return OpCostFactors(cpu_cycles=elements * working.dtype.lane_count,
                             block_local_memory_load_bytes=sum(reads),
                             block_local_memory_store_bytes=written)


def _type_node(value):
    return Node('__operand_type', 'builtin.var', (), value, attrs={'name': '__operand_type'})


def operand_pack_axes(packing, rank):
    lanes = tuple(packing['lanes'])
    axes = tuple(packing['axes']) if 'axes' in packing else (packing['axis'],) * len(lanes)
    return normalize_axes(axes, rank)


__all__ = ["VectorizedBinary"]
