# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Arithmetic owns its public output lane mapping and minimum tail mask."""

from math import prod
from dataclasses import replace

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.math.vectorized_binary import VectorizedBinary
from triton.flagmega.ir.ops.tensors.pack import axis_lane_products
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def result_module(shape, lanes, axes, old_lanes=(), *, binary_op='add', metadata=None, packed_rhs=False):
    dtype = fm.vector_type('bfloat16', old_lanes) if old_lanes else 'bfloat16'
    products = axis_lane_products(lanes, axes)
    pads = tuple((-dim) % products.get(axis, 1) for axis, dim in enumerate(shape))
    packing = {'lanes': lanes, 'axes': axes, 'padding': {'pad_end': pads, 'pad_value': 0.5}}
    source = fm.tensor_type(dtype, shape)
    rhs = VectorizedBinary.operand_representation(source, 0, {'lhs_pack': packing}) if packed_rhs else source
    module = primitive_module(VectorizedBinary, (source, rhs),
                              binary_op=binary_op, lhs_pack=packing, rhs_pack=None if packed_rhs else packing,
                              result_unpack={'lanes': lanes, 'axes': axes, 'shape': shape})
    if metadata:
        module = replace(module, nodes=tuple(replace(fm.with_execution_mesh(node, '*'), metadata={
            **node.metadata, 'execution_mesh': '*', 'kernel_program_kind': 'persistent', 'kernel_max_ctas': 2,
        }) if node.op == 'math.vectorized_binary' else node for node in module.nodes))
    return module


@pytest.mark.parametrize('shape,lanes,axes,old_lanes', [
    ((3, 10), (2, 4), (0, 1), ()),
    ((4103,), (4,), (0,), ()),
    ((15, 17), (2, 4), (0, 1), (3,)),
    ((4103,), (2, 4), (0, 0), (3,)),
])
@pytest.mark.parametrize('binary_op', ['add', 'mul'])
def test_result_lanes_keep_values_shape_and_rounding(tmp_path, shape, lanes, axes, old_lanes, binary_op):
    module = result_module(shape, lanes, axes, old_lanes, binary_op=binary_op)
    fm.verify_module(module)
    physical_shape = (*shape, *old_lanes)
    lhs = ((torch.arange(prod(physical_shape)) % 29 - 14).float() / 16).bfloat16().reshape(physical_shape)
    rhs = ((torch.arange(prod(physical_shape)) % 23 - 11).float() / 32).bfloat16().reshape(physical_shape)
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {'lhs': lhs, 'rhs': rhs})[0]
    expected = lhs + rhs if binary_op == 'add' else lhs * rhs
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert tuple(dim.fixed_value for dim in module.node_map['output'].type.shape) == shape
    assert fm.load_module(fm.emit_module(module, tmp_path / 'result.py')).semantic_hash == module.semantic_hash


def test_result_distribution_restores_the_original_scalar_grain():
    source = fm.tensor_type('float32', (4103,))
    place = fm.Placement((2,), 'x', 'b')
    scalar = fm.DistributedType(source, (fm.SBP.split_block_cyclic((0,), 8),), place)
    packing = {'lanes': (4,), 'axes': (0,), 'padding': {'pad_end': (1,), 'pad_value': 0.0}}
    attrs = {'binary_op': 'add', 'lhs_pack': packing, 'rhs_pack': packing,
             'result_unpack': {'lanes': (4,), 'axes': (0,), 'shape': (4103,)}}
    node = fm.Node('x', 'builtin.var', (), scalar, attrs={'name': 'x'})
    result = VectorizedBinary.infer_type((node, node), attrs)
    assert result == scalar
    assert VectorizedBinary.infer_distributed_input_types(result, (source, source), attrs) == ((scalar, scalar),)
    assert VectorizedBinary.distributed_output_type_candidates(((scalar,), (scalar,)), source, attrs) == (scalar,)


@pytest.mark.parametrize('result_unpack', [
    {'lanes': (2,), 'axes': (0,), 'shape': (63,)},
    {'lanes': (4,), 'axes': (0,), 'shape': (60,)},
    {'lanes': (4,), 'axes': (0,), 'shape': (65,)},
])
def test_result_contract_rejects_incompatible_lanes_and_nonminimal_trimming(result_unpack):
    vector = fm.tensor_type(fm.vector_type('float32', (4,)), (16,))
    with pytest.raises(IRSchemaError):
        primitive_module(VectorizedBinary, (vector, vector), binary_op='add', result_unpack=result_unpack)


def test_shared_vector_output_survives_the_result_lane_rewrite():
    from triton.flagmega.compiler import Compiler
    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.tensor_type(fm.vector_type('float32', (4,)), (16,)))
            vector = fm.F.math.vectorized_binary(x, x, binary_op='add', name='vector')
            scalar = fm.F.tensors.unpack(vector, axes=(0,))
            exact = fm.F.tensors.slice_to_shape(scalar, shape=(63,), name='exact')
            self.function('main', (x,), (vector, exact))
    source = Graph(dialect='high_level', stage='imported', entry='main').build()
    module = Compiler().compile(source, stop_after='propose-distribution').module
    assert not any(n.op == 'tensors.slice_to_shape' for n in module.nodes)
    assert module.node_map['vector'].type.dtype == fm.vector_type('float32', (4,))
    assert module.node_map['exact'].op == 'math.vectorized_binary'
    value = torch.arange(64).float().reshape(16, 4)
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {'x': value})
    torch.testing.assert_close(actual[0], value * 2, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], (value * 2).flatten()[:63], rtol=0, atol=0)


@pytest.mark.parametrize('shape,lanes,axes,old_lanes,packed_rhs', [
    ((15, 257), (2, 4), (0, 1), (), False),
    ((4103,), (4,), (0,), (), False),
    ((4103,), (2, 4), (0, 0), (), False),
    ((4103,), (2, 4), (0, 0), (3,), False),
    ((4103,), (4,), (0,), (), True),
])
def test_device_result_mapping_writes_exact_addresses(tmp_path, shape, lanes, axes, old_lanes, packed_rhs):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    module = result_module(shape, lanes, axes, old_lanes, metadata=True, packed_rhs=packed_rhs)
    compiled = Compiler().compile(module).module
    path = write_artifact(compiled, tmp_path / 'artifact', target='nvidia-sm90', emit_executable=True)
    runtime = load(path, device='cuda:0')
    physical_shape = (*shape, *old_lanes)
    generator = torch.Generator(device='cuda').manual_seed(197)
    lhs = (torch.randint(-32, 33, physical_shape, generator=generator, device='cuda').float() / 16).bfloat16()
    rhs = (torch.randint(-32, 33, physical_shape, generator=generator, device='cuda').float() / 32).bfloat16()
    expected = lhs + rhs
    if packed_rhs:
        products = axis_lane_products(lanes, axes)
        padded_shape = tuple(dim + (-dim) % products.get(axis, 1) for axis, dim in enumerate(shape))
        padded = rhs.new_full((*padded_shape, *old_lanes), 0.5)
        padded[tuple(slice(0, dim) for dim in shape)] = rhs
        rhs = pack_physical(padded, len(shape), lanes, axes, old_lanes)
    try:
        runtime.prepare(lhs, rhs)
        actual = runtime.run(lhs, rhs)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
        recipe = TaskLaunchRecipe.from_data(runtime.codegen['kernels'][0]['launch_recipe'])
        assert recipe.work_size >= prod(physical_shape)
        reports = runtime.resource_report
        assert all(report['spill_bytes'] == 0 for report in (reports if isinstance(reports, list) else [reports]))
    finally:
        runtime.close()
