# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Operand lane access belongs to vector arithmetic, including scalar inputs."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.math.vectorized_binary import VectorizedBinary
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


@pytest.mark.parametrize('binary_op', ['add', 'mul'])
def test_scalar_operands_produce_real_multiaxis_vector_elements(tmp_path, binary_op):
    scalar = fm.tensor_type('bfloat16', (4, 12))
    packing = {'lanes': (2, 3), 'axes': (0, 1)}
    module = primitive_module(VectorizedBinary, (scalar, scalar), binary_op=binary_op,
                              lhs_pack=packing, rhs_pack=packing)
    fm.verify_module(module)
    assert module.node_map['output'].type == fm.tensor_type(fm.vector_type('bfloat16', (2, 3)), (2, 4))
    lhs = (torch.arange(48).reshape(4, 12).float() / 32).bfloat16()
    rhs = ((torch.arange(48).reshape(4, 12).float() % 7 - 3) / 16).bfloat16()
    scalar_result = lhs + rhs if binary_op == 'add' else lhs * rhs
    expected = pack_physical(scalar_result, 2, (2, 3), (0, 1))
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {'lhs': lhs, 'rhs': rhs})[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fm.load_module(fm.emit_module(module, tmp_path / 'contract.py')).semantic_hash == module.semantic_hash


def test_one_operand_already_packed_preserves_existing_lane_order():
    scalar = fm.tensor_type('float32', (4, 12))
    vector = fm.tensor_type(fm.vector_type('float32', (2, 3)), (2, 4))
    packing = {'lanes': (2, 3), 'axes': (0, 1)}
    module = primitive_module(VectorizedBinary, (scalar, vector), binary_op='add', lhs_pack=packing)
    lhs = torch.arange(48).float().reshape(4, 12)
    rhs_scalar = lhs / 16
    rhs = pack_physical(rhs_scalar, 2, (2, 3), (0, 1))
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {'lhs': lhs, 'rhs': rhs})[0]
    torch.testing.assert_close(actual, pack_physical(lhs + rhs_scalar, 2, (2, 3), (0, 1)), rtol=0, atol=0)


def test_packing_existing_vectors_keeps_their_trailing_lanes():
    source = fm.tensor_type(fm.vector_type('float32', (3,)), (16,))
    packing = {'lanes': (2, 4), 'axes': (0, 0)}
    module = primitive_module(VectorizedBinary, (source, source), binary_op='add',
                              lhs_pack=packing, rhs_pack=packing)
    lhs = torch.arange(48).float().reshape(16, 3)
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {'lhs': lhs, 'rhs': lhs / 8})[0]
    expected = pack_physical(lhs + lhs / 8, 1, (2, 4), (0, 0), (3,))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert module.node_map['output'].type.dtype.lanes == (2, 4, 3)


def test_tail_padding_is_loaded_by_arithmetic_and_fills_real_output_lanes(tmp_path):
    source = fm.tensor_type('bfloat16', (3, 10))
    packing = {'lanes': (2, 4), 'axes': (0, 1), 'padding': {'pad_end': (1, 2), 'pad_value': 0.5}}
    module = primitive_module(VectorizedBinary, (source, source), binary_op='add',
                              lhs_pack=packing, rhs_pack=packing)
    lhs = torch.arange(30).float().reshape(3, 10).bfloat16() / 16
    padded = torch.nn.functional.pad(lhs, (0, 2, 0, 1), value=0.5)
    expected = pack_physical(padded + padded, 2, (2, 4), (0, 1))
    actual = TorchEvaluator(DictWeightResolver({})).run(module, {'lhs': lhs, 'rhs': lhs})[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert fm.load_module(fm.emit_module(module, tmp_path / 'tail.py')).semantic_hash == module.semantic_hash


def test_operand_pack_distribution_scales_units_and_rejects_cut_lanes():
    from triton.flagmega.errors import IRSchemaError
    source = fm.tensor_type('float32', (64,))
    place = fm.Placement((2,), 'x', 'b')
    packing = {'lanes': (4,), 'axes': (0,)}
    attrs = {'binary_op': 'add', 'lhs_pack': packing, 'rhs_pack': packing}
    scalar = fm.DistributedType(source, (fm.SBP.split_block_cyclic((0,), 8),), place)
    left = fm.Node('left', 'builtin.var', (), scalar, attrs={'name': 'left'})
    result = VectorizedBinary.infer_type((left, left), attrs)
    assert result.axis_policies == (fm.SBP.split_block_cyclic((0,), 2),)
    assert VectorizedBinary.infer_distributed_input_types(result, (source, source), attrs) == ((scalar, scalar),)
    cut = fm.DistributedType(source, (fm.SBP.split_block_cyclic((0,), 2),), place)
    with pytest.raises(IRSchemaError, match='lane|scale'):
        VectorizedBinary.infer_type((fm.Node('cut', 'builtin.var', (), cut, attrs={'name': 'cut'}), left), attrs)
    partial = fm.DistributedType(source, (fm.SBP.broadcast(),), place, partial=fm.SBP.partial((0,)))
    assert not tuple(VectorizedBinary.distributed_input_type_tuples(((partial,), (partial,)), attrs))


def test_nested_pack_access_is_composed_without_standalone_adapters():
    from triton.flagmega.compiler import Compiler
    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.tensor_type('float32', (4, 12)))
            inner = fm.F.tensors.pack(x, lanes=(3,), axes=(1,), name='inner')
            outer = fm.F.tensors.pack(inner, lanes=(2,), axes=(0,), name='outer')
            y = fm.F.math.vectorized_binary(outer, outer, binary_op='add', name='y')
            self.function('main', (x,), (y,))
    source = Graph(dialect='high_level', stage='imported', entry='main').build()
    module = Compiler().compile(source, stop_after='propose-distribution').module
    assert not any(n.op == 'tensors.pack' for n in module.nodes)
    x = torch.arange(48).float().reshape(4, 12)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(module, {'x': x})[0], evaluator.run(source, {'x': x})[0], rtol=0, atol=0)


def test_default_vector_contract_keeps_legacy_attributes():
    vector = fm.tensor_type(fm.vector_type('float32', (4,)), (16,))
    module = primitive_module(VectorizedBinary, (vector, vector), binary_op='add')
    assert dict(module.node_map['output'].attrs) == {'binary_op': 'add'}


def test_non_power_of_two_lane_payload_has_a_legal_scalar_tile():
    from triton.flagmega.compiler import Compiler
    vector = fm.tensor_type(fm.vector_type('bfloat16', (2, 4, 3)), (512,))
    module = Compiler().compile(primitive_module(VectorizedBinary, (vector, vector), binary_op='add')).module
    kernel = next(fm.kernel_dispatch_for_call(module, n) for n in module.nodes
                  if fm.kernel_dispatch_for_call(module, n) is not None)
    tile = kernel.parameters['vector_schedule']['physical']['elements_per_program']
    assert tile == 1024
    assert tile & (tile - 1) == 0


@pytest.mark.parametrize('packing', [
    {'lanes': (True,), 'axes': (0,)},
    {'lanes': (4,), 'axes': (0, 1)},
    {'lanes': (4,), 'axes': (2,)},
    {'lanes': (5,), 'axes': (1,)},
])
def test_invalid_operand_pack_is_rejected(packing):
    from triton.flagmega.errors import IRSchemaError
    scalar = fm.tensor_type('float32', (4, 12))
    with pytest.raises(IRSchemaError):
        primitive_module(VectorizedBinary, (scalar, scalar), binary_op='add', lhs_pack=packing, rhs_pack=packing)


@pytest.mark.parametrize('shape,lanes,axes,old_lanes', [
    ((16, 256), (2, 4), (0, 1), ()),
    ((4096,), (2, 4), (0, 0), ()),
    ((4096,), (2, 4), (0, 0), (3,)),
])
def test_device_operand_pack_reads_exact_scalar_addresses(tmp_path, shape, lanes, axes, old_lanes):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from math import prod
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    dtype = fm.vector_type('bfloat16', old_lanes) if old_lanes else 'bfloat16'
    source = fm.tensor_type(dtype, shape)
    packing = {'lanes': lanes, 'axes': axes}
    module = primitive_module(VectorizedBinary, (source, source), binary_op='mul',
                              lhs_pack=packing, rhs_pack=packing)
    path = write_artifact(Compiler().compile(module).module, tmp_path / 'artifact',
                          target='nvidia-sm90', emit_executable=True)
    runtime = load(path, device='cuda:0')
    try:
        physical_shape = (*shape, *old_lanes)
        lhs = ((torch.arange(prod(physical_shape), device='cuda') % 23 - 11).float() / 16).bfloat16().reshape(physical_shape)
        rhs = ((torch.arange(prod(physical_shape), device='cuda') % 19 - 9).float() / 32).bfloat16().reshape(physical_shape)
        runtime.prepare(lhs, rhs)
        actual = runtime.run(lhs, rhs)
        torch.cuda.synchronize()
        expected = pack_physical(lhs * rhs, len(shape), lanes, axes, old_lanes)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runtime.resource_report['spill_bytes'] == 0
    finally:
        runtime.close()


def test_device_fused_binary_keeps_bf16_intermediate_rounding(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.tensor_type('bfloat16', (4096,)))
            y = self.input('y', x.type)
            summed = fm.F.math.add(x, y, name='summed')
            result = fm.F.math.silu(summed, name='result')
            self.function('main', (x, y), (result,))
    module = Compiler().compile(Graph(dialect='high_level', stage='imported', entry='main').build()).module
    path = write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90', emit_executable=True)
    runtime = load(path, device='cuda:0')
    try:
        lhs = torch.full((4096,), 1.015625, device='cuda', dtype=torch.bfloat16)
        rhs = torch.full_like(lhs, 0.00390625)
        rounded = (lhs + rhs).float()
        expected = (rounded * torch.sigmoid(rounded)).bfloat16()
        unrounded = lhs.float() + rhs.float()
        incorrect = (unrounded * torch.sigmoid(unrounded)).bfloat16()
        assert torch.any(expected != incorrect)
        runtime.prepare(lhs, rhs)
        actual = runtime.run(lhs, rhs)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runtime.resource_report['spill_bytes'] == 0
    finally:
        runtime.close()


def test_device_nonzero_tail_fill_writes_all_packed_output_lanes(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.runtime import load
    source = fm.tensor_type('bfloat16', (3, 10))
    packing = {'lanes': (2, 4), 'axes': (0, 1), 'padding': {'pad_end': (1, 2), 'pad_value': 0.5}}
    module = primitive_module(VectorizedBinary, (source, source), binary_op='add', lhs_pack=packing, rhs_pack=packing)
    artifact = write_artifact(Compiler().compile(module).module, tmp_path / 'artifact',
                              target='nvidia-sm90', emit_executable=True)
    runtime = load(artifact, device='cuda:0')
    try:
        value = torch.arange(30, device='cuda').reshape(3, 10).bfloat16() / 16
        runtime.prepare(value, value)
        actual = runtime.run(value, value)
        torch.cuda.synchronize()
        padded = torch.nn.functional.pad(value, (0, 2, 0, 1), value=0.5)
        expected = pack_physical(padded + padded, 2, (2, 4), (0, 1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runtime.resource_report['spill_bytes'] == 0
    finally:
        runtime.close()
