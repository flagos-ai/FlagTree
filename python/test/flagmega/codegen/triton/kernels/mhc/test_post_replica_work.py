# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Sharing hidden tiles preserves complete mHC post outputs and rounding."""

import importlib.util
import sys
from dataclasses import replace
from math import prod
from pathlib import Path
from types import ModuleType

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.targets import NvidiaSm90Target

from python.test.flagmega.codegen.triton.kernels.dense_matmul.test_replica_owner_groups import fixture

from .test_mhc_runtime import _run, _skip_without_sm90


@pytest.fixture(scope='module')
def optimization():
    root = (Path(__file__).resolve().parents[6]
            / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/local_optimizations')
    name = '_flagmega_mhc_post_replica_tests'
    package = ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    previous = TritonTemplateRegistry.render
    try:
        spec = importlib.util.spec_from_file_location(name + '.mhc_post_work', root / 'mhc_post_work.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        TritonTemplateRegistry.render = previous
    try:
        yield module
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + '.'):
                sys.modules.pop(key)


@pytest.mark.parametrize('cause', ['address', 'partial', 'exclusive', 'private'])
def test_post_hidden_tasks_reject_unproved_physical_sharing(optimization, cause):
    call, context = fixture(owner_axes=())
    abi = call['outputs'][0]['buffers'][0]['abi']
    if cause == 'address':
        call['residual_offset'] = 'shard_coord1 * 16384'
    elif cause == 'private':
        abi['memory_sharing_scope'] = 'block'
    else:
        abi['distributed_type'][cause] = {'axes': [1]}
    assert optimization.hidden_group(call, context) is None


def _graph(tokens, hc, hidden, dtype, mode, placement, *, private_input=False, consume_output=False):
    class Graph(fm.Module):
        def forward(self):
            types = {'value': fm.tensor_type('float32' if private_input else 'bfloat16', (tokens, hidden)),
                     'residual': fm.tensor_type('float32' if private_input else 'bfloat16', (tokens, hc, hidden)),
                     'post': fm.tensor_type('float32', (tokens, hc)),
                     'comb': fm.tensor_type('float32', (tokens, hc, hc))}
            values = tuple(self.input(name, kind, id=name) for name, kind in types.items())
            operands = values
            if private_input:
                operands = tuple(fm.F.distributed.sharded_view(value, fm.DistributedType(
                    value.type, (fm.SBP.broadcast(),) * value.type.rank, placement)) for value in values)
                operands = tuple(fm.F.tensors.cast(value, 'bfloat16') if index < 2 else value
                                 for index, value in enumerate(operands))
            result = fm.F.nn.mhc_post(*operands, output_dtype=dtype, accumulation_mode=mode, name='carry')
            if consume_output:
                result = fm.F.tensors.cast(result, 'float32')
                result = fm.F.distributed.force_boxing(result, fm.logical_type(result.type))
            kind = fm.tensor_type('float32', (128,))
            sentinel = self.input('sentinel', kind, id='sentinel')
            split = fm.DistributedType(kind, (fm.SBP.split_contiguous((0, 1), 1),), placement)
            local = fm.F.distributed.force_boxing(sentinel, split)
            copied = fm.F.distributed.force_boxing(fm.F.math.add(local, local), kind)
            self.function('main', (*values, sentinel), (result, copied))

    return Graph(dialect='distributed', stage='frozen_constants', entry='main',
                 metadata={'auto_distribution': {'placement': placement.to_data()}}).build()


@pytest.mark.parametrize('tokens,hc,hidden,dtype,mode,warps', [
    (1, 4, 4096, 'float32', 'value_fma_first', 8),
    (3, 3, 777, 'float32', 'value_fma_first', 8),
    (2, 4, 4104, 'bfloat16', 'residual_then_value', 8),
    (2, 1, 128, 'bfloat16', 'value_fma_first', 8),
    (1, 4, 4096, 'float32', 'value_fma_first', 16),
])
def test_hidden_tile_sharing_keeps_outputs_guards_and_fma_order(
        tmp_path, monkeypatch, optimization, tokens, hc, hidden, dtype, mode, warps):
    torch = pytest.importorskip('torch')
    _skip_without_sm90(torch)
    placement = fm.Placement((8, 16), 'yx', 'bb')
    compiler = Compiler(CompileOptions(placement=placement))
    base = NvidiaSm90Target().triton_implementation_model
    catalog = replace(base, implementations=tuple(
        replace(value, parameters={**value.parameters, 'compute_num_warps': warps})
        if value.family == 'mhc_post' else value for value in base.implementations))
    compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    module = compiler.compile(_graph(tokens, hc, hidden, dtype, mode, placement)).module
    baseline = write_artifact(module, tmp_path / 'baseline', target='nvidia-sm90', emit_executable=True)
    with monkeypatch.context() as patch:
        patch.setattr(TritonTemplateRegistry, 'render', optimization.override)
        candidate = write_artifact(module, tmp_path / 'candidate', target='nvidia-sm90', emit_executable=True)
    assert (candidate / 'generated_kernels.py').read_bytes() != (baseline / 'generated_kernels.py').read_bytes()
    runtimes = [load(path, device='cuda:0') for path in (baseline, candidate)]
    generator = torch.Generator().manual_seed(351)
    values = {'value': torch.randn((tokens, hidden), generator=generator).bfloat16().cuda(),
              'residual': torch.randn((tokens, hc, hidden), generator=generator).bfloat16().cuda(),
              'post': torch.randn((tokens, hc), generator=generator).cuda(),
              'comb': torch.randn((tokens, hc, hc), generator=generator).cuda(),
              'sentinel': torch.arange(128, dtype=torch.float32, device='cuda')}
    results, guards = [], []
    for runtime in runtimes:
        names = runtime.ir_module.function_map[runtime.ir_module.entry].outputs
        outputs, protected = {}, []
        for name, shape, kind in zip(names, ((tokens, hc, hidden), (128,)),
                                     (getattr(torch, dtype), torch.float32), strict=True):
            storage = torch.full((prod(shape) + 32,), 123., dtype=kind, device='cuda')
            outputs[name] = storage[:-32].reshape(shape)
            protected.append(storage[-32:])
        results.append(outputs)
        guards.append(protected)

    def run(expected=None):
        for runtime, outputs in zip(runtimes, results, strict=True):
            _run(runtime, values, outputs, torch)
            assert runtime.resource_report['grid'] == [128, 1, 1]
            assert runtime.resource_report['spill_bytes'] == 0
        first, second = (next(iter(value.values())) for value in results)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
        if expected is not None:
            torch.testing.assert_close(second, torch.full_like(second, expected), rtol=0, atol=0)
        for output, protected in zip(results, guards, strict=True):
            torch.testing.assert_close(tuple(output.values())[1], values['sentinel'] * 2, rtol=0, atol=0)
            assert all(torch.all(value == 123.) for value in protected)

    try:
        run()
        values['residual'].add_(1 / 128)
        run()
        if mode == 'value_fma_first':
            values['value'].fill_(1.5)
            values['residual'].zero_()
            values['residual'][:, 0].fill_(-1.5)
            values['post'].fill_(1 + 2 ** -23)
            values['comb'].zero_()
            values['comb'][:, 0].fill_(1.)
            run(3 * 2 ** -24)
            if hc == 4:
                values['value'].zero_()
                values['residual'].fill_(1.)
                values['post'].zero_()
                values['comb'].copy_(torch.tensor([2 ** 24, 1., -(2 ** 24), 2.], device='cuda')
                                      .view(1, 4, 1).expand(tokens, 4, 4))
                run(2.)
    finally:
        for runtime in runtimes:
            runtime.close()


def test_private_read_replicas_preserve_hidden_task_sharing(tmp_path, monkeypatch, optimization):
    torch = pytest.importorskip('torch')
    _skip_without_sm90(torch)
    from python.test.flagmega.runtime.test_glm_native_mla_private_input import _profile
    profile = _profile(monkeypatch)
    mesh = fm.Placement((8, 16), 'yx', 'bb')
    compiler = Compiler(CompileOptions(placement=mesh))
    compiler.target = NvidiaSm90Target(triton_implementation_model=profile.native_implementation_model())
    module = compiler.compile(_graph(1, 4, 777, 'bfloat16', 'value_fma_first', mesh,
                                     private_input=True, consume_output=True)).module
    plan = fm.verify_buffer_plan(module)
    call, = (node for node in module.nodes if node.op == 'tir.call'
             and (dispatch := fm.kernel_dispatch_for_call(module, node)) is not None
             and dispatch.microkernel.family == 'mhc_post')
    assert all(plan.buffer_map[value].storage == 'block_local_data' for value in call.inputs[:2])
    assert plan.buffer_map[call.id].storage == 'workspace'
    baseline = write_artifact(module, tmp_path / 'baseline', target='nvidia-sm90', emit_executable=True)
    groups = []
    actual_group = optimization.hidden_group

    def record_group(call, context):
        group = actual_group(call, context)
        groups.append(group)
        return group

    with monkeypatch.context() as patch:
        patch.setattr(optimization, 'hidden_group', record_group)
        patch.setattr(TritonTemplateRegistry, 'render', optimization.override)
        candidate = write_artifact(module, tmp_path / 'candidate', target='nvidia-sm90', emit_executable=True)
    assert any(group is not None and group['count'] == 128 for group in groups)
    runtimes = [load(path, device='cuda:0') for path in (baseline, candidate)]
    values = {'value': ((torch.arange(777, device='cuda') % 17 - 8).float() / 8).reshape(1, 777),
              'residual': ((torch.arange(4 * 777, device='cuda') % 13 - 6).float() / 8).reshape(1, 4, 777),
              'post': torch.full((1, 4), 1 + 2 ** -23, device='cuda'),
              'comb': torch.eye(4, device='cuda').reshape(1, 4, 4),
              'sentinel': torch.arange(128, dtype=torch.float32, device='cuda')}
    results, guards = [], []
    for runtime in runtimes:
        names = runtime.ir_module.function_map[runtime.ir_module.entry].outputs
        output = torch.full((64 + 4 * 777 + 64,), -123, dtype=torch.float32, device='cuda')
        sentinel = torch.full((64 + 128 + 64,), -125, dtype=torch.float32, device='cuda')
        results.append({names[0]: output[64:-64].reshape(1, 4, 777), names[1]: sentinel[64:-64]})
        guards.append((output[:64], output[-64:], sentinel[:64], sentinel[-64:]))
    try:
        for mutation in range(4):
            if mutation:
                values['value'].add_(1 / 128)
                values['residual'].sub_(1 / 64)
            for runtime, outputs in zip(runtimes, results, strict=True):
                for output in outputs.values():
                    output.fill_(float('nan'))
                _run(runtime, values, outputs, torch)
                assert runtime.resource_report['spill_bytes'] == 0
                assert runtime.resource_report['grid'] == [128, 1, 1]
                torch.testing.assert_close(tuple(outputs.values())[1], values['sentinel'] * 2, rtol=0, atol=0)
            expected, actual = (next(iter(outputs.values())) for outputs in results)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert all(bool((left == -123).all() & (right == -123).all()
                            & (tail_left == -125).all() & (tail_right == -125).all())
                       for left, right, tail_left, tail_right in guards)
    finally:
        for runtime in runtimes:
            runtime.close()
