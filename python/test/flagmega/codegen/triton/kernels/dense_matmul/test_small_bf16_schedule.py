# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Small BF16 projections select an existing GEMV through the normal plan."""

from dataclasses import replace
import importlib.util
from math import prod
from pathlib import Path
import sys

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.targets import NvidiaSm90Target


@pytest.fixture(scope='module')
def policy():
    root = (Path(__file__).resolve().parents[6]
            / 'tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800/local_optimizations')
    name = '_flagmega_small_bf16_test_profile'
    previous = TritonTemplateRegistry.render
    spec = importlib.util.spec_from_file_location(name, root / '__init__.py', submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    try:
        spec.loader.exec_module(package)
        package.activate_base()
        render = TritonTemplateRegistry.render
        module = __import__(name + '.small_bf16_schedule', fromlist=['select'])
    finally:
        TritonTemplateRegistry.render = previous
    try:
        yield module, render
    finally:
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + '.'):
                sys.modules.pop(key)


def _graph(m, k, n, placement, output='bfloat16', axes=()):
    b = fm.SBP.broadcast()
    owners = prod(placement.hierarchy[axis] for axis in axes)
    groups = (n // 8 + owners - 1) // owners

    class Graph(fm.Module):
        def forward(self):
            x = self.input('x', fm.DistributedType(fm.tensor_type('bfloat16', (m, k)), (b, b), placement), id='x')
            w = self.input('w', fm.DistributedType(
                fm.tensor_type(fm.vector_type('bfloat16', (8, 2, 8)), (k // 16, n // 8)),
                (b, fm.SBP.split_contiguous(axes, groups) if axes else b), placement), id='w')
            no = fm.F.builtin.none()
            projected = fm.F.ntt.packed_matmul(x, w, no, no, output_data_type=output, name='projection',
                metadata={'selected_vectorization': 'vectorization.matmul.n',
                          'selected_vector_axes': (1,), 'selected_vector_lanes': (8,)})
            sentinel = self.input('sentinel', fm.tensor_type('float32', (128,)), id='sentinel')
            split = fm.DistributedType(sentinel.type, (fm.SBP.split_contiguous((0, 1), 1),), placement)
            local = fm.F.distributed.force_boxing(sentinel, split)
            copied = fm.F.distributed.force_boxing(fm.F.math.add(local, local), sentinel.type)
            self.function('main', (x, w, sentinel), (projected, copied))

    return Graph(dialect='distributed', stage='frozen_constants', entry='main',
                 metadata={'auto_distribution': {'placement': placement.to_data()}}).build()


def _compile(source, placement, policy, selected):
    class Target(NvidiaSm90Target):
        def propose_tir(self, module):
            return policy.select(super().propose_tir(module))

    base = NvidiaSm90Target().triton_implementation_model
    identity = 'tir.dense_matmul.packed_k_major_mma'
    catalog = replace(base, implementations=tuple(
        replace(value, parameters={**value.parameters, 'compute_num_warps': 8})
        if value.id == identity else value for value in base.implementations),
        preferences={**base.preferences, 'dense_matmul': tuple(dict.fromkeys((identity, *base.preferences['dense_matmul'])))})
    catalog = policy.implementation_model(catalog)
    compiler = Compiler(CompileOptions(placement=placement))
    compiler.target = (Target if selected else NvidiaSm90Target)(triton_implementation_model=catalog)
    return compiler.compile(source).module


@pytest.mark.parametrize('m,k,n,output,selected', [
    (1, 4096, 16, 'bfloat16', True),
    (1, 1152, 8, 'bfloat16', True),
    (1, 512, 8, 'bfloat16', False),
    (1, 1088, 8, 'bfloat16', False),
    (1, 4096, 24, 'bfloat16', False),
    (2, 4096, 16, 'bfloat16', False),
    (1, 4096, 16, 'float32', False),
])
def test_selection_records_and_serialized_implementation_agree(tmp_path, policy, m, k, n, output, selected):
    helper, _ = policy
    mesh = fm.Placement((1, 1), 'yx', 'bb')
    module = _compile(_graph(m, k, n, mesh, output), mesh, helper, True)
    record = module.selection_map['tir.projection']
    definitions = [value for value in module.kernel_definitions if value.dispatch.microkernel.family == 'dense_matmul']
    assert len(definitions) == 1
    micro = definitions[0].dispatch.microkernel
    assert (record.candidate_id == helper.IMPLEMENTATION) is selected
    assert record.candidate_id == micro.implementation
    if selected:
        assert record.policy == helper.POLICY
        assert micro.parameters['block_k'] == 128
        assert micro.parameters['compute_num_warps'] == 8
    path = fm.emit_module(module, tmp_path / 'selected.py')
    resumed = fm.load_module(path)
    assert resumed.semantic_hash == module.semantic_hash
    assert fm.verify_buffer_plan(resumed) == fm.verify_buffer_plan(module)


@pytest.mark.parametrize('k,n,axes', [(4096, 128, (1,)), (4096, 2048, (0, 1)), (1152, 128, (1,))])
def test_gemv_profile_preserves_complete_projection_and_product_precision(
        tmp_path, monkeypatch, policy, k, n, axes):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90 CUDA')
    helper, render = policy
    mesh = fm.Placement((8, 16), 'yx', 'bb')
    source = _graph(1, k, n, mesh, axes=axes)
    modules = [_compile(source, mesh, helper, selected) for selected in (False, True)]
    assert modules[1].selection_map['tir.projection'].candidate_id == helper.IMPLEMENTATION
    with monkeypatch.context() as patch:
        patch.setattr(TritonTemplateRegistry, 'render', render)
        paths = [write_artifact(module, tmp_path / str(i), target='nvidia-sm90', emit_executable=True)
                 for i, module in enumerate(modules)]
    runtimes = [load(path, device='cuda:0') for path in paths]
    x = torch.empty((1, k), dtype=torch.bfloat16, device='cuda')
    w = torch.empty((k, n), dtype=torch.bfloat16, device='cuda')
    packed = torch.empty((k // 16, n // 8, 8, 2, 8), dtype=torch.bfloat16, device='cuda')
    sentinel = torch.arange(128, dtype=torch.float32, device='cuda')
    outputs, guards = [], []
    for runtime in runtimes:
        storage = torch.full((n + 32,), 123., dtype=torch.bfloat16, device='cuda')
        copied = torch.full((160,), 123., device='cuda')
        names = [spec['value'] for spec in runtime.external_arguments if spec['role'] == 'result']
        outputs.append(dict(zip(names, (storage[:n].reshape(1, n // 8, 8), copied[:128]), strict=True)))
        guards.extend((storage[n:], copied[128:]))
    try:
        for mutation in range(3):
            if mutation < 2:
                x.copy_((((torch.arange(k, device='cuda') % 13) - 6 + mutation) / 128).reshape(1, k))
                w.copy_((((torch.arange(k * n, device='cuda') % 17) - 8) / 128).reshape(k, n))
            else:
                x.zero_(); w.zero_()
                x[0, 0] = 129 / 128; x[0, 1] = 1.
                w[0].fill_(129 / 128); w[1].fill_(-65 / 64)
            packed.copy_(w.reshape(k // 16, 2, 8, n // 8, 8).permute(0, 3, 4, 1, 2))
            expected = (x.cpu().double() @ w.cpu().double()).bfloat16().cuda()
            for runtime, output in zip(runtimes, outputs, strict=True):
                values = dict(x=x, w=packed, sentinel=sentinel, **output)
                arguments = tuple(values[spec['value']] for spec in runtime.external_arguments)
                GeneratedTirCallGraphModule.prepare(runtime, *arguments)
                GeneratedTirCallGraphModule.run_into(runtime, *arguments)
                torch.cuda.synchronize()
                torch.testing.assert_close(next(iter(output.values())).reshape(1, n), expected, rtol=0, atol=0)
                torch.testing.assert_close(tuple(output.values())[1], sentinel * 2, rtol=0, atol=0)
                assert runtime.resource_report['grid'] == [128, 1, 1]
                assert runtime.resource_report['spill_bytes'] == 0
            assert all(torch.all(value == 123.) for value in guards)
            if mutation == 2:
                assert torch.all(expected == 2 ** -14)
        if k == 4096 and n == 2048:
            # Retain the random input on which MMA differs at five BF16
            # outputs. The selected GEMV is checked against an independent
            # FP64 dot, rather than treating another reduction tree as exact.
            generator = torch.Generator(device='cuda').manual_seed(712)
            x.copy_(torch.randn((1, k), device='cuda', generator=generator))
            w.copy_(torch.randn((k, n), device='cuda', generator=generator) * .01)
            packed.copy_(w.reshape(k // 16, 2, 8, n // 8, 8).permute(0, 3, 4, 1, 2))
            expected = (x.cpu().double() @ w.cpu().double()).bfloat16().cuda()
            runtime, output = runtimes[1], outputs[1]
            values = dict(x=x, w=packed, sentinel=sentinel, **output)
            arguments = tuple(values[spec['value']] for spec in runtime.external_arguments)
            GeneratedTirCallGraphModule.run_into(runtime, *arguments)
            torch.cuda.synchronize()
            torch.testing.assert_close(next(iter(output.values())).reshape(1, n), expected, rtol=0, atol=0)
            assert all(torch.all(value == 123.) for value in guards)
    finally:
        for runtime in runtimes:
            runtime.close()
