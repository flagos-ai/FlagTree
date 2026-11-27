# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Mixed execution domains retain fork/join lifetimes, launches and graphs."""
from copy import deepcopy
from dataclasses import replace
import importlib.util

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.codegen.triton.execution_programs import plan_execution_programs
from triton.flagmega.codegen.triton.kernel_dispatch import selected_kernel_nodes
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime import load
from triton.flagmega.runtime.kernel_programs import verified_kernel_programs
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
from triton.flagmega.targets import NvidiaSm90Target


def compile_mesh_fork():
    class Graph(fm.Module):
        def forward(self):
            kind = fm.tensor_type('float32', (4096,))
            x, y = self.input('x', kind, id='x'), self.input('y', kind, id='y')
            a = fm.F.math.add(x, y, name='prefix', metadata={'execution_mesh': (4, 8)})
            b = fm.F.math.add(a, x, name='left', metadata={'execution_mesh': (8, 16)})
            c = fm.F.math.mul(a, y, name='right', metadata={'execution_mesh': '*',
                'kernel_program_kind': 'ordinary', 'kernel_num_warps': 4})
            d = fm.F.math.add(b, c, name='join', metadata={'execution_mesh': (4, 16)})
            self.function('main', (x, y), (d,))

    class Target(NvidiaSm90Target):
        def select_microkernels(self, module):
            module = super().select_microkernels(module)
            kernels = selected_kernel_nodes(module)
            module = replace(module, metadata={**module.metadata,
                'launch_contract': {**module.metadata.get('launch_contract', {}),
                                    **self.plan_launch(module, kernels)}})
            module = plan_execution_programs(module, self, kernels)
            requests = module.metadata['kernel_program_partition']
            assert len(requests) == 4
            parents = ((), (0,), (0,), (1, 2))
            return replace(module, metadata={**module.metadata, 'kernel_program_partition': tuple(
                {**request, 'depends_on': tuple(requests[p]['symbol'] for p in dependencies)}
                for request, dependencies in zip(requests, parents, strict=True))})

    compiler = Compiler()
    compiler.target = Target()
    return compiler.compile(Graph(dialect='nn', stage='imported', entry='main').build()).module


def test_mesh_dag_render_reload_and_typed_launch_verification(tmp_path):
    module = compile_mesh_fork()
    package = render_triton_package(module, tmp_path)
    assert package['kernel_program_schema'] == 'flagmega.kernel-programs/v7-mesh-dag'
    specs = package['kernels']
    assert specs[1]['stream_index'] != specs[2]['stream_index']
    assert specs[3]['depends_on'] == [specs[1]['symbol'], specs[2]['symbol']]
    assert [s['grid'] for s in specs] == [[32, 1, 1], [128, 1, 1], [32, 1, 1], [64, 1, 1]]
    spec = importlib.util.spec_from_file_location('mesh_dag_source', tmp_path / 'generated_kernels.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    kernels = tuple(getattr(source, s['symbol']) for s in specs)
    assert len(verified_kernel_programs(package, kernels, module)) == 4
    reloaded = fm.load_module(fm.emit_module(module, tmp_path / 'module.py'))
    assert reloaded.semantic_hash == module.semantic_hash
    assert len(verified_kernel_programs(package, kernels, reloaded)) == 4
    forged = deepcopy(package)
    forged['kernels'][2]['stream_index'] = specs[1]['stream_index']
    source.FLAGMEGA_KERNEL_PROGRAMS = forged['kernels']
    with pytest.raises(ArtifactError, match='typed program plan'):
        verified_kernel_programs(forged, kernels, module)


def test_mesh_dag_device_independent_numerics_and_cuda_graph(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    module = compile_mesh_fork()
    runtime = load(write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90',
        emit_executable=True), device='cuda:0')
    x = torch.linspace(-1., 1., 4096, device='cuda')
    y = torch.linspace(.25, .75, 4096, device='cuda')
    output = torch.full_like(x, float('nan'))
    try:
        arguments = tuple(output if s['role'] == 'result' else {'x': x, 'y': y}[s['value']]
            for s in runtime.external_arguments)
        Runtime.prepare(runtime, *arguments)
        assert runtime._program_dag is not None
        assert len(runtime._program_dag.streams) == 2
        caller = torch.cuda.Stream()
        caller.wait_stream(torch.cuda.current_stream())
        Runtime.run_into(runtime, *arguments, stream=caller)
        torch.cuda.synchronize()
        reference = ((x.double() + y.double()) + x.double()) + (x.double() + y.double()) * y.double()
        torch.testing.assert_close(output.double(), reference, rtol=2e-6, atol=2e-6)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=caller):
            Runtime.run_into(runtime, *arguments, stream=caller)
        output.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.double(), reference, rtol=2e-6, atol=2e-6)
    finally:
        runtime.close()
