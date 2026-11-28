# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A logical first residual operand receives a legal distributed view."""
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


def test_float32_residual_with_bf16_projection_cast(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    target = {(9, 0): 'nvidia-sm90', (12, 0): 'nvidia-sm120'}.get(torch.cuda.get_device_capability())
    if target is None:
        pytest.skip('requires an executable FlagMega CUDA target')

    class Graph(fm.Module):

        def forward(self):
            a = self.input('a', fm.tensor_type('float32', (1, 5120)), id='a')
            b = self.input('b', fm.tensor_type('bfloat16', (1, 5120)), id='b')
            c = fm.F.math.add(a, fm.F.tensors.cast(b, 'float32'), name='residual')
            stats = fm.F.nn.norm_stats(c, axis=-1, use_mean=False, name='stats')
            self.function('main', (a, b), (c, stats))

    module = Graph(dialect='high_level', stage='imported', entry='main').build()
    compiled = Compiler(CompileOptions(target=target)).compile(module).module
    artifact = write_artifact(compiled, tmp_path / 'artifact', target=target, emit_executable=True)
    runtime = load(artifact, device='cuda:0')
    torch.manual_seed(312)
    inputs = {
        'a': torch.randn((1, 5120), device='cuda'), 'b': torch.randn((1, 5120), dtype=torch.bfloat16, device='cuda')
    }
    values = {}
    for arg in runtime.external_arguments:
        buf = runtime.buffer_plan.buffer_map[arg['buffer']]
        lanes = buf.dtype.lanes if isinstance(buf.dtype, fm.VectorType) else ()
        dtype = buf.dtype.elem_type if lanes else buf.dtype
        shape = (*buf.shape, *lanes)
        values[buf.id] = (torch.empty(shape, device='cuda', dtype=getattr(torch, dtype.value))
                          if arg['role'] == 'result' else inputs[arg['value']].reshape(shape))
    arguments = tuple(values[arg['buffer']] for arg in runtime.external_arguments)
    outputs = runtime.buffer_plan.function_map['main'].outputs
    actual, stats = (values[outputs[i][1][0]] for i in (0, 1))
    GeneratedTirCallGraphModule.prepare(runtime, *arguments)
    for step in range(3):
        inputs['a'].add_(.125)
        wanted = inputs['a'] + inputs['b'].float()
        GeneratedTirCallGraphModule.run_into(runtime, *arguments)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.reshape_as(wanted), wanted, rtol=0, atol=0)
        torch.testing.assert_close(stats.flatten(), wanted.square().sum().reshape(1), rtol=2e-6, atol=1e-3)
    assert runtime.resource_report['spill_bytes'] == 0
