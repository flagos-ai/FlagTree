# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Chunk-style numerical boundaries survive normal codegen and packed ABIs."""

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from python.test.flagmega.codegen.triton.kernels.kda_gated_recurrent_core.test_kda_gated_recurrent_runtime import (
    _module,
    _values,
)


@pytest.mark.parametrize('packing', ['scalar', 'direct', 'demand', 'head_split'])
@pytest.mark.parametrize('initial_state', ['zero', 'nonzero', 'subnormal_update'])
@pytest.mark.parametrize('l2norm', [False, True])
def test_chunk_device_numerics_and_abi(tmp_path, packing, initial_state, l2norm):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('SM90 CUDA is required')
    attrs = {'output_lanes': (2, 4)} if packing in ('direct', 'head_split') else {}
    if packing == 'demand':
        attrs['consumer_lanes'] = (2, 4)
    if packing == 'head_split':
        attrs.update(placement=fm.Placement((2,), 'x', 'b'), head_policy=fm.SBP.split_contiguous((0,)))
    module = _module(heads=2, dim=128, arithmetic='chunk_bf16', query_scale=128 ** -.5, l2norm=l2norm, **attrs)
    host = _values(torch, heads=2, dim=128)
    if initial_state in ('zero', 'subnormal_update'):
        host['state'].zero_()
    if initial_state == 'subnormal_update':
        # Pinned FlashKDA CUDA build uses fast math: its state FMA flushes
        # subnormal FP32 inputs/results while BF16 output arithmetic does not.
        # A tiny but representable outer product must not enter the cache.
        for name in ('q', 'k', 'gate'):
            host[name].zero_()
        host['q'][..., 0] = 1
        host['k'][..., 0] = 1
        host['k'][..., 1] = 2. ** -62
        host['v'].fill_(2. ** -65)
        host['beta'].zero_()
        host['gate_bias'].zero_()
        host['decay_rate'].fill_(1)
    # CPU transcendental functions need not match PTX approximations bitwise;
    # BF16 conversion and the state export contract must still be observable.
    reference_module = _module(heads=2, dim=128, arithmetic='chunk_bf16', query_scale=128 ** -.5, l2norm=l2norm)
    expected = TorchEvaluator(DictWeightResolver({})).run(reference_module, host)
    path = write_artifact(Compiler().compile(module).module, tmp_path / 'chunk',
                          target='nvidia-sm90', emit_executable=True)
    source = (path / 'generated_kernels.py').read_text()
    assert '# flagmega-kernel: pack/' not in source
    assert 'tanh.approx.f32' in source
    runtime = load(path, device='cuda:0')
    try:
        shape = (1, 2, 128) if packing == 'scalar' else (1, 2, 16, 2, 4)
        buffers = (torch.empty(shape, dtype=torch.bfloat16, device='cuda'),
                   torch.empty_like(host['state'], device='cuda'))
        outputs = {spec['value']: tensor for spec, tensor in zip(
            (s for s in runtime.external_arguments if s['role'] == 'result'), buffers, strict=True)}
        values = {name: tensor.cuda().clone() for name, tensor in host.items()}
        args = tuple(outputs[spec['value']] if spec['role'] == 'result' else values[spec['value']]
                     for spec in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *args)
        GeneratedTirCallGraphModule.run_into(runtime, *args)
        actual = (buffers[0].cpu().reshape(1, 2, 128), buffers[1].cpu())
        torch.testing.assert_close(actual, expected, rtol=.01, atol=1e-4)
        assert torch.equal(actual[1], actual[1].bfloat16().float())
        if initial_state == 'subnormal_update':
            assert torch.count_nonzero(actual[1][:, :, 1]) == 0
            assert torch.count_nonzero(actual[1][:, :, 0]) > 0
        assert runtime.resource_report['spill_bytes'] == 0
    finally:
        runtime.close()
