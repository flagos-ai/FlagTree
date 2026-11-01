# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""BF16 expert projections preserve ordered K fragments and masked tails."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load
from triton.flagmega.importer import MemoryCheckpoint
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import stage_module


@pytest.mark.parametrize('definition,packed', ((SparseExpertsGateUp, False), (SparseExpertsGateUp, True),
                                               (SparseExpertsDown, False), (SparseExpertsDown, True)))
def test_ordered_mma_matches_independent_eager(tmp_path, definition, packed):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    attrs = dict(round_projections=True, round_activation=True) if definition is SparseExpertsGateUp else {}
    module = stage_module(definition, tokens=32, hidden=2048, intermediate=40, packed=packed,
                          reduction_order='mma_k16', **attrs)
    parameters = module.function_map['main'].parameters
    scales = {p for p in parameters if p.endswith('_scale')}
    module = fm.verify_module(replace(module, stage='imported', nodes=tuple(
        replace(n, op='builtin.splat_const', attrs={'value': 1.}) if n.id in scales else n for n in module.nodes),
        functions=(replace(module.function_map['main'], parameters=tuple(p for p in parameters if p not in scales)),)))
    compiled = Compiler().compile(module).module
    artifact = write_artifact(compiled, tmp_path/'artifact', target='nvidia-sm90', emit_executable=True,
                              checkpoint=MemoryCheckpoint({}, {}))
    runtime = load(artifact, device='cuda:0')
    generator = torch.Generator().manual_seed(20261023)
    values = {}
    for name in module.function_map['main'].parameters:
        kind = fm.logical_type(module.node_map[name].type)
        shape = tuple(d.fixed_value for d in kind.shape)
        if name == 'router_expert_ids':
            value = (torch.arange(32*3).reshape(32,3) * 3 + 1).remainder(5).int()
        else:
            lanes = getattr(kind.dtype, 'lanes', ())
            value = (torch.randn((*shape,*lanes),generator=generator) / 16).bfloat16()
        values[name] = value.cuda()
    expected, = TorchEvaluator(DictWeightResolver({})).run(module, values)
    args = tuple(values[p] for p in module.function_map['main'].parameters)
    try:
        runtime.prepare(*args)
        actual = runtime.run(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reports = runtime.resource_report
        if isinstance(reports, dict):
            reports = (reports,)
        assert all(r['spill_bytes']==0 for r in reports)
    finally:
        runtime.close()
