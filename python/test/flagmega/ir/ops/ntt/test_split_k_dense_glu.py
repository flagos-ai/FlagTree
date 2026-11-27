# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The split result is FP32; the complete GU owns the BF16 rounding."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.ntt.split_k_dense_glu import SplitKDenseGluProjection as Projection, SplitKDenseGluEpilogue as Epilogue


def value(name, dtype, shape):
    return fm.Node(name, 'builtin.var', (), fm.tensor_type(dtype, shape), attrs={'name': name})


def test_projection_and_epilogue_keep_explicit_scalar_types():
    a, w = value('a', 'bfloat16', (1, 2048)), value('w', 'bfloat16', (2048, 12288))
    p = Projection.prepare((a, w), {})
    assert p.result_type == fm.tensor_type('float32', (2, 12288))
    assert Epilogue.prepare((value('p', 'float32', (2, 12288)),), {}).result_type == fm.tensor_type('bfloat16', (1, 6144))


@pytest.mark.parametrize('which,dtype,shape', [('a','float32',(1,2048)), ('a','bfloat16',(2,2048)),
    ('w','bfloat16',(12288,2048)), ('w','bfloat16',(2048,6144))])
def test_rejects_reinterpreted_or_partial_projection_shapes(which, dtype, shape):
    a, w = value('a', 'bfloat16', (1, 2048)), value('w', 'bfloat16', (2048, 12288))
    with pytest.raises(IRSchemaError):
        Projection.prepare((value('a', dtype, shape) if which == 'a' else a,
                            value('w', dtype, shape) if which == 'w' else w), {})


@pytest.mark.parametrize('bad', ['mesh', 'partial', 'sharded_input'])
def test_projection_requires_complete_inputs_in_exact_physical_domain(bad):
    placement = fm.Placement((48, 2), 'xy', 'bb')
    b = fm.SBP.broadcast()
    a, w = value('a', 'bfloat16', (1, 2048)), value('w', 'bfloat16', (2048, 12288))
    a, w = [replace(n, type=fm.DistributedType(n.type, (b, b), placement)) for n in (a, w)]
    if bad == 'mesh':
        wrong = fm.Placement((32, 2), 'xy', 'bb')
        a, w = [replace(n, type=replace(n.type, placement=wrong)) for n in (a,w)]
    elif bad == 'partial':
        a = replace(a, type=replace(a.type, partial=fm.SBP.partial((0,))))
    else:
        a = replace(a, type=replace(a.type, axis_policies=(b, fm.SBP.split_contiguous((1,)))))
    with pytest.raises(IRSchemaError):
        Projection.prepare((a, w), {})


def test_complete_sum_rounds_gate_and_up_before_silu_without_rounding_silu():
    torch = pytest.importorskip('torch')
    generator = torch.Generator().manual_seed(76193)
    p = torch.randn((2,12288), generator=generator) * 0.75
    node = value('p', 'float32', (2,12288))
    actual = Epilogue.evaluate(node, (p,), SimpleNamespace(torch=torch))
    complete = p[0]+p[1]
    gate, up = complete[:6144].to(torch.bfloat16).float(), complete[6144:].to(torch.bfloat16).float()
    expected = (torch.nn.functional.silu(gate)*up).to(torch.bfloat16).reshape(1,6144)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    unrounded = (torch.nn.functional.silu(complete[:6144])*complete[6144:]).to(torch.bfloat16).reshape(1,6144)
    extra_round = (torch.nn.functional.silu(gate).to(torch.bfloat16).float()*up).to(torch.bfloat16).reshape(1,6144)
    assert torch.count_nonzero(actual != unrounded) > 0
    assert torch.count_nonzero(actual != extra_round) > 0
