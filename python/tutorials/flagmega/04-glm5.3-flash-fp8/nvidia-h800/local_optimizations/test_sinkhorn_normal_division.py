"""Keep exceptional divisions and the complete stabilized RN32 recurrence."""
import ast
from jinja2 import Environment
from local_optimizations.sinkhorn_unroll import transform


def stabilized_source():
    return '''def mix(x):
    _fm_pre = libdevice.div_rn(1., 1. + libdevice.exp(-x))
    _fm_post = libdevice.div_rn(1., 1. + libdevice.exp(-x))
    _fm_comb = libdevice.div_rn(_fm_comb, tl.sum(_fm_comb, 1)[:, None])
    _fm_comb = tl.where(_fm_pair_active, _fm_comb + {{ call.hc_epsilon }}, 0.)
    _fm_comb = libdevice.div_rn(_fm_comb, tl.sum(_fm_comb, 0)[None, :] + {{ call.hc_epsilon }})
    for _fm_iteration in tl.range(0, {{ call.sinkhorn_iterations - 1 }}):
        _fm_comb = libdevice.div_rn(_fm_comb, tl.sum(_fm_comb, 1)[:, None] + {{ call.hc_epsilon }})
        _fm_comb = libdevice.div_rn(_fm_comb, tl.sum(_fm_comb, 0)[None, :] + {{ call.hc_epsilon }})
'''


def test_stabilized_recurrence_uses_native_normal_path_and_keeps_exceptional_divisions():
    rendered = Environment().from_string(transform(stabilized_source())).render(
        call=dict(hc_epsilon=1e-6, sinkhorn_iterations=20))
    tree = ast.parse(rendered)
    general = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and ast.unparse(n.func) == 'libdevice.div_rn']
    normalized = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == 'tl.inline_asm_elementwise']
    assert len(general) == 3  # pre, post, and softmax before epsilon.
    assert len(normalized) == 3
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For))
    assert ast.unparse(loop.iter) == 'tl.range(0, 19, loop_unroll_factor=19)'
    denominators = [ast.unparse(next(k.value for k in n.keywords if k.arg == 'args').elts[1]) for n in normalized]
    assert denominators.count('tl.sum(_fm_comb, 0)[None, :] + 1e-06') == 2
    assert denominators.count('tl.sum(_fm_comb, 1)[:, None] + 1e-06') == 1


import pytest
from fractions import Fraction
from local_optimizations.sinkhorn_normal_division import domain_certificate
from triton.flagmega.errors import CodegenError


def test_complete_domain_bound_stays_normal_through_all_39_divisions():
    certificate = domain_certificate(dict(hc=4, stream_active='4', sinkhorn_iterations=20, hc_epsilon='1e-06'))
    first = Fraction(1, 2**23)
    lower = first * (Fraction(1, 4) * (1 - Fraction(1, 2**17)))**38
    assert lower > Fraction(1, 2**100) > Fraction(1, 2**126)
    assert certificate['full_iterations'] == 20


@pytest.mark.parametrize('field,value', [('hc',8),('stream_active','3'),('sinkhorn_iterations',64),('hc_epsilon',1e-30)])
def test_unproven_neighbor_domains_are_rejected(field,value):
    call=dict(hc=4,stream_active='4',sinkhorn_iterations=20,hc_epsilon=1e-6)
    call[field]=value
    with pytest.raises(CodegenError,match='proven complete'):
        domain_certificate(call)
