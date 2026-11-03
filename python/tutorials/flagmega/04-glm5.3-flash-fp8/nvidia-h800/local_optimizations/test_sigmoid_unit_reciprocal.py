"""Check the real activated template and keep the non-unit divisions intact."""
from pathlib import Path
from local_optimizations.kernel_recipe import activate
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry


def _check_actual_recipe_rewrites_only_unit_sigmoid_divisions():
    activate()
    r=TritonTemplateRegistry()
    r.environment.get_template('module.py.jinja')
    s=r.environment.loader.get_source(r.environment,'kernels/mhc_pre/cooperative_stream_fma.py.jinja')[0]
    assert s.count('def _flagmega_sigmoid_unit_reciprocal(')==1
    assert '_fm_pre = _flagmega_sigmoid_unit_reciprocal(' in s
    assert '_fm_post = _flagmega_sigmoid_unit_reciprocal(' in s
    assert 'libdevice.div_rn(_fm_comb' in s
    assert 'loop_unroll_factor={{ call.sinkhorn_iterations - 1 }}' in s
    fused=(Path(__file__).parent/'fused-body.py.jinja').read_text()
    assert '_flagmega_sigmoid_unit_reciprocal(1.+libdevice.exp(-logits))' in fused
    assert 'BlockEncoding([1,4],[8,4],[8,1],[0,1])' in fused
    assert 'total=total+tl.reshape(v0' in fused


def test_actual_recipe_rewrites_only_unit_sigmoid_divisions():
    # The recipe installs process-global renderer hooks in a fresh worker.
    # Neighboring tests also replace those hooks; exercise the real recipe
    # in its own interpreter so their hook lifetime cannot change this check.
    import subprocess
    import sys
    source = Path(__file__).resolve()
    code = ("import sys, runpy; sys.path.insert(0, " + repr(str(source.parents[1])) + "); "
            "runpy.run_path(" + repr(str(source)) + ")['_check_actual_recipe_rewrites_only_unit_sigmoid_divisions']()")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_sigmoid_scaling_keeps_reciprocal_and_residuals_in_normal_domain():
    from fractions import Fraction
    from local_optimizations.sigmoid_unit_reciprocal import SCALE,FTZ_LIMIT
    assert Fraction(SCALE)==Fraction(1,2**64)
    assert Fraction(FTZ_LIMIT)==Fraction(2**126)
    # For binary32 y>=1, y*2^-64 is exact and stays in [2^-64,2^64).
    # A product of 24-bit significands near one has a nonzero residual >=2^-47.
    # Thus even the smallest reciprocal times that residual exceeds 2^-126.
    assert Fraction(1,2**64)*Fraction(1,2**47)>Fraction(1,2**126)
    for power in (0,1,63,64,124,125,126,127):
        for mantissa in (0,1,2**22,2**23-1):
            y=Fraction((2**23+mantissa)*2**power,2**23)
            scaled=y*Fraction(1,2**64)
            assert Fraction(1,2**64)<=scaled<Fraction(2**64)
            assert Fraction(1,2**64)<1/scaled<=Fraction(2**64)
            if y>Fraction(2**126):assert 1/y<Fraction(1,2**126)


def test_changed_sigmoid_denominator_is_rejected():
    import pytest
    from local_optimizations.sigmoid_unit_reciprocal import transform_template
    from triton.flagmega.errors import CodegenError
    with pytest.raises(CodegenError,match='unit sigmoid'):
        transform_template('libdevice.div_rn(1., 0.5 + libdevice.exp(-x))')


def test_early_mhc_scope_declares_helper_and_empty_program_is_byte_identical():
    from jinja2 import Environment
    from local_optimizations.sigmoid_unit_reciprocal import with_helper
    source='@triton.jit\ndef early_mhc(x):\n    return _flagmega_sigmoid_unit_reciprocal(x)\n'
    template=Environment().from_string(with_helper(source))
    early=template.render(render_calls=[dict(family='mhc_pre')])
    assert early.count('def _flagmega_sigmoid_unit_reciprocal(')==1
    original=Environment().from_string(source).render()
    assert template.render(render_calls=[])==original
    assert template.render(render_calls=[dict(family='partitioned_linear')])==original
