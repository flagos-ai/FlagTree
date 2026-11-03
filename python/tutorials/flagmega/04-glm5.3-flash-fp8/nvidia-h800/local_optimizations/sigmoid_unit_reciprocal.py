"""Exact RN32/FTZ reciprocal for the complete mHC sigmoid denominator."""
import hashlib
from pathlib import Path

from triton.flagmega.errors import CodegenError

SYMBOL = '_flagmega_sigmoid_unit_reciprocal'
SCALE = 2.0 ** -64
FTZ_LIMIT = 2.0 ** 126


def with_helper(source):
    # Some complete mHC projection programs precede the RMS-fused prefix.
    # Declare their scalar helper in their own generated namespace as well.
    # Empty programs retain their exact source text and compilation keys.
    helper = Path(__file__).with_name('sigmoid-reciprocal.py.jinja').read_text()
    return ('{% if render_calls | selectattr("family", "equalto", "mhc_pre") | list %}'
            + helper + '\n{% endif %}' + source)


def transform_template(source):
    for name, count in (('pre', 2), ('post', 1)):
        before = f'libdevice.div_rn(1., 1. + libdevice.exp(-_fm_{name}_logits))'
        if source.count(before) != count:
            raise CodegenError('Original complete mHC unit sigmoid division changed')
        source = source.replace(before, f'{SYMBOL}(1. + libdevice.exp(-_fm_{name}_logits))')
    return with_helper(source)


def describe(package):
    from .sinkhorn_normal_division import domain_certificate
    calls = [c for c in package['render_calls']
             if c['family'] == 'mhc_pre' and c['variant'] == 'cooperative_stream_fma']
    if not calls:
        raise CodegenError('Unit reciprocal requires complete mHC sigmoid calls')
    for call in calls:
        domain_certificate(call)
    package['declared_sigmoid_unit_reciprocal_domain'] = dict(
        selected_calls=[c['call'] for c in calls], numerator=1.0,
        finite_denominator_minimum=1.0, scaling_power=-64,
        normal_RCP_input_and_output=True, FTZ_limit=FTZ_LIMIT,
        full_pre_post_epsilon_and_initial_softmax_preserved=True,
        full_4096_RMS_reduction_order_preserved=True,
        exception_behavior='positive infinity yields zero; NaN propagates')
    for name in ('sigmoid_unit_reciprocal.py', 'sigmoid-reciprocal.py.jinja'):
        path = Path(__file__).with_name(name)
        package['implementation_override']['inputs'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return package
