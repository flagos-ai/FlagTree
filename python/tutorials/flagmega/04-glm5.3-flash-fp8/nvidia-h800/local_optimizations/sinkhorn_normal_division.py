"""Use the native RN32 normal path only in the epsilon-stabilized recurrence."""
import hashlib
from pathlib import Path

from triton.flagmega.errors import CodegenError

# This is the instruction sequence emitted by ptxas for div.rn.ftz.f32's
# normal path: refine the approximate reciprocal, then correct the quotient.
# Inputs and quotient here are normal; the three pre-epsilon divisions retain
# their full FTZ/exceptional handling.
ASM = (
    '{ .reg .f32 r,e,q,ny; '
    'rcp.approx.ftz.f32 r,$2; neg.f32 ny,$2; '
    'fma.rn.f32 e,ny,r,0f3f800000; fma.rn.f32 r,r,e,r; '
    'mul.rn.f32 q,r,$1; fma.rn.f32 e,ny,q,$1; '
    'fma.rn.f32 $0,r,e,q; }'
)


def transform_template(source):
    for axis, suffix, count in ((0, '[None, :]', 2), (1, '[:, None]', 1)):
        denominator = f'tl.sum(_fm_comb, {axis}){suffix} + {{{{ call.hc_epsilon }}}}'
        before = f'libdevice.div_rn(_fm_comb, {denominator})'
        if source.count(before) != count:
            raise CodegenError('Original stabilized Sinkhorn divisions changed')
        after = (f'tl.inline_asm_elementwise({ASM!r}, constraints="=f,f,f", '
                 f'args=[_fm_comb, {denominator}], dtype=tl.float32, is_pure=True, pack=1)')
        source = source.replace(before, after)
    return source


def domain_certificate(call):
    if (call['hc'] != 4 or call['stream_active'] != '4'
            or call['sinkhorn_iterations'] != 20 or float(call['hc_epsilon']) != 1e-6):
        raise CodegenError('Normalized division requires the proven complete 4x4/20-step domain')
    # Binary32 epsilon > 2^-20. The first stabilized column division is
    # > 2^-23; each of the remaining 38 divisions loses at most a factor
    # 4/(1-2^-17), including RN32 rounding. Bernoulli's bound leaves >2^-100.
    # Every denominator is in [epsilon,4*(1+2^-18)], so reciprocal and quotient
    # are normal and finite. NaNs propagate; no finite value uses a fallback.
    return dict(matrix=[4, 4], full_iterations=20, epsilon=1e-6,
                finite_minimum_quotient_power=-100,
                finite_denominator_minimum_power=-20,
                finite_denominator_maximum=4 * (1 + 2 ** -18),
                original_pre_post_and_initial_softmax_divisions_preserved=True)


def describe(package):
    calls = [c for c in package['render_calls']
             if c['family'] == 'mhc_pre' and c['variant'] == 'cooperative_stream_fma']
    proofs = [domain_certificate(c) for c in calls]
    package['implementation_override']['inputs']['sinkhorn_normal_division.py'] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    package['declared_stabilized_division_domain'] = proofs
    return package
