# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One warp per coefficient row; convert only after coefficient projection."""
import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

original_render = TritonTemplateRegistry.render


def override(self, name, context):
    if name != 'module.py.jinja':
        return original_render(self, name, context)
    key = 'kernels/mhc_pre/local.py.jinja'
    source = (self.root/key).read_text()
    marker = '{{ entry.kernel_decorator(call) }}'
    source = source.replace(marker, '''import triton.experimental.tle.language as tle
{{ call.symbol }}__prepost_layout = tl.constexpr(tle.gpu.BlockEncoding(
    [1, 1], [1, 32], [{{ num_warps }}, 1], [1, 0],
))
{{ call.symbol }}__prepost_result = tl.constexpr(tle.gpu.SlicedEncoding(
    1, {{ call.symbol }}__prepost_layout.value,
))
{{ call.symbol }}__prepost_input = tl.constexpr(tle.gpu.SlicedEncoding(
    0, {{ call.symbol }}__prepost_layout.value,
))
{{ call.symbol }}__comb_projection = tl.constexpr(tle.gpu.BlockEncoding(
    [1, 1, 1], [1, 1, 32], [4, {{ num_warps // 4 }}, 1], [2, 1, 0],
))
{{ call.symbol }}__comb_accumulator = tl.constexpr(tle.gpu.SlicedEncoding(
    2, {{ call.symbol }}__comb_projection.value,
))
{{ call.symbol }}__comb_input = tl.constexpr(tle.gpu.SlicedEncoding(
    0, tle.gpu.SlicedEncoding(1, {{ call.symbol }}__comb_projection.value),
))
{{ call.symbol }}__coefficient_lanes = tl.constexpr(tle.gpu.BlockEncoding(
    [1], [32], [{{ num_warps }}], [0],
))
{{ call.symbol }}__collapse = tl.constexpr(tle.gpu.BlockEncoding(
    [1, 1], [4, 8], [1, {{ num_warps }}], [1, 0],
))
{{ call.symbol }}__sinkhorn = tl.constexpr(tle.gpu.BlockEncoding(
    [1, 1], [4, 8], [1, {{ num_warps }}], [1, 0],
))
''' + marker)
    marker = '    _fm_pair_active = _fm_stream_active[:, None] & _fm_stream_active[None, :]'
    source = source.replace(marker, marker + '''
    _fm_g = tl.arange(0, {{ 2 * call.stream_lanes }})
    _fm_g_row = tl.where(_fm_g < {{ call.stream_lanes }}, _fm_g,
                         _fm_g - {{ call.stream_lanes }} + {{ call.hc }})
    _fm_g_active = tl.where(_fm_g < {{ call.stream_lanes }},
                            _fm_g < {{ call.hc }},
                            _fm_g - {{ call.stream_lanes }} < {{ call.hc }})
''')
    old = '''        _fm_pre_mix = tl.zeros(({{ call.stream_lanes }}, ), tl.float32)
        _fm_post_mix = tl.zeros(({{ call.stream_lanes }}, ), tl.float32)'''
    new = '''        _fm_prepost_mix = tle.gpu.set_layout(
            tl.zeros(({{ 2 * call.stream_lanes }},), tl.float32),
            {{ call.symbol }}__prepost_result,
        )'''
    assert old in source
    source = source.replace(old, new)
    marker = '        for _fm_start in tl.range(0, {{ call.elements }}, {{ call.matvec_tile }}):'
    source = source.replace(marker, '''        _fm_comb_mix = tle.gpu.set_layout(_fm_comb_mix, {{ call.symbol }}__comb_accumulator)
''' + marker)
    marker = '            _fm_square_sum +='
    source = source.replace(marker, """            _fm_values_prepost = tle.gpu.set_layout(_fm_values, {{ call.symbol }}__prepost_input)
            _fm_values_comb = tle.gpu.set_layout(_fm_values, {{ call.symbol }}__comb_input)
""" + marker)
    begin = source.index('            _fm_row_active = ')
    end = source.index('        # RMS without a mean term', begin)
    source = source[:begin] + '''            _fm_prepost_weights = tle.gpu.set_layout(
                tl.load({{ call.projection }} + ({{ call.projection_pre_offset | replace('_fm_i[:, None]', '_fm_g_row[:, None]') }}),
                        mask=_fm_g_active[:, None] & _fm_flat_active[None, :],
                        other=0.).to(tl.float32), {{ call.symbol }}__prepost_layout)
            _fm_prepost_mix += tl.sum(_fm_prepost_weights * _fm_values_prepost[None, :], 1)
            _fm_comb_weights = tle.gpu.set_layout(
                tl.load({{ call.projection }} + ({{ call.projection_comb_offset }}),
                        mask=_fm_pair_active[:, :, None] & _fm_flat_active[None, None, :],
                        other=0.).to(tl.float32), {{ call.symbol }}__comb_projection)
            _fm_comb_mix += tl.sum(_fm_comb_weights * _fm_values_comb[None, None, :], 2)
        _fm_prepost_coefficients = tle.gpu.set_layout(
            _fm_prepost_mix, {{ call.symbol }}__coefficient_lanes)
        _fm_pre_mix = tl.gather(_fm_prepost_coefficients, _fm_i, 0)
        _fm_post_mix = tl.gather(_fm_prepost_coefficients, _fm_i + {{ call.stream_lanes }}, 0)
''' + source[end:]
    begin = source.index('        # Softmax over the sink axis.')
    end = source.index('        tl.store(', begin)
    sinkhorn = source[begin:end].replace('_fm_pair_active', '_fm_sink_pair_active')
    sinkhorn = sinkhorn.replace('_fm_stream_active[:, None]', '_fm_sink_row_active')
    source = source[:begin] + '''        _fm_comb = tle.gpu.set_layout(_fm_comb, {{ call.symbol }}__sinkhorn)
        _fm_sink_pair_active = tle.gpu.set_layout(_fm_pair_active, {{ call.symbol }}__sinkhorn)
        _fm_sink_row_active = tle.gpu.set_layout(_fm_stream_active[:, None], {{ call.symbol }}__sinkhorn)
''' + sinkhorn + source[end:]
    source = source.replace('        # Pass two: collapse the streams', """        _fm_collapse_pre = tle.gpu.set_layout(_fm_pre[:, None], {{ call.symbol }}__collapse)
        # Pass two: collapse the streams""")
    source = source.replace('_fm_pre[:, None] * _fm_plane', '_fm_collapse_pre * _fm_plane')
    self.environment.loader = ChoiceLoader([DictLoader({key: source}), self.environment.loader])
    print(json.dumps({'trial': 'coefficient rows across warps, K across lanes, packed pre/post projection',
                      'override_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), flush=True)
    return original_render(self, name, context)


TritonTemplateRegistry.render = override
