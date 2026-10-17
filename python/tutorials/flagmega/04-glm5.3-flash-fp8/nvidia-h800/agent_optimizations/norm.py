# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Candidate: explicit two-stage finalization for one scalar per CTA."""
import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

from . import boxing  # noqa: F401 -- install the preceding kernel policy

original_render = TritonTemplateRegistry.render


def override(self, name, context):
    if name != 'module.py.jinja':
        return original_render(self, name, context)
    key = 'kernels/norm_stats/_function.py.jinja'
    source = (self.root/key).read_text()
    helper = '''import triton.experimental.tle.language as tle
FLAGMEGA_NORM_WARP_ROWS = tl.constexpr(tle.gpu.BlockEncoding(
    [1, 1], [1, 32], [{{ num_warps }}, 1], [1, 0],
))
FLAGMEGA_NORM_REPLICATED_PARTIALS = tl.constexpr(tle.gpu.SlicedEncoding(
    0, FLAGMEGA_NORM_WARP_ROWS.value,
))

@triton.jit
def _flagmega_norm_stats_finalize(values):
    if values.shape[0] >= 32 and values.shape[0] <= {{ num_warps * 32 }}:
        rows = tl.reshape(values, (values.shape[0] // 32, 32))
        rows = tle.gpu.set_layout(rows, FLAGMEGA_NORM_WARP_ROWS)
        partials = tl.sum(rows, axis=1)
        partials = tle.gpu.set_layout(partials, FLAGMEGA_NORM_REPLICATED_PARTIALS)
        return tl.sum(partials, axis=0)
    else:
        return tl.sum(values, axis=0)

'''
    for value in ('norm_sum', 'norm_square_sum'):
        original = 'tl.sum(' + value + ', axis=0)'
        source = source.replace(original,
            '{% if call.norm_stats_reduction_layout | default("default") == "default" %}'
            + '_flagmega_norm_stats_finalize(' + value + ')'
            + '{% else %}' + original + '{% endif %}')
    assert '_flagmega_norm_stats_finalize(norm_square_sum)' in source
    source = helper + source
    self.environment.loader = ChoiceLoader([DictLoader({key: source}), self.environment.loader])
    print(json.dumps({'trial': 'norm finalization gathers warp partials before the last warp-local reduction',
                      'override_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), flush=True)
    return original_render(self, name, context)


TritonTemplateRegistry.render = override
