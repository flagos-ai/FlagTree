# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Experimental: explicit CTA shared coefficients preserve ordered FMA reductions."""
import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

original_render = TritonTemplateRegistry.render


def projection_template(source, row_count=8):
    assert row_count == 8
    source = source.replace('_flagmega_mhc_project_pair', '_flagmega_mhc_project_shared')
    source = source.replace('streams, projection, row_start,', 'streams, projection, coefficients, row_start,')
    source = source.replace('rows = row_start + tl.arange(0, 2)', 'rows = row_start + tl.arange(0, 8)')
    source = source.replace('tl.zeros((2, SPLITS, 128), tl.float32)', 'tl.zeros((8, SPLITS, 128), tl.float32)')
    before = '''    first = tl.sum(tl.gather(result, tl.full((1,), 0, tl.int32), 0), 0)
    second = tl.sum(tl.gather(result, tl.full((1,), 1, tl.int32), 0), 0)
    return first, second'''
    assert source.count(before) == 1
    source = source.replace(before, '    tl.store(coefficients + rows, result, rows < COUNT)')
    before = '''    mixes = tle.gpu.set_layout(tl.zeros((MIX_LANES,), tl.float32), MIX_LAYOUT)
    for start in tl.range(0, COUNT, 2):
        first, second = _flagmega_mhc_project_shared(
            streams, projection, start, HC, HIDDEN, SPLITS,'''
    assert source.count(before) == 1
    source = source.replace(before, '''    storage = tle.gpu.alloc((MIX_LANES,), tl.float32, nv_mma_shared_layout=False)
    coefficients = tl.reshape(tle.gpu.local_ptr(storage, (tl.arange(0, 1),), [1]), ())
    for start in tl.range(0, COUNT, 8):
        _flagmega_mhc_project_shared(
            streams, projection, coefficients, start, HC, HIDDEN, SPLITS,''')
    before = '''        mixes = tl.where(tl.arange(0, MIX_LANES) == start, first, mixes)
        mixes = tl.where(tl.arange(0, MIX_LANES) == start + 1, second, mixes)
    return mixes, square_sum'''
    assert source.count(before) == 1
    source = source.replace(before, '''    tl.debug_barrier()
    indices = tle.gpu.set_layout(tl.arange(0, MIX_LANES), MIX_LAYOUT)
    mixes = tl.load(coefficients + indices, indices < COUNT, other=0.)
    return mixes, square_sum''')
    return source

def override(self, name, context):
    if name != 'module.py.jinja':
        return original_render(self, name, context)
    original_render(self, name, context)
    key = 'kernels/mhc_pre/_stream_projection.py.jinja'
    source = self.environment.loader.get_source(self.environment, key)[0]
    projection = projection_template(source)
    key = 'kernels/mhc_pre/stream_fma.py.jinja'
    source = self.environment.loader.get_source(self.environment, key)[0]
    before = '    [{{ [num_warps // 4, 2] | min }}, {{ num_warps // (4 * ([num_warps // 4, 2] | min)) }}, 4], [2, 1, 0],'
    assert source.count(before) == 1
    source = source.replace(before, '''{% if mhc_projection_variant | default("stream_fma") == "stream_fma" %}
    [{{ num_warps }}, 1, 1], [2, 1, 0],
{% else %}
''' + before + '\n{% endif %}')
    self.environment.loader = ChoiceLoader([
        DictLoader({'kernels/mhc_pre/_stream_projection.py.jinja': projection, key: source}),
        self.environment.loader,
    ])
    print(json.dumps(dict(trial='eight mHC rows via declared CTA shared coefficients; ordered reductions',
                          override_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())), flush=True)
    return self.environment.get_template(name).render(**context)


TritonTemplateRegistry.render = override
