# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Candidate: register-local MHC and one output per warp for narrow GEMV."""
import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

from . import mhc  # noqa: F401 -- install the preceding kernel policy

original_render = TritonTemplateRegistry.render


def override(self, name, context):
    if name != 'module.py.jinja':
        return original_render(self, name, context)
    key = 'kernels/dense_matmul/_function.py.jinja'
    source = (self.root/key).read_text()
    marker = '{{ entry.kernel_decorator(call) }}'
    assert source.count(marker) == 1
    condition = '''{% set narrow_warp_rows = (not dense_matmul_uses_descriptor)
    and call.variant in ("packed_k_major_gemv", "split_k_n_packed_k_major_gemv")
    and call.tile_n >= num_warps and call.block_k >= 32 %}
{% if narrow_warp_rows %}
import triton.experimental.tle.language as tle
{{ call.symbol }}__warp_rows = tl.constexpr(tle.gpu.BlockEncoding(
    [1, 1], [1, 32], [{{ num_warps }}, 1], [1, 0],
))
{% endif %}
'''
    source = source.replace(marker, condition + marker)
    marker = '''            dense_accumulator += (
                dense_weight_value.to(tl.float32) * dense_source_value[None, :].to(tl.float32)
'''
    assert source.count(marker) == 1
    source = source.replace(marker, '''{% if narrow_warp_rows %}
            dense_weight_value = tle.gpu.set_layout(dense_weight_value, {{ call.symbol }}__warp_rows)
{% endif %}
''' + marker)
    self.environment.loader = ChoiceLoader([DictLoader({key: source}), self.environment.loader])
    print(json.dumps({'trial': 'SIMT GEMV maps output rows across warps; K reduction stays within warp',
                      'override_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), flush=True)
    return original_render(self, name, context)


TritonTemplateRegistry.render = override
