# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Candidate: warp-local reduction for small partial-Boxing matrices."""
import hashlib
import json
from pathlib import Path

from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

from . import gemv  # noqa: F401 -- install the preceding kernel policy

original_render = TritonTemplateRegistry.render


def override(self, name, context):
    if name != 'module.py.jinja':
        return original_render(self, name, context)
    key = 'kernels/distributed_boxing/_function.py.jinja'
    source = (self.root/key).read_text()
    prefix = '''import triton.experimental.tle.language as tle
{% for rows in (1, 2, 4, 8, 16, 32) %}
FLAGMEGA_BOXING_PARTIAL_LAYOUT_R{{ rows }} = tl.constexpr(tle.gpu.BlockEncoding(
    [1, 1], [{{ rows }}, {{ 32 // rows }}], [1, {{ num_warps }}], [1, 0],
))
{% endfor %}
'''
    source = prefix + source
    marker = '    boxing_reduced = {{ leaf.reduction_expression }}'
    assert source.count(marker) == 1
    source = source.replace(marker, '''{% if leaf.partial_owner_tile <= 32 and leaf.partial_owner_tile * leaf.tile <= num_warps * 32 %}
    boxing_partial_values = tle.gpu.set_layout(
        boxing_partial_values, FLAGMEGA_BOXING_PARTIAL_LAYOUT_R{{ leaf.partial_owner_tile }},
    )
{% endif %}
''' + marker)
    self.environment.loader = ChoiceLoader([DictLoader({key: source}), self.environment.loader])
    print(json.dumps({'trial': 'small partial Boxing reduces owner rows within a warp',
                      'override_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), flush=True)
    return original_render(self, name, context)


TritonTemplateRegistry.render = override
