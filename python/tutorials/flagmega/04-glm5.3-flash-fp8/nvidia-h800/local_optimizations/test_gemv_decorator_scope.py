# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""GEMV warp-row layout must preserve the selected entry decorator options."""
import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

from . import gemv


@pytest.mark.parametrize('decorator', [
    '{{ entry.kernel_decorator(call) }}',
    '{{ entry.kernel_decorator(call, prefer_inline=true) }}',
])
def test_warp_rows_preserve_decorator_arguments(monkeypatch, tmp_path, decorator):
    registry = TritonTemplateRegistry()
    key = 'kernels/dense_matmul/_function.py.jinja'
    source = (registry.root / key).read_text()
    current = '{{ entry.kernel_decorator(call, prefer_inline=true) }}'
    assert source.count(current) == 1
    source = source.replace(current, decorator)
    path = tmp_path / key
    path.parent.mkdir(parents=True)
    path.write_text(source)
    registry.root = tmp_path
    monkeypatch.setattr(gemv, 'original_prepare', lambda *args: None)
    gemv.prepare(registry, 'module.py.jinja', {})
    transformed = registry.environment.loader.get_source(registry.environment, key)[0]
    assert transformed.count(decorator) == 1
    assert transformed.index('__warp_rows = tl.constexpr') < transformed.index(decorator)
    assert 'dense_weight_value = tle.gpu.set_layout' in transformed
    layout = '{% if narrow_warp_rows %}\n' \
             '            dense_weight_value = tle.gpu.set_layout(dense_weight_value, {{ call.symbol }}__warp_rows)\n' \
             '{% endif %}\n'
    without_layout = transformed.replace(layout, '')
    begin = without_layout.index('{% set narrow_warp_rows =')
    end = without_layout.index(decorator, begin)
    assert without_layout[:begin] + without_layout[end:] == source


def test_non_module_template_keeps_original_prepare(monkeypatch):
    registry = TritonTemplateRegistry()
    expected = object()
    monkeypatch.setattr(gemv, 'original_prepare', lambda *args: expected)
    assert gemv.prepare(registry, 'kernels/dense_matmul/packed_k_major_gemv.py.jinja', {}) is expected
